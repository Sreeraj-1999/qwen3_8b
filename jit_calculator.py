"""
JIT Arrival Calculator
======================
Pure math — no ML, no external APIs.
Takes vessel state + ETB, returns speed recommendation and fuel saving.
"""

from datetime import datetime, timezone
import math

import json
import os

def get_vessel_config(imo):
    try:
        config_path = os.path.join(os.path.dirname(__file__), "vessel_config.json")
        with open(config_path) as f:
            configs = json.load(f)
        return configs.get(str(imo), {
            "fuel_price_usd_per_tonne": 600,
            "min_speed_knots": 8.0,
            "max_speed_knots": 22.0,
            "design_speed_knots": 14.0
        })
    except:
        return {
            "fuel_price_usd_per_tonne": 600,
            "min_speed_knots": 8.0,
            "max_speed_knots": 22.0,
            "design_speed_knots": 14.0
        }


def calculate_distance_nm(lat1, lon1, lat2, lon2):
    """
    Haversine formula — distance between two lat/lon points in nautical miles
    """
    R = 3440.065  # Earth radius in nautical miles

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return round(R * c, 2)


def calculate_fuel_at_speed(base_fuel, base_speed, new_speed):
    """
    Cubic law — fuel consumption scales with speed cubed.
    base_fuel  : current fuel burn in kg/h
    base_speed : current speed in knots
    new_speed  : target speed in knots
    returns    : estimated fuel burn at new speed in kg/h
    """
    if base_speed <= 0:
        return base_fuel
    return round(base_fuel * (new_speed / base_speed) ** 3, 2)


def calculate_berth_confidence(etb_datetime, hours_until_etb):
    """
    Simple rule-based confidence scoring.
    In future this will use voyage plan historical accuracy per port.
    """
    # Far out = less confident
    if hours_until_etb > 48:
        base_confidence = 55
    elif hours_until_etb > 24:
        base_confidence = 68
    elif hours_until_etb > 12:
        base_confidence = 78
    elif hours_until_etb > 6:
        base_confidence = 87
    else:
        base_confidence = 93

    return base_confidence


def run_jit_calculation(
    imo,
    current_lat,
    current_lon,
    current_speed,       # SOG in knots — if 0 use sailing average
    current_fuel,        # ME fuel kg/h — if None use sailing average
    avg_speed,           # sailing average SOG
    avg_fuel,            # sailing average fuel kg/h
    destination_lat,
    destination_lon,
    etb_iso,             # ETB as ISO string e.g. "2026-03-10T04:00:00Z"
):
    """
    Main JIT calculation.
    Returns a dict with recommendation, speeds, fuel saving, confidence.
    """
    config = get_vessel_config(imo)
    FUEL_PRICE_USD_PER_TONNE = config["fuel_price_usd_per_tonne"]
    MIN_SPEED_KNOTS = config["min_speed_knots"]
    MAX_SPEED_KNOTS = config["max_speed_knots"]
    result = {}

    # --- Use averages if current values unavailable (ship in port) ---
    speed = current_speed if (current_speed and current_speed > 2) else avg_speed
    fuel  = current_fuel  if (current_fuel  and current_fuel  > 0) else avg_fuel

    if not speed or not fuel:
        return {"error": "Insufficient speed/fuel data for JIT calculation"}

    # --- Parse ETB ---
    try:
        etb = datetime.fromisoformat(etb_iso.replace("Z", "+00:00"))
    except Exception as e:
        return {"error": f"Invalid ETB format: {e}. Use ISO format e.g. 2026-03-10T04:00:00Z"}

    now = datetime.now(timezone.utc)
    hours_until_etb = (etb - now).total_seconds() / 3600

    if hours_until_etb <= 0:
        return {"error": "ETB is in the past"}

    if hours_until_etb < 1:
        return {"error": "Less than 1 hour to ETB — too late for JIT adjustment"}

    # --- Distance to port ---
    distance_nm = calculate_distance_nm(
        current_lat, current_lon,
        destination_lat, destination_lon
    )

    # --- Required speed ---
    required_speed = round(distance_nm / hours_until_etb, 2)
    confidence = calculate_berth_confidence(etb, hours_until_etb)

    # --- Check feasibility ---
    if required_speed < MIN_SPEED_KNOTS:
        recommendation = "SLOW_DOWN"
        note = f"Required speed {required_speed} kn is below minimum {MIN_SPEED_KNOTS} kn. Consider delaying departure or notifying port of early arrival."
        required_speed = MIN_SPEED_KNOTS
    elif required_speed > MAX_SPEED_KNOTS:
        recommendation = "IMPOSSIBLE"
        note = f"Required speed {required_speed} kn exceeds vessel maximum {MAX_SPEED_KNOTS} kn. Cannot make ETB — notify port agent to revise ETB or reroute."
        # Don't run fuel math on impossible speed
        return {
            "recommendation": "IMPOSSIBLE",
            "note": note,
            "required_speed_kn": required_speed,
            "max_speed_kn": MAX_SPEED_KNOTS,
            "distance_to_port_nm": distance_nm,
            "hours_until_etb": round(hours_until_etb, 2),
            "berth_confidence_pct": 0,
            "anchorage_risk_pct": 0,
            "etb": etb_iso,
            "calculated_at": now.isoformat()
        }
        
    elif required_speed < speed - 0.5:
        recommendation = "SLOW_DOWN"
        note = f"Reduce speed from {speed} kn to {required_speed} kn to arrive just-in-time."
    elif required_speed > speed + 0.5:
        recommendation = "SPEED_UP"
        note = f"Increase speed from {speed} kn to {required_speed} kn to make ETB."
    else:
        recommendation = "MAINTAIN"
        note = f"Current speed {speed} kn is optimal for JIT arrival. No adjustment needed."

    # --- Fuel saving calculation ---
    fuel_at_required  = calculate_fuel_at_speed(fuel, speed, required_speed)
    fuel_diff_per_hour = fuel - fuel_at_required         # kg/h saved (negative = extra burn)
    total_fuel_saving_kg   = round(fuel_diff_per_hour * hours_until_etb, 1)
    total_fuel_saving_tonnes = round(total_fuel_saving_kg / 1000, 2)
    fuel_saving_usd = round(total_fuel_saving_tonnes * FUEL_PRICE_USD_PER_TONNE, 0)

    # --- Confidence ---
    

    # --- Anchorage risk if ignored ---
    # Simple heuristic — if arriving significantly early, high anchorage risk
    eta_at_current_speed = round(distance_nm / speed, 2)  # hours
    early_by_hours = round(hours_until_etb - eta_at_current_speed, 2)
    
    if early_by_hours > 4:
        anchorage_risk = 85
    elif early_by_hours > 2:
        anchorage_risk = 65
    elif early_by_hours > 1:
        anchorage_risk = 45
    elif early_by_hours > 0:
        anchorage_risk = 25
    else:
        anchorage_risk = 10  # arriving late — no anchorage risk

    result = {
        "recommendation":          recommendation,
        "note":                    note,
        "current_speed_kn":        speed,
        "required_speed_kn":       required_speed,
        "distance_to_port_nm":     distance_nm,
        "hours_until_etb":         round(hours_until_etb, 2),
        "eta_at_current_speed_h":  eta_at_current_speed,
        "early_by_hours":          early_by_hours if early_by_hours > 0 else 0,
        "fuel_at_current_speed_kgph":  fuel,
        "fuel_at_required_speed_kgph": fuel_at_required,
        "fuel_saving_kg":          total_fuel_saving_kg,
        "fuel_saving_tonnes":      total_fuel_saving_tonnes,
        "fuel_saving_usd":         fuel_saving_usd,
        "berth_confidence_pct":    confidence,
        "anchorage_risk_pct":      anchorage_risk,
        "etb":                     etb_iso,
        "calculated_at":           now.isoformat(),
    }

    return result


# --- Quick test ---
if __name__ == "__main__":
    result = run_jit_calculation(
        imo="9841938",
        current_lat=29.82,
        current_lon=-90.00,
        current_speed=0,
        current_fuel=None,
        avg_speed=11.84,
        avg_fuel=511.91,
        destination_lat=25.77,
        destination_lon=-80.19,
        etb_iso="2026-03-10T04:00:00Z"
    )
    import json
    print(json.dumps(result, indent=2))