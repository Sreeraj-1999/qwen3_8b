"""
Offline Location Resolver for Maritime Vessels
===============================================
Resolves lat/lon → human-readable location string WITHOUT any external API.

Uses:
  1. Ocean/sea zone polygonal boundaries for zone identification
  2. Nearest port from ports.json using Haversine
  3. Cardinal bearing to nearest port

Output example: "North Atlantic Ocean, 165 NM ESE of Port Canaveral"

Drop-in replacement for coords_to_place() in mcp_telemetry.py
"""

import math
import json
import os
import logging

logger = logging.getLogger(__name__)

# ============================================================
# HAVERSINE — distance in nautical miles
# ============================================================

def _haversine_nm(lat1, lon1, lat2, lon2):
    """Distance between two points in nautical miles"""
    R = 3440.065  # Earth radius in NM
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return round(R * 2 * math.asin(math.sqrt(a)), 1)


def _bearing(lat1, lon1, lat2, lon2):
    """Initial bearing from point1 to point2 in degrees (0-360)"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def _bearing_to_cardinal(bearing):
    """Convert bearing degrees to 16-point compass direction"""
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    idx = round(bearing / 22.5) % 16
    return directions[idx]


def _reverse_cardinal(cardinal):
    """Reverse a cardinal direction — if port is NE of vessel, vessel is SW of port"""
    opposites = {
        "N": "S", "S": "N", "E": "W", "W": "E",
        "NE": "SW", "SW": "NE", "NW": "SE", "SE": "NW",
        "NNE": "SSW", "SSW": "NNE", "NNW": "SSE", "SSE": "NNW",
        "ENE": "WSW", "WSW": "ENE", "ESE": "WNW", "WNW": "ESE",
    }
    return opposites.get(cardinal, cardinal)


# ============================================================
# OCEAN / SEA ZONE DEFINITIONS
# ============================================================
# Each zone is defined as a bounding box [min_lat, max_lat, min_lon, max_lon]
# or a polygon (list of [lat, lon] vertices) for complex shapes.
#
# Order matters — more specific zones (seas, gulfs) come FIRST so they
# match before the broad ocean fallback.
# ============================================================

# Point-in-polygon test (ray casting)
def _point_in_polygon(lat, lon, polygon):
    """Ray casting algorithm for point-in-polygon test"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# Zone definitions: list of (name, polygon_or_bbox)
# Using polygons for accuracy on complex coastlines, bounding boxes for open ocean
MARITIME_ZONES = [
    # === ENCLOSED / SEMI-ENCLOSED SEAS (check first — most specific) ===

    ("Strait of Hormuz", [
        [25.5, 56.0], [26.5, 56.0], [27.0, 56.5], [26.5, 57.0],
        [25.5, 57.0], [25.0, 56.5],
    ]),

    ("Persian Gulf", [
        [30.0, 48.0], [29.9, 49.5], [28.5, 50.0], [27.0, 50.0],
        [26.0, 51.5], [25.0, 52.5], [24.0, 53.5], [23.5, 55.0],
        [24.5, 57.0], [25.5, 57.5], [26.5, 57.0], [27.0, 56.5],
        [28.0, 56.0], [29.0, 53.0], [30.0, 50.0], [30.5, 48.5],
    ]),

    ("Gulf of Oman", [
        [23.5, 55.0], [22.5, 58.0], [22.0, 60.5], [24.0, 62.0],
        [26.0, 62.0], [26.5, 57.0], [25.5, 57.5], [24.5, 57.0],
    ]),
    ("Bab el-Mandeb Strait", [
        [11.0, 42.5], [13.2, 42.5], [13.8, 45.0], [12.0, 45.0],
        [10.8, 44.0],
    ]),
    

    ("Red Sea", [
        [12.5, 43.0], [13.0, 43.0], [15.0, 41.0], [18.0, 38.5],
        [20.0, 37.0], [22.0, 36.0], [24.0, 35.5], [27.0, 34.0],
        [28.5, 33.0], [30.0, 32.5], [30.0, 34.5], [28.0, 35.0],
        [27.0, 36.0], [24.0, 38.0], [20.0, 40.0], [16.0, 42.5],
        [13.5, 44.0], [12.0, 44.5],
    ]),
    ("Strait of Gibraltar", [
        [35.7, -6.0], [36.3, -5.8], [36.3, -5.0], [35.7, -5.0],
    ]),

    ("Mediterranean Sea", [
        [30.0, -6.0], [35.5, -6.0], [37.0, -5.5], [38.0, -2.0],
        [41.0, 1.0], [43.5, 3.0], [43.5, 8.0], [44.0, 12.0],
        [42.0, 16.0], [40.0, 19.0], [39.5, 20.0], [38.5, 21.0],
        [36.5, 22.0], [35.0, 24.5], [35.0, 27.0], [36.0, 29.0],
        [37.0, 36.0], [36.5, 36.0], [34.5, 36.0], [33.0, 35.5],
        [32.0, 34.5], [31.5, 32.0], [31.0, 30.0], [30.0, 25.0],
    ]),

    ("Black Sea", [
        [41.0, 27.5], [41.5, 28.5], [42.0, 29.5], [43.0, 33.0],
        [44.0, 34.5], [45.5, 36.5], [46.5, 38.5], [46.0, 40.0],
        [44.0, 41.5], [42.5, 41.5], [41.5, 41.0], [41.0, 40.0],
        [41.0, 37.0], [41.5, 33.0], [41.0, 29.0],
    ]),

    ("Gulf of Finland", [
        [58.5, 21.5], [59.0, 23.0], [59.5, 24.5], [60.0, 26.0],
        [60.5, 28.0], [61.5, 28.0], [61.5, 23.0], [60.5, 21.0],
        [59.5, 21.0],
    ]),

    ("Baltic Sea", [
    [53.5, 9.5], [54.5, 12.0], [55.0, 14.0], [55.5, 16.0],
    [56.0, 19.5], [57.0, 20.0], [58.0, 20.5], [59.0, 21.0],
    [60.0, 21.0], [61.0, 21.5], [63.0, 19.5], [64.0, 21.0],
    [65.0, 25.5], [66.0, 26.0], [66.0, 23.0], [65.0, 21.0],
    [64.0, 18.0], [63.0, 17.0], [61.0, 17.0], [59.0, 16.5],
    [57.0, 13.0], [55.5, 10.5], [54.0, 9.5],
    ]),

    ("North Sea", [
        [51.0, -2.0], [51.5, 1.0], [53.0, 4.0], [54.0, 6.0],
        [55.5, 8.0], [57.5, 8.0], [58.5, 6.0], [60.0, 3.0],
        [61.5, 0.0], [61.0, -3.0], [58.0, -4.0], [56.0, -3.0],
        [54.0, -1.0], [52.0, 0.0],
    ]),

    ("English Channel", [
        [48.5, -6.0], [49.0, -5.0], [49.5, -3.0], [50.0, -1.0],
        [50.5, 0.5], [51.0, 1.5], [51.5, 2.0], [51.0, 2.0],
        [50.5, 1.5], [49.5, -1.0], [48.5, -5.0],
    ]),

    ("Gulf of Mexico", [
        [18.0, -98.0], [18.5, -95.0], [19.5, -92.0], [21.0, -87.0],
        [22.0, -84.0], [23.5, -83.0], [25.0, -81.0], [27.0, -80.0],
        [29.0, -82.0], [30.0, -84.0], [30.5, -88.0], [30.0, -90.0],
        [29.5, -93.5], [28.0, -96.0], [26.0, -97.5], [22.0, -98.0],
    ]),

    ("Caribbean Sea", [
        [8.0, -84.0], [9.5, -83.0], [11.0, -82.0], [12.5, -81.0],
        [14.0, -78.0], [16.0, -73.0], [18.0, -68.0], [18.5, -64.0],
        [18.0, -60.0], [15.0, -60.0], [12.0, -61.0], [10.0, -62.0],
        [9.0, -63.0], [10.5, -67.0], [12.0, -72.0], [11.0, -76.0],
        [9.5, -80.0],
    ]),
    

    ("South China Sea", [
        [-3.0, 100.0], [3.0, 105.0], [7.0, 108.0],
        [10.0, 109.0], [15.0, 110.0], [18.0, 111.0], [21.0, 113.0],
        [23.0, 117.0], [22.0, 121.0], [18.0, 122.0], [14.0, 121.0],
        [10.0, 119.0], [7.0, 117.0], [5.0, 115.0], [3.0, 110.0],
        [0.0, 107.0], [-2.0, 105.0], [-3.0, 103.0],
    ]),
    ("Taiwan Strait", [
    [22.5, 117.0], [23.5, 117.5], [25.0, 119.0], [26.5, 120.5],
    [26.5, 122.0], [24.5, 122.0], [23.0, 120.0], [22.0, 118.0],
    ]),

    ("East China Sea", [
        [23.0, 117.0], [25.0, 120.0], [26.0, 122.0], [28.0, 123.0],
        [30.0, 124.0], [32.0, 125.5], [33.0, 128.0], [31.0, 130.0],
        [29.0, 129.0], [27.0, 127.0], [25.0, 123.0], [24.0, 121.0],
    ]),

    ("Sea of Japan", [
    [34.0, 127.0], [35.0, 128.0], [37.0, 128.0], [39.0, 129.0],
    [41.0, 130.5], [43.0, 132.0], [46.0, 137.0], [48.0, 140.0],
    [52.0, 141.5], [51.0, 143.0], [46.0, 142.0], [43.0, 140.5],
    [40.0, 140.0], [37.0, 138.0], [35.0, 134.0], [34.0, 131.0],
]),

    ("Bay of Bengal", [
        [5.5, 80.0], [6.0, 82.0], [8.0, 80.0], [10.0, 79.5],
        [13.0, 80.0], [15.0, 80.0], [17.0, 82.0], [19.0, 85.0],
        [21.0, 87.5], [22.5, 89.0], [23.0, 91.0], [22.0, 92.0],
        [20.0, 93.0], [16.0, 95.0], [14.0, 96.0], [10.0, 98.0],
        [7.0, 97.5], [5.0, 95.0], [3.0, 92.0], [5.0, 85.0],
    ]),
    ("Gulf of Aden", [
        [10.5, 43.0], [11.5, 43.5], [12.0, 45.0], [12.5, 47.0],
        [13.5, 49.0], [14.5, 51.0], [15.5, 53.0], [15.0, 54.0],
        [12.0, 53.0], [10.5, 49.0], [10.0, 46.0], [10.0, 43.5],
    ]),

    ("Arabian Sea", [
        [5.0, 50.0], [8.0, 51.0], [10.0, 52.0], [12.5, 44.0],
        [14.5, 43.0], [16.0, 42.5], [20.0, 58.0], [23.0, 62.0],
        [24.0, 64.0], [25.0, 67.0], [23.0, 68.5], [20.0, 72.0],
        [15.0, 74.0], [10.0, 76.0], [8.0, 77.0], [5.0, 76.0],
        [2.0, 72.0], [0.0, 65.0], [0.0, 55.0], [3.0, 50.0],
    ]),

    ("Strait of Malacca", [
        [0.5, 103.0], [1.0, 103.5], [1.5, 103.0], [2.5, 102.0],
        [4.0, 100.0], [5.5, 98.0], [6.5, 96.0], [5.5, 95.0],
        [4.0, 98.0], [2.5, 100.5], [1.0, 102.5],
    ]),

    ("Suez Canal Approaches", [
        [29.5, 32.0], [30.5, 32.0], [31.5, 32.0], [31.5, 33.5],
        [30.5, 33.5], [29.5, 33.0],
    ]),

    ("Panama Canal Approaches", [
        [8.5, -80.0], [9.5, -80.0], [9.5, -79.0], [8.5, -79.0],
    ]),

    

    

    ("Sunda Strait", [
        [-5.5, 104.5], [-5.5, 106.5], [-7.0, 106.5], [-7.0, 104.5],
    
    ]),

    ("Lombok Strait", [
        [-8.0, 115.2], [-8.0, 116.0], [-8.8, 116.0], [-8.8, 115.2],
    ]),

    

    ("Drake Passage", [
        [-55.0, -70.0], [-55.0, -55.0], [-62.0, -55.0], [-62.0, -70.0],
    ]),

    ("Bering Strait", [
        [65.0, -170.0], [65.0, -168.0], [66.5, -168.0], [66.5, -170.0],
    ]),

    ("Mozambique Channel", [
        [-11.0, 39.0], [-12.0, 40.5], [-15.0, 41.0], [-18.0, 42.0],
        [-22.0, 43.0], [-25.5, 44.5], [-26.0, 40.0], [-23.0, 36.0],
        [-20.0, 35.0], [-16.0, 36.5], [-13.0, 38.0],
    ]),

    
    ("Barents Sea", [
    [70.0, 15.0], [70.0, 60.0], [80.0, 60.0], [80.0, 15.0],
    ]),

    ("Baffin Bay", [
        [68.0, -80.0], [68.0, -55.0], [78.0, -55.0], [78.0, -80.0],
    ]),

    ("Hawaiian Waters", [
        [18.0, -162.0], [18.0, -154.0], [23.0, -154.0], [23.0, -162.0],
    ]),

    ("Fiji Waters", [
        [-21.0, 174.0], [-21.0, 180.0], [-15.0, 180.0], [-15.0, 174.0],
    ]),

    ("Java Sea", [
        [-3.0, 106.0], [-3.5, 108.0], [-4.0, 110.0], [-5.0, 112.0],
        [-6.0, 114.0], [-7.0, 116.0], [-7.5, 114.0], [-7.0, 111.0],
        [-6.5, 108.5], [-5.5, 106.0], [-4.0, 105.0],
    ]),

    ("Coral Sea", [
        [-10.0, 145.0], [-10.0, 155.0], [-12.0, 158.0], [-18.0, 162.0],
        [-24.0, 160.0], [-28.0, 155.0], [-25.0, 150.0], [-20.0, 148.0],
        [-15.0, 145.0],
    ]),

    ("Tasman Sea", [
        [-28.0, 150.0], [-30.0, 155.0], [-33.0, 160.0], [-38.0, 165.0],
        [-44.0, 170.0], [-48.0, 172.0], [-48.0, 168.0], [-45.0, 162.0],
        [-42.0, 150.0], [-38.0, 148.0], [-34.0, 149.0],
    ]),

    ("Gulf of Guinea", [
        [-5.0, -5.0], [-2.0, -3.0], [0.0, -1.0], [2.0, 1.0],
        [4.0, 3.0], [5.0, 5.0], [6.0, 7.5], [5.5, 10.5],
        [4.0, 11.0], [2.0, 10.0], [0.0, 8.5], [-2.0, 7.0],
        [-4.0, 5.0], [-5.0, 2.0],
    ]),

    # Bering Sea crosses dateline — handled specially in _get_ocean_zone()

    # === BROAD OCEAN BASINS (check last — fallback) ===

    ("North Atlantic Ocean", [
        [0.0, -82.0], [0.0, -20.0], [10.0, -15.0], [20.0, -18.0],
        [30.0, -10.0], [40.0, -10.0], [50.0, -10.0], [60.0, -10.0],
        [65.0, -20.0], [70.0, -40.0], [60.0, -55.0], [50.0, -60.0],
        [45.0, -65.0], [40.0, -72.0], [35.0, -77.0], [30.0, -81.5],
        [25.0, -81.5], [20.0, -82.0], [10.0, -82.0],
    ]),

    ("South Atlantic Ocean", [
        [0.0, -60.0], [0.0, 15.0], [-10.0, 15.0], [-20.0, 15.0],
        [-30.0, 18.0], [-40.0, 20.0], [-55.0, 20.0], [-55.0, -70.0],
        [-40.0, -65.0], [-30.0, -55.0], [-20.0, -45.0], [-10.0, -40.0],
    ]),

    ("North Pacific Ocean", [
        [0.0, 120.0], [0.0, 180.0], [0.0, -180.0], [0.0, -80.0],
        [10.0, -80.0], [20.0, -105.0], [30.0, -120.0], [40.0, -125.0],
        [50.0, -130.0], [60.0, -145.0], [65.0, -170.0], [60.0, 180.0],
        [50.0, 160.0], [40.0, 145.0], [30.0, 130.0], [20.0, 120.0],
        [10.0, 118.0],
    ]),

    ("Indian Ocean", [
    [-55.0, 20.0], [-55.0, 120.0], [-10.0, 120.0], [-5.0, 100.0],
    [0.0, 100.0], [5.0, 80.0], [10.0, 75.0], [15.0, 73.0],
    [22.0, 60.0], [25.0, 57.0], [30.0, 32.0], [20.0, 38.0],
    [12.0, 43.0], [5.0, 42.0], [0.0, 42.0], [-10.0, 40.0],
    [-20.0, 35.0], [-30.0, 25.0], [-40.0, 20.0],
    ]),

    ("South Pacific Ocean", [
        [0.0, 130.0], [0.0, -80.0], [-10.0, -80.0], [-20.0, -75.0],
        [-30.0, -75.0], [-55.0, -75.0], [-55.0, 170.0], [-40.0, 170.0],
        [-30.0, 160.0], [-20.0, 155.0], [-10.0, 145.0],
    ]),

    ("Southern Ocean", [
        [-55.0, -180.0], [-55.0, 180.0], [-90.0, 180.0], [-90.0, -180.0],
    ]),

    ("Arctic Ocean", [
        [70.0, -180.0], [70.0, 180.0], [90.0, 180.0], [90.0, -180.0],
    ]),
]


# ============================================================
# ZONE RESOLVER
# ============================================================

def _get_ocean_zone(lat, lon):
    """Determine which ocean/sea zone a coordinate falls in"""
    # Special handling for Bering Sea (crosses dateline)
    if 52.0 <= lat <= 66.0:
        if lon >= 165.0 or lon <= -160.0:
            return "Bering Sea"

    for zone_name, polygon in MARITIME_ZONES:
        if _point_in_polygon(lat, lon, polygon):
            return zone_name
    # Absolute fallback based on hemisphere
    if lat >= 0:
        if -80 <= lon <= 0:
            return "North Atlantic Ocean"
        elif 0 < lon <= 120:
            return "Indian Ocean" if lat < 30 else "North Atlantic Ocean"
        else:
            return "North Pacific Ocean"
    else:
        if -70 <= lon <= 20:
            return "South Atlantic Ocean"
        elif 20 < lon <= 130:
            return "Indian Ocean"
        else:
            return "South Pacific Ocean"


# ============================================================
# PORTS LOADER (cached)
# ============================================================

_ports_cache = None

def _load_ports():
    """Load ports.json and cache it"""
    global _ports_cache
    if _ports_cache is not None:
        return _ports_cache

    ports_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ports.json")
    try:
        with open(ports_path, "r") as f:
            _ports_cache = json.load(f)
        logger.info(f"Loaded {len(_ports_cache)} ports from ports.json")
    except Exception as e:
        logger.warning(f"Could not load ports.json: {e}")
        _ports_cache = {}
    return _ports_cache


# ============================================================
# NEAREST PORT FINDER
# ============================================================

def _find_nearest_port(lat, lon):
    """Find the nearest port from ports.json
    Returns (port_name, distance_nm, bearing_from_port_to_vessel)
    """
    ports = _load_ports()
    if not ports:
        return None, None, None

    best_name = None
    best_dist = float("inf")
    best_bearing = 0

    for port_name, port_data in ports.items():
        plat = port_data.get("lat", 0)
        plon = port_data.get("lon", 0)
        if plat == 0 and plon == 0:
            continue

        dist = _haversine_nm(lat, lon, plat, plon)
        if dist < best_dist:
            best_dist = dist
            best_name = port_name
            # Bearing FROM port TO vessel (so we can say "X NM [direction] of [port]")
            best_bearing = _bearing(plat, plon, lat, lon)

    return best_name, best_dist, best_bearing


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

AT_BERTH_THRESHOLD_NM = 2.0       # Within 2 NM of port = "at berth / in port"
NEAR_PORT_THRESHOLD_NM = 10.0     # Within 10 NM = "approaching / near port"

def resolve_location(lat, lon):
    """
    Resolve lat/lon to a human-readable maritime location string.
    
    Returns examples:
      - "Singapore Container Terminal (at berth)"
      - "Near Singapore Container Terminal, Strait of Malacca"
      - "North Atlantic Ocean, 165 NM ESE of Port Canaveral"
      - "Arabian Sea, 340 NM SW of Mumbai Port"
    
    Fully offline — no external API calls.
    """
    if lat is None or lon is None:
        return "Position unavailable"

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return "Invalid coordinates"

    # Sanity check
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        return f"{lat}, {lon}"

    # Find nearest port
    port_name, dist_nm, bearing_deg = _find_nearest_port(lat, lon)

    # Get ocean/sea zone
    zone = _get_ocean_zone(lat, lon)

    # Build location string
    if port_name and dist_nm is not None:
        if dist_nm <= AT_BERTH_THRESHOLD_NM:
            return f"{port_name} (at berth)"
        
        elif dist_nm <= NEAR_PORT_THRESHOLD_NM:
            return f"Near {port_name}, {zone}"
        
        else:
            cardinal = _bearing_to_cardinal(bearing_deg)
            dist_rounded = round(dist_nm)
            return f"{zone}, {dist_rounded} NM {cardinal} of {port_name}"
    
    else:
        # No ports loaded — just return zone
        return zone


# ============================================================
# QUICK TEST
# ============================================================

# if __name__ == "__main__":
#     test_cases = [
#         (28.041, -77.051, "Atlantic east of Bahamas"),
#         (1.26, 103.82, "Singapore"),
#         (25.77, -80.19, "Miami"),
#         (51.9, 1.4, "North Sea"),
#         (35.0, 139.7, "Tokyo Bay area"),
#         (21.3, 39.1, "Red Sea near Jeddah"),
#         (-33.8, 151.2, "Sydney area"),
#         (60.0, 25.0, "Baltic near Helsinki"),
#         (10.0, 75.0, "Arabian Sea off Kerala"),
#         (5.0, 98.0, "Strait of Malacca"),
#         (0.0, 0.0, "Gulf of Guinea"),
#     ]

#     for lat, lon, expected in test_cases:
#         result = resolve_location(lat, lon)
#         print(f"  ({lat:8.3f}, {lon:8.3f}) → {result}  [expected near: {expected}]")

if __name__ == "__main__":

    test_cases = [
        # Chokepoints
        (36.0, -5.5, "Strait of Gibraltar"),
        (26.2, 56.3, "Strait of Hormuz"),
        (12.6, 43.3, "Bab el-Mandeb Strait"),
        (2.5, 101.0, "Strait of Malacca"),
        (-6.5, 105.5, "Sunda Strait"),
        (-8.5, 115.6, "Lombok Strait"),
        (24.5, 119.5, "Taiwan Strait"),
        (30.0, 32.5, "Suez Canal Approaches"),
        (9.0, -79.5, "Panama Canal Approaches"),
        (-56.0, -65.0, "Drake Passage"),

        # Enclosed seas
        (27.0, 51.0, "Persian Gulf"),
        (23.0, 59.0, "Gulf of Oman"),
        (20.0, 38.0, "Red Sea"),
        (35.0, 18.0, "Mediterranean Sea"),
        (43.0, 35.0, "Black Sea"),
        (60.0, 25.0, "Gulf of Finland"),
        (56.0, 19.0, "Baltic Sea"),
        (55.0, 3.0, "North Sea"),
        (50.0, -1.0, "English Channel"),
        (25.0, -90.0, "Gulf of Mexico"),
        (15.0, -70.0, "Caribbean Sea"),
        (12.0, 115.0, "South China Sea"),
        (28.0, 124.0, "East China Sea"),
        (40.0, 135.0, "Sea of Japan"),
        (15.0, 85.0, "Bay of Bengal"),
        (15.0, 65.0, "Arabian Sea"),
        (13.0, 48.0, "Gulf of Aden"),
        (-15.0, 40.0, "Mozambique Channel"),
        (-5.0, 110.0, "Java Sea"),
        (-20.0, 152.0, "Coral Sea"),
        (-38.0, 160.0, "Tasman Sea"),
        (3.0, 5.0, "Gulf of Guinea"),

        # Broad oceans
        (28.0, -77.0, "North Atlantic Ocean"),
        (-25.0, -10.0, "South Atlantic Ocean"),
        (30.0, -170.0, "North Pacific Ocean"),
        (-25.0, -130.0, "South Pacific Ocean"),
        (-10.0, 80.0, "Indian Ocean"),
        (-60.0, 50.0, "Southern Ocean"),
        (75.0, 0.0, "Arctic Ocean"),

        # Special zones
        (21.3, -157.8, "Hawaiian Waters"),
        (-17.7, 177.1, "Fiji Waters"),
        (75.0, 40.0, "Barents Sea"),
        (73.0, -70.0, "Baffin Bay"),

        # Vessel positions (real-world test)
        (29.82, -90.0, "Gulf of Mexico near New Orleans"),
        (1.26, 103.82, "Singapore at berth"),
        (10.0, 75.0, "Arabian Sea off Kerala"),
    ]

    print("\n=== COMPREHENSIVE LOCATION TEST ===\n")

    for lat, lon, expected in test_cases:
        result = resolve_location(lat, lon)
        match = "OK" if expected.split()[0].lower() in result.lower() or expected.split()[-1].lower() in result.lower() else "??"
        print(f"[{match}] ({lat}, {lon})")
        print(f"  Result  : {result}")
        print(f"  Expected: {expected}")
        print()