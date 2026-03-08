r"""
Vessel Telemetry MCP Server
============================
Standalone MCP server that provides tool access to vessel telemetry SQLite databases.
Designed for Qwen3-8B tool calling integration.

Usage:
  python mcp_telemetry.py

Runs on port 5010. Tools are exposed as function definitions that Qwen3 can call.

Database pattern:
  r"C:\Users\User\Desktop\Main Engine Diagnostics\{Ship Name} (IMO {number})\Telemetry Data\imo_{number}.s3db"
"""

from flask import Flask, request, jsonify
import requests
import sqlite3
import json
import os
import glob
import logging
from datetime import datetime, timedelta
from location_resolver import resolve_location

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask('mcp_telemetry')

# Base path for all vessel databases
DB_BASE_PATH = r"C:\Users\User\Desktop\Main Engine Diagnostics"


def coords_to_place(lat, lon):
    return resolve_location(lat, lon)

# ============================================================
# SENSOR CATEGORIES — for organized output
# ============================================================
SENSOR_GROUPS = {
    "navigation": {
        "V_GPSLAT_act_deg@LAST": "Latitude",
        "V_GPSLON_act_deg@LAST": "Longitude",
        "V_SOG_act_kn@AVG": "Speed Over Ground (kn)",
        "V_STW_act_kn@AVG": "Speed Through Water (kn)",
        "V_HDG_act_deg@AVG": "Heading (deg)",
        "V_COG_act_deg@AVG": "Course Over Ground (deg)",
        "V_ROT_act_degPmin@AVG": "Rate of Turn (deg/min)",
        "V_RUA_act_deg@AVG": "Rudder Angle (deg)",
    },
    "main_engine": {
        "ME_RPM": "ME RPM",
        "ME_Load@AVG": "ME Load (%)",
        "SA_POW_act_kW@AVG": "Shaft Power (kW)",
        "SA_SPD_act_rpm@AVG": "Shaft Speed (RPM)",
        "SA_TQU_act_kNm@AVG": "Shaft Torque (kNm)",
        "SHAFT_POWER": "Shaft Power (raw)",
        "SHAFT_TORQUE": "Shaft Torque (raw)",
        "ME_SCAV_AIR_PRESS": "Scav Air Pressure",
        "ME_MAIN_LO_IN_PRESS": "Main LO Inlet Pressure",
        "ME_MAIN_LO_IN_TEMP": "Main LO Inlet Temp",
        "ME_TC_IN_TEMP": "TC Inlet Temp",
        "ME_NO_1_TC_RPM": "TC No.1 RPM",
        "ME_NO_2_TC_RPM": "TC No.2 RPM",
        "ME_NO_1_TC_EXH_GAS_IN_TEMP": "TC No.1 Exh Gas In Temp",
        "ME_NO_2_TC_EXH_GAS_IN_TEMP": "TC No.2 Exh Gas In Temp",
        "ME_NO_1_TC_EXH_GAS_OUT_TEMP": "TC No.1 Exh Gas Out Temp",
        "ME_NO_2_TC_EXH_GAS_OUT_TEMP": "TC No.2 Exh Gas Out Temp",
        "ME_NO_1_TC_LO_IN_PRESS": "TC No.1 LO Inlet Pressure",
        "ME_NO_2_TC_LO_IN_PRESS": "TC No.2 LO Inlet Pressure",
        "ME_NO_1_TC_LO_OUT_TEMP": "TC No.1 LO Outlet Temp",
        "ME_NO_2_TC_LO_OUT_TEMP": "TC No.2 LO Outlet Temp",
    },
    "fuel": {
        "ME_FMS_act_kgPh@AVG": "ME Fuel Consumption (kg/h)",
        "ME_HFO_FMS_act_kgPh@AVG": "ME HFO Consumption (kg/h)",
        "ME_MDO_FMS_act_kgPh@AVG": "ME MDO Consumption (kg/h)",
        "AE_FMS_act_kgPh@AVG": "AE Total Fuel (kg/h)",
        "AE_HFO_FMS_act_kgPh@AVG": "AE HFO Consumption (kg/h)",
        "AE_MDO_FMS_act_kgPh@AVG": "AE MDO Consumption (kg/h)",
        "ME_FDS_act_kgPm3@AVG": "ME Fuel Density (kg/m3)",
        "ME_FTS_act_dgC@AVG": "ME Fuel Temp (°C)",
        "ME_HFO_FTS_act_dgC@AVG": "ME HFO Temp (°C)",
        "AE_FTS_act_dgC@AVG": "AE Fuel Temp (°C)",
        "FMS1_act_kgPh@AVG": "FMS1 Flow (kg/h)",
        "FMS2_act_kgPh@AVG": "FMS2 Flow (kg/h)",
        "FMS3_act_kgPh@AVG": "FMS3 Flow (kg/h)",
    },
    "generators": {
        "Dg1_active": "DG1 Active",
        "Dg2_active": "DG2 Active",
        "Dg3_active": "DG3 Active",
        "Dg4_active": "DG4 Active",
        "NO_1_G_E_RUNNING": "GE1 Running",
        "NO_2_G_E_RUNNING": "GE2 Running",
        "NO_3_G_E_RUNNING": "GE3 Running",
        "NO_4_G_E_RUNNING": "GE4 Running",
        "AE1_POW_act_kW@AVG": "AE1 Power (kW)",
        "AE2_POW_act_kW@AVG": "AE2 Power (kW)",
        "AE3_POW_act_kW@AVG": "AE3 Power (kW)",
        "AE4_POW_act_kW@AVG": "AE4 Power (kW)",
        "auxiliaryEngine1Percent@AVG": "AE1 Load (%)",
        "auxiliaryEngine2Percent@AVG": "AE2 Load (%)",
        "auxiliaryEngine3Percent@AVG": "AE3 Load (%)",
        "auxiliaryEngine4Percent@AVG": "AE4 Load (%)",
        "totalAuxEnginesElecPower@AVG": "Total AE Power (kW)",
        "totalAuxEnginesElecPowerPercent@AVG": "Total AE Load (%)",
        "NO_1_G_E_VOLTAGE": "GE1 Voltage",
        "NO_2_G_E_VOLTAGE": "GE2 Voltage",
        "NO_3_G_E_VOLTAGE": "GE3 Voltage",
        "NO_4_G_E_VOLTAGE": "GE4 Voltage",
        "NO_1_G_E_FREQUENCY": "GE1 Frequency",
        "NO_2_G_E_FREQUENCY": "GE2 Frequency",
        "NO_3_G_E_FREQUENCY": "GE3 Frequency",
        "NO_4_G_E_FREQUENCY": "GE4 Frequency",
    },
    "weather": {
        "WEA_WST_act_kn@AVG": "Wind Speed True (kn)",
        "WEA_WDT_act_deg@AVG": "Wind Direction True (deg)",
        "WEA_WSR_act_kn@AVG": "Wind Speed Relative (kn)",
        "WEA_WDR_act_deg@AVG": "Wind Direction Relative (deg)",
        "E_R_AMBIENT_TEMP": "ER Ambient Temp",
    },
}

# Alarm keys — any key with these patterns that equals 1 means active alarm
ALARM_PATTERNS = [
    "ALARM", "SHUTDOWN", "SHUT_DOWN", "FAILURE", "FAIL", "TRIP", "STOP",
    "OVERSPEED", "OVERLOAD", "ABNORMAL", "ABN", "HIGH", "LOW", "FIRE",
    "BLACKOUT", "RELEASE", "LEAKAGE", "MIST", "DETECT",
]

# Keys to EXCLUDE from alarm detection (they're sensor values, not alarm flags)
ALARM_EXCLUDE = {
    "ME_RPM", "ME_Load@AVG", "SHAFT_POWER", "SHAFT_TORQUE", "SHAFT_REVOLUTIONS",
    "GPS_Status", "ME_TC_IN_TEMP", "ME_MAIN_LO_IN_TEMP", "ME_MAIN_LO_IN_PRESS",
    "ME_SCAV_AIR_PRESS", "Local_time", "GMT", "IMO", "time", "google", "MDC", "Spm",
    "Aconis", "upload_mbps", "download_mbps", "Gps_autopilot", "Echosounder_gyro",
    "Speedlog_windsensor", "Aux_genertors",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_db_path(imo: str) -> str:
    """Find the s3db file for a given IMO number"""
    # Direct path
    pattern = os.path.join(DB_BASE_PATH, f"*IMO {imo}*", "Telemetry Data", f"imo_{imo}.s3db")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    
    # Try without space variations
    pattern2 = os.path.join(DB_BASE_PATH, f"*(IMO {imo})*", "Telemetry Data", f"imo_{imo}.s3db")
    matches2 = glob.glob(pattern2)
    if matches2:
        return matches2[0]
    
    # Try broader search
    pattern3 = os.path.join(DB_BASE_PATH, "**", f"imo_{imo}.s3db")
    matches3 = glob.glob(pattern3, recursive=True)
    if matches3:
        return matches3[0]
    
    return None


def get_db_connection(imo: str):
    """Get SQLite connection for a vessel"""
    db_path = find_db_path(imo)
    if not db_path:
        return None, f"No database found for IMO {imo}"
    
    if not os.path.exists(db_path):
        return None, f"Database file not found: {db_path}"
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn, None


def parse_payload(payload_str):
    """Parse JSON payload string"""
    try:
        return json.loads(payload_str)
    except (json.JSONDecodeError, TypeError):
        return None


def is_alarm_key(key):
    """Check if a key is an alarm flag (not a sensor value)"""
    if key in ALARM_EXCLUDE:
        return False
    # Check for any key that contains alarm-related patterns
    key_upper = key.upper()
    return any(p in key_upper for p in ALARM_PATTERNS)


# ============================================================
# TOOL DEFINITIONS — these are what Qwen3 sees
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_latest_readings",
            "description": "Get the latest sensor readings for a vessel. Returns current values for navigation, main engine, fuel, generators, and weather. Use this when asked about current status, latest values, or real-time data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imo": {
                        "type": "string",
                        "description": "IMO number of the vessel (e.g., '9665671')"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["all", "navigation", "main_engine", "fuel", "generators", "weather"],
                        "description": "Category of readings to return. Use 'all' for complete snapshot."
                    }
                },
                "required": ["imo"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "get_jit_snapshot",
        "description": "Get current vessel state for JIT arrival calculation. Returns current SOG, position, fuel consumption, RPM, shaft power, and sailing averages from recent history.",
        "parameters": {
            "type": "object",
            "properties": {
                "imo": {
                    "type": "string",
                    "description": "IMO number of the vessel"
                }
            },
            "required": ["imo"]
        }
    }
},
    {
        "type": "function",
        "function": {
            "name": "get_active_alarms",
            "description": "Get all currently active alarms for a vessel. Scans the latest telemetry for any alarm/shutdown/failure/trip flags that are active (value = 1). Use when asked about current alarms, faults, warnings, or abnormal conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imo": {
                        "type": "string",
                        "description": "IMO number of the vessel"
                    }
                },
                "required": ["imo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sensor_history",
            "description": "Get historical values for a specific sensor over a time range. Returns timestamped readings. Use when asked about trends, changes over time, or historical data for a specific parameter like RPM, power, temperature, fuel consumption etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imo": {
                        "type": "string",
                        "description": "IMO number of the vessel"
                    },
                    "sensor_key": {
                        "type": "string",
                        "description": "The exact JSON key of the sensor in the payload (e.g., 'ME_RPM', 'SA_POW_act_kW@AVG', 'ME_FMS_act_kgPh@AVG')"
                    },
                    "hours_back": {
                        "type": "integer",
                        "description": "How many hours of history to retrieve (default: 24, max: 168 = 1 week)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of data points to return (default: 50)"
                    }
                },
                "required": ["imo", "sensor_key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_generator_status",
            "description": "Get detailed status of all 4 diesel generators/auxiliary engines for a vessel. Shows which are running, their power output, load percentage, voltages, and frequencies. Use when asked about generator status, DG status, auxiliary engines, or electrical power.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imo": {
                        "type": "string",
                        "description": "IMO number of the vessel"
                    }
                },
                "required": ["imo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_telemetry",
            "description": "Run a custom query on vessel telemetry data. For advanced queries not covered by other tools. The database has one table 'VesselData' with columns: id, payload (JSON text), fk_vessel, createdAt, vesselTime, vesselTimeStamp. The payload contains ~300 sensor keys as JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imo": {
                        "type": "string",
                        "description": "IMO number of the vessel"
                    },
                    "query_type": {
                        "type": "string",
                        "enum": ["latest_n", "time_range", "custom_sql"],
                        "description": "Type of query"
                    },
                    "n": {
                        "type": "integer",
                        "description": "For latest_n: number of recent records (default: 10)"
                    },
                    "start_time": {
                        "type": "string",
                        "description": "For time_range: start datetime (ISO format)"
                    },
                    "end_time": {
                        "type": "string",
                        "description": "For time_range: end datetime (ISO format)"
                    },
                    "sql": {
                        "type": "string",
                        "description": "For custom_sql: raw SQL query (SELECT only, no modifications allowed)"
                    },
                    "sensor_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific sensor keys to extract from payload"
                    }
                },
                "required": ["imo", "query_type"]
            }
        }
    }
]


# ============================================================
# TOOL ENDPOINTS
# ============================================================

@app.route('/mcp/tools', methods=['GET'])
def list_tools():
    """List available tools — Qwen3 needs this for tool definitions"""
    return jsonify({"tools": TOOLS})


@app.route('/mcp/call', methods=['POST'])
def call_tool():
    """Execute a tool call from the LLM"""
    try:
        data = request.get_json()
        tool_name = data.get('name')
        arguments = data.get('arguments', {})
        
        logger.info(f"Tool call: {tool_name} with args: {arguments}")
        
        if tool_name == "get_latest_readings":
            return jsonify(handle_latest_readings(arguments))
        elif tool_name == "get_active_alarms":
            return jsonify(handle_active_alarms(arguments))
        elif tool_name == "get_sensor_history":
            return jsonify(handle_sensor_history(arguments))
        elif tool_name == "get_generator_status":
            return jsonify(handle_generator_status(arguments))
        elif tool_name == "query_telemetry":
            return jsonify(handle_query_telemetry(arguments))
        elif tool_name == "get_jit_snapshot":
            return jsonify(handle_jit_snapshot(arguments))
        else:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 400
    
    except Exception as e:
        logger.error(f"Tool call error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# TOOL HANDLERS
# ============================================================

def handle_latest_readings(args):
    """Get latest sensor readings"""
    imo = args.get('imo')
    category = args.get('category', 'all')
    
    conn, error = get_db_connection(imo)
    if error:
        return {"error": error}
    
    try:
        cursor = conn.execute(
       "SELECT payload, vesselTimeStamp FROM VesselData ORDER BY vesselTimeStamp DESC LIMIT 1"
      )
        row = cursor.fetchone()
        
        if not row:
            return {"error": "No data found"}
        
        payload = parse_payload(row[0])
        if not payload:
            return {"error": "Failed to parse payload"}
        
        timestamp = row[1] or "unknown"
        
        result = {
            "vessel_imo": imo,
            "timestamp": timestamp,
            "local_time": payload.get("Local_time", "unknown"),
        }
        
        if category == "all":
            for group_name, keys in SENSOR_GROUPS.items():
                group_data = {}
                for key, label in keys.items():
                    if key in payload and payload[key] is not None:
                        group_data[label] = payload[key]
                if group_data:
                    result[group_name] = group_data
            # Add human-readable location to navigation
            lat = payload.get("V_GPSLAT_act_deg@LAST")
            lon = payload.get("V_GPSLON_act_deg@LAST")
            if lat and lon:
                result.setdefault("navigation", {})["current_location"] = coords_to_place(lat, lon)

        elif category in SENSOR_GROUPS:
            group_data = {}
            for key, label in SENSOR_GROUPS[category].items():
                if key in payload and payload[key] is not None:
                    group_data[label] = payload[key]
            result[category] = group_data
            # Add human-readable location if navigation category
            if category == "navigation":
                lat = payload.get("V_GPSLAT_act_deg@LAST")
                lon = payload.get("V_GPSLON_act_deg@LAST")
                if lat and lon:
                    result["navigation"]["current_location"] = coords_to_place(lat, lon)
        else:
            return {"error": f"Unknown category: {category}. Use: all, navigation, main_engine, fuel, generators, weather"}
        
        return result
    
    finally:
        conn.close()


def handle_active_alarms(args):
    """Get all currently active alarms"""
    imo = args.get('imo')
    
    conn, error = get_db_connection(imo)
    if error:
        return {"error": error}
    
    try:
        cursor = conn.execute(
            "SELECT payload, vesselTimeStamp FROM VesselData ORDER BY vesselTimeStamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        
        if not row:
            return {"error": "No data found"}
        
        payload = parse_payload(row[0])
        if not payload:
            return {"error": "Failed to parse payload"}
        
        active_alarms = []
        for key, value in payload.items():
            if is_alarm_key(key) and value == 1:
                active_alarms.append({
                    "alarm": key,
                    "value": value,
                    "description": key.replace("_", " ").title()
                })
        
        return {
            "vessel_imo": imo,
            "timestamp": row[1] or "unknown",
            "local_time": payload.get("Local_time", "unknown"),
            "active_alarm_count": len(active_alarms),
            "active_alarms": active_alarms,
            "status": "ALL CLEAR" if len(active_alarms) == 0 else "ALARMS ACTIVE"
        }
    
    finally:
        conn.close()


def handle_sensor_history(args):
    """Get historical sensor data"""
    imo = args.get('imo')
    sensor_key = args.get('sensor_key')
    hours_back = min(args.get('hours_back', 24), 168)  # Max 1 week
    limit = min(args.get('limit', 50), 200)  # Max 200 points
    
    conn, error = get_db_connection(imo)
    if error:
        return {"error": error}
    
    try:
        # Get the cutoff timestamp
        # cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        # cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        
        # # Query records — try vesselTimeStamp first, fall back to createdAt
        # cursor = conn.execute("""
        #     SELECT payload, vesselTimeStamp, createdAt 
        #     FROM VesselData 
        #     WHERE vesselTimeStamp >= ? OR createdAt >= ?
        #     ORDER BY vesselTimeStamp DESC
        #     LIMIT ?
        # """, (cutoff_str, cutoff_str, limit))
        # vesselTimeStamp is unix epoch — calculate cutoff as unix timestamp
        import time
        cutoff_unix = int(time.time()) - (hours_back * 3600)
        
        # First try with cutoff
        cursor = conn.execute("""
            SELECT payload, vesselTimeStamp, createdAt 
            FROM VesselData 
            WHERE CAST(vesselTimeStamp AS INTEGER) >= ?
            ORDER BY vesselTimeStamp DESC
            LIMIT ?
        """, (cutoff_unix, limit))
        
        rows = cursor.fetchall()
        
        # If no data in time range, get latest N records instead
        if not rows:
            cursor = conn.execute("""
                SELECT payload, vesselTimeStamp, createdAt 
                FROM VesselData 
                ORDER BY vesselTimeStamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
        
        # rows = cursor.fetchall()
        
        if not rows:
            return {"error": f"No data found in the last {hours_back} hours"}
        
        data_points = []
        for row in reversed(rows):  # Chronological order
            payload = parse_payload(row[0])
            if payload and sensor_key in payload:
                value = payload[sensor_key]
                if value is not None:
                    data_points.append({
                        "timestamp": payload.get("Local_time") or row[1] or row[2],
                        "value": value
                    })
        
        if not data_points:
            # Sensor key might not exist — show available keys that match
            sample_payload = parse_payload(rows[0][0])
            suggestions = [k for k in sample_payload.keys() if sensor_key.lower() in k.lower()] if sample_payload else []
            return {
                "error": f"Sensor key '{sensor_key}' not found or has no data",
                "suggestions": suggestions[:10]
            }
        
        # Calculate basic stats
        values = [dp["value"] for dp in data_points if isinstance(dp["value"], (int, float))]
        stats = {}
        if values:
            stats = {
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "avg": round(sum(values) / len(values), 2),
                "latest": values[-1],
                "points_count": len(values)
            }
        
        return {
            "vessel_imo": imo,
            "sensor_key": sensor_key,
            "hours_back": hours_back,
            "statistics": stats,
            "data": data_points
        }
    
    finally:
        conn.close()


def handle_generator_status(args):
    """Get detailed generator/AE status"""
    imo = args.get('imo')
    
    conn, error = get_db_connection(imo)
    if error:
        return {"error": error}
    
    try:
        cursor = conn.execute(
            "SELECT payload, vesselTimeStamp FROM VesselData ORDER BY vesselTimeStamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        
        if not row:
            return {"error": "No data found"}
        
        payload = parse_payload(row[0])
        if not payload:
            return {"error": "Failed to parse payload"}
        
        generators = []
        for i in range(1, 5):
            gen = {
                "generator": f"DG{i}",
                "active": payload.get(f"Dg{i}_active", 0),
                "running": payload.get(f"NO_{i}_G_E_RUNNING", 0),
                "power_kw": payload.get(f"AE{i}_POW_act_kW@AVG", 0),
                "load_percent": payload.get(f"auxiliaryEngine{i}Percent@AVG", 0),
                "voltage": payload.get(f"NO_{i}_G_E_VOLTAGE", 0),
                "frequency": payload.get(f"NO_{i}_G_E_FREQUENCY", 0),
                "bus_current": payload.get(f"NO_{i}_G_E_BUS_CURRENT", 0),
                "fo_inlet_temp": payload.get(f"G_E{i}_FO_IN_TEMP", 0),
                "lo_inlet_temp": payload.get(f"G_E{i}_LO_IN_TEMP", 0),
                "lo_inlet_press": payload.get(f"G_E{i}_LO_INLET_PRESS", 0),
                "alarms": {
                    "shutdown": payload.get(f"G_E{i}_COMMON_SHUTDOWN", 0),
                    "emergency_shutdown": payload.get(f"G_E{i}_EMERGENCY_SHUTDOWN", 0),
                    "abnormal": payload.get(f"G_E{i}_COMMON_ABNORMAL_ALARM", 0),
                    "start_failure": payload.get(f"G_E{i}_START_FAILURE", 0),
                    "stop_failure": payload.get(f"G_E{i}_STOP_FAILURE", 0),
                    "governor_fail": payload.get(f"G_E{i}_GOVERNOR_FAIL", 0),
                    "overspeed_1st": payload.get(f"G_E{i}_OVERSPEED_TRIP_1ST_", 0),
                    "overspeed_2nd": payload.get(f"G_E{i}_OVERSPEED_TRIP_2ND_", 0),
                }
            }
            generators.append(gen)
        
        active_count = sum(1 for g in generators if g["active"] == 1)
        total_power = payload.get("totalAuxEnginesElecPower@AVG", 0)
        total_load = payload.get("totalAuxEnginesElecPowerPercent@AVG", 0)
        
        return {
            "vessel_imo": imo,
            "timestamp": row[1] or "unknown",
            "local_time": payload.get("Local_time", "unknown"),
            "summary": {
                "active_generators": active_count,
                "total_power_kw": total_power,
                "total_load_percent": total_load,
            },
            "generators": generators
        }
    
    finally:
        conn.close()


def handle_query_telemetry(args):
    """Handle custom telemetry queries"""
    imo = args.get('imo')
    query_type = args.get('query_type')
    
    conn, error = get_db_connection(imo)
    if error:
        return {"error": error}
    
    try:
        if query_type == "latest_n":
            n = min(args.get('n', 10), 100)
            sensor_keys = args.get('sensor_keys', [])
            
            cursor = conn.execute(
                "SELECT payload, vesselTimeStamp FROM VesselData ORDER BY vesselTimeStamp DESC LIMIT ?",
                (n,)
            )
            rows = cursor.fetchall()
            
            results = []
            for row in reversed(rows):
                payload = parse_payload(row[0])
                if payload:
                    if sensor_keys:
                        entry = {"timestamp": payload.get("Local_time", row[1])}
                        for key in sensor_keys:
                            entry[key] = payload.get(key, None)
                        results.append(entry)
                    else:
                        results.append({
                            "timestamp": payload.get("Local_time", row[1]),
                            "ME_RPM": payload.get("ME_RPM"),
                            "SA_POW_act_kW@AVG": payload.get("SA_POW_act_kW@AVG"),
                            "ME_FMS_act_kgPh@AVG": payload.get("ME_FMS_act_kgPh@AVG"),
                            "V_SOG_act_kn@AVG": payload.get("V_SOG_act_kn@AVG"),
                        })
            
            return {"vessel_imo": imo, "record_count": len(results), "data": results}
        
        elif query_type == "time_range":
            start_time = args.get('start_time')
            end_time = args.get('end_time')
            sensor_keys = args.get('sensor_keys', [])
            
            if not start_time or not end_time:
                return {"error": "start_time and end_time required for time_range query"}
            
            # cursor = conn.execute("""
            #     SELECT payload, vesselTimeStamp FROM VesselData 
            #     WHERE vesselTimeStamp BETWEEN ? AND ?
            #     ORDER BY vesselTimeStamp ASC
            #     LIMIT 200
            # """, (start_time, end_time))
            from datetime import datetime as dt
            try:
                start_unix = int(dt.fromisoformat(start_time.replace('Z', '+00:00')).timestamp())
                end_unix = int(dt.fromisoformat(end_time.replace('Z', '+00:00')).timestamp())
            except:
                return {"error": f"Invalid date format. Use ISO format like 2026-02-10T00:00:00Z"}

            cursor = conn.execute("""
                SELECT payload, vesselTimeStamp FROM VesselData 
                WHERE CAST(vesselTimeStamp AS INTEGER) BETWEEN ? AND ?
                ORDER BY vesselTimeStamp ASC
                LIMIT 200
            """, (start_unix, end_unix))
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                payload = parse_payload(row[0])
                if payload:
                    entry = {"timestamp": payload.get("Local_time", row[1])}
                    for key in (sensor_keys or ["ME_RPM", "SA_POW_act_kW@AVG"]):
                        entry[key] = payload.get(key, None)
                    results.append(entry)
            
            return {"vessel_imo": imo, "record_count": len(results), "data": results}
        
        elif query_type == "custom_sql":
            sql = args.get('sql', '')
            
            # Safety: only allow SELECT
            if not sql.strip().upper().startswith("SELECT"):
                return {"error": "Only SELECT queries are allowed"}
            
            # Block dangerous keywords
            dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "EXEC"]
            sql_upper = sql.upper()
            for keyword in dangerous:
                if keyword in sql_upper:
                    return {"error": f"Keyword '{keyword}' is not allowed"}
            
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            
            results = []
            for row in rows[:100]:  # Max 100 results
                results.append(dict(zip(columns, row)))
            
            return {"vessel_imo": imo, "columns": columns, "record_count": len(results), "data": results}
        
        else:
            return {"error": f"Unknown query_type: {query_type}"}
    
    finally:
        conn.close()
def handle_jit_snapshot(args):
    """Get current vessel state for JIT calculation"""
    imo = args.get('imo')
    
    conn, error = get_db_connection(imo)
    if error:
        return {"error": error}
    
    try:
        # Get latest record
        row = conn.execute(
            "SELECT payload, vesselTimeStamp FROM VesselData ORDER BY vesselTimeStamp DESC LIMIT 1"
        ).fetchone()
        
        if not row:
            return {"error": "No data found"}
        
        payload = parse_payload(row[0])
        if not payload:
            return {"error": "Failed to parse payload"}
        
        # Get average fuel burn from last 100 SAILING records (SOG > 5)
        rows = conn.execute(
            "SELECT payload FROM VesselData ORDER BY vesselTimeStamp DESC LIMIT 5000"
        ).fetchall()
        
        sailing_fuel = []
        sailing_sog = []
        for r in rows:
            p = parse_payload(r[0])
            if not p:
                continue
            sog = p.get("V_SOG_act_kn@AVG")
            fuel = p.get("ME_FMS_act_kgPh@AVG")
            if sog and fuel and float(sog) > 5 and float(fuel) > 0:
                sailing_fuel.append(float(fuel))
                sailing_sog.append(float(sog))
            if len(sailing_fuel) >= 100:
                break
        
        avg_fuel = round(sum(sailing_fuel) / len(sailing_fuel), 2) if sailing_fuel else None
        avg_sog  = round(sum(sailing_sog)  / len(sailing_sog),  2) if sailing_sog  else None
        
        return {
            "vessel_imo": imo,
            "timestamp": row[1],
            "current_state": {
                "location": coords_to_place(payload.get("V_GPSLAT_act_deg@LAST"), payload.get("V_GPSLON_act_deg@LAST")),
                "sog_knots":      payload.get("V_SOG_act_kn@AVG"),
                "stw_knots":      payload.get("V_STW_act_kn@AVG"),
                "rpm":            payload.get("SA_SPD_act_rpm@AVG"),
                "shaft_power_kw": payload.get("SA_POW_act_kW@AVG"),
                "fuel_kgph":      payload.get("ME_FMS_act_kgPh@AVG"),
                "latitude":       payload.get("V_GPSLAT_act_deg@LAST"),
                "longitude":      payload.get("V_GPSLON_act_deg@LAST"),
                "heading":        payload.get("V_HDG_act_deg@AVG"),
                "wind_direction": payload.get("WEA_WDT_act_deg@AVG"),
            },
            "sailing_averages": {
                "avg_sog_knots":  avg_sog,
                "avg_fuel_kgph":  avg_fuel,
                "samples_used":   len(sailing_fuel)
            }
        }
    
    finally:
        conn.close()

# ============================================================
# HEALTH CHECK
# ============================================================
@app.route('/mcp/health', methods=['GET'])
def health():
    """Health check — also lists available vessel databases"""
    # Find all s3db files
    pattern = os.path.join(DB_BASE_PATH, "**", "*.s3db")
    databases = glob.glob(pattern, recursive=True)
    
    vessels = []
    for db_path in databases:
        filename = os.path.basename(db_path)
        if filename.startswith("imo_"):
            imo = filename.replace("imo_", "").replace(".s3db", "")
            # Get folder name for ship name
            parts = db_path.split(os.sep)
            ship_folder = [p for p in parts if "IMO" in p]
            ship_name = ship_folder[0] if ship_folder else "Unknown"
            vessels.append({"imo": imo, "name": ship_name, "db_path": db_path})
    
    return jsonify({
        "status": "healthy",
        "available_vessels": vessels,
        "vessel_count": len(vessels)
    })


# ============================================================
# STARTUP
# ============================================================
if __name__ == '__main__':
    logger.info("Starting Vessel Telemetry MCP Server...")
    logger.info(f"Database base path: {DB_BASE_PATH}")
    
    # Check for databases on startup
    pattern = os.path.join(DB_BASE_PATH, "**", "*.s3db")
    databases = glob.glob(pattern, recursive=True)
    logger.info(f"Found {len(databases)} vessel database(s)")
    for db in databases:
        logger.info(f"  → {db}")
    
    app.run(host='0.0.0.0', port=5010, debug=False)