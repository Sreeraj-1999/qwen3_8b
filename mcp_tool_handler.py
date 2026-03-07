"""
Tool Call Handler for gpu_service.py
=====================================
Routes Qwen3 tool calls to the correct MCP server:
  - Telemetry tools (5) → port 5010 (mcp_telemetry.py)
  - PMS tools (8)       → port 5011 (mcp_pms.py)
"""

import json
import requests
import logging

logger = logging.getLogger(__name__)

MCP_TELEMETRY_URL = "http://localhost:5010"
MCP_PMS_URL = "http://localhost:5011"

# ============================================================
# TELEMETRY TOOL DEFINITIONS (5 tools)
# ============================================================

TELEMETRY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_latest_readings",
            "description": "Get the latest sensor readings for a vessel. Returns current values for navigation, main engine, fuel, generators, and weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imo": {"type": "string", "description": "IMO number of the vessel"},
                    "category": {"type": "string", "enum": ["all", "navigation", "main_engine", "fuel", "generators", "weather"], "description": "Category of readings to return"}
                },
                "required": ["imo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_alarms",
            "description": "Get all currently active alarms for a vessel. Scans latest telemetry for any alarm/shutdown/failure flags that are active.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imo": {"type": "string", "description": "IMO number of the vessel"}
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
            "name": "get_sensor_history",
            "description": "Get historical values for a specific sensor over a time range. Use for trends and historical data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imo": {"type": "string", "description": "IMO number of the vessel"},
                    "sensor_key": {"type": "string", "description": "Exact JSON key of the sensor (e.g., 'ME_RPM', 'SA_POW_act_kW@AVG')"},
                    "hours_back": {"type": "integer", "description": "Hours of history (default: 24, max: 168)"},
                    "limit": {"type": "integer", "description": "Max data points (default: 50)"}
                },
                "required": ["imo", "sensor_key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_generator_status",
            "description": "Get detailed status of all 4 diesel generators/auxiliary engines including power, load, voltages, and alarms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imo": {"type": "string", "description": "IMO number of the vessel"}
                },
                "required": ["imo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_telemetry",
            "description": "Run custom query on vessel telemetry. Types: 'latest_n' for recent records, 'time_range' for date range, 'custom_sql' for raw SELECT queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imo": {"type": "string", "description": "IMO number of the vessel"},
                    "query_type": {"type": "string", "enum": ["latest_n", "time_range", "custom_sql"]},
                    "n": {"type": "integer", "description": "For latest_n: number of records"},
                    "start_time": {"type": "string", "description": "For time_range: start datetime (ISO)"},
                    "end_time": {"type": "string", "description": "For time_range: end datetime (ISO)"},
                    "sql": {"type": "string", "description": "For custom_sql: SELECT query only"},
                    "sensor_keys": {"type": "array", "items": {"type": "string"}, "description": "Specific sensor keys to extract"}
                },
                "required": ["imo", "query_type"]
            }
        }
    }
]

# ============================================================
# PMS TOOL DEFINITIONS (8 tools)
# ============================================================

PMS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_equipment",
            "description": "Search the vessel equipment registry. Find equipment by name, code, maker, type, department, or location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Search keyword (e.g., 'turbocharger', 'boiler', 'pump', 'MAN')"},
                    "field": {"type": "string", "enum": ["all", "equipment_name", "equipment_code", "maker", "equipment_type", "department", "location"], "description": "Which field to search. Use 'all' for general search."},
                    "limit": {"type": "integer", "description": "Max results (default: 20)"}
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_maintenance_schedule",
            "description": "Get planned/scheduled maintenance jobs. Shows job title, frequency, job type, next due date, remaining days. Use for maintenance schedule, planned jobs, inspections due.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Equipment name or code (e.g., 'turbocharger', '651.053')"},
                    "job_type": {"type": "string", "enum": ["all", "INSPECTION", "TESTING", "OVERHAUL", "SERVICE", "MAINTENANCE", "CALIBRATION", "REPLACEMENT"], "description": "Filter by job type"},
                    "critical_only": {"type": "boolean", "description": "Only safety-critical jobs"},
                    "limit": {"type": "integer", "description": "Max results (default: 20)"}
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pending_jobs",
            "description": "Get pending/due/overdue maintenance jobs with due dates, priority, and status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Equipment name/code, or 'all' for all pending jobs"},
                    "priority": {"type": "string", "enum": ["all", "URGENT", "HIGH", "NORMAL", "LOW"], "description": "Filter by priority"},
                    "overdue_only": {"type": "boolean", "description": "Only show overdue jobs"},
                    "limit": {"type": "integer", "description": "Max results (default: 30)"}
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_history",
            "description": "Get completed maintenance history. Shows what was done, when, job reports, condition after, who did it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Equipment name or code (e.g., 'main engine', '551.001')"},
                    "limit": {"type": "integer", "description": "Max results (default: 20)"}
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_running_hours",
            "description": "Get equipment running hour readings and trends over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Equipment name or code"},
                    "latest_only": {"type": "boolean", "description": "Only latest reading per equipment"},
                    "limit": {"type": "integer", "description": "Max results (default: 20)"}
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_spare_parts",
            "description": "Search spare parts inventory. Shows stock levels (ROB), min stock, reorder levels, mapped equipment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Part name or equipment (e.g., 'gasket', 'bearing', 'turbocharger')"},
                    "low_stock_only": {"type": "boolean", "description": "Only parts where ROB <= minimum stock"},
                    "equipment_code": {"type": "string", "description": "Filter by equipment code"},
                    "limit": {"type": "integer", "description": "Max results (default: 25)"}
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_maintenance_summary",
            "description": "High-level PMS dashboard: counts of pending, completed, overdue jobs, low-stock spares, critical items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_details": {"type": "boolean", "description": "Include lists of overdue jobs and low-stock parts"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_equipment_full_status",
            "description": "COMPLETE equipment status: details + scheduled maintenance + pending jobs + history + running hours + spare parts in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equipment_code": {"type": "string", "description": "Equipment code (e.g., '651.053')"},
                    "equipment_name": {"type": "string", "description": "Equipment name if code unknown"}
                },
                "required": []
            }
        }
    }
]

# ============================================================
# COMBINED TOOLS + ROUTING
# ============================================================

ALL_TOOLS = TELEMETRY_TOOLS + PMS_TOOLS

PMS_TOOL_NAMES = {
    "search_equipment", "get_maintenance_schedule", "get_pending_jobs",
    "get_job_history", "get_running_hours", "search_spare_parts",
    "get_maintenance_summary", "get_equipment_full_status"
}

TELEMETRY_TOOL_NAMES = {
    "get_latest_readings", "get_active_alarms", "get_sensor_history",
    "get_generator_status", "query_telemetry","get_jit_snapshot"
}


def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Route tool call to the correct MCP server"""
    try:
        if tool_name in PMS_TOOL_NAMES:
            url = f"{MCP_PMS_URL}/mcp/call"
            server_label = "PMS"
        elif tool_name in TELEMETRY_TOOL_NAMES:
            url = f"{MCP_TELEMETRY_URL}/mcp/call"
            server_label = "Telemetry"
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        logger.info(f"Routing {tool_name} → {server_label} server")
        
        response = requests.post(
            url,
            json={"name": tool_name, "arguments": arguments},
            timeout=30
        )
        result = response.json()
        
        # Truncate large arrays to avoid token overflow
        for key in list(result.keys()):
            if isinstance(result[key], list) and len(result[key]) > 20:
                total = len(result[key])
                result[key] = result[key][:20]
                result[f"_{key}_note"] = f"Showing 20 of {total} records"
        
        # return json.dumps(result, indent=2)
        return json.dumps(result, separators=(',', ':'))
    
    except requests.exceptions.ConnectionError:
        server = "PMS (port 5011)" if tool_name in PMS_TOOL_NAMES else "Telemetry (port 5010)"
        return json.dumps({"error": f"MCP {server} server not running."})
    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        return json.dumps({"error": str(e)})


def parse_tool_call(response_text: str):
    """Parse <tool_call> from model response"""
    if "<tool_call>" not in response_text:
        return None, None
    
    try:
        tool_json = response_text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        parsed = json.loads(tool_json)
        return parsed.get("name"), parsed.get("arguments", {})
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to parse tool call: {e}")
        return None, None


def needs_tool_call(question: str, imo: str = None) -> bool:
    """Heuristic: does this question need any tool (telemetry OR PMS)?"""
    
    telemetry_keywords = [
        "current", "latest", "right now", "reading", "sensor",
        "rpm", "power", "speed", "temperature", "pressure",
        "alarm", "fault", "warning", "shutdown", "failure", "trip",
        "generator", "dg", "auxiliary", "ae1", "ae2", "ae3", "ae4",
        "trend", "last 24", "last hour", "over time",
        "telemetry", "live", "real-time", "realtime",
        "position", "latitude", "longitude", "heading", "sog", "stw",
        "wind", "weather", "turbocharger rpm", "tc rpm",
        "anchored", "at sea", "sailing", "underway",
    ]
    
    pms_keywords = [
        "maintenance", "job plan", "pending job", "overdue", "due job",
        "completed job", "job history", "job report", "inspection",
        "spare part", "spares", "stock", "rob", "reorder", "inventory",
        "running hour", "counter reading", "equipment list", "maker",
        "serial number", "pms", "planned maintenance",
        "schedule", "frequency", "overhaul", "calibration",
        "service history", "testing", "condition", "last done", "next due",
        "what needs to be done", "what is due", "what was done",
        "boiler maintenance", "pump maintenance", "compressor", "purifier",
        "low stock", "critical", "safety critical",
        "maintenance summary", "maintenance dashboard", "pms status",
    ]
    
    q_lower = question.lower()
    
    needs_telemetry = any(kw in q_lower for kw in telemetry_keywords) and imo is not None
    needs_pms = any(kw in q_lower for kw in pms_keywords)
    
    return needs_telemetry or needs_pms