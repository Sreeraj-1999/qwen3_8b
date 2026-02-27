"""
Tool Call Handler for gpu_service.py
=====================================
Add this to gpu_service.py to handle Qwen3 tool calls.

When Qwen3 generates a <tool_call> in its response, this module:
1. Parses the tool call
2. Sends it to the MCP telemetry server (port 5010)
3. Returns the result back to Qwen3 for final answer generation
"""

import json
import requests
import logging

logger = logging.getLogger(__name__)

MCP_SERVER_URL = "http://localhost:5010"

# Tool definitions — pass these to Qwen3 via apply_chat_template(tools=TOOLS)
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


def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Send tool call to MCP server and return result as string"""
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/mcp/call",
            json={"name": tool_name, "arguments": arguments},
            timeout=30
        )
        result = response.json()
        
        # Truncate large data arrays to avoid token overflow
        if "data" in result and isinstance(result["data"], list) and len(result["data"]) > 20:
            result["data"] = result["data"][:20]
            result["_truncated"] = f"Showing 20 of {len(result['data'])} records"
        
        return json.dumps(result, indent=2)
    
    except requests.exceptions.ConnectionError:
        return json.dumps({"error": "MCP telemetry server not running. Start it with: python mcp_telemetry.py"})
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
    """Quick heuristic: does this question likely need telemetry data?"""
    telemetry_keywords = [
        "current", "latest", "right now", "status", "reading", "sensor",
        "rpm", "power", "speed", "temperature", "pressure", "fuel",
        "alarm", "fault", "warning", "shutdown", "failure", "trip",
        "generator", "dg", "auxiliary", "ae1", "ae2", "ae3", "ae4",
        "trend", "history", "last 24", "last hour", "over time",
        "telemetry", "live", "real-time", "realtime",
        "position", "latitude", "longitude", "heading", "sog", "stw",
        "wind", "weather", "turbocharger", "tc rpm",
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in telemetry_keywords) and imo is not None