"""
Condition Resolver for Maritime Alert Rules
=============================================
Handles two input types:
  1. Explicit tag-based:  "if @ME_RPM > 100 and @V_SOG < 5"
  2. Natural language:    "alert me if average exhaust temp of all cylinders exceeds 400"

Both produce LaTeX output for the frontend.

Natural language flow:
  User text → LLM resolves to tag-based condition → LaTeX conversion

Architecture:
  - This module is imported by main2.py
  - LLM calls go to gpu_llama.py (port 5005) for NL resolution
  - LaTeX calls go to latex_service.py (port 5020) for conversion
"""

import re
import json
import logging
import requests

LATEX_SYSTEM_PROMPT = r"""Convert conditions to LaTeX. Output ONLY the LaTeX expression, nothing else.

Examples:
Input: if @ME_RPM is greater than 100 and @V_SOG is less than 5
Output: \text{ME_RPM} > 100 \land \text{V_SOG} < 5

Input: if @DG5_Power_kW is more than 8 or @ME_Torque_kNm equals zero
Output: \text{DG5_Power_kW} > 8 \lor \text{ME_Torque_kNm} = 0

Input: if @ME_LOAD is not equal to 50 and @AE_TEMP is at least 200
Output: \text{ME_LOAD} \neq 50 \land \text{AE_TEMP} \geq 200

Now convert:"""

logger = logging.getLogger(__name__)

# ============================================================
# MAIN ENGINE SENSOR REGISTRY
# ============================================================
# Keys are the exact telemetry payload keys.
# "aliases" are what a human might say in natural language.
# "group" allows selection by category (e.g., "all cylinders").
#
# >>> REPLACE THESE WITH REAL KEYS FROM YOUR TELEMETRY PAYLOAD <<<
# ============================================================

ME_SENSOR_REGISTRY = {
    # --- Cylinder Exhaust Gas Outlet Temperatures ---
    "exhaust_gas_after_cyl_1_temp_high": {
        "label": "Cyl 1 Exhaust Gas Temp",
        "unit": "°C",
        "group": "cylinder_exhaust_temp",
        "aliases": ["cyl 1 exhaust", "cylinder 1 exhaust temp", "no 1 cylinder exhaust"]
    },
    "exhaust_gas_after_cyl_2_temp_high": {
        "label": "Cyl 2 Exhaust Gas Temp",
        "unit": "°C",
        "group": "cylinder_exhaust_temp",
        "aliases": ["cyl 2 exhaust", "cylinder 2 exhaust temp", "no 2 cylinder exhaust"]
    },
    "exhaust_gas_after_cyl_3_temp_high": {
        "label": "Cyl 3 Exhaust Gas Temp",
        "unit": "°C",
        "group": "cylinder_exhaust_temp",
        "aliases": ["cyl 3 exhaust", "cylinder 3 exhaust temp", "no 3 cylinder exhaust"]
    },
    "exhaust_gas_after_cyl_4_temp_high": {
        "label": "Cyl 4 Exhaust Gas Temp",
        "unit": "°C",
        "group": "cylinder_exhaust_temp",
        "aliases": ["cyl 4 exhaust", "cylinder 4 exhaust temp", "no 4 cylinder exhaust"]
    },
    "exhaust_gas_after_cyl_5_temp_high": {
        "label": "Cyl 5 Exhaust Gas Temp",
        "unit": "°C",
        "group": "cylinder_exhaust_temp",
        "aliases": ["cyl 5 exhaust", "cylinder 5 exhaust temp", "no 5 cylinder exhaust"]
    },

    # --- Piston Cooling Oil Outlet Temperatures ---
    "cyl_1_piston_cooling_oil_outlet_temp_high": {
        "label": "Cyl 1 Piston CO Outlet Temp",
        "unit": "°C",
        "group": "piston_cooling_temp",
        "aliases": ["cyl 1 piston cooling", "no 1 piston co temp"]
    },
    "cyl_2_piston_cooling_oil_outlet_temp_high": {
        "label": "Cyl 2 Piston CO Outlet Temp",
        "unit": "°C",
        "group": "piston_cooling_temp",
        "aliases": ["cyl 2 piston cooling", "no 2 piston co temp"]
    },
    "cyl_3_piston_cooling_oil_outlet_temp_high": {
        "label": "Cyl 3 Piston CO Outlet Temp",
        "unit": "°C",
        "group": "piston_cooling_temp",
        "aliases": ["cyl 3 piston cooling", "no 3 piston co temp"]
    },
    "cyl_4_piston_cooling_oil_outlet_temp_high": {
        "label": "Cyl 4 Piston CO Outlet Temp",
        "unit": "°C",
        "group": "piston_cooling_temp",
        "aliases": ["cyl 4 piston cooling", "no 4 piston co temp"]
    },
    "cyl_5_piston_cooling_oil_outlet_temp_high": {
        "label": "Cyl 5 Piston CO Outlet Temp",
        "unit": "°C",
        "group": "piston_cooling_temp",
        "aliases": ["cyl 5 piston cooling", "no 5 piston co temp"]
    },

    # --- Jacket Cooling Water Outlet Temperatures ---
    "cyl_1_jacket_cooling_water_outlet_temp_high": {
        "label": "Cyl 1 JCW Outlet Temp",
        "unit": "°C",
        "group": "jacket_cooling_temp",
        "aliases": ["cyl 1 jacket cooling", "no 1 jcw temp"]
    },
    "cyl_2_jacket_cooling_water_outlet_temp_high": {
        "label": "Cyl 2 JCW Outlet Temp",
        "unit": "°C",
        "group": "jacket_cooling_temp",
        "aliases": ["cyl 2 jacket cooling", "no 2 jcw temp"]
    },
    "cyl_3_jacket_cooling_water_outlet_temp_high": {
        "label": "Cyl 3 JCW Outlet Temp",
        "unit": "°C",
        "group": "jacket_cooling_temp",
        "aliases": ["cyl 3 jacket cooling", "no 3 jcw temp"]
    },
    "cyl_4_jacket_cooling_water_outlet_temp_high": {
        "label": "Cyl 4 JCW Outlet Temp",
        "unit": "°C",
        "group": "jacket_cooling_temp",
        "aliases": ["cyl 4 jacket cooling", "no 4 jcw temp"]
    },
    "cyl_5_jacket_cooling_water_outlet_temp_high": {
        "label": "Cyl 5 JCW Outlet Temp",
        "unit": "°C",
        "group": "jacket_cooling_temp",
        "aliases": ["cyl 5 jacket cooling", "no 5 jcw temp"]
    },
    "scavenge_air_box_1_temp_high": {
        "label": "Scav Air Box 1 Temp",
        "unit": "°C",
        "group": "scav_air_temp",
        "aliases": ["scav air 1", "scavenge box 1 temp"]
    },
    "scavenge_air_box_2_temp_high": {
        "label": "Scav Air Box 2 Temp",
        "unit": "°C",
        "group": "scav_air_temp",
        "aliases": ["scav air 2", "scavenge box 2 temp"]
    },
    "scavenge_air_box_3_temp_high": {
        "label": "Scav Air Box 3 Temp",
        "unit": "°C",
        "group": "scav_air_temp",
        "aliases": ["scav air 3", "scavenge box 3 temp"]
    },
    "scavenge_air_box_4_temp_high": {
        "label": "Scav Air Box 4 Temp",
        "unit": "°C",
        "group": "scav_air_temp",
        "aliases": ["scav air 4", "scavenge box 4 temp"]
    },
    "scavenge_air_box_5_temp_high": {
        "label": "Scav Air Box 5 Temp",
        "unit": "°C",
        "group": "scav_air_temp",
        "aliases": ["scav air 5", "scavenge box 5 temp"]
    },

    # --- Turbocharger ---
    "M_E_T_C_RPM": {
        "label": "TC RPM",
        "unit": "RPM",
        "group": "tc_rpm",
        "aliases": ["tc rpm", "turbocharger rpm", "turbo rpm", "turbocharger speed"]
    },
    "T_C_inlet_exhaust_gas_temp_high": {
        "label": "TC Exhaust Gas Inlet Temp",
        "unit": "°C",
        "group": "tc_exhaust_temp",
        "aliases": ["tc exhaust inlet", "turbocharger inlet temp"]
    },
    "T_C_outlet_exhaust_gas_temp_high": {
        "label": "TC Exhaust Gas Outlet Temp",
        "unit": "°C",
        "group": "tc_exhaust_temp",
        "aliases": ["tc exhaust outlet", "turbocharger outlet temp"]
    },

    # --- Main Engine Core Parameters ---
    "M_E_ENGINE_RPM": {
        "label": "ME RPM",
        "unit": "RPM",
        "group": "me_core",
        "aliases": ["main engine rpm", "me speed", "engine rpm", "rpm"]
    },
    "M_E_shaft_power": {
        "label": "ME Shaft Power",
        "unit": "kW",
        "group": "me_core",
        "aliases": ["me shaft power", "engine power"]
    },
    "M_E_FUEL_INDEX": {
        "label": "ME Fuel Index",
        "unit": "",
        "group": "me_core",
        "aliases": ["fuel index", "me fuel index"]
    },
    "ME_Load@AVG": {
        "label": "ME Load",
        "unit": "%",
        "group": "me_core",
        "aliases": ["main engine load", "me load", "engine load"]
    },
    "SA_POW_act_kW@AVG": {
        "label": "Shaft Power",
        "unit": "kW",
        "group": "me_core",
        "aliases": ["shaft power", "power output"]
    },
    "SA_SPD_act_rpm@AVG": {
        "label": "Shaft Speed",
        "unit": "RPM",
        "group": "me_core",
        "aliases": ["shaft speed", "shaft rpm"]
    },
    "SA_TQU_act_kNm@AVG": {
        "label": "Shaft Torque",
        "unit": "kNm",
        "group": "me_core",
        "aliases": ["shaft torque", "torque", "me torque"]
    },
    # --- Pressures ---
    "ME_SCAV_AIR_PRESS": {
        "label": "Scavenge Air Pressure",
        "unit": "bar",
        "group": "me_pressure",
        "aliases": ["scav air pressure", "scavenge pressure", "charge air pressure"]
    },
    "scavenge_air_inlet_press_Indi": {
        "label": "Scavenge Air Inlet Pressure",
        "unit": "bar",
        "group": "me_pressure",
        "aliases": ["scav air pressure", "scavenge pressure", "charge air pressure"]
    },
    "jacket_cooling_water_outlet_press_indi": {
        "label": "JCW Outlet Pressure",
        "unit": "bar",
        "group": "me_pressure",
        "aliases": ["jacket cooling pressure", "jcw pressure"]
    },
    "lube_oil_press_T_C_inlet_low": {
        "label": "TC LO Inlet Pressure",
        "unit": "bar",
        "group": "me_pressure",
        "aliases": ["lube oil pressure", "lo pressure", "tc lo pressure"]
    },

    # --- Temperatures ---
    "jacket_CW_inlet_temp_low": {
        "label": "JCW Inlet Temp",
        "unit": "°C",
        "group": "me_temperature",
        "aliases": ["jacket cooling water temp", "jcw inlet temp"]
    },
    "cylinder_L_O_inlet_temp_high": {
        "label": "Cylinder LO Inlet Temp",
        "unit": "°C",
        "group": "me_temperature",
        "aliases": ["cylinder lube oil temp", "cyl lo temp"]
    },
    "lube_oil_temp_T_C_outlet_high": {
        "label": "TC LO Outlet Temp",
        "unit": "°C",
        "group": "me_temperature",
        "aliases": ["tc lube oil temp", "turbocharger lo temp"]
    },
    "intermediate_bearing_temp_high": {
        "label": "Intermediate Bearing Temp",
        "unit": "°C",
        "group": "me_temperature",
        "aliases": ["intermediate bearing", "bearing temp"]
    },
    "thrust_bearing_segment_temp_high": {
        "label": "Thrust Bearing Temp",
        "unit": "°C",
        "group": "me_temperature",
        "aliases": ["thrust bearing", "thrust bearing temp"]
    },
    "sterntube_aft_bearing_temp_high": {
        "label": "Sterntube Aft Bearing Temp",
        "unit": "°C",
        "group": "me_temperature",
        "aliases": ["sterntube bearing", "stern tube temp"]
    },

    # --- Fuel ---
    "ME_FMS_act_kgPh@AVG": {
        "label": "ME Fuel Consumption",
        "unit": "kg/h",
        "group": "me_fuel",
        "aliases": ["fuel consumption", "me fuel", "fuel burn", "fuel rate"]
    },
    "AE_FMS_act_kgPh@AVG": {
        "label": "AE Fuel Consumption",
        "unit": "kg/h",
        "group": "ae_fuel",
        "aliases": ["ae fuel", "auxiliary fuel", "generator fuel"]
    },
}

# ============================================================
# GROUP DEFINITIONS — for natural language group references
# ============================================================
# When user says "all cylinders" or "average of cylinder temps",
# the LLM maps to a group name, and we resolve to all keys in that group.

GROUP_DESCRIPTIONS = {
    "cylinder_exhaust_temp": {
        "description": "Main engine cylinder exhaust gas outlet temperatures (all cylinders)",
        "natural_phrases": [
            "cylinder exhaust", "exhaust gas temp", "exhaust temperature",
            "cylinder temps", "all cylinders exhaust", "cyl exhaust",
            "exhaust gas outlet", "cylinder gas temp"
        ]
    },
    "scav_air_temp": {
        "description": "Scavenge air box temperatures (all cylinders)",
        "natural_phrases": ["scavenge air temp", "scav air temp", "scavenge box temp", "scav air box"]
    },
    "me_temperature": {
        "description": "Main engine temperatures (bearings, lube oil, cooling water)",
        "natural_phrases": ["engine temperature", "me temperature", "bearing temp"]
    },
    "piston_cooling_temp": {
        "description": "Piston cooling oil outlet temperatures (all cylinders)",
        "natural_phrases": [
            "piston cooling", "piston co temp", "piston oil temp",
            "cooling oil outlet"
        ]
    },
    "jacket_cooling_temp": {
        "description": "Jacket cooling water outlet temperatures (all cylinders)",
        "natural_phrases": [
            "jacket cooling", "jcw temp", "jacket water temp",
            "cooling water outlet"
        ]
    },
    "tc_rpm": {
        "description": "Turbocharger RPM values",
        "natural_phrases": ["turbocharger rpm", "tc rpm", "turbo rpm", "turbocharger speed"]
    },
    "tc_exhaust_temp": {
        "description": "Turbocharger exhaust gas temperatures (inlet and outlet)",
        "natural_phrases": ["turbocharger temp", "tc exhaust temp", "turbo temperature"]
    },
    "me_core": {
        "description": "Core main engine parameters (RPM, load, power, torque)",
        "natural_phrases": [
            "main engine", "engine parameters", "me parameters"
        ]
    },
    "me_pressure": {
        "description": "Main engine pressures (scavenge air, lube oil)",
        "natural_phrases": [
            "engine pressure", "me pressure"
        ]
    },
    "me_fuel": {
        "description": "Main engine fuel consumption",
        "natural_phrases": [
            "me fuel", "main engine fuel", "me fuel consumption", "engine fuel burn"
        ]
    },
    "ae_fuel": {
        "description": "Auxiliary engine fuel consumption",
        "natural_phrases": [
            "ae fuel", "auxiliary fuel", "generator fuel", "ae fuel consumption"
        ]
    },
}


# ============================================================
# HELPER: Detect if input is tag-based or natural language
# ============================================================

def is_tag_based(condition: str) -> bool:
    """
    Returns True if the condition contains explicit @TAG references.
    e.g., "if @ME_RPM > 100" → True
    e.g., "alert me if exhaust temp exceeds 400" → False
    """
    return bool(re.search(r'@\w+', condition))


# ============================================================
# HELPER: Get keys by group
# ============================================================

def get_keys_by_group(group_name: str) -> list:
    """Return all sensor keys belonging to a group"""
    return [key for key, info in ME_SENSOR_REGISTRY.items() if info["group"] == group_name]


def get_all_group_names() -> list:
    """Return all available group names"""
    return list(GROUP_DESCRIPTIONS.keys())


# ============================================================
# HELPER: Build sensor context for LLM prompt
# ============================================================

def _build_sensor_context() -> str:
    """Build a compact sensor reference for the LLM prompt"""
    lines = []
    lines.append("AVAILABLE SENSOR GROUPS (use group name when user refers to multiple sensors):")
    for group, info in GROUP_DESCRIPTIONS.items():
        keys = get_keys_by_group(group)
        lines.append(f"  {group}: {info['description']}")
        lines.append(f"    Keys: {', '.join(keys)}")
    
    lines.append("\nAVAILABLE INDIVIDUAL SENSORS:")
    for key, info in ME_SENSOR_REGISTRY.items():
        aliases_str = ", ".join(info["aliases"][:3])
        lines.append(f"  @{key} — {info['label']} ({info['unit']}) [aliases: {aliases_str}]")
    
    return "\n".join(lines)


# ============================================================
# CORE: Resolve natural language to structured condition
# ============================================================

NL_SYSTEM_PROMPT = r"""You are a maritime sensor condition parser. Convert natural language alert conditions into structured tag-based expressions.

RULES:
1. Output ONLY a JSON object. No explanation, no markdown, no backticks.
2. Map natural language to exact sensor keys from the registry below.
3. If user refers to a GROUP of sensors (e.g., "all cylinders", "average of exhaust temps"), use the group name.
4. Supported aggregations: average, max, min, sum, any, all
5. Supported operators: greater_than, less_than, greater_equal, less_equal, equal, not_equal
6. If no aggregation is mentioned and multiple sensors are implied, default to "average".
7. If user mentions a specific cylinder number, use that specific sensor key.
8. Generate a short human-readable rule_name.

OUTPUT FORMAT:
{
    "rule_name": "Short descriptive name",
    "mode": "single" | "group",
    "sensor_key": "@EXACT_KEY (only for mode=single)",
    "group": "group_name (only for mode=group)",
    "aggregation": "average|max|min|sum|any|all (only for mode=group)",
    "operator": "greater_than|less_than|greater_equal|less_equal|equal|not_equal",
    "threshold": number,
    "compound": null | {"logic": "and|or", "conditions": [...]}
}

For compound conditions (AND/OR), use the "compound" field with nested conditions.

SENSOR REGISTRY:
{sensor_context}

EXAMPLES:

Input: "alert me if average exhaust gas temperature of all cylinders exceeds 400"
Output: {{"rule_name":"High Avg Cylinder Exhaust Temp","mode":"group","group":"cylinder_exhaust_temp","aggregation":"average","operator":"greater_than","threshold":400,"compound":null}}

Input: "notify if ME RPM drops below 50"
Output: {{"rule_name":"Low ME RPM","mode":"single","sensor_key":"@ME_RPM","operator":"less_than","threshold":50,"compound":null}}

Input: "alert if cylinder 3 exhaust temp is above 450 or shaft power exceeds 8000"
Output: {{"rule_name":"Cyl 3 Exhaust or High Power","mode":"single","compound":{{"logic":"or","conditions":[{{"sensor_key":"@ME_NO_3_CYL_EXH_GAS_OUTLET_TEMP","operator":"greater_than","threshold":450}},{{"sensor_key":"@SA_POW_act_kW@AVG","operator":"greater_than","threshold":8000}}]}}}}

Input: "check if max turbocharger RPM is above 15000 and scavenge air pressure is below 2"
Output: {{"rule_name":"TC Overspeed with Low Scav Pressure","mode":"group","compound":{{"logic":"and","conditions":[{{"group":"turbocharger","aggregation":"max","operator":"greater_than","threshold":15000}},{{"sensor_key":"@ME_SCAV_AIR_PRESS","operator":"less_than","threshold":2}}]}}}}

Now parse:"""


def resolve_natural_language(condition: str, llm_url: str = "http://localhost:5005/gpu/llm/generate") -> dict:
    """
    Send natural language condition to LLM, get back structured rule JSON.
    
    Returns:
        {
            "success": True/False,
            "rule": { ... parsed rule ... },
            "tag_condition": "@TAG > value ...",  # reconstructed tag-based string
            "error": "..." (if failed)
        }
    """
    sensor_context = _build_sensor_context()
    system_prompt = NL_SYSTEM_PROMPT.replace("{sensor_context}", sensor_context)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": condition}
    ]
    
    try:
        response = requests.post(
            llm_url,
            json={"messages": messages, "response_type": "condition_parse", "enable_thinking": False},
            timeout=120
        )
        raw = response.json().get("response", "").strip()
        
        # Clean LLM output
        raw = raw.replace("```json", "").replace("```", "").strip()
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        
        # Clean common LLM JSON mistakes
        raw = raw.replace("'", '"')  # single quotes to double
        raw = re.sub(r',\s*}', '}', raw)  # trailing comma before }
        raw = re.sub(r',\s*]', ']', raw)  # trailing comma before ]
        raw = raw.replace('{{', '{').replace('}}', '}')

        logger.info(f"LLM RAW OUTPUT: {raw}")

        # Parse JSON
        rule = json.loads(raw)
        
        # Validate but don't block — just warn
        validation = _validate_rule(rule)
        warning = None
        if not validation["valid"]:
            warning = validation["error"]
            logger.warning(f"Rule validation warning (non-blocking): {warning}")
        
        # Reconstruct tag-based condition string for LaTeX conversion
        tag_condition = _rule_to_tag_condition(rule)
        
        return {
            "success": True,
            "rule": rule,
            "tag_condition": tag_condition,
            "warning": warning
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        return {"success": False, "error": f"Failed to parse LLM response as JSON: {e}", "raw_response": raw}
    except Exception as e:
        logger.error(f"NL resolution failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================
# VALIDATION
# ============================================================

VALID_OPERATORS = {"greater_than", "less_than", "greater_equal", "less_equal", "equal", "not_equal"}
VALID_AGGREGATIONS = {"average", "max", "min", "sum", "any", "all"}

def _validate_rule(rule: dict) -> dict:
    """Validate a parsed rule against the sensor registry"""
    
    if not isinstance(rule, dict):
        return {"valid": False, "error": "Rule must be a JSON object"}
    
    # Check compound conditions
    if rule.get("compound"):
        compound = rule["compound"]
        if compound.get("logic") not in ("and", "or"):
            return {"valid": False, "error": f"Invalid compound logic: {compound.get('logic')}"}
        
        for i, cond in enumerate(compound.get("conditions", [])):
            v = _validate_single_condition(cond)
            if not v["valid"]:
                return {"valid": False, "error": f"Condition {i+1}: {v['error']}"}
        
        return {"valid": True}
    
    # Single or group condition
    return _validate_single_condition(rule)


def _validate_single_condition(cond: dict) -> dict:
    """Validate a single condition (not compound)"""
    
    mode = cond.get("mode", "single")
    
    if mode == "single":
        sensor_key = cond.get("sensor_key", "")
        if sensor_key.startswith("@"):
            sensor_key = sensor_key[1:]  # Only strip the leading @
        if sensor_key and sensor_key not in ME_SENSOR_REGISTRY:
            # Try fuzzy match
            close = [k for k in ME_SENSOR_REGISTRY if sensor_key.lower() in k.lower()]
            if close:
                return {"valid": False, "error": f"Unknown sensor '{sensor_key}'. Did you mean: {', '.join(close[:3])}?"}
            return {"valid": False, "error": f"Unknown sensor key: {sensor_key}"}
    
    elif mode == "group":
        group = cond.get("group", "")
        if group not in GROUP_DESCRIPTIONS:
            return {"valid": False, "error": f"Unknown group: {group}. Available: {', '.join(GROUP_DESCRIPTIONS.keys())}"}
        
        agg = cond.get("aggregation", "average")
        if agg not in VALID_AGGREGATIONS:
            return {"valid": False, "error": f"Invalid aggregation: {agg}. Use: {', '.join(VALID_AGGREGATIONS)}"}
    
    operator = cond.get("operator", "")
    if operator and operator not in VALID_OPERATORS:
        return {"valid": False, "error": f"Invalid operator: {operator}. Use: {', '.join(VALID_OPERATORS)}"}
    
    threshold = cond.get("threshold")
    if threshold is not None and not isinstance(threshold, (int, float)):
        return {"valid": False, "error": f"Threshold must be a number, got: {type(threshold).__name__}"}
    
    return {"valid": True}


# ============================================================
# RULE → TAG-BASED CONDITION STRING
# ============================================================

OPERATOR_SYMBOLS = {
    "greater_than": ">",
    "less_than": "<",
    "greater_equal": ">=",
    "less_equal": "<=",
    "equal": "=",
    "not_equal": "!=",
}

def _rule_to_tag_condition(rule: dict) -> str:
    """Convert a structured rule back to a tag-based condition string for LaTeX"""
    
    if rule.get("compound"):
        compound = rule["compound"]
        logic = " and " if compound["logic"] == "and" else " or "
        parts = [_single_condition_to_tag(c) for c in compound["conditions"]]
        return logic.join(parts)
    
    return _single_condition_to_tag(rule)


def _single_condition_to_tag(cond: dict) -> str:
    """Convert a single condition to tag-based string"""
    
    mode = cond.get("mode", "single")
    operator = OPERATOR_SYMBOLS.get(cond.get("operator", ""), ">")
    threshold = cond.get("threshold", 0)
    
    if mode == "group":
        group = cond.get("group", "")
        agg = cond.get("aggregation", "average")
        keys = get_keys_by_group(group)
        
        if not keys:
            return f"UNKNOWN_GROUP({group}) {operator} {threshold}"
        
        # Build aggregation expression
        tag_refs = " + ".join(f"@{k}" for k in keys)
        count = len(keys)
        
        if agg == "any":
            parts = [f"@{k} {operator} {threshold}" for k in keys]
            return " or ".join(parts)
        elif agg == "all":
            parts = [f"@{k} {operator} {threshold}" for k in keys]
            return " and ".join(parts)
        elif agg == "average":
            return f"({tag_refs}) / {count} {operator} {threshold}"
        elif agg == "max":
            return f"max({', '.join(f'@{k}' for k in keys)}) {operator} {threshold}"
        elif agg == "min":
            return f"min({', '.join(f'@{k}' for k in keys)}) {operator} {threshold}"
        elif agg == "sum":
            return f"({tag_refs}) {operator} {threshold}"
        else:
            return f"({tag_refs}) / {count} {operator} {threshold}"
    
    else:
        # sensor_key = cond.get("sensor_key", "").replace("@", "")
        sensor_key = cond.get("sensor_key", "")
        if sensor_key.startswith("@"):
            sensor_key = sensor_key[1:]
        return f"@{sensor_key} {operator} {threshold}"


# ============================================================
# CONVERT TAG CONDITION → LATEX
# ============================================================

def convert_to_latex_local(tag_condition: str) -> str:
    """
    Convert tag-based condition to LaTeX WITHOUT calling the LaTeX service.
    This is a deterministic conversion — no LLM needed.
    
    Input:  "(@ME_NO_1_CYL_EXH + @ME_NO_2_CYL_EXH) / 5 > 400"
    Output: "\\frac{\\text{ME_NO_1_CYL_EXH} + \\text{ME_NO_2_CYL_EXH}}{5} > 400"
    """
    latex = tag_condition
    
    # Replace @TAG_NAME with \text{TAG_NAME}
    latex = re.sub(r'@(\w[\w@.]*)', r'\\text{\1}', latex)
    
    # Replace operators
    latex = latex.replace(">=", r" \geq ")
    latex = latex.replace("<=", r" \leq ")
    latex = latex.replace("!=", r" \neq ")
    latex = latex.replace(">", r" > ")
    latex = latex.replace("<", r" < ")
    latex = latex.replace("=", r" = ")
    # Fix double-replaced >= and <=
    latex = latex.replace(r" \g", r" \g")
    latex = latex.replace(r" \l", r" \l")
    latex = latex.replace(r" \n", r" \n")
    
    # Replace "and" / "or" with LaTeX
    latex = re.sub(r'\band\b', r'\\land', latex)
    latex = re.sub(r'\bor\b', r'\\lor', latex)
    
    # Handle average: (A + B + C) / N → \frac{A + B + C}{N}
    avg_match = re.search(r'\(([^)]+)\)\s*/\s*(\d+)', latex)
    if avg_match:
        numerator = avg_match.group(1)
        denominator = avg_match.group(2)
        frac = f"\\frac{{{numerator}}}{{{denominator}}}"
        latex = latex[:avg_match.start()] + frac + latex[avg_match.end():]
    
    # Handle max/min functions
    latex = re.sub(r'max\(([^)]+)\)', r'\\max\\left(\1\\right)', latex)
    latex = re.sub(r'min\(([^)]+)\)', r'\\min\\left(\1\\right)', latex)
    
    # Clean up extra spaces
    latex = re.sub(r'\s+', ' ', latex).strip()
    
    return latex


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def resolve_condition(condition: str, llm_url: str = "http://localhost:5005/gpu/llm/generate",
                      latex_url: str = None) -> dict:
    condition = condition.strip()
    
    if not condition:
        return {"success": False, "error": "No condition provided"}
    
    result = {
        "original": condition,
        "success": False,
    }
    
    if is_tag_based(condition):
        # ===== PATH A: Has @tags — send straight to LLM for LaTeX =====
        result["input_type"] = "tag_based"
        result["tag_condition"] = condition
        
        try:
            messages = [
                {"role": "system", "content": LATEX_SYSTEM_PROMPT},
                {"role": "user", "content": condition}
            ]
            resp = requests.post(llm_url, json={"messages": messages, "response_type": "latex_conversion"}, timeout=60)
            latex = resp.json().get("response", "").strip()
            latex = latex.replace("```latex", "").replace("```", "").strip().strip("$")
            if "<think>" in latex:
                latex = latex.split("</think>")[-1].strip()
            result["latex"] = latex
            result["success"] = True
        except Exception as e:
            result["error"] = f"LaTeX conversion failed: {e}"
    
    else:
        # ===== PATH B: Natural language — resolve to tags, then LaTeX =====
        result["input_type"] = "natural_language"
        
        nl_result = resolve_natural_language(condition, llm_url)
        
        if not nl_result["success"]:
            # Validation failed but we still try to make LaTeX from whatever we got
            if nl_result.get("raw_response"):
                # Try to extract any tag condition from the raw response and convert anyway
                try:
                    raw = nl_result["raw_response"]
                    raw = raw.replace("'", '"').replace('{{', '{').replace('}}', '}')
                    raw = re.sub(r',\s*}', '}', raw)
                    raw = re.sub(r',\s*]', ']', raw)
                    rule = json.loads(raw)
                    tag_condition = _rule_to_tag_condition(rule)
                    result["tag_condition"] = tag_condition
                    result["rule"] = rule
                    result["warning"] = nl_result.get("error", "")
                except:
                    pass
            
            if not result.get("tag_condition"):
                result["error"] = nl_result.get("error", "NL resolution failed")
                return result
        
        result["rule"] = nl_result["rule"]
        result["tag_condition"] = nl_result["tag_condition"]
        
        # Now send the resolved tag condition to LLM for LaTeX
        try:
            messages = [
                {"role": "system", "content": LATEX_SYSTEM_PROMPT},
                {"role": "user", "content": nl_result["tag_condition"]}
            ]
            resp = requests.post(llm_url, json={"messages": messages, "response_type": "latex_conversion"}, timeout=60)
            latex = resp.json().get("response", "").strip()
            latex = latex.replace("```latex", "").replace("```", "").strip().strip("$")
            if "<think>" in latex:
                latex = latex.split("</think>")[-1].strip()
            result["latex"] = latex
            result["success"] = True
        except Exception as e:
            result["error"] = f"LaTeX conversion failed: {e}"
    
    return result


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    print("=== TAG-BASED TESTS (no LLM needed) ===\n")
    
    tag_tests = [
        "if @ME_RPM > 100 and @V_SOG < 5",
        "@ME_Load@AVG >= 85",
        "@SA_POW_act_kW@AVG > 8000 or @ME_RPM < 40",
        "@ME_SCAV_AIR_PRESS <= 1.5",
        "@ME_NO_1_CYL_EXH_GAS_OUTLET_TEMP != 0",
    ]
    
    for cond in tag_tests:
        result = resolve_condition(cond, llm_url="dummy", latex_url="dummy")
        print(f"Input : {cond}")
        print(f"Type  : {result.get('input_type')}")
        print(f"LaTeX : {result.get('latex')}")
        print(f"OK    : {result.get('success')}")
        print()
    
    print("=== NATURAL LANGUAGE DETECTION ===\n")
    
    nl_tests = [
        "alert me if average exhaust gas temperature of all cylinders exceeds 400",
        "check if ME RPM drops below 50",
        "notify when scavenge air pressure is less than 2 bar",
        "if @ME_RPM > 100",  # This should be detected as tag-based
    ]
    
    for cond in nl_tests:
        detected = "TAG-BASED" if is_tag_based(cond) else "NATURAL LANGUAGE"
        print(f"  '{cond[:60]}...' → {detected}")