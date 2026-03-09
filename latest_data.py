# import sqlite3
# import json

# db_path = r"C:\Users\User\Desktop\Main Engine Diagnostics\Flora Schulte (IMO 9841938)\Telemetry Data\imo_9841938.s3db"

# conn = sqlite3.connect(db_path)
# row = conn.execute("SELECT payload FROM VesselData ORDER BY vesselTimeStamp DESC LIMIT 1").fetchone()
# conn.close()

# payload = json.loads(row[0])

# print(f"Total keys: {len(payload)}\n")
# for i, (key, value) in enumerate(sorted(payload.items()), 1):
#     print(f"{i:4d}. {key:60s} = {value}")
##############
# import sqlite3, json

# db_path = r"C:\Users\User\Desktop\Main Engine Diagnostics\Flora Schulte (IMO 9841938)\Telemetry Data\imo_9841938.s3db"

# conn = sqlite3.connect(db_path)
# row = conn.execute("SELECT payload FROM VesselData ORDER BY vesselTimeStamp DESC LIMIT 1").fetchone()
# conn.close()

# payload = json.loads(row[0])

# # Print all ME/engine related keys with current values
# for key in sorted(payload.keys()):
#     k = key.upper()
#     if any(x in k for x in ['ME_', 'CYL', 'SHAFT', 'SA_', 'TC_', 'SCAV', 'PISTON', 'TURBO', 'TORSION', 'LO_', 'COOL', 'EXH']):
#         print(f"{key} = {payload[key]}")
###### test `1` ###############
# import json

# def explore_json(data, indent=0):
#     space = "  " * indent

#     if isinstance(data, dict):
#         for key, value in data.items():
#             print(f"{space}Key: {key} | Type: {type(value).__name__}")
#             explore_json(value, indent + 1)

#     elif isinstance(data, list):
#         print(f"{space}List of {len(data)} items")
#         if len(data) > 0:
#             print(f"{space}First item type: {type(data[0]).__name__}")
#             explore_json(data[0], indent + 1)

#     else:
#         print(f"{space}Value: {data}")


# # load json
# with open(r"C:\Users\User\Desktop\Main Engine Diagnostics\APL_HOUSTON (IMO 9597537)\Telemetry Data\db 2.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# print("=== JSON STRUCTURE ===")
# explore_json(data)

#
# import json

# with open(r"C:\Users\User\Desktop\Main Engine Diagnostics\APL_HOUSTON (IMO 9597537)\Telemetry Data\db 2.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # models = set()

# # for item in data:
# #     if isinstance(item, dict) and "model" in item:
# #         models.add(item["model"])

# # print("Models found:\n")
# # for m in sorted(models):
# #     print(m)

# for item in data:
#     if item.get("model") == "LogApp.mqttdata":
#         print("\nPK:", item.get("pk"))
#         print("FIELDS:")
        
#         for k, v in item.get("fields", {}).items():
#             print(f"{k}: {v}")
        
#         print("-" * 40)
#         break 
# 
import json
import sqlite3
import os

# Input JSON
json_path = r"C:\Users\User\Desktop\Main Engine Diagnostics\APL_HOUSTON (IMO 9597537)\Telemetry Data\db 2.json"

# Output s3db (this is what mcp_telemetry.py expects)
db_path = r"C:\Users\User\Desktop\Main Engine Diagnostics\APL_HOUSTON (IMO 9597537)\Telemetry Data\imo_9597537.s3db"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

conn = sqlite3.connect(db_path)
conn.execute("""
    CREATE TABLE IF NOT EXISTS VesselData (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        payload TEXT,
        fk_vessel TEXT,
        createdAt TEXT,
        vesselTime TEXT,
        vesselTimeStamp TEXT
    )
""")

count = 0
for item in data:
    if item.get("model") == "LogApp.mqttdata":
        fields = item.get("fields", {})
        payload = fields.get("payload", {})
        
        # payload might be a dict or a string
        if isinstance(payload, dict):
            payload_str = json.dumps(payload)
            timestamp = str(payload.get("time", fields.get("vesselutctimestamp", "")))
        else:
            payload_str = str(payload)
            timestamp = str(fields.get("vesselutctimestamp", ""))
        
        conn.execute(
            "INSERT INTO VesselData (payload, fk_vessel, createdAt, vesselTime, vesselTimeStamp) VALUES (?, ?, ?, ?, ?)",
            (
                payload_str,
                str(fields.get("fk_vessel", "")),
                fields.get("created_at", ""),
                "",
                timestamp
            )
        )
        count += 1

conn.commit()
conn.close()

print(f"Done. Inserted {count} records into {db_path}")   