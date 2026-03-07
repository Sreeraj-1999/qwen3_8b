# import sqlite3
# import sys
# import json

# DB_PATH = r"C:\Users\User\Downloads\imo_9841938.s3db" # change this

# conn = sqlite3.connect(DB_PATH)
# total = conn.execute("SELECT COUNT(*) FROM VesselData").fetchone()[0]
# print(f"Searching all {total} records...")

# rows = conn.execute("SELECT payload FROM VesselData").fetchall()

# target = "SA_SPD_act_rpm@AVG"
# found = []

# for row in rows:
#     try:
#         p = json.loads(row[0])
#         val = p.get(target)
#         if val is not None:
#             found.append(val)
#     except:
#         pass

# print(f"'{target}' found in {len(found)}/{total} records")
# if found:
#     print(f"Sample values: {found[:10]}")
# else:
#     print("NOT present in Flora's data at all")

# conn.close()

# # Basic counts and date range
# # row = conn.execute("""
# #     SELECT 
# #         COUNT(*) as total_rows,
# #         MIN(vesselTimeStamp) as earliest,
# #         MAX(vesselTimeStamp) as latest
# #     FROM VesselData
# # """).fetchone()

# # print(f"Total rows    : {row[0]}")
# # print(f"Earliest      : {row[1]}")
# # print(f"Latest        : {row[2]}")

# # # Sample one payload to confirm SOG/position keys exist
# # sample = conn.execute("SELECT payload FROM VesselData ORDER BY vesselTimeStamp DESC LIMIT 1").fetchone()
# # if sample:
# #     import json
# #     p = json.loads(sample[0])
# #     keys_we_need = ["V_SOG_act_kn@AVG", "V_STW_act_kn@AVG", "V_GPSLAT_act_deg@LAST", 
# #                     "V_GPSLON_act_deg@LAST", "ME_FMS_act_kgPh@AVG", "ME_RPM"]
# #     print("\nKey data availability:")
# #     for k in keys_we_need:
# #         val = p.get(k)
# #         print(f"  {k}: {val}")

# # conn.close()

import requests

r = requests.get(
    "https://nominatim.openstreetmap.org/reverse",
    params={"lat": 29.82469, "lon": -90.00139, "format": "json"},
    headers={"User-Agent": "MarineAI/1.0"},
    timeout=5
)
print(r.json())