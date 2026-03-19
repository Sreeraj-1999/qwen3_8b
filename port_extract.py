import json

with open("ports.json", "r") as f:
    data = json.load(f)

port_names = list(data.keys())

print(port_names)