import json

# Specify the path to your JSON file
json_file_path = "./fix/amount/small_result.json"
# json_file_path = "./dynamic/purterbation/move/time_36/cube.json"


# Open and read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)  # Load the JSON content into a Python dictionary

scoop_failures = 0
spillage = 0
success = 0

for i in data["spillage_scoop"]:
    if i[0] != 0:
        spillage += 1
    if i[1] == 0:
        scoop_failures += 1
    if i[0] == 0 and i[1] != 0:
        success += 1

print("Scoop success       :", len(data["spillage_scoop"]) - scoop_failures)
print("without Spillage    :", len(data["spillage_scoop"]) - spillage)
print("Successes rate      :", success/len(data["spillage_scoop"]))
