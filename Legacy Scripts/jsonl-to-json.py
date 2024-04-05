# Revised approach to handle JSONL to JSON conversion considering the specific structure of the provided data
import json

jsonl_file_path = 'conversation_trees_dual_v3.jsonl'
json_file_path = 'conversation_trees_dual_v3.json'

# Reading the JSONL file and converting it to JSON format
json_data = []

with open(jsonl_file_path, 'r') as file:
    # Split the content by the actual newline character (\n) and process each JSON object
    lines = file.read().split('\n')
    for line in lines:
        if line:  # Check if the line is not empty
            json_data.append(json.loads(line))

# Writing the converted data to a JSON file
with open(json_file_path, 'w') as file:
    json.dump(json_data, file, indent=4) 