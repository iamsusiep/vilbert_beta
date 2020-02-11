import json

with open('val_result.json') as f:
    data = json.loads(f.read())

print(data[0])
