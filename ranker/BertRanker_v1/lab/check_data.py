import json

with open("../../data/cnndm/train_data.json") as f:
    data = json.load(f)

print(len(data))

with open("../../data/cnndm/dev_data.json") as f:
    data = json.load(f)

print(len(data))

with open("../../data/cnndm/test_data.json") as f:
    data = json.load(f)

print(len(data))