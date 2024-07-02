import json

path = '_annotations.coco.json'

# Read the JSON file
with open(path, 'r') as file:
    data = json.load(file)

# Extract bounding boxes
bounding_boxes = [annotation['bbox'] for annotation in data['annotations']]

# Print bounding boxes
for bbox in bounding_boxes:
    print(bbox)
