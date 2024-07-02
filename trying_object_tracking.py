import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Define the class indices to track, in this case, 'person' which is class index 0
class_ids_to_track = [0]

# Perform tracking on the video
results = model.track(
    source='video.mp4', 
    show=True, 
    tracker='bytetrack.yaml',
    classes=class_ids_to_track  # Add the class filter here
)
