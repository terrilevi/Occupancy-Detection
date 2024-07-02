import cv2
import os

path = r"/home/terry/Desktop/chairs/video.mp4"
# Extrayendo un frame
# Load the video
cap = cv2.VideoCapture(path)
# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
middle_frame_index = total_frames // 2
cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
ret, frame = cap.read()
output_image_path = os.path.join(os.path.dirname(path), "foto.png")
cv2.imwrite(output_image_path, frame)
