
# Downloads the YOLO.v8 model
# If file "yolov8n.pt" already exists, it will not be downloaded again

from ultralytics import YOLO
YOLO("yolo11n.pt")  # Downloads the model
