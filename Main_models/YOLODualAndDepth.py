# Running YOLO on both cameras and calculating depth using stereo vision
# YOLO is used on its own model and with a custom model trained on pallets

print("\nStarting...")

# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time

## -- Manual Input Parameters -- ##
processor = 'GPU'               # 'CPU' or 'GPU'
fps = 60                        # Frames per second
brightness_value = 120          # (0-255)
width = 720; height = width     
depth_distance = 20             # Maximum distance for depth calculation (in meters)
scale_factor = 2.2              # Scale factor for depth calculation
remove_time = 0.1               # Time (in seconds) after which an object is considered 'stale' and removed
object_threshold = 0.4          # Confidence threshold for object detection
pallet_threshold = 0.6          # Confidence threshold for pallet detection
## ---------------------------- ##

# Suppress YOLO console output
logging.disable(logging.CRITICAL)

## -- Load data -- ##
calib_data = np.load("Calibration/stereo_calibration.npz")
mtxL, distL, mtxR, distR = calib_data["mtxL"], calib_data["distL"], calib_data["mtxR"], calib_data["distR"]
FOCAL_LENGTH_L, BASELINE = calib_data["FOCAL_LENGTH_L"], calib_data["BASELINE"]
## --------------- ##

## -- Initializing cameras -- ##
camera_left = cv2.VideoCapture(1, cv2.CAP_DSHOW)
camera_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)
if not camera_left.isOpened():
    print("Error: Unable to open left camera.")
    exit(1)
elif not camera_right.isOpened():
    print("Error: Unable to open right camera.")
    exit(1)
else:
    print('\033[92mCameras initialized.\033[0m')
## ---------------------------- ##

## -- Load YOLO and Custom Model -- ##
yolo_model = YOLO("yolov8n.pt")
pallet_model = YOLO("training/pallets_trained.pt")
if processor == 'GPU':
    yolo_model.to('cuda')
    pallet_model.to('cuda')
else:
    yolo_model.to('cpu')
    pallet_model.to('cpu')
print(f'\033[93mRunning on {processor} at {fps} FPS.')
## ----------------------------------- ##

## -- Detection -- ##
def run_detection(frame, model, conf_threshold):
    results = model(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            if box.conf[0] < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"         # Get the object name and confidence score
            detections.append((x1, y1, x2, y2, label, int(box.cls[0])))
    return detections
## ---------------- ##

## -- Camera settings -- ##
camera_left.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
camera_right.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
camera_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
print(f"Camera resolution set to {width}x{height}.")
print(f"Brightness set to {brightness_value}.")
## ---------------------- ##

## -- Parameters for disparity calculation -- ##
blockSize = 5; P1 = 8*3*blockSize**2; P2 = 32*3*blockSize**2
stereoSGBM = cv2.StereoSGBM_create(minDisparity=0,
    numDisparities=16*16,
    uniquenessRatio=5,
    speckleWindowSize=100,
    speckleRange=16,
    disp12MaxDiff=1,
    blockSize=blockSize,
    P1=P1,
    P2=P2
    )
## ------------------------------------------- ##

# Initializing empty list
object_data = {}

print("\033[92mVisualizing...\033[0m")
frame_time = 1 / fps
while camera_left.isOpened() and camera_right.isOpened():
    ret_left, frame_left = camera_left.read()
    ret_right, frame_right = camera_right.read()
    if not ret_left or not ret_right:
        print("\033[91mError: Could not read from both cameras\033[0m")
        break
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Disparity calculation
    disparity = stereoSGBM.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    # Run object and pallet detection
    all_detections = run_detection(frame_left, yolo_model, object_threshold) + run_detection(frame_left, pallet_model, pallet_threshold)
    
    # Loops through all detections and calculates depth
    for x1, y1, x2, y2, label, cls_id in all_detections:
        object_id = f"{cls_id}_{x1}_{y1}_{x2}_{y2}"         # Unique ID for each object
        RoI = disparity[y1:y2, x1:x2]                       # Region of interest

        # Filter invalid disparities
        valid_disparities = RoI[(RoI > 0) & (RoI < 255)]
        if valid_disparities.size > 0:
            depth = (FOCAL_LENGTH_L * BASELINE) / (np.median(valid_disparities) * scale_factor) / 100
            # Filter objects based on depth
            if depth > depth_distance:
                depth = None
            else:
                object_data[object_id] = {"bbox": (x1, y1, x2, y2), "depth": depth, "label": label, "last_seen": time.time()}
        else:
            depth = None

    # Remove objects that haven't been seen in 'remove_time' seconds
    object_data = {k: v for k, v in object_data.items() if time.time() - v["last_seen"] < remove_time}

    for obj in object_data.values():
        x1, y1, x2, y2 = obj["bbox"]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Displaying bounding box and label, midpoint circle.
        cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_left, obj["label"], (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame_left, obj["label"], (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(frame_left, (center_x, center_y), 4, (0, 0, 255), -1)
        
        # Displaying depth besides the midpoint circle if depth is valid.
        if obj["depth"] is not None:
            depth_text = f"{obj['depth']:.2f}m"
            cv2.putText(frame_left, depth_text, (center_x - 20, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame_left, depth_text, (center_x - 20, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display Object Detection + Depth and the Disparity map
    cv2.imshow("Disparity Map", cv2.applyColorMap(cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET))
    cv2.imshow("Left Camera - Object Detection + Depth", frame_left)

    # Stops the code when pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\033[91mExiting...\033[0m")
        break
    
    key = cv2.waitKey(int(frame_time * 1000))

camera_left.release()
camera_right.release()
cv2.destroyAllWindows()
