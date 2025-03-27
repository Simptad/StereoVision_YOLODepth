# Running YOLO on both cameras and calculating depth using stereo vision
# YOLO is used on its own model and with a custom model trained on pallets

print("\nStarting...")

#################################################################
#################################################################
# -------------------------- Imports -------------------------- #
# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
logging.disable(logging.CRITICAL)   # Suppress YOLO console output
import os

# Load data
calib_data = np.load("Calibration/stereo_calibration.npz")
mtxL, distL, mtxR, distR = calib_data["mtxL"], calib_data["distL"], calib_data["mtxR"], calib_data["distR"]
# FOCAL_LENGTH_L, BASELINE = calib_data["FOCAL_LENGTH_L"], calib_data["BASELINE"]
camera_propeties = [calib_data["FOCAL_LENGTH_L"], calib_data["BASELINE"]]
object_data = {}    # Initializing empty list
print('\t\033[92mImport \u2714\033[0m')

# ------------------------ END Imports ------------------------ #
#################################################################
#################################################################



###########################################################################
###########################################################################
# ------------------- Global Variables and Parameters ------------------- #
# Path to .pt files
yolo_model = YOLO("yolo11n.pt")
pallet_model = YOLO("training/palletbox_trained.pt")
target_class = ["person", "bottom"]    # Classes pallet {'block', 'bottom', 'stringer', 'top'}

# Set camera ID (0 is for laptop camera, >0 is for external cameras)
LCameraID = 1; RCameraID = 2

# Manual Input Parameters
processor = 'GPU'               # 'CPU' or 'GPU'
fps = 30                        # Frames per second
brightness_value = 120          # (0-255)
width = 1080; height = width     
depth_distance = 20             # Maximum distance for depth calculation (in meters)
scale_factor = 1.9                # Scale factor for depth calculation
remove_time = 0.1               # Time (in seconds) after which an object is considered 'stale' and removed
object_threshold = 0.4          # Confidence threshold for object detection
pallet_threshold = 0.6          # Confidence threshold for pallet detection

# Parameters for disparity calculation
blockSize = 5; P1 = 8*3*blockSize**2; P2 = 32*3*blockSize**2
stereoSGBM = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*16,
    uniquenessRatio=5,
    speckleWindowSize=100,
    speckleRange=16,
    disp12MaxDiff=1,
    blockSize=blockSize, P1=P1, P2=P2
    )
print('\t\033[92mVariables and Parameters \u2714\033[0m')

# ----------------- END Global Variables and Parameters ----------------- #
###########################################################################
###########################################################################



#########################################################################
#########################################################################
# --------------------------- DEFINITIONS ----------------------------- #
# Initialize Cameras
def init_cameras(LCameraID, RCameraID, w, h, bvalue):
    print("\t\033[93mInitializing cameras..\033[0m")
    camera_left = cv2.VideoCapture(LCameraID, cv2.CAP_DSHOW)
    camera_right = cv2.VideoCapture(RCameraID, cv2.CAP_DSHOW)
    if not camera_left.isOpened():
        print("\t\t\033[91mError: Unable to open left camera.\033[0m")
        exit(1)
    elif not camera_right.isOpened():
        print("\t\t\033[91mError: Unable to open right camera.\033[0m")
        exit(1)
    else:
        camera_left.set(cv2.CAP_PROP_BRIGHTNESS, bvalue)
        camera_right.set(cv2.CAP_PROP_BRIGHTNESS, bvalue)
        camera_left.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        camera_left.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        camera_right.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        camera_right.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        print(f"\t\tCamera resolution set to {w}x{h}.")
        print(f"\t\tBrightness set to {bvalue}.")
        print('\t\033[92mCameras initialized \u2714 \033[0m')
    return camera_left, camera_right

# Object Detection
def run_detection(frame, model, conf_threshold, target_class):
    results = model(frame)
    BoundingBox = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])                      # Class id
            class_name = model.names[class_id]              # Class name that corresponds to its id
            box_conf = float(box.conf)                      # Confidence score for the bounding 
            x1, y1, x2, y2 = map(int, box.xyxy[0])          # Extract box corner coordinates

            # Filtering using confidence threshold and target class
            if (box_conf < conf_threshold) or (target_class is not None and class_name not in target_class):
                continue

            label = f"{class_name} {box_conf:.2f}"
            BoundingBox.append((x1, y1, x2, y2, label, class_id))
    return BoundingBox

def append_depth_values(depthdata):
    file_path = "Validation/Depth_data/depth_values.txt"; data = []
    start_meter = 10

    depth, depth_new = depthdata[0], depthdata[1]
    data.append(f"Before: {depth:.3f}\n")
    data.append(f"After: {depth_new:.3f}\n")

    # Ensure the file exists before reading
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            file.write("")  # Create an empty file if it doesn't exist
    
    # Open the file in read mode to check existing content
    with open(file_path, "r") as file:
        content = file.readlines()
    
    # Open the file in write mode to modify or append content
    with open(file_path, "w") as file:
        found_section = False
        # Write existing content back, looking for the specified meter section
        for line in content:
            file.write(line)
            if line.strip() == f"({start_meter}) meter":
                found_section = True
        
        # If the section wasn't found, add a new section header
        if not found_section:
            file.write(f"\n({start_meter}) meter\n")
        
    # Open the file again in append mode to add the data without overwriting
    with open(file_path, "a") as file:
        file.writelines(data)

# Object Depth Calculation
def depth_calculation(detections, disparity, camera_propeties, scale_factor):

        for x1, y1, x2, y2, label, class_id in detections:
            object_id = f"{class_id}_{x1}_{y1}_{x2}_{y2}"           # Unique ID for each object
            RoI = disparity[y1:y2, x1:x2]                           # Maps out the Region of interest

            # Filter invalid disparities
            valid_disparities = RoI[(RoI > 0) & (RoI < 255)]
            if valid_disparities.size > 0:
                depth = (camera_propeties[0] * camera_propeties[1]) / (np.median(valid_disparities) * scale_factor)/100
                depth_new = (-0.7388+(0.7388**2-4*0.0478*(0.3123-depth))**0.5)/(2*0.0478)

                # Filter objects based on a predefined depth (How far do you want to look)
                if depth < depth_distance:
                    object_data[object_id] = {"BoundingBox": (x1, y1, x2, y2), "depth": depth, "label": label, "last_seen": time.time()}

        # Calculated the depth map
        disparity[disparity == 0] = 0.1
        depth_map = (camera_propeties[0] * camera_propeties[1]) / (disparity * scale_factor)/100

        # Normalize depth for visualization
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply color map
        depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

        append_depth_values([depth, depth_new])
        return object_data, depth_colormap

def disparity_calculation(frame_left, frame_right):
    # Converting images to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Disparity calculation for the whole scene
    disparity = stereoSGBM.compute(gray_left, gray_right).astype(np.float32) / 16.0

    return disparity

# Visualization
def draw_visuals (frame_left, bbox, label, depth):
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
    # Displaying bounding box, label.
    cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_left, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame_left, label, (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw midpoint circle.
    cv2.circle(frame_left, (center_x, center_y), 4, (0, 0, 255), -1)

    # Draw arrows from borders to the center.
    # cv2.arrowedLine(frame_left, (x1, center_y), (center_x//2, center_y), (255, 0, 0), 2, tipLength=0.2)  # Horizontal arrow
    # cv2.arrowedLine(frame_left, (center_x//2, y1), (center_x//2, center_y), (0, 255, 255), 2, tipLength=0.2)  # Vertical arrow

    # Displaying depth besides the midpoint circle if depth is valid.
    if depth is not None:
        depth_text = f"{obj['depth']:.2f}m"
        cv2.putText(frame_left, depth_text, (center_x - 20, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame_left, depth_text, (center_x - 20, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Set processor for the models
def set_processor(model1, model2, processor):
    if processor == 'GPU':
        model1.to('cuda'); model2.to('cuda')
    else:
        model1.to('cpu');  model2.to('cpu')
    print(f'\tRunning on {processor} at {fps} FPS.')

# --------------------------- END DEFINITIONS ----------------------------- #
#############################################################################
#############################################################################



########################################################################
########################################################################
# --------------------------- MAIN LOGIC ----------------------------- #
camera_left, camera_right = init_cameras(LCameraID, RCameraID, width, height, brightness_value)
set_processor(yolo_model, pallet_model, processor)

## ---- MAIN LOOP ---- ##
print("Visualizing...")
frame_time = 1 / fps
while camera_left.isOpened() and camera_right.isOpened():
    _, frame_left = camera_left.read()
    _, frame_right = camera_right.read()

    # Calculate the disparity for the combined image
    disparity = disparity_calculation(frame_left, frame_right)

    # Run object and pallet detection
    all_detections = run_detection(frame_left, yolo_model, object_threshold, target_class) + run_detection(frame_left, pallet_model, pallet_threshold, target_class)

    # Depth calculations
    object_data, depth_colormap = depth_calculation(all_detections, disparity, camera_propeties, scale_factor)
    object_data = {k: v for k, v in object_data.items() if time.time() - v["last_seen"] < remove_time}  # Remove objects that haven't been seen in 'remove_time' seconds

    # Visualize bounding boxes, labels and depth
    for obj in object_data.values():
        draw_visuals(frame_left, obj["BoundingBox"], obj["label"], obj["depth"])
    
    # Display Object Detection + Depth and the Disparity map
    # cv2.imshow("Disparity Map", cv2.applyColorMap(cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET))
    # cv2.imshow("Depth Map", depth_colormap)
    cv2.imshow("Left Camera - Object Detection + Depth", frame_left)

    # Breaks out of the loop when pressing 'Q' and stopping code
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\033[91mStopping...\033[0m\n")
        break
    cv2.waitKey(int(frame_time * 1000))
## ------- END MAIN LOGIC ------- ##

# Stop and Quit
camera_left.release(); camera_right.release(); cv2.destroyAllWindows()
