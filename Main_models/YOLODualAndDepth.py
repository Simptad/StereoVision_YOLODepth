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
# pallet_model = YOLO("training/palletbox_trained.pt")
pallet_model = YOLO("training/blocktunnel_trained.pt")
target_class = ["person", "block", "tunnel"]          # Classes pallet {'block', 'tunnel'}.
# target_class = None                       # Use this to detect all objects.

# Camera
LCameraID = 1; RCameraID = 2    # Set camera ID (0 is for laptop camera, >0 is for external cameras)
brightness_value = 120          # (0-255)
width = 720; height = width     # Resolution [pixels]

# Computing
processor = 'GPU'               # 'CPU' or 'GPU'  
fps = 15                        # Frames per second 

# Depth Estimation
depth_distance = 20             # Maximum distance for depth calculation (in meters)
scale_factor = 1.9              # Scale factor for depth calculation

# Object Detection
remove_time = 0.1               # Time (in seconds) after which an object is considered 'stale' and removed
object_threshold = 0.4          # Confidence threshold for object detection
pallet_threshold = 0.3          # Confidence threshold for pallet detection

# Visualization
cross_size = 5; cross_width = 2         # Center cross dimensions
red = (0,0,255); green = (0,255,0); blue = (255,0,0)
white = (255,255,255); black = (0,0,0);                     
purple = (255, 0, 255)


# Disparity calculation
blockSize = 5; P1 = 8*3*blockSize**2; P2 = 32*3*blockSize**2
stereoSGBM = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*21,           # Depth resolution
    uniquenessRatio=5,              # Higher value to reduce noise
    speckleWindowSize=100,          # Larger window to remove speckles
    speckleRange=16,                # Larger range to handle more variations
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

# Set processor for the models
def set_processor(model1, model2, processor):
    if processor == 'GPU':
        model1.to('cuda'); model2.to('cuda')
    else:
        model1.to('cpu');  model2.to('cpu')
    print(f'\tRunning on {processor} at {fps} FPS.')

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

# Created for validation script, will be remove later
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
        latest_depth = None
        for x1, y1, x2, y2, label, class_id in detections:
            object_id = f"{class_id}_{x1}_{y1}_{x2}_{y2}"           # Unique ID for each object
            RoI = disparity[y1:y2, x1:x2]                           # Maps out the Region of interest

            # Filter invalid disparities
            valid_disparities = RoI[(RoI > 0) & (RoI < 255)]
            disparity = cv2.bilateralFilter(disparity.astype(np.uint8), 9, 75, 75)
            if valid_disparities.size > 0:
                depth = (camera_propeties[0] * camera_propeties[1]) / (np.median(valid_disparities) * scale_factor)/100
                depth = (-0.7388+np.sqrt(0.7388**2-4*0.0478*(0.3123-depth)))/(2*0.0478)     # Adjusted depth

                # Filter objects based on a predefined depth (How far do you want to look)
                if depth < depth_distance:
                    object_data[object_id] = {"BoundingBox": (x1, y1, x2, y2), "depth": depth, "label": label, "last_seen": time.time()}
                    latest_depth = depth
                else:
                    object_data[object_id] = {"BoundingBox": (x1, y1, x2, y2), "depth": latest_depth, "label": label, "last_seen": time.time()}
            else:
                object_data[object_id] = {"BoundingBox": (x1, y1, x2, y2), "depth": None, "label": label, "last_seen": time.time()}

        return object_data

# def calculate_tunnel_depth(camera_propeties, scale_factor, disparity, tunnel_center_points):
#     # Calculate tunnel center depths
#     tunnel_depth = []
#     for mid_x, mid_y in tunnel_center_points:
#         # Ensure the point is within the disparity map bounds
#         if 0 <= mid_x < disparity.shape[1] and 0 <= mid_y < disparity.shape[0]:
#             disparity_value = disparity[mid_y, mid_x]

#             # Skip invalid disparity values
#             if disparity_value > 0:
#                 depth = (camera_propeties[0] * camera_propeties[1]) / (disparity_value * scale_factor) / 100  # Convert to meters
#                 depth = (-0.7388 + np.sqrt(0.7388**2 - 4 * 0.0478 * (0.3123 - depth))) / (2 * 0.0478)  # Adjusted depth
#                 tunnel_depth.append((mid_x, mid_y, depth))
#             else:
#                 tunnel_depth.append((mid_x, mid_y, None))  # Invalid depth
#         else:
#             tunnel_depth.append((mid_x, mid_y, None))  # Out of bounds
#     return tunnel_depth

# Depth Map Calculation
def depth_map(disparity, camera_propeties, scale_factor):
    # Calculated the depth map
    disparity[disparity <= 0] = 0.1
    depthmap_values = (camera_propeties[0] * camera_propeties[1]) / (disparity * scale_factor)/100
    depthmap_values = (-0.7388+np.sqrt(0.7388**2-4*0.0478*(0.3123-depthmap_values)))/(2*0.0478)
    
    # Clip depth values to a reasonable range
    depthmap_values = np.clip(depthmap_values, 0.5, 15)

    # Normalize depth and apply color for visualization
    depthmap_normalized = cv2.normalize(depthmap_values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depthmap = cv2.applyColorMap(depthmap_normalized, cv2.COLORMAP_PLASMA)

    return depthmap

# Disparity Calculation
def disparity_calculation(frame_left, frame_right):
    # Converting images to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Disparity calculation for the whole scene
    disparity = stereoSGBM.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Clip disparity values to resonable values for disparity map
    disparitymap_clip = np.clip(disparity, 0.01, 255)
    # Normalize depth and apply color for visualization
    disparity_normalized = cv2.normalize(disparitymap_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disparitymap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_CIVIDIS)

    return disparity, disparitymap

def find_tunnel_center(detections):
    tunnel_center_points = []
    for x1, y1, x2, y2, label, *_ in detections:
        if "tunnel" in label.lower():
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            tunnel_center_points.append((mid_x, mid_y))
    return tunnel_center_points

def find_tunnel_center_blocks(detections, y_tolerance=20):
    # Detects the pallet blocks and calculates the midpoint between them.
    # The midpoint should land in the center of the pallet tunnel.
    # IF only the left and right block are detected, the midpoint will land at the center block.

    # Filter detections for blocks only
    block_detections = [d for d in detections if "block" in d[4].lower()]
    tunnel_center_points_blocks = []

    # Sort blocks by their y1 coordinate (top of the bounding box)
    block_detections.sort(key=lambda b: b[1])

    # Pair blocks based on y-level proximity
    for i, block1 in enumerate(block_detections):
        x1_1, y1_1, x2_1, y2_1, _, _ = block1
        mid_x1, mid_y1 = (x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2

        for j, block2 in enumerate(block_detections[i + 1:], start=i + 1):
            x1_2, y1_2, x2_2, y2_2, _, _ = block2
            mid_x2, mid_y2 = (x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2

            # Check if the blocks are at roughly the same y-level
            if abs(mid_y1 - mid_y2) <= y_tolerance:
                # Calculate the midpoint between the two blocks
                mid_x_blocks = (mid_x1 + mid_x2) // 2
                mid_y_blocks = (mid_y1 + mid_y2) // 2
                tunnel_center_points_blocks.append((mid_x_blocks, mid_y_blocks))
    return tunnel_center_points_blocks

def calculate_camera_offset(frame_center, tunnel_center_points, tunnel_center_points_blocks):
    # Prioritize tunnel center points if available, otherwise use block center points
    # if tunnel_center_points:                                    
    #     closest_tunnel_center = min(tunnel_center_points, key=lambda p: (p[0] - frame_center[0])**2 + (p[1] - frame_center[1])**2)
    # elif tunnel_center_points_blocks:                           
    #     closest_tunnel_center = min(tunnel_center_points_blocks, key=lambda p: (p[0] - frame_center[0])**2 + (p[1] - frame_center[1])**2)
    #     print(f"Closest: {closest_tunnel_center}")
    # else:
    #     return (None, None), None  # No center points available
    all_center_points = tunnel_center_points + tunnel_center_points_blocks
    if all_center_points:  # If there are any center points
        # Find the closest center point to the frame center
        closest_tunnel_center = min(
            all_center_points,
            key=lambda p: (p[0] - frame_center[0])**2 + (p[1] - frame_center[1])**2
        )
    else:  # No center points available
        return (None, None), None

    # Calculate the offset in pixels
    offset_x = closest_tunnel_center[0] - frame_center[0]
    offset_y = closest_tunnel_center[1] - frame_center[1]

    return (offset_x, offset_y), closest_tunnel_center

# Visualization
def visualization(frame_left, object_data, tunnel_center_points, frame_center, camera_offset, closest_tunnel_center, tunnel_center_points_blocks):
    # ------ Bounding box visualization ------ #
    for obj in object_data.values():
        bbox, label, depth = obj["BoundingBox"], obj["label"], obj["depth"]
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
        # Displaying bounding box.
        cv2.rectangle(frame_left, (x1, y1), (x2, y2), black, 2)
        
        # Draw bounding box midpoint circle.
        cv2.circle(frame_left, (center_x, center_y), 4, red, -1)

        # Draw bounding box labels and confidence score
        if label.split()[0] == 'block' or label.split()[0] == 'tunnel':
            cv2.putText(frame_left, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)                      # Top Label text
        else:
            cv2.putText(frame_left, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)                          # Top Label text
            cv2.putText(frame_left, label, (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)                    # Bottom Label text
        
        if depth is not None:
            depth_text = f"{obj['depth']:.2f}m"
            cv2.putText(frame_left, depth_text, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)  # Depth below boundingbox
    #------------------------------------------#

    # ------ Draw 'x' in the center of the camera frame ------ #
    frame_center_x, frame_center_y = frame_center[0], frame_center[1]
    cv2.line(frame_left, (frame_center_x - cross_size, frame_center_y - cross_size), 
            (frame_center_x + cross_size, frame_center_y + cross_size), purple, cross_width)
    cv2.line(frame_left, (frame_center_x - cross_size, frame_center_y + cross_size), 
            (frame_center_x + cross_size, frame_center_y - cross_size), purple, cross_width)
    #---------------------------------------------------#

    # ------ Visualize camera offset to pallet center ------ #
    visualization_center = closest_tunnel_center
    if camera_offset:
        offset_x, offset_y = camera_offset[0], camera_offset[1]
        cv2.putText(frame_left, f"Offset: ({offset_x}, {offset_y}) [pixels]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, purple, 2)
        
        # Determine which center to use for visualization
        # if tunnel_center_points:  # Prioritize tunnel if visible
        #     visualization_center = closest_tunnel_center
        # elif tunnel_center_points_blocks:  # Use blocks if tunnel is not visible
        #     visualization_center = tunnel_center_points_blocks[0]  # Use the first block pair center
        if visualization_center:
            # Draw arrows pointing to the selected center
            cv2.arrowedLine(frame_left, frame_center, (visualization_center[0], frame_center_y), purple, 2, tipLength=0.2)  # X-axis arrow
            cv2.arrowedLine(frame_left, frame_center, (frame_center_x, visualization_center[1]), purple, 2, tipLength=0.2)  # Y-axis arrow

            # Display offset values near the arrows
            cv2.putText(frame_left, f"X: {offset_x}", (visualization_center[0] + 10, frame_center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 2)
            cv2.putText(frame_left, f"Y: {offset_y}", (frame_center_x + 10, visualization_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 2)
    #--------------------------------------------------------#

    # ------ Pallet Tunnel Center Visualization ------ #
    for mid_x, mid_y in tunnel_center_points:                   # Object "tunnel"
        color = purple if visualization_center and (mid_x, mid_y) == visualization_center else blue
        cv2.circle(frame_left, (mid_x, mid_y), 5, color, -1)

    for mid_x, mid_y in tunnel_center_points_blocks:            # Object "blocks"
        color = purple if visualization_center and (mid_x, mid_y) == visualization_center else red
        cv2.circle(frame_left, (mid_x, mid_y), 5, color, -1)  
    #--------------------------------------------------#

# Show camera feed
def display(frame_left, depthmap, disparitymap):
        # Crop window
        crop_left, crop_right = (340, 0)

        # Disparity Map
        disparitymap_cropped = disparitymap[:, crop_left:disparitymap.shape[1] - crop_right]
        # Depth Map
        depthmap_cropped = depthmap[:, crop_left:depthmap.shape[1] - crop_right]

        # Show camera feed
        # cv2.imshow("Disparity Map", disparitymap_cropped)
        # cv2.imshow("Depth Map", depthmap_cropped)
        cv2.imshow("Left Camera - Object Detection + Depth", frame_left)

# --------------------------- END DEFINITIONS ----------------------------- #
#############################################################################
#############################################################################



########################################################################
########################################################################
# --------------------------- MAIN LOGIC ----------------------------- #
camera_left, camera_right = init_cameras(LCameraID, RCameraID, width, height, brightness_value)
set_processor(yolo_model, pallet_model, processor)

# Read the first frame to calculate center of the frame
_, frame_left = camera_left.read()
frame_center = (frame_left.shape[1] // 2, frame_left.shape[0] // 2)     # Calculate the center of the frame

## ---- MAIN LOOP ---- ##
frame_time = 1 / fps
print("Visualizing... (press 'Q' to stop)")
while camera_left.isOpened() and camera_right.isOpened():
    _, frame_left = camera_left.read()
    _, frame_right = camera_right.read()
    
    # Calculate the disparity for the combined image
    disparity, disparitymap = disparity_calculation(frame_left, frame_right)

    # Run object and pallet detection
    all_detections = run_detection(frame_left, yolo_model, object_threshold, target_class) + run_detection(frame_left, pallet_model, pallet_threshold, target_class)

    # Object Depth Calculations
    object_data = depth_calculation(all_detections, disparity, camera_propeties, scale_factor)
    object_data = {k: v for k, v in object_data.items() if time.time() - v["last_seen"] < remove_time}  # Remove objects that haven't been seen in 'remove_time' seconds

    # Depth map
    depthmap = depth_map(disparity, camera_propeties, scale_factor)
    
    # Find pallet tunnel center
    tunnel_center_points_blocks= find_tunnel_center_blocks(all_detections)
    tunnel_center_points = find_tunnel_center(all_detections)

    # Calculate camera offset to the pallet tunnel center
    camera_offset, closest_tunnel_center = calculate_camera_offset(frame_center, tunnel_center_points, tunnel_center_points_blocks)

    # Calculate tunnel depth
    # tunnel_depth = calculate_tunnel_depth(camera_propeties, scale_factor, disparity, tunnel_center_points)

    # Visualize
    visualization(frame_left, object_data, tunnel_center_points, frame_center, camera_offset, closest_tunnel_center, tunnel_center_points_blocks)

    # Display figures
    display(frame_left, depthmap, disparitymap)

    # Breaks out of the loop when pressing 'Q' and stopping code
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\033[91mStopping...\033[0m\n")
        break
    cv2.waitKey(int(frame_time * 1000))
## ------- END MAIN LOGIC ------- ##

# Stop and Quit
camera_left.release(); camera_right.release(); cv2.destroyAllWindows()
