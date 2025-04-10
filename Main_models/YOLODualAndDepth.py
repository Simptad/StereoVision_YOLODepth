# Running YOLO on both cameras and calculating depth using stereo vision
# Using YOLO's own trained model together with custom models.
# The custom models includes
# - Pallet detection
# - Pallet tunnel & pallet block detection

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
import math
from collections import deque

logging.disable(logging.CRITICAL)   # Suppress YOLO console output

# Load data
calib_data = np.load("Calibration/stereo_calibration.npz")
mtxL, distL, mtxR, distR = calib_data["mtxL"], calib_data["distL"], calib_data["mtxR"], calib_data["distR"]
camera_propeties = [calib_data["FOCAL_LENGTH_L"], calib_data["BASELINE"], 78]   # 78 Diagonal FoV [deg] (from manufacturer spec.)
object_data = {}    # Initializing empty list
print('\t\033[92mImport \u2714\033[0m')

# ------------------------ END Imports ------------------------ #
#################################################################
#################################################################



###########################################################################
###########################################################################
# ------------------- Global Variables and Parameters ------------------- #
# Object Detection
remove_time = 0.2               # Time (in seconds) after which an object is considered 'stale' and removed
object_threshold = 0.5          # Confidence threshold for object detection
pallet_threshold = 0.7          # Confidence threshold for pallet detection
blocktunnel_threshold = 0.1     # Confidence treshhold for tunnel and block detection

# Path to .pt files (trained_model, threshold)
models = [
    (YOLO("yolo11n.pt"), object_threshold),
    (YOLO("training/blocktunnel_trained.pt"), blocktunnel_threshold),
    (YOLO("training/pallets_trained.pt"), pallet_threshold)
]
target_class = ["person", "block", "tunnel", "pallet"]          # Classes pallet {'block', 'tunnel'}.
# target_class = None                                           # Use this to detect all objects.

# Camera
LCameraID = 0; RCameraID = 2    # Set camera ID (0 is for laptop camera, >0 is for external cameras)
brightness_value = 120          # (0-255)
width = 720; height = width     # Resolution [pixels]

# Computing
processor = 'GPU'               # 'CPU' or 'GPU'  
fps = 30                        # Frames per second 
smoothing = 10                  # Number of frames it calculates the mean of

# Depth Estimation
depth_distance = 20             # Maximum distance for depth calculation (in meters)
scale_factor = 1.9              # Scale factor for depth calculation

# Visualization
cross_size = 5; cross_width = 2         # Center cross dimensions
red = (0,0,255); green = (0,255,0); blue = (255,0,0)
white = (255,255,255); black = (0,0,0);                     
camera_offset_color = (255, 0, 255)


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
def set_processor(models, processor):
    for model, _ in models:
        if processor == 'GPU':
            model.to('cuda')
        else:
            model.to('cpu')

    print(f'\tRunning on {processor} at {fps} FPS.')

# Calculate the cameras horizontal and vertical field of view.
def calculate_fov(diagonal_fov, height, width):

    horizontal_fov = diagonal_fov*(height/math.sqrt(width**2+height**2))
    vertical_fov = diagonal_fov*(width/math.sqrt(width**2+height**2))
    
    return horizontal_fov, vertical_fov

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


# Object Depth Calculation
def depth_calculation(detections, disparity, camera_propeties, scale_factor):
        latest_depth = None
        for x1, y1, x2, y2, label, class_id in detections:
            object_id = f"{class_id}_{x1}_{y1}_{x2}_{y2}"           # Unique ID for each object
            RoI = disparity[y1:y2, x1:x2]                           # Maps out the Region of interest

            # Filter invalid disparities
            valid_disparities = RoI[(RoI > 0) & (RoI < 255)]
            # disparity = cv2.bilateralFilter(disparity.astype(np.uint8), 9, 75, 75)
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

# Depth Map Calculation
def depth_map(disparity, camera_propeties, scale_factor):
    # Calculated the depth map
    disparity[disparity <= 0] = 0.1
    depthmap_values = (camera_propeties[0] * camera_propeties[1]) / (disparity*scale_factor*100)
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

# Checks if the center of "tunnel" or "block" is inside of pallet
def is_inside(center, detection):
    cx, cy = center
    for x1, y1, x2, y2, label, *_ in detection:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False

# Finds the tunnel center from the detection of pallet "tunnel"
def find_tunnel_center(detections):
    # Checks if the center of a "tunnel" or "block" is inside of a pallet bounding box
    tunnel_center_points = []
    is_inside_tunnel = []
    tunnel_coords = []

    detected_pallets = [d for d in detections if "pallet" in d[4].lower()]  # Find all pallet detections

    for x1, y1, x2, y2, label, *_ in detections:
        if "tunnel" in label.lower():
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            is_inside_tunnel = is_inside((mid_x, mid_y), detected_pallets)
            if is_inside_tunnel:
                tunnel_center_points.append((mid_x, mid_y))
                tunnel_coords.append((x1, y1, x2, y2))

    return tunnel_center_points, is_inside_tunnel, tunnel_coords

# Finds the tunnel center from the detection of pallet "blocks"
def find_tunnel_center_blocks(detections, y_tolerance=20):
    # Detects the pallet blocks and calculates the midpoint between them.
    # The midpoint should land in the center of the pallet tunnel.
    # IF only the left and right block are detected, the midpoint will land at the center block.

    # Find all block and pallet detections
    block_detections = [d for d in detections if "block" in d[4].lower()]
    detected_pallets = [d for d in detections if "pallet" in d[4].lower()]
    tunnel_center_points_blocks = []
    is_inside_blocks = []

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

                # Checks if center points are inside of 'pallet'
                is_inside_blocks = is_inside((mid_x_blocks, mid_y_blocks), detected_pallets)
                if is_inside_blocks:
                    tunnel_center_points_blocks.append((mid_x_blocks, mid_y_blocks))

    return tunnel_center_points_blocks, is_inside_blocks

# Calculates the camera offset from the frame center to the closest pallet tunnel
def calculate_camera_offset(frame_center, tunnel_center_points, tunnel_center_points_blocks, disparity):
    # Global variabel for the depth offset from the camera center to pallet tunnel

    # Find the closest center point to the frame center
    combined_center_points = tunnel_center_points + tunnel_center_points_blocks
    if combined_center_points:
        closest_tunnel_center = min(combined_center_points,key=lambda p: (p[0] - frame_center[0])**2 + (p[1] - frame_center[1])**2)
        closest_disp = disparity[closest_tunnel_center[1], closest_tunnel_center[0]]
    else:
        closest_tunnel_center, closest_disp = None, 0

    if 0 < closest_disp < 255: 
        valid_disp = closest_disp 
    else: 
        valid_disp = None

    if valid_disp and closest_tunnel_center:
        # Fetch cameras horizontal and vertical FoV
        horizontal_fov, vertical_fov = calculate_fov(camera_propeties[2], height, width)

        offset_x_pixels = closest_tunnel_center[0] - frame_center[0]
        offset_y_pixels = closest_tunnel_center[1] - frame_center[1]
        offset_angle_x = offset_x_pixels*(horizontal_fov/width)
        offset_angle_y = offset_y_pixels*(vertical_fov/height)

        offset_length = (offset_x_pixels, offset_y_pixels)
        offset_angle = (round(offset_angle_x, 2), round(offset_angle_y, 2))
    else:
        offset_length, offset_angle = (None, None), (None, None)

    return offset_length, closest_tunnel_center, offset_angle


# Visualization
def visualization(frame_left, object_data, tunnel_center_points, frame_center, camera_offset, closest_tunnel_center, tunnel_center_points_blocks, angle_offset,
                  is_inside_tunnel, is_inside_blocks, tunnel_coords):
    # ------ Bounding box visualization ------ #
    tunnel_detected = False

    for tunnel_coord in tunnel_coords:
        x1t, y1t, x2t, y2t = tunnel_coord  # Extract each set of coordinates

        # Draws 'tunnel' if its found inside of a 'pallet'
        for obj in object_data.values():
            # Gets all object data
            label = obj["label"]
            bbox, depth = obj["BoundingBox"], obj["depth"]
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            if "tunnel" in label:
                center_xt, center_yt = (x1t + x2t) // 2, (y1t + y2t) // 2
                tunnel_detected = True

                # Checks if 'tunnel' is inside of pallet bounding box
                if is_inside_tunnel:
                    cv2.rectangle(frame_left, (x1t, y1t), (x2t, y2t), (0, 255, 0), 2)
                    cv2.circle(frame_left, (center_xt, center_yt), 4, (0, 0, 255), -1)
                    cv2.putText(frame_left, label, (x1t + 5, y1t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Top Label text

                    if obj["depth"] is not None:
                        depth_text = f"{obj['depth']:.2f}m"
                        cv2.putText(frame_left, depth_text, (x1t + 5, y2t + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Depth below boundingbox
                    
                    # Draw closest tunnel center circle with the color "camera_offset_color"
                    for mid_x, mid_y in tunnel_center_points:
                        color = camera_offset_color if closest_tunnel_center and (mid_x, mid_y) == closest_tunnel_center else blue
                        cv2.circle(frame_left, (mid_x, mid_y), 5, color, -1)
            
    # Draws 'block' if its found inside of a 'pallet' and "tunnel" is not found
    if not tunnel_detected:
        for obj in object_data.values():
            # Gets all object data
            label = obj["label"]
            bbox, depth = obj["BoundingBox"], obj["depth"]
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2 
            
            if "block" in label.lower():
                if is_inside_blocks:
                    cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame_left, (center_x, center_y), 4, (0, 0, 255), -1)
                    cv2.putText(frame_left, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Top Label text

                    if obj["depth"] is not None:
                        depth_text = f"{obj['depth']:.2f}m"
                        cv2.putText(frame_left, depth_text, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Depth below bounding box

                    # Draw center circle
                    for mid_x, mid_y in tunnel_center_points_blocks: 
                        color = camera_offset_color if closest_tunnel_center and (mid_x, mid_y) == closest_tunnel_center else red
                        cv2.circle(frame_left, (mid_x, mid_y), 5, color, -1)

    # Draws everything else
    for obj in object_data.values():
        label = obj["label"]
        bbox, depth = obj["BoundingBox"], obj["depth"]
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        if "block" not in label.lower() and "tunnel" not in label.lower():
            cv2.rectangle(frame_left, (x1, y1), (x2, y2), green, 2)
            cv2.circle(frame_left, (center_x, center_y), 4, red, -1)
            cv2.putText(frame_left, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)  # Top Label text
            cv2.putText(frame_left, label, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)  # Bottom Label text

            if depth is not None:
                depth_text = f"{obj['depth']:.2f}m"
                cv2.putText(frame_left, depth_text, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)  # Depth below bounding box
    # ------------------------------------------ #

    # ------ Draw 'x' in the center of the camera frame ------ #
    frame_center_x, frame_center_y = frame_center[0], frame_center[1]
    cv2.line(frame_left, (frame_center_x - cross_size, frame_center_y - cross_size), 
            (frame_center_x + cross_size, frame_center_y + cross_size), camera_offset_color, cross_width)
    cv2.line(frame_left, (frame_center_x - cross_size, frame_center_y + cross_size), 
            (frame_center_x + cross_size, frame_center_y - cross_size), camera_offset_color, cross_width)
    #---------------------------------------------------#

    # ------ Visualize camera offset to pallet center ------ #
    if camera_offset and closest_tunnel_center:
        angle_x, angle_y = angle_offset
        height, _, _ = frame_left.shape
        tunnel_center_x, tunnel_center_y = closest_tunnel_center

        # Start and end points for the angle offsets
        start_point_x = (frame_center_x, height)
        end_point_x = (tunnel_center_x, tunnel_center_y)
        start_point_y = (0, frame_center_y)
        end_point_y = (tunnel_center_x, tunnel_center_y)

        # Display the angle offset lines
        cv2.line(frame_left, start_point_x, end_point_x, camera_offset_color, 2)
        cv2.putText(frame_left, f"{angle_x} deg", (frame_center_x+10, height-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, camera_offset_color, 1, cv2.LINE_AA)
        
        cv2.line(frame_left, start_point_y, end_point_y, camera_offset_color, 2)
        cv2.putText(frame_left, f"{angle_y} deg", (10, frame_center_y+2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, camera_offset_color, 1, cv2.LINE_AA)

    #--------------------------------------------------------#

# Show camera feed
def display(frame_left, depthmap, disparitymap):
        # Crop window
        crop_left, crop_right = (340, 0)

        # Disparity Map
        # disparitymap_cropped = disparitymap[:, crop_left:disparitymap.shape[1] - crop_right]
        # Depth Map
        # depthmap_cropped = depthmap[:, crop_left:depthmap.shape[1] - crop_right]

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
set_processor(models, processor)

# Read the first frame to calculate center of the frame
_, frame_left = camera_left.read()
frame_center = (frame_left.shape[1] // 2, frame_left.shape[0] // 2)     # Calculate the center of the frame

## ---- MAIN LOOP ---- ##
frame_time = 1/fps
print("Visualizing... (press 'Q' to stop)")
while camera_left.isOpened() and camera_right.isOpened():
    _, frame_left = camera_left.read()
    _, frame_right = camera_right.read()
    
    # Calculate the disparity for the combined image
    disparity, disparitymap = disparity_calculation(frame_left, frame_right)

    # Run object and pallet detection
    all_detections = []
    for model, threshold in models:
        detection = run_detection(frame_left, model, threshold, target_class)
        all_detections.extend(detection)

    # Object Depth Calculations
    object_data = depth_calculation(all_detections, disparity, camera_propeties, scale_factor)
    object_data = {k: v for k, v in object_data.items() if time.time() - v["last_seen"] < remove_time}  # Remove objects that haven't been seen in 'remove_time' seconds
    # clean_object_data(object_data, remove_time)

    # Depth map
    depthmap = depth_map(disparity, camera_propeties, scale_factor)
    
    # Find pallet tunnel center
    tunnel_center_points_blocks, is_inside_blocks= find_tunnel_center_blocks(all_detections)
    tunnel_center_points, is_inside_tunnel, tunnel_coords = find_tunnel_center(all_detections)

    # Calculate camera offset to the pallet tunnel center
    camera_offset, closest_tunnel_center, camera_angle_offset = calculate_camera_offset(frame_center, tunnel_center_points, tunnel_center_points_blocks, disparity)

    # Visualize
    visualization(frame_left, object_data, tunnel_center_points, frame_center, camera_offset, closest_tunnel_center, tunnel_center_points_blocks, camera_angle_offset,
                  is_inside_tunnel, is_inside_blocks, tunnel_coords)

    # Display figures
    display(frame_left, depthmap, disparitymap)

    # Breaks out of the loop when pressing 'Q' and stopping code
    if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord('q'):
        print("\033[91mStopping...\033[0m\n")
        break
## ------- END MAIN LOGIC ------- ##

# Stop and Quit
camera_left.release(); camera_right.release(); cv2.destroyAllWindows()
