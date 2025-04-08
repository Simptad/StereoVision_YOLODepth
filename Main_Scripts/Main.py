# Running YOLO on both cameras and calculating depth using stereo vision
# Using YOLO's own trained model together with custom models.
# The custom models includes
# - Pallet detection
# - Pallet tunnel & pallet block detection

print("\nStarting...")

#################################################################
#################################################################
# -------------------------- Imports -------------------------- #

# Import
from Parameters import * 
from CameraModelInit import *
from Disparity import *
from Detection import *
from Depth import *
from TunnelCenter import *
from CameraOffset import *
from Visualization import *

# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time

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


########################################################################
########################################################################
# --------------------------- MAIN LOGIC ----------------------------- #
camera_left, camera_right = init_cameras(LCameraID, RCameraID, width, height, brightness_value)
set_processor(models, processor)
print(f'\tRunning on {processor} at {fps} FPS.')

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
    disparity, disparitymap = disparity_calculation(frame_left, frame_right, stereoSGBM)

    # Run object and pallet detection
    all_detections = []
    for model, threshold in models:
        detection = run_detection(frame_left, model, threshold, target_class)
        all_detections.extend(detection)

    # Object Depth Calculations
    object_data = depth_calculation(all_detections, disparity, camera_propeties, scale_factor, depth_distance, object_data)
    object_data = {k: v for k, v in object_data.items() if time.time() - v["last_seen"] < remove_time}  # Remove objects that haven't been seen in 'remove_time' seconds

    # Depth map
    # depthmap = depth_map(disparity, camera_propeties, scale_factor)
    
    # Find pallet tunnel center
    tunnel_center_points_blocks, is_inside_blocks= find_tunnel_center_blocks(all_detections)
    tunnel_center_points, is_inside_tunnel, tunnel_coords = find_tunnel_center(all_detections)

    # Calculate camera offset to the pallet tunnel center
    camera_offset, closest_tunnel_center, camera_angle_offset = calculate_camera_offset(frame_center, tunnel_center_points, tunnel_center_points_blocks, disparity)

    # Visualize
    visualization(frame_left, object_data, tunnel_center_points, frame_center, camera_offset, closest_tunnel_center, tunnel_center_points_blocks, camera_angle_offset,
                  is_inside_tunnel, is_inside_blocks, tunnel_coords)

    # Display figures
    display(frame_left, _, disparitymap)

    # Breaks out of the loop when pressing 'Q' and stopping code
    if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord('q'):
        print("\033[91mStopping...\033[0m\n")
        break
# ------------------------ END MAIN LOGIC ------------------------ #
####################################################################
####################################################################

# Stop and Quit
camera_left.release(); camera_right.release(); cv2.destroyAllWindows()
