#################################################################
#################################################################
# -------------------------- Imports -------------------------- #
# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
import logging

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
target_class = ["person", "block", "tunnel", "pallet"]          # Classes pallet {'pallet', 'block', 'tunnel'}.
# target_class = None                                           # Use this to detect all objects.

# Camera
LCameraID = 0; RCameraID = 1    # Set camera ID (0 is for laptop camera, >0 is for external cameras)
brightness_value = 120          # (0-255)
width = 720; height = width     # Resolution [pixels]

# Computing
processor = 'GPU'               # 'CPU' or 'GPU'  
fps = 30                        # Frames per second 
smoothing = 10                  # Number of frames it calculates the mean of

# Depth Estimation
depth_distance = 20             # Maximum distance for depth calculation (in meters)
scale_factor = 1.9              # Scale factor for depth calculation


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
