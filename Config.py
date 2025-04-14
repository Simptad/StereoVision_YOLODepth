# ------------------------------------------------------------------------
# All the necessary parameters are defined here.






# ------------------------------------------------------------------------
# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
import logging

logging.disable(logging.CRITICAL)   # Suppress YOLO console output

# ------------------ Detection ------------------ #
remove_time = 0.2                   # Object removal time [s]
object_threshold = 0.5              # Threshold for the YOLO model
pallet_threshold = 0.7              # Threshold for the pallet model
blocktunnel_threshold = 0.1         # Treshold for the pallet tunnel and pallet block model
models = [
    (YOLO("Models/yolo11n.pt"), object_threshold),
    (YOLO("Models/blocktunnel_trained.pt"), blocktunnel_threshold),
    (YOLO("Models/pallets_trained.pt"), pallet_threshold)
]
target_class = ["person", "block", "tunnel", "pallet"]          # Classes pallet {'pallet', 'block', 'tunnel'}.
# target_class = None  # Detect all

# ------------------ Camera ------------------ #
CameraID = (1, 2)               # Camera IDs for left and right cameras
brightness_value = 120          # Brightness value (0-255)
width = 960; height = 540       # Target resolution

# ------------------ Calibration ------------------ #
calib_data = np.load("Models/stereo_calibration.npz")
mtxL, distL, mtxR, distR = calib_data["mtxL"], calib_data["distL"], calib_data["mtxR"], calib_data["distR"]
camera_propeties = [
    calib_data["FOCAL_LENGTH_L"],               # Calculated focal length from calibration
    np.linalg.norm(calib_data["T"]),            # Calculated baseline from calibration
    78                                          # Diagonal FoV from manufacturer specification [deg]
]
original_calibdata_res = (1920, 1080)           # Resolution of the original calibration data

# ------------------ Processing ------------------ #
processor = 'GPU'           # 'CPU' or 'GPU'
fps = 30
frame_time = 1/fps

# ------------------ Depth ------------------ #
depth_distance = 20         # When to cut off the depth estimation [m]
scale_factor = 1.9          # Scale factor. Should not be changed.

# ------------------ Disparity ------------------ #
blockSize = 5
P1 = 8*3*blockSize**2
P2 = 32*3*blockSize**2
stereoSGBM = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,
    uniquenessRatio=5,
    speckleWindowSize=100,
    speckleRange=16,
    disp12MaxDiff=1,
    blockSize=blockSize,
    P1=P1, P2=P2
)


