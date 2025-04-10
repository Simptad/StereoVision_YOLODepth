#################################################################
#################################################################
# # -------------------------- Imports -------------------------- #
# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
import logging

logging.disable(logging.CRITICAL)   # Suppress YOLO console output

# ------------------ Detection ------------------ #
remove_time = 0.2
object_threshold = 0.5
pallet_threshold = 0.7
blocktunnel_threshold = 0.1
models = [
    (YOLO("yolo11n.pt"), object_threshold),
    (YOLO("training/blocktunnel_trained.pt"), blocktunnel_threshold),
    (YOLO("training/pallets_trained.pt"), pallet_threshold)
]
# target_class = ["person", "block", "tunnel", "pallet"]          # Classes pallet {'pallet', 'block', 'tunnel'}.
target_class = None  # detect all

# ------------------ Camera ------------------ #
LCameraID = 0
RCameraID = 2
brightness_value = 120
width = 960
height = 540

# ------------------ Calibration ------------------ #
calib_data = np.load("Main_Scripts/stereo_calibration.npz")
mtxL, distL, mtxR, distR = calib_data["mtxL"], calib_data["distL"], calib_data["mtxR"], calib_data["distR"]
camera_propeties = [
    calib_data["FOCAL_LENGTH_L"],
    np.linalg.norm(calib_data["T"]),
    78  # Diagonal FoV from manufacturer specification [deg]
]

# ------------------ Processing ------------------ #
processor = 'GPU'           # 'CPU' or 'GPU'
fps = 30
smoothing = 10
frame_time = 1/fps

# ------------------ Depth ------------------ #
depth_distance = 20
scale_factor = 1.9

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


