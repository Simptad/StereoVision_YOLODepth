
# Runs YOLO on the left cameras and calculates depth using stereo vision (both cameras).
# Make sure to have the calibration data and YOLO model downloaded.

import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")
model.to('cuda')  # Run YOLO on GPU
# model.to('cpu')  # Run YOLO on CPU

# Open both cameras (Left = Camera 0, Right = Camera 1)
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

# Set camera resolution (optional, adjust based on your cameras)
width, height = 640, 480
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Create stereo depth matcher
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15) # StereoBM (default)
# Using StereoSGBM for better results
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # More disparities = better depth accuracy
    blockSize=15,  # Adjust for your scene
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    P1=8*3*15**2,  # Controls smoothness
    P2=32*3*15**2
)

# Define camera parameters (adjust based on your setup)
# Load calibration data
calib_data = np.load("stereo_calibration.npz")
mtxL, distL, mtxR, distR, FOCAL_LENGTH = calib_data["mtxL"], calib_data["distL"], calib_data["mtxR"], calib_data["distR"]

FOCAL_LENGTH = 700  # Pixels. Hide this if you have calibration data.
BASELINE = 6  # Distance between cameras in cm

while cap_left.isOpened() and cap_right.isOpened():
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Error: Could not read from both cameras")
        break

    # Undistort frames before depth calculation
    frame_left = cv2.undistort(frame_left, mtxL, distL)
    frame_right = cv2.undistort(frame_right, mtxR, distR)

    # Convert to grayscale for depth calculation
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Compute depth map (disparity)
    disparity = stereo.compute(gray_left, gray_right)
    disparity = np.float32(disparity) / 16.0  # Normalize disparity

    # Run YOLO only on the LEFT camera
    results = model(frame_left)

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"

            # Calculate center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Get disparity value at object center
            disparity_value = disparity[center_y, center_x]

            # Avoid division by zero (disparity must be > 0)
            if disparity_value > 0:
                depth_cm = (FOCAL_LENGTH * BASELINE) / disparity_value
            else:
                depth_cm = 9999  # If disparity is invalid, set depth very high

            # Draw bounding box and label
            cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_left, f"{label}, {depth_cm:.1f}cm", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Show images
    cv2.imshow("Left Camera - YOLO + Depth", frame_left)
    cv2.imshow("Depth Map", cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
