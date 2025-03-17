# Runs YOLO on the left cameras and calculates depth using stereo vision (both cameras).
# Make sure to have the calibration data and YOLO model downloaded.

# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
import logging 

# Suppress YOLO console output
logging.disable(logging.CRITICAL)

## --  Load data -- ##
# Inital paramters
# init_data = np.load("Calibration\InitialParameters.npz")
# width = float(init_data["width"])
# height = float(init_data["height"])
width = 720
height = 720
# Calibration parameters
calib_data = np.load("Calibration/stereo_calibration.npz")
mtxL, distL, mtxR, distR = calib_data["mtxL"], calib_data["distL"], calib_data["mtxR"], calib_data["distR"]
FOCAL_LENGTH_L, FOCAL_LENGTH_R = calib_data["FOCAL_LENGTH_L"], calib_data["FOCAL_LENGTH_R"]
BASELINE = calib_data["BASELINE"]
# --------------------------------- ##

## -- Load YOLO model. Comment/Uncomment to use GPU or CPU -- ##
model = YOLO("yolov8n.pt")
# model.to('cuda')  # Run YOLO on GPU
model.to('cpu')  # Run YOLO on CPU
device = 'GPU' if model.device.type == 'cuda' else 'CPU'
print(f"\nYOLO.v8 on {device}")
# ----------------------------------------------------------- ##

## -- Initialzing cameras -- ##
print("Initializing cameras...")
cap_left = cv2.VideoCapture(1, cv2.CAP_DSHOW)
print("Left camera initialized.")
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)
print("Right camera initialized.\n")
# ---------------------------- ##

## -- Set camera resolution -- ##
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
print(f"Camera resolution set to {width} x {height}")
# ------------------------------ ##

# Create stereo depth matcher
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5) # StereoBM (default)
# Using StereoSGBM for better results
blockSize = 15
P1 = 8*3*blockSize**2
P2 = 32*3*blockSize**2
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,           # More disparities = better depth accuracy
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    blockSize=blockSize,
    P1=P1,                      # Controls smoothness
    P2=P2
)

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

    confidence_threshold = 0.3
    # Process detections
    for result in results:
        for box in result.boxes:
            if box.conf[0] < confidence_threshold:
                continue  # Skip detections below the confidence threshold

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"

            # Calculate center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Avoid division by zero (disparity must be > 0)
            # Mask out invalid disparity values (disparity > 0 is valid)
            # Get disparity value at object center
            disparity_value = disparity[center_y, center_x]
            print(f"Disparity at object center ({center_x}, {center_y}): {disparity_value}")
            if disparity_value > 0 and disparity_value < 255:  # 255 is a typical upper limit for disparity values
                depth_cm = (FOCAL_LENGTH_L * BASELINE) / disparity_value
                depth_m = depth_cm / 100  # Convert depth from cm to meters
            else:
                depth_m = 99.99  # If disparity is invalid, set depth to a high value

            # Apply Gaussian smoothing to disparity map to reduce noise
            disparity_smoothed = cv2.GaussianBlur(disparity, (5, 5), 0)

            # Draw bounding box and label
            cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_left, f"{label}, {depth_m:.2f}m", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
    
    # Show images
    cv2.imshow("Left Camera - YOLO + Depth", frame_left)
    # cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Depth Map", cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))


    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nExiting...")
        break

# Cleanup
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

