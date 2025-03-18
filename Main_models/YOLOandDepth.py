# Runs YOLO on the left cameras and calculates depth using stereo vision (both cameras).
# Make sure to have the calibration data and YOLO model downloaded.

# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
import logging 

# Set camera resolution
width = 720
height = 720

# Suppress YOLO console output
logging.disable(logging.CRITICAL)

## --  Load data -- ##
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

## -- Increase the brightness of the cameras -- ##
brightness_value = 180  # (0-255)
cap_left.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
cap_right.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
print(f"Camera brightness set to {brightness_value}")
# ------------------------------ ##

## -- Set camera resolution -- ##
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
print(f"Camera resolution set to {width} x {height}")
# ------------------------------ ##

# Create stereo depth matcher
# stereo = cv2.StereoBM_create(numDisparities=16*12, blockSize=11) # StereoBM (default)
# Using StereoSGBM for better results
blockSize = 7
P1 = 8*3*blockSize**2
P2 = 32*3*blockSize**2
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*16,           # More disparities = better depth accuracy
    uniquenessRatio=5,           
    speckleWindowSize=100,                      
    speckleRange=16,
    disp12MaxDiff=1,    
    blockSize=blockSize,
    P1=P1,                      
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

    # Apply Histogram Equalization
    gray_left = cv2.equalizeHist(gray_left)
    gray_right = cv2.equalizeHist(gray_right)

    # Compute depth map, Fix invalid disparities
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0  # Normalize disparity
    disparity[disparity < 0] = 0.1
    # Apply Median and Gaussian Blur for Smoother Disparity
    disparity = cv2.medianBlur(disparity, 5)  # Reduce high-frequency noise
    disparity = cv2.GaussianBlur(disparity, (5,5), 0)  # Smooth sharp jumps

    # Run YOLO detections on both cameras
    results_left = model(frame_left)
    results_right = model(frame_right)
    confidence_threshold = 0.75

    # Process detections
    for result in results_left:
        for box in result.boxes:
            if box.conf[0] < confidence_threshold:
                continue  # Skip detections below the confidence threshold

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"

            # Check if the object is visible in the right camera (by checking for a similar bounding box)
            corresponding_box_right = None
            for result_right in results_right:
                for box_right in result_right.boxes:
                    # Check if the object in the right camera corresponds to the object in the left camera
                    if box.cls[0] == box_right.cls[0]:
                        # You can further refine this by comparing position or overlap, for now, we check class
                        corresponding_box_right = box_right
                        break

            # If a corresponding object was found in both cameras, calculate depth
            if corresponding_box_right is not None:
                center_x_left = (x1 + x2) // 2
                center_y_left = (y1 + y2) // 2
                center_x_right = (int(corresponding_box_right.xyxy[0][0]) + int(corresponding_box_right.xyxy[0][2])) // 2
                center_y_right = (int(corresponding_box_right.xyxy[0][1]) + int(corresponding_box_right.xyxy[0][3])) // 2

                # If the object is present in both cameras, compute disparity and depth
                disparity_value = disparity[center_y_left, center_x_left]  # Get disparity value from left camera position
                if 0 < disparity_value < 255:
                    depth_cm = (FOCAL_LENGTH_L * BASELINE) / (disparity_value*2)
                    depth_m = depth_cm / 100  # Convert depth to meters
                    if depth_m < 10:
                        # Draw bounding box and label with depth
                        cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_left, f"{label}, {depth_m:.2f}m", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (0, 255, 0), 2)
                    else:
                        # Do not display any depth value if disparity is invalid
                        cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame_left, f"{label}, Invalid Depth", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (0, 255, 255), 2)
            else:
                # Do not display any depth value if disparity is invalid
                cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_left, f"{label}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 255), 2)
    
    # Show images
    cv2.imshow("Left Camera - YOLO + Depth", frame_left)
    cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Depth Map", cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    # cv2.imshow("Rectified Left", frame_left_rect)
    # cv2.imshow("Rectified Right", frame_right_rect)


    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nExiting...")
        break

# Cleanup
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

