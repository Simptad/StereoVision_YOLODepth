# Runs YOLO on the left cameras and calculates depth using stereo vision (both cameras).
# Make sure to have the calibration data and YOLO model downloaded.

# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
import matplotlib.pyplot as plt

## -- Manual Input Parameters -- ##
width = 720; height = width
fps = 60
processor = 'CPU'           # 'CPU' or 'GPU'
scale_factor = 2.2          # Scale factor for depth calculation
smoothing_factor = 1.0      # Smoothing factor for depth values
window_size = 8             # Window size for median filtering
# ------------------------ ##

# Suppress YOLO console output
logging.disable(logging.CRITICAL)

## --  Load data -- ##
# Calibration parameters
calib_data = np.load("Calibration/stereo_calibration.npz")
mtxL, distL, mtxR, distR = calib_data["mtxL"], calib_data["distL"], calib_data["mtxR"], calib_data["distR"]
FOCAL_LENGTH_L, FOCAL_LENGTH_R = calib_data["FOCAL_LENGTH_L"], calib_data["FOCAL_LENGTH_R"]
BASELINE = calib_data["BASELINE"]
mapL1, mapL2 = calib_data["mapL1"], calib_data["mapL2"]
mapR1, mapR2 = calib_data["mapR1"], calib_data["mapR2"]
# --------------------------------- ##

## -- Load YOLO model. Comment/Uncomment to use GPU or CPU -- ##
model = YOLO("yolov8n.pt")
if processor == 'GPU':
    model.to('cuda')  # Run YOLO on GPU
else:
    model.to('cpu')  # Run YOLO on CPU
print(f"\nYOLO.v8 on {processor} at {fps} FPS")
# ----------------------------------------------------------- ##

## -- Initialize a variable to store smoothed depth values for each object -- ##
smoothed_depths = {}
object_data = {}
# --------------------------------- ##

## -- Initialzing cameras -- ##
print("Initializing cameras...")
cap_left = cv2.VideoCapture(1, cv2.CAP_DSHOW)
print("Left camera initialized.")
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)
print("Right camera initialized.\n")
# ---------------------------- ##

## -- Camera settings -- ##
# Set camera brightness
brightness_value = 150  # (0-255)
cap_left.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
cap_right.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
print(f"Camera brightness set to {brightness_value}")
# Set camera resolution
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
print(f"Camera resolution set to {width} x {height}")
# ------------------------------ ##

# Create stereo depth matcher
# stereo = cv2.StereoBM_create(numDisparities=16*12, blockSize=11) # StereoBM (default)
# Using StereoSGBM for better results
blockSize = 5
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

def draw_objects_on_frame(frame, object_data):
    for obj_id, data in object_data.items():
        x1, y1, x2, y2 = data["bbox"]
        label = data["label"]
        depth = data["depth"]

        # Color logic based on depth
        if depth is None:  
            color = (0, 0, 255)  # Red for single-camera detection (no depth)
            text_color = (0, 0, 255)
            text = f"{label}"  # No depth shown
        elif depth < 10:
            color = (0, 255, 0)  # Green for depth < 10m
            text_color = (0, 0, 0)  # Black text
            text = f"{label} ({depth:.2f}m)"
        else:
            color = (0, 255, 255)  # Yellow for depth >= 10m
            text_color = (0, 0, 0)  # Black text
            text = f"{label} ({depth:.2f}m)"

        # Draw bounding box, label and depth
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)
    return frame


frame_time = 1/fps
while cap_left.isOpened() and cap_right.isOpened():
    start_time = time.time()

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

    # Run YOLO detections on both cameras
    results_left = model(frame_left)
    results_right = model(frame_right)
    confidence_threshold = 0.6

    # Process detections
    for result in results_left:
        for box in result.boxes:
            if box.conf[0] < confidence_threshold:
                continue  # Skip detections below the confidence threshold

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"

            object_id = int(box.cls[0])
            current_time = time.time()
            # Check if the object is visible in the right camera (by checking for a similar bounding box)
            corresponding_box_right = None
            for result_right in results_right:
                for box_right in result_right.boxes:

                    # Check if the object in the right camera corresponds to the object in the left camera
                    if box.cls[0] == box_right.cls[0]:  # Match based on class only
                        corresponding_box_right = box_right
                        break
                if corresponding_box_right is not None:
                    break

            # If a corresponding object was found in both cameras, calculate depth
            if corresponding_box_right is not None:
                center_x_left = (x1 + x2) // 2
                center_y_left = (y1 + y2) // 2  

                # If the object is present in both cameras, compute disparity and depth
                disparity_value = disparity[center_y_left, center_x_left]  # Get disparity value from left camera position

                # Use Median Filtering for Disparity 
                half_window = window_size // 2
                
                if (center_y_left - half_window >= 0 and center_y_left + half_window < disparity.shape[0] and 
                    center_x_left - half_window >= 0 and center_x_left + half_window < disparity.shape[1]):

                    region = disparity[center_y_left - half_window:center_y_left + half_window + 1,
                                       center_x_left - half_window:center_x_left + half_window + 1]
                    disparity_value = np.median(region)  # Median filtering
                else:
                    disparity_value = disparity[center_y_left, center_x_left]  # Fallback

                if 0 < disparity_value < 255:
                    depth_cm = (FOCAL_LENGTH_L * BASELINE) / (disparity_value*scale_factor)
                    depth_m = depth_cm / 100  # Convert depth to meters

                    # Apply Exponential Moving Average (EMA) for smoothing
                    if object_id in smoothed_depths:
                        smoothed_depths[object_id] = (smoothing_factor*depth_m) + (1-smoothing_factor) * smoothed_depths[object_id]
                    else:
                        smoothed_depths[object_id] = depth_m
                    
                    depth_m = smoothed_depths[object_id]


                    # Ensure object_data is updated safely
                    if object_id in object_data:
                        object_data[object_id].update({
                            "bbox": (x1, y1, x2, y2),
                            "depth": depth_m,
                            "label": label,
                            "last_seen": current_time  # Ensure 'last_seen' is always updated
                        })
                    else:
                        object_data[object_id] = {
                            "bbox": (x1, y1, x2, y2),
                            "depth": depth_m,
                            "label": label,
                            "last_seen": current_time  # Ensure 'last_seen' is initialized
                        }
            else:
                # Do not display any depth value if disparity is invalid
                cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_left, f"{label}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 255), 2)


    # Remove stale objects safely at the end of the loop
    object_data = {obj_id: data for obj_id, data in object_data.items() if "last_seen" in data and current_time - data["last_seen"] < 1.0}
        
    # Draw bounding boxes and depth info
    frame_left = draw_objects_on_frame(frame_left, object_data)

    # Show images
    cv2.imshow("Left Camera - YOLO + Depth", frame_left)
    # cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Disparity Map", cv2.applyColorMap(cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET))


    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nExiting...")
        break

    elapsed_time = time.time() - start_time
    if elapsed_time < frame_time:
        time.sleep(frame_time - elapsed_time)

# Cleanup
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
