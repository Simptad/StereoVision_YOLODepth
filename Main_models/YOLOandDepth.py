# Runs YOLO on the left camera and calculates depth using stereo vision (both cameras).
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
window_size = 15             # Window size for median filtering
remove_time = 0.1           # Time (in seconds) after which an object is considered 'stale' and removed
confidence_threshold = 0.6  # Confidence threshold for detections
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
detected_objects = {}
# --------------------------------- ##

## -- Initialzing cameras -- ##
print("Initializing cameras...")
cap_left, cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW), cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap_left.isOpened():
    print("Error: Unable to open left camera.")
    exit(1)
elif not cap_right.isOpened():
    print("Error: Unable to open right camera.")
    exit(1)
else:
    print("Cameras initialized.")
# ---------------------------- ##

## -- Camera settings -- ##
# Set camera brightness
brightness_value = 120  # (0-255)
cap_left.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
cap_right.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
print(f"\nCamera brightness set to {brightness_value}")
# Set camera resolution
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
print(f"Camera resolution set to {width} x {height}")
# ------------------------------ ##

# Create stereo depth matcher
blockSize = 5
P1 = 8*3*blockSize**2
P2 = 32*3*blockSize**2
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*16,
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

        # Calculate the center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Color logic based on depth
        # if depth is None:  
        #     color = (0, 0, 255)             # Red for when only the left camera sees the object
        #     text_color = (0, 0, 255)
        #     depth_text = "No depth"         # No depth shown
        if depth < 50:
            color = (0, 255, 0)             # Green for depth < 10m
            text_color = (0, 0, 0)          # Black text
            depth_text = f"{depth:.2f}m"
        else:
            continue
        # else:
        #     color = (0, 255, 255)           # Yellow for depth >= 10m
        #     text_color = (0, 0, 0)          # Black text
        #     depth_text = f"{depth:.2f}m"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw a circle at the center of the bounding box
        cv2.circle(frame, (center_x, center_y), 5, color, -1)

        # Draw category label at the top-left corner of the bounding box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)

        # Draw depth text above and below the circle
        cv2.putText(frame, depth_text, (center_x - 30, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, depth_text, (center_x - 30, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

frame_time = 1/fps
while cap_left.isOpened() and cap_right.isOpened():
    start_time = time.time()

    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Error: Could not read from both cameras")
        break

    ## -- Preprocess frames for depth calculation -- ##
    # Undistort frames
    frame_left, frame_right = cv2.undistort(frame_left, mtxL, distL), cv2.undistort(frame_right, mtxR, distR)
    # Convert to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY); gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization
    gray_left = cv2.equalizeHist(gray_left); gray_right = cv2.equalizeHist(gray_right)
    ## ----------------------- ##

    # Compute depth map, Fix invalid disparities
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0  # Normalize disparity

    # Run YOLO detections on both cameras
    results_left = model(frame_left); results_right = model(frame_right)
    
    current_time = time.time()
    detected_objects.clear()

    # Process detections
    for result in results_left:
        for box in result.boxes:
            if box.conf[0] < confidence_threshold:
                continue  # Skip detections below the confidence threshold

            # Get bounding box and label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"

            # Object ID based on class 
            # object_id = int(box.cls[0])
            object_id = f"{int(box.cls[0])}_{x1}_{y1}_{x2}_{y2}"

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
                    depth = (FOCAL_LENGTH_L * BASELINE) / (disparity_value * scale_factor) / 100

                    # Apply Exponential Moving Average (EMA) for smoothing
                    if object_id in smoothed_depths:
                        smoothed_depths[object_id] = (smoothing_factor*depth) + (1-smoothing_factor) * smoothed_depths[object_id]
                    else:
                        smoothed_depths[object_id] = depth
                    
                    depth = smoothed_depths[object_id]

                    # Ensure object_data is updated safely
                    if object_id in object_data:
                        object_data[object_id].update({
                            "bbox": (x1, y1, x2, y2),
                            "depth": depth,
                            "label": label,
                            "last_seen": current_time  # Ensure 'last_seen' is always updated
                            })
                    else:
                        object_data[object_id] = {
                            "bbox": (x1, y1, x2, y2),
                            "depth": depth,
                            "label": label,
                            "last_seen": current_time  # Ensure 'last_seen' is initialized
                        }

    # Remove stale objects safely at the end of the loop
    object_data = {obj_id: data for obj_id, data in object_data.items() if "last_seen" in data and current_time - data["last_seen"] < remove_time}
        
    # Draw bounding boxes and depth info
    frame_left = draw_objects_on_frame(frame_left, object_data)

    # Show images
    cv2.imshow("Disparity Map", cv2.applyColorMap(cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET))
    cv2.imshow("Left Camera - YOLO + Depth", frame_left)
    # cv2.imshow("Right Camera", frame_right)

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
