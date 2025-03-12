# This script captures simultanously images from two cameras and saves them in folders Calibrationpictures_L/H

import cv2
import os

# Initialize cameras (Change indexes if needed)
cameraL = cv2.VideoCapture(0)  # Left Camera
cameraR = cv2.VideoCapture(1)  # Right Camera

# Resolution [pixels]
width = 1920
height = 1080

# Set resolution (Modify as needed)
if cameraL.isOpened():
    cameraL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cameraL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
else:
    print("Left camera (0) not detected.")

if cameraR.isOpened():
    cameraR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cameraR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
else:
    print("Right camera (1) not detected.")

image_count = 0

while True:
    # Capture frames.
    frameL, frameR = None, None
    retL, retR = False, False
    
    # Read frames from cameras
    retL, frameL = cameraL.read() 
    retR, frameR = cameraR.read()
    
    if frameL is not None:
        cv2.imshow("Left Camera", frameL)
    if frameR is not None:
        cv2.imshow("Right Camera", frameR)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 13:  # Enter key to capture image
        if retL:
            filenameL = f"Calibration/Calibrationpictures_L/left_{image_count:03d}.jpg"
            cv2.imwrite(filenameL, frameL)
            print(f"Captured {filenameL}")
        if retR:
            filenameR = f"Calibration/Calibrationpictures_H/right_{image_count:03d}.jpg"
            cv2.imwrite(filenameR, frameR)
            print(f"Captured {filenameR}")
        image_count += 1
    
    elif key == ord('q'):  # Press 'q' to exit
        break

# Release resources
if cameraL.isOpened():
    cameraL.release()
if cameraR.isOpened():
    cameraR.release()
cv2.destroyAllWindows()
print("Camera capture stopped.")