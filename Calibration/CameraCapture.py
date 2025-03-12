# This script captures simultanously images from two cameras and saves them in folders Calibrationpictures_L/H

import cv2
import os

## -- User input -- ##
# Resolution [pixels]
width = 1920
height = 1080
internal_width = 10
internal_height = 7

# Initialize cameras (Change indexes if needed)
cameraL = cv2.VideoCapture(0)  # Left Camera
cameraR = cv2.VideoCapture(1)  # Right Camera


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

def detect_and_draw_chessboard_corners(image, pattern_size=(internal_width, internal_height), camera_name=""):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if ret:
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)
    else:
        print(f"Corners not found for *{camera_name}* camera.")
    return image

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
            # Run chessboard corner detection on the captured image
            chessboard_image_L = detect_and_draw_chessboard_corners(frameL, camera_name="Left")
            cv2.imshow("Chessboard Detection Left", chessboard_image_L)  # Show the result
        if retR:
            filenameR = f"Calibration/Calibrationpictures_H/right_{image_count:03d}.jpg"
            cv2.imwrite(filenameR, frameR)
            print(f"Captured {filenameR}")
            # Run chessboard corner detection on the captured image
            chessboard_image_R = detect_and_draw_chessboard_corners(frameR, camera_name="Right")
            cv2.imshow("Chessboard Detection Right", chessboard_image_R)  # Show the result
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
