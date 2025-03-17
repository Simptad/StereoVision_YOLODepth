import cv2
import os
import numpy as np

## -- Parameters -- ##
# Resolution [pixels]
width = 1920
height = 1080
# Board parameters
square_size = 3.5  # Size of one square in cm
internal_width = 10
internal_height = 7
# Camera parameters
BASELINE = 39  # Distance between cameras in cm

# Initialize cameras
print("\nInitializing cameras....")
cameraL = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Left Camera
print("Left camera initialized.")
cameraR = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Right Camera
print("Right camera initialized.\n")

# Sets the resolution for both cameras
if cameraL.isOpened():
    cameraL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cameraL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print("Camera_L resolution set to", width, "x", height)
else:
    print("Left camera (0) not detected.")

if cameraR.isOpened():
    cameraR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cameraR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print("Camera_R resolution set to", width, "x", height)
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
    return ret, image

print("Starting capture....\n")
while True:
    # Capture frames from both cameras
    retL, frameL = cameraL.read()
    retR, frameR = cameraR.read()

    if frameL is not None:
        cv2.imshow("Left Camera", frameL)
    if frameR is not None:
        cv2.imshow("Right Camera", frameR)

    key = cv2.waitKey(1) & 0xFF 

    # Enter key to capture image
    if key == 13:  # Enter key
        if retL and retR:
            # Check chessboard detection on both cameras
            cornersL_detected, chessboard_image_L = detect_and_draw_chessboard_corners(frameL, camera_name="Left")
            cornersR_detected, chessboard_image_R = detect_and_draw_chessboard_corners(frameR, camera_name="Right")

            if cornersL_detected and cornersR_detected:
                # Save images if both cameras detect corners
                filenameL = f"Calibration/Calibrationpictures_L/left_{image_count:03d}.jpg"
                filenameR = f"Calibration/Calibrationpictures_R/right_{image_count:03d}.jpg"
                cv2.imwrite(filenameL, frameL)
                cv2.imwrite(filenameR, frameR)
                print(f"Captured {filenameL} and {filenameR}")
                
                cv2.imshow("Left camera", chessboard_image_L)
                cv2.imshow("Right camera", chessboard_image_R)
                cv2.setWindowTitle("Left camera", f"{filenameL}")
                cv2.setWindowTitle("Right camera", f"{filenameR}")
                
                image_count += 1
            else:
                print("Corners not detected on both cameras, please adjust your setup.")

    # Press 'q' to exit
    elif key == ord('q'):
        break

    # Press 'spacebar' to redo image and go back to the previous image
    elif key == ord('-'):
        if image_count > 0:
            image_count -= 1
            if os.path.exists(f"Calibration/Calibrationpictures_L/left_{image_count:03d}.jpg"):
                os.remove(f"Calibration/Calibrationpictures_L/left_{image_count:03d}.jpg")
            if os.path.exists(f"Calibration/Calibrationpictures_R/right_{image_count:03d}.jpg"):
                os.remove(f"Calibration/Calibrationpictures_R/right_{image_count:03d}.jpg")
            print(f"\nDeleted image {image_count:03d}.")
        else:
            print("No images to delete.")

# Release resources
if cameraL.isOpened():
    cameraL.release()
if cameraR.isOpened():
    cameraR.release()
cv2.destroyAllWindows()
print("\nCamera capture stopped.")

# Save parameters to a file
np.savez("Calibration/InitialParameters.npz",
         width=width, height=height,
         square_size=square_size,
         internal_width=internal_width, internal_height=internal_height,
         BASELINE=BASELINE)
print("Initial parameters saved.\n")
