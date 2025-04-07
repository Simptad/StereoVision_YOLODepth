import cv2
import numpy as np
import time

## -- Parameters -- ##
# Resolution [pixels]
width = 1920; height = 1080
# Board parameters
square_size = 3.3       # Size of one square in cm
internal_width = 10     # Number of internal corners (horizontal)
internal_height = 7     # Number of internal corners (vertical)
# Camera parameters
BASELINE = 39           # Distance between cameras in cm

# Initialize cameras
print("\nInitializing cameras....")
cameraL = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Left Camera
print("Left camera initialized.")
cameraR = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Right Camera
print("Right camera initialized.\n")

# Sets the resolution for both cameras
cameraL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cameraL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cameraR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cameraR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
print("Camera resolution set to", width, "x", height)

image_count = 0
image_saved_text = ""  # Text to display when an image is saved
text_display_time = 0  # Timer to clear text

# Detection and drawing function
def detect_and_draw_chessboard_corners(image, pattern_size=(internal_width, internal_height), camera_name=""):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if ret:
        image_with_corners = image.copy()
        cv2.drawChessboardCorners(image_with_corners, pattern_size, corners, ret)
        return ret, image_with_corners
    return ret, image

print("Starting capture....\n")
while True:
    # Capture frames from both cameras
    retL, frameL = cameraL.read()
    retR, frameR = cameraR.read()

    if frameL is not None:
        cornersL_detected, chessboard_image_L = detect_and_draw_chessboard_corners(frameL, camera_name="Left")
        if cornersL_detected:
            cv2.putText(chessboard_image_L, "Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(chessboard_image_L, "Not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display "Image XXX saved"
        if time.time() - text_display_time < 2:  # Show text for 2 seconds
            cv2.putText(chessboard_image_L, image_saved_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Left Camera", chessboard_image_L)

    if frameR is not None:
        cornersR_detected, chessboard_image_R = detect_and_draw_chessboard_corners(frameR, camera_name="Right")
        if cornersR_detected:
            cv2.putText(chessboard_image_R, "Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(chessboard_image_R, "Not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display "Image XXX saved"
        if time.time() - text_display_time < 2:  # Show text for 2 seconds
            cv2.putText(chessboard_image_R, image_saved_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Right Camera", chessboard_image_R)

    key = cv2.waitKey(1) & 0xFF  

    # Press 'q' to exit
    if key == ord('q'):
        break

    # Enter key to capture image
    if key == 13:  # Enter key
        if retL and retR and cornersL_detected and cornersR_detected:
            filenameL = f"Calibration/Calibrationpictures_L/left_{image_count:03d}.jpg"
            filenameR = f"Calibration/Calibrationpictures_R/right_{image_count:03d}.jpg"
            cv2.imwrite(filenameL, frameL)
            cv2.imwrite(filenameR, frameR)
            image_saved_text = f"Image {image_count:03d} saved"  # Update text to display
            text_display_time = time.time()  # Start timer
            image_count += 1

# Stop capturing
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
