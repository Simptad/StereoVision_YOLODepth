import cv2
import numpy as np
import glob

# Set checkerboard size
CHECKERBOARD = (9, 6)  # Adjust based on your pattern
square_size = 2.5  # Size of one square in cm

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# Arrays to store object and image points
objpoints = []
imgpointsL = []
imgpointsR = []

# Load calibration images. Make sure to have the same number of images for both cameras, and atleast 20 pictures.
left_images = glob.glob("left/*.jpg")   # Folder with left camera images
right_images = glob.glob("right/*.jpg") # Folder with right camera images

for left_img, right_img in zip(left_images, right_images):
    imgL = cv2.imread(left_img)
    imgR = cv2.imread(right_img)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

# Calibrate cameras
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# Get the focal length in pixels. In this case, for the left camera (assuming both cameras are identical).
FOCAL_LENGTH = mtxL[0, 0]

# Save calibration results
np.savez("stereo_calibration.npz", mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, FOCAL_LENGTH=FOCAL_LENGTH)
print("Calibration done! Results saved.")
