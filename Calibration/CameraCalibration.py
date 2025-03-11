import cv2
import numpy as np
import glob

## ------ Input parameters ------ ##
# Real world dimensions of the checkerboard pattern
CHECKERBOARD = (9, 6)   # Number of corners
square_size = 2.5       # Size of one square in cm
left_images = glob.glob("left/*.jpg")   # Left camera images
right_images = glob.glob("right/*.jpg") # Right camera images
# ------------------------------------#

## --- Preparing empty lists to store object points and image points from all images --- ##
# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size
objpoints = []      # 3D points in real world
imgpointsL = []     # 2D points for left camera
imgpointsR = []     # 2D points for right camera
# -----------------------------------------------------------------------------------------#

# Image calibration loop
for left_img, right_img in zip(left_images, right_images):
    # This part converts the images to grayscale
    imgL = cv2.imread(left_img)
    imgR = cv2.imread(right_img)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find Checkerboard corners
    retL, cornersL = cv2.findCheckerboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findCheckerboardCorners(grayR, CHECKERBOARD, None)

    # Stores the 2D coordinates if corners are found in both images
    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

# Calibrate individual cameras
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# Stereo Calibration: Compute rotation (R) and translation (T) between cameras
flags = cv2.CALIB_FIX_INTRINSIC  # Keep individual calibration fixed
retS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    flags=flags
)

# Stereo Rectification (Aligns both camera images to same plane)
R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1], R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
)

# Save calibration results
np.savez("stereo_calibration.npz",
         mtxL=mtxL, distL=distL, 
         mtxR=mtxR, distR=distR,
         R=R, T=T, E=E, F=F,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

print("Stereo Calibration done! Results saved.")
