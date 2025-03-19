import cv2
import numpy as np
import glob

## -- Load initial parameters -- ##
init_data = np.load("Calibration/InitialParameters.npz")
# CHECKERBOARD = (init_data["internal_width"], init_data["internal_length"])
CHECKERBOARD = (10, 7)
BASELINE = init_data["BASELINE"]
square_size = init_data["square_size"]
# ------------------------------- ##

## --- Load calibration images --- ##
left_images = glob.glob("Calibration/Calibrationpictures_L/*.jpg")   # Left camera images
right_images = glob.glob("Calibration/Calibrationpictures_R/*.jpg") # Right camera images
# ------------------------------------#

## --- Preparing empty lists to store object points and image points from all images --- ##
# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size
objpoints = []      # 3D points in real world
imgpointsL = []     # 2D points for left camera
imgpointsR = []     # 2D points for right camera
# -----------------------------------------------------------------------------------------#

denied_images = 0; proccessed_images = 0
print("\nStarting image calibration....")
# Image calibration loop
for left_img, right_img in zip(left_images, right_images):
    # This part converts the images to grayscale
    imgL = cv2.imread(left_img)
    imgR = cv2.imread(right_img)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # # Optionally apply histogram equalization to enhance contrast
    # grayL = cv2.equalizeHist(grayL)
    # grayR = cv2.equalizeHist(grayR)

    # # Optionally apply Gaussian Blur to smooth the image and reduce noise
    # grayL = cv2.GaussianBlur(grayL, (5, 5), 0)
    # grayR = cv2.GaussianBlur(grayR, (5, 5), 0)

    # Find Checkerboard corners
    # retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    # retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    # Better chessboard corner detection using SB
    retL, cornersL = cv2.findChessboardCornersSB(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCornersSB(grayR, CHECKERBOARD, None)

    # Stores the 2D coordinates if corners are found in both images
    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        print(f"Image {proccessed_images+denied_images} Processed.")
        proccessed_images += 1
    else:
        print(f"Image {proccessed_images+denied_images} Denied.")
        denied_images += 1
    
print("Done\n")

print("Individual Stereo Calibration...")
# Calibrate individual cameras
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
print("Done\n")

print("Computing Rotation and Translation...")
# Stereo Calibration: Compute rotation (R) and translation (T) between cameras
flags = cv2.CALIB_FIX_INTRINSIC  # Keep individual calibration fixed
retS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    flags=flags
)
print("Done\n")

print("Computing Stereo Rectification...")
# Stereo Rectification (Aligns both camera images to same plane)
R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1], R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
) 
print("Done\n")

# Compute rectification maps
mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_16SC2)

# Extract focal length from the camera matrix
FOCAL_LENGTH_L = mtxL[0, 0]
FOCAL_LENGTH_R = mtxR[0, 0]

# Print the calculated focal lengths
print(f"Calculated Focal Length (Left Camera): {FOCAL_LENGTH_L} pixels")
print(f"Calculated Focal Length (Right Camera): {FOCAL_LENGTH_R} pixels\n")

# Save calibration results
np.savez("Calibration/stereo_calibration.npz",
         mtxL=mtxL, distL=distL, 
         mtxR=mtxR, distR=distR,
         R=R, T=T, E=E, F=F,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, 
         BASELINE=BASELINE, 
         FOCAL_LENGTH_L=FOCAL_LENGTH_L, 
         FOCAL_LENGTH_R=FOCAL_LENGTH_R,
         mapL1=mapL1, mapL2=mapL2,
         mapR1=mapR1, mapR2=mapR2
)

print("Stereo Calibration done! Results saved.")
print(f"Total images processed/denied: {proccessed_images}/{denied_images} ({proccessed_images+denied_images})")

## -- Verify calibration -- ##
print("\nStarting camera verification....")
reprojection_error_L = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], mtxL, distL, rvecsL, tvecsL)[0]
reprojection_error_R = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], mtxR, distR, rvecsR, tvecsR)[0]
stereo_error = retS
print(f"Reprojection Error (Left Camera): {reprojection_error_L}")
print(f"Reprojection Error (Right Camera): {reprojection_error_R}")
print(f"Stereo Calibration Reprojection Error: {stereo_error}")
print("Done\n")

