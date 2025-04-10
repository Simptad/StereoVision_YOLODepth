import cv2
import numpy as np
import glob

# Load initial parameters
CHECKERBOARD = (10, 7)
square_size = 3.3
BASELINE = 39

# Load calibration images
left_images = sorted(glob.glob("Calibration/Calibrationpictures_L/*.jpg"))
right_images = sorted(glob.glob("Calibration/Calibrationpictures_R/*.jpg"))

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size
objpoints = []
imgpointsL = []
imgpointsR = []

denied_images = 0
proccessed_images = 0
print("\nStarting image calibration....")
for left_img, right_img in zip(left_images, right_images):
    imgL = cv2.imread(left_img)
    imgR = cv2.imread(right_img)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCornersSB(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCornersSB(grayR, CHECKERBOARD, None)

    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        print(f"Image {proccessed_images + denied_images} Processed.")
        proccessed_images += 1
    else:
        print(f"Image {proccessed_images + denied_images} Denied.")
        denied_images += 1
print(f"Done. Processed: {proccessed_images}, Denied: {denied_images}\n")

print("Individual Camera Calibration...")
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
print(f"Left Camera Reprojection Error: {retL:.4f}")
print(f"Right Camera Reprojection Error: {retR:.4f}\n")

print("Stereo Calibration...")
# Try without fixing intrinsics first
flags_try1 = cv2.CALIB_USE_INTRINSIC_GUESS  # Use initial guesses from individual calibration
retS1, mtxL1, distL1, mtxR1, distR1, R1_try1, T1_try1, E1, F1 = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL.copy(), distL.copy(), mtxR.copy(), distR.copy(),
    grayL.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    flags=flags_try1
)
print(f"Stereo Calibration (Try 1 - CALIB_USE_INTRINSIC_GUESS) Reprojection Error: {retS1:.4f}")
print(f"Computed baseline (Try 1): {np.linalg.norm(T1_try1):.4f} units\n")

# Try fixing intrinsics (your original approach)
flags_try2 = cv2.CALIB_FIX_INTRINSIC
retS2, mtxL2, distL2, mtxR2, distR2, R2_try2, T2_try2, E2, F2 = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL.copy(), distL.copy(), mtxR.copy(), distR.copy(),
    grayL.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    flags=flags_try2
)
print(f"Stereo Calibration (Try 2 - CALIB_FIX_INTRINSIC) Reprojection Error: {retS2:.4f}")
print(f"Computed baseline (Try 2): {np.linalg.norm(T2_try2):.4f} units\n")

# Choose the calibration with the lower reprojection error
if retS1 < retS2:
    retS = retS1
    mtxL_final = mtxL1
    distL_final = distL1
    mtxR_final = mtxR1
    distR_final = distR1
    R_final = R1_try1
    T_final = T1_try1
    print("Using calibration with CALIB_USE_INTRINSIC_GUESS for rectification.")
else:
    retS = retS2
    mtxL_final = mtxL2
    distL_final = distL2
    mtxR_final = mtxR2
    distR_final = distR2
    R_final = R2_try2
    T_final = T2_try2
    print("Using calibration with CALIB_FIX_INTRINSIC for rectification.")

print(f"Final Stereo Reprojection Error: {retS:.4f}\n")

print("Stereo Rectification...")
# Try different alpha values for stereoRectify and evaluate ROI area
alpha_values = [0, 0.25, 0.5, 0.75, 1]
best_alpha = -1
max_roi_area = -1
best_rect_params = None

for alpha in alpha_values:
    R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(
        mtxL_final, distL_final, mtxR_final, distR_final,
        grayL.shape[::-1], R_final, T_final,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
    )
    print(f"Stereo Rectification (alpha={alpha}): ROI Left = {roiL}, ROI Right = {roiR}")

    # Calculate the area of the intersection of the ROIs
    x_overlap = max(roiL[0], roiR[0])
    y_overlap = max(roiL[1], roiR[1])
    width_overlap = min(roiL[0] + roiL[2], roiR[0] + roiR[2]) - x_overlap
    height_overlap = min(roiL[1] + roiL[3], roiR[1] + roiR[3]) - y_overlap
    overlap_area = max(0, width_overlap) * max(0, height_overlap)

    print(f"  Overlap ROI Area: {overlap_area}")

    if overlap_area > max_roi_area:
        max_roi_area = overlap_area
        best_alpha = alpha
        best_rect_params = (R1, R2, P1, P2, Q, roiL, roiR)

if best_alpha != -1:
    print(f"\nBest alpha value found: {best_alpha} with maximum overlap ROI area: {max_roi_area}")
    R1, R2, P1, P2, Q, roiL, roiR = best_rect_params
else:
    print("\nNo valid alpha value found.")
    # Use a default alpha if no overlap was found (shouldn't happen with typical setups)
    R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(
        mtxL_final, distL_final, mtxR_final, distR_final,
        grayL.shape[::-1], R_final, T_final,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.5  # Default alpha
    )
    best_alpha = 0.5

print("Done\n")

# Compute rectification maps with the best alpha
mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL_final, distL_final, R1, P1, grayL.shape[::-1], cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR_final, distR_final, R2, P2, grayL.shape[::-1], cv2.CV_16SC2)

# Extract focal length
FOCAL_LENGTH_L = mtxL_final[0, 0]
FOCAL_LENGTH_R = mtxR_final[0, 0]

print(f"Calculated Focal Length (Left): {FOCAL_LENGTH_L:.2f} pixels")
print(f"Calculated Focal Length (Right): {FOCAL_LENGTH_R:.2f} pixels\n")

# Save calibration results (include best alpha)
np.savez("Calibration/stereo_calibration3.npz",
         mtxL=mtxL_final, distL=distL_final,
         mtxR=mtxR_final, distR=distR_final,
         R=R_final, T=T_final, E=(E1 if retS1 < retS2 else E2), F=(F1 if retS1 < retS2 else F2),
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         BASELINE=BASELINE,
         FOCAL_LENGTH_L=FOCAL_LENGTH_L,
         FOCAL_LENGTH_R=FOCAL_LENGTH_R,
         mapL1=mapL1, mapL2=mapL2,
         mapR1=mapR1, mapR2=mapR2,
         obj_points=objpoints,
         img_points_l=imgpointsL,
         img_points_r=imgpointsR,
         best_alpha=best_alpha  # Save the best alpha value
)

print("Stereo Calibration done! Results saved to stereo_calibration3.npz")
print(f"Total images: {proccessed_images + denied_images}, Processed: {proccessed_images}, Denied: {denied_images}")

## -- Verification (Stereo Reprojection Error) -- ##
print("\nVerifying Stereo Calibration...")
mean_error, _, _, _, _, _, _, _, _ = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL_final.copy(), distL_final.copy(), mtxR_final.copy(), distR_final.copy(),
    grayL.shape[::-1], R_final.copy(), T_final.copy(),
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    flags=(cv2.CALIB_FIX_INTRINSIC if retS2 < retS1 else cv2.CALIB_USE_INTRINSIC_GUESS)
)
print(f"Final Stereo Reprojection Error (Verification): {mean_error:.4f}")
print(f"Computed baseline from calibration (||T||): {np.linalg.norm(T_final):.4f} units\n")

print("Done.")

print(best_alpha)

