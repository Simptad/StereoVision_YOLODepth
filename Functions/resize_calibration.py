import cv2
import numpy as np

def resize_calibration(calib_data, calib_imagesize, target_imagesize):
    print("\t\033[93mResizing calibration data..\033[0m")

    if calib_imagesize == target_imagesize:
        print("\t\tNo resizing needed.")
        return (calib_data["mtxL"], calib_data["mtxR"]), (calib_data["mapL1"], calib_data["mapL2"], calib_data["mapR1"], calib_data["mapR2"]), calib_data["Q"]

    # Extract the original calibration data
    mtxL = calib_data["mtxL"]
    mtxR = calib_data["mtxR"]
    distL = calib_data["distL"]
    distR = calib_data["distR"]
    R = calib_data["R"]
    T = calib_data["T"]

    # Calculate the scaling factors
    scaling_factor_x = target_imagesize[0] / calib_imagesize[0]
    scaling_factor_y = target_imagesize[1] / calib_imagesize[1]

    # Scale the intrinsic matrices (focal lengths and principal points)
    mtxL_new = mtxL.copy()
    mtxR_new = mtxR.copy()
    mtxL_new[0, 0] *= scaling_factor_x  # fx
    mtxL_new[1, 1] *= scaling_factor_y  # fy
    mtxL_new[0, 2] *= scaling_factor_x  # cx
    mtxL_new[1, 2] *= scaling_factor_y  # cy

    mtxR_new[0, 0] *= scaling_factor_x  # fx
    mtxR_new[1, 1] *= scaling_factor_y  # fy
    mtxR_new[0, 2] *= scaling_factor_x  # cx
    mtxR_new[1, 2] *= scaling_factor_y  # cy

    # Recompute the rectification maps for the new resolution
    R1_new, R2_new, P1_new, P2_new, Q_new, roiL_new, roiR_new = cv2.stereoRectify(
        mtxL_new, distL, mtxR_new, distR,
        target_imagesize, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    # Generate new undistort-rectify maps
    mapL1_new, mapL2_new = cv2.initUndistortRectifyMap(mtxL_new, distL, R1_new, P1_new, target_imagesize, cv2.CV_16SC2)
    mapR1_new, mapR2_new = cv2.initUndistortRectifyMap(mtxR_new, distR, R2_new, P2_new, target_imagesize, cv2.CV_16SC2)

    print(f"\t\tOld size: {calib_imagesize}p, New size: {target_imagesize}p")
    print('\t\033[92mResizing \u2714\033[0m')

    return (mtxL_new, mtxR_new), (mapL1_new, mapL2_new, mapR1_new, mapR2_new), Q_new