import cv2
import numpy as np

# Disparity Calculation
def disparity_calculation(frame_left, frame_right, stereoSGBM, map_new):

    # Undistort and rectify the images
    frame_left_rect = cv2.remap(frame_left, map_new[0], map_new[1], interpolation=cv2.INTER_LINEAR)
    frame_right_rect = cv2.remap(frame_right, map_new[2], map_new[3], interpolation=cv2.INTER_LINEAR)

    # Converting images to grayscale
    gray_left = cv2.cvtColor(frame_left_rect, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right_rect, cv2.COLOR_BGR2GRAY)

    # Disparity calculation for the whole scene
    disparity = stereoSGBM.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Clip disparity values to resonable values for disparity map
    disparitymap_clip = np.clip(disparity, 0, 255)

    # Normalize depth and apply color for visualization
    disparity_normalized = cv2.normalize(disparitymap_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disparitymap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

    return disparity, disparitymap, (frame_left_rect, frame_right_rect)
