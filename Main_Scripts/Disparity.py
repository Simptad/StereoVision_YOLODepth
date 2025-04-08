import cv2
import numpy as np

# Disparity Calculation
def disparity_calculation(frame_left, frame_right, stereoSGBM):
    # Converting images to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Disparity calculation for the whole scene
    disparity = stereoSGBM.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Clip disparity values to resonable values for disparity map
    disparitymap_clip = np.clip(disparity, 0.01, 255)

    # Normalize depth and apply color for visualization
    disparity_normalized = cv2.normalize(disparitymap_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disparitymap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

    return disparity, disparitymap