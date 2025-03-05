
# Run this to check if cameras are detected

import cv2

for i in range(2): # Change range for number of cameras
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
