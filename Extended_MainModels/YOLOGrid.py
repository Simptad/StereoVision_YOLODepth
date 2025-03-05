import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("StereoVision_YOLODepth\yolov8n.pt")
model.to('cpu')  # Run YOLO on CPU

# Open the camera
cap = cv2.VideoCapture(0)

# Define middle and big box sizes (adjust as needed)
middle_box_size = 300  # Size of the middle box (width/height)
big_box = (0, 0, 640, 480)  # The entire frame as the big box (change as per your resolution)

# Define the center of the frame for the middle box
h, w, _ = None, None, None  # Set these later when reading the frame

def draw_detections(frame, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box for detection
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera")
        break

    # Get the frame dimensions
    h, w, _ = frame.shape
    
    # Define the middle box coordinates based on the frame center
    center_x, center_y = w // 2, h // 2
    middle_box = (center_x - middle_box_size // 2, center_y - middle_box_size // 2,
                  center_x + middle_box_size // 2, center_y + middle_box_size // 2)

    # Run YOLO on the frame
    results = model(frame)

    # Process YOLO detections
    person_detected_in_middle = False
    person_detected_in_big_box = False

    # Check if a person is detected in the middle box or big box
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls[0])] == "cell phone":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Check if the person is in the middle box
                if x1 < middle_box[2] and x2 > middle_box[0] and y1 < middle_box[3] and y2 > middle_box[1]:
                    person_detected_in_middle = True

                # Check if the person is in the big box (entire screen)
                if x1 < big_box[2] and x2 > big_box[0] and y1 < big_box[3] and y2 > big_box[1]:
                    person_detected_in_big_box = True

    # Apply color overlays based on detection presence in boxes
    if person_detected_in_middle:
        # Faded red for middle box if a person is detected
        overlay = frame.copy()
        cv2.rectangle(overlay, (middle_box[0], middle_box[1]), (middle_box[2], middle_box[3]), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    if person_detected_in_big_box:
        # Faded yellow for the area around the middle box if a person is detected in the big box
        overlay = frame.copy()
        # Left side of the big box (left of middle box)
        cv2.rectangle(overlay, (0, 0), (middle_box[0], h), (0, 255, 255), -1)  
        # Right side of the big box (right of middle box)
        cv2.rectangle(overlay, (middle_box[2], 0), (w, h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Draw the middle box and the big box in the frame
    # cv2.rectangle(frame, (middle_box[0], middle_box[1]), (middle_box[2], middle_box[3]), (255, 0, 0), 2)  # Blue middle box
    # cv2.rectangle(frame, (big_box[0], big_box[1]), (big_box[2], big_box[3]), (0, 255, 0), 2)  # Green big box

    # Draw YOLO bounding boxes
    draw_detections(frame, results)

    # Show the camera feed
    cv2.imshow("Camera - YOLO with Boxes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
