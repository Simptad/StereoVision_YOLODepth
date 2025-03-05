import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")
model.to('cuda')  # Run YOLO on GPU
# model.to('cpu')  # Run YOLO on CPU

# Open both cameras (change index if needed)
cap1 = cv2.VideoCapture(0)  # Laptop camera
cap2 = cv2.VideoCapture(1)  # External camera (change index if needed)

while cap1.isOpened() and cap2.isOpened():
    # Read frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Error: Could not read from both cameras")
        break

    # Run YOLO on both frames
    results1 = model(frame1)  # YOLO on Camera 1
    results2 = model(frame2)  # YOLO on Camera 2

    # Function to draw YOLO detections
    def draw_detections(frame, results, color):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)

    # Draw detections
    draw_detections(frame1, results1, (0, 255, 0))  # Green for camera 1
    draw_detections(frame2, results2, (255, 0, 0))  # Blue for camera 2

    # Show both camera feeds
    cv2.imshow("Camera 1 - YOLO", frame1)
    cv2.imshow("Camera 2 - YOLO", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap1.release()
cap2.release()
cv2.destroyAllWindows()
