import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("StereoVision_YOLODepth\yolov8n.pt")
# model.to('cuda')  # Run YOLO on GPU
model.to('cpu')  # Run YOLO on CPU

# Open the camera (change index if needed)
cap = cv2.VideoCapture(0)  # Use the default camera

while cap.isOpened():
    # Read frame from camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read from camera")
        break

    # Run YOLO on the frame
    results = model(frame)

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

    # Draw detections on the frame
    draw_detections(frame, results, (0, 255, 0))  # Green for detections

    # Show the camera feed
    cv2.imshow("Camera - YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
