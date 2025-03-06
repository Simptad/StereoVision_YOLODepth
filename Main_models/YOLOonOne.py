import cv2
from ultralytics import YOLO

#### ---- User input ---- ####
# Run model on GPU (nvidia) or CPU?
run_gpu = True

# Confidence threshold for detections
conf_threshold = 0.6

############################################

# Load YOLO models
default_model = YOLO("yolov8n.pt")
trained_model = YOLO("training/pallets_trained.pt")

if run_gpu:
    default_model.to('cuda')
    trained_model.to('cuda')
else:
    default_model.to('cpu')
    trained_model.to('cpu')

# Open the camera (change index if needed)
cap = cv2.VideoCapture(0)  # Use the default camera

while cap.isOpened():
    # Read frame from camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read from camera")
        break

    # Run YOLO on the frame
    default_results = default_model(frame)
    trained_results = trained_model(frame)

    # Function to draw YOLO detections
    def draw_detections(frame, results, model_names, color, conf_threshold):
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{model_names[int(box.cls[0])]} {box.conf[0]:.2f}"
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, color, 2)

    # Draw detections on the frame
    draw_detections(frame, default_results, default_model.names, (0, 255, 0), conf_threshold)  # Green for default model detections
    draw_detections(frame, trained_results, trained_model.names, (0, 255, 255), conf_threshold)  # Yellow for trained model detections

    # Show the camera feed
    cv2.imshow("Camera - YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()