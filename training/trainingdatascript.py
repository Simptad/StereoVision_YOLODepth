
from ultralytics import YOLO

# Resume training? Choose true or false and set the number of epochs to resume training for.
Training_resume = False
resume_nmbr = 1

try:
    # Trains on data from trainingdata.yaml
    if Training_resume == False:

        # Load a YOLO model
        model = YOLO(r"yolov8n.pt")  # or any other YOLO model

        # Train the model
        model.train(data=r"training/trainingdata.yaml", epochs=100, imgsz=640)  # 100 epochs

    else:
        # Load the saved model
        model = YOLO(r"training/pallets_trained.pt")

        # Resume training for additional epochs
        model.train(data=r"training/trainingdata.yaml", epochs=resume_nmbr, imgsz=640, resume=True)

except KeyboardInterrupt:
    print("Training stopped by user. Saving the model...")
    model.save(r"training/pallets_trained.pt")
    print("Model saved as 'pallets_trained.pt'")

# Save the trained model weights
model.save(r"training/pallets_trained.pt")
print("Model saved as 'pallets_trained.pt'")
