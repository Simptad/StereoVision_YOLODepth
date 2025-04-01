
from ultralytics import YOLO

# Resume training? Choose true or false and set the number of epochs to resume training for.
Training_resume = False
resume_nmbr = 1
epochs = 100

if __name__ == '__main__':
    try:
        # Trains on data from trainingdata.yaml
        if Training_resume == False:

            # Load a YOLO model
            model = YOLO("yolo11n.pt")  # or any other YOLO model

            # Train the model
            model.train(data="training/trainingdata.yaml", epochs=epochs, imgsz=640)  # 100 epochs

        else:
            # Load the saved model
            model = YOLO("training/blocktunnel_trained.pt")

            # Resume training for additional epochs
            model.train(data="training/trainingdata.yaml", epochs=resume_nmbr, imgsz=640, resume=True)

    except KeyboardInterrupt:
        print("Training stopped by user. Saving the model...")
        model.save(r"training/blocktunnel_trained.pt")
        print("Model saved as 'blocktunnel_trained.pt'")

    # Save the trained model weights
    model.save(r"training/blocktunnel_trained.pt")
    print("Model saved as 'blocktunnel_trained.pt'")
