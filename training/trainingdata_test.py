from ultralytics import YOLO
import os
import cv2

# Set confidence thresholds for detections
conf1 = 0.7
conf2 = 0.5

# Load your trained model
model = YOLO("training/palletbox_trained.pt")           # Path to your trained model
testing_images = ("training/datasets/pallets_testing\images")     # Path for testing images

# Run inference on a folder of images with two different confidence thresholds
results1 = model(testing_images, conf=conf1)
results2 = model(testing_images, conf=conf2)

# Ensure the output folder exists, create it if it doesn't
output_folder = "training/datasets/palletbox_testingOutput"
os.makedirs(output_folder, exist_ok=True)

# Dictionary to store window names and corresponding images
window_images = {}
active_window = None  # Track the currently selected window
current_index = 0  # Start at the first image


# Callback function to track the clicked window
def on_mouse_click(event, x, y, flags, param):
    global active_window
    if event == cv2.EVENT_LBUTTONDOWN:
        active_window = param  # Set the clicked window as active


# Function to display the next image
def show_next_image():
    global current_index
    if current_index < len(results1):
        result1 = results1[current_index]
        result2 = results2[current_index]
        
        img = cv2.imread(result1.path)  # Read the original image
        
        # Draw bounding boxes for the first set of results with green color
        high_conf_boxes = []
        for box in result1.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            high_conf_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color for higher confidence threshold
            cv2.putText(img, f"{box.conf.item():.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw bounding boxes for the second set of results with yellow color
        for box in result2.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x1, y1, x2, y2) not in high_conf_boxes:  # Avoid duplicates
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow color for lower confidence threshold
                cv2.putText(img, f"{box.conf.item():.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        window_name = f"Result {current_index}"

        # Show the image and set up click tracking
        cv2.imshow(window_name, img)
        cv2.setMouseCallback(window_name, on_mouse_click, window_name)

        # Store image reference
        window_images[window_name] = img

        current_index += 1  # Move to the next image


# Show the first image
show_next_image()
print("\nPress 'spacebar' to open the next image, 's' to save the selected image, or 'q' to quit.\n")

while True:
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):  # Quit the program
        break
    elif key == ord(' '):  # Show next image
        show_next_image()
    elif key == ord('s') and active_window:  # Save the selected window's image
        save_path = os.path.join(output_folder, f"{active_window}.jpg")
        cv2.imwrite(save_path, window_images[active_window])
        print(f"Image saved to {save_path}")

# Close all OpenCV windows
cv2.destroyAllWindows()