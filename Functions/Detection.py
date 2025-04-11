# Object Detection
def run_detection(frame, model, conf_threshold, target_class):
    results = model(frame)
    BoundingBox = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])                      # Class id
            class_name = model.names[class_id]              # Class name that corresponds to its id
            box_conf = float(box.conf)                      # Confidence score for the bounding 
            x1, y1, x2, y2 = map(int, box.xyxy[0])          # Extract box corner coordinates

            # Filtering using confidence threshold and target class
            if (box_conf < conf_threshold) or (target_class is not None and class_name not in target_class):
                continue

            label = f"{class_name} {box_conf:.2f}"
            BoundingBox.append((x1, y1, x2, y2, label, class_id))
    return BoundingBox