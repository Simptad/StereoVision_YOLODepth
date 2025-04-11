import numpy as np
import time

# Object Depth Calculation
def depth_calculation(detections, disparity, camera_propeties, scale_factor, depth_distance, object_data):
        latest_depth = None
        for x1, y1, x2, y2, label, class_id in detections:
            object_id = f"{class_id}_{x1}_{y1}_{x2}_{y2}"           # Unique ID for each object
            RoI = disparity[y1:y2, x1:x2]                           # Maps out the Region of interest

            # Filter invalid disparities
            valid_disparities = RoI[(RoI > 0) & (RoI < 255)]

            if valid_disparities.size > 0:
                depth = (camera_propeties[0] * camera_propeties[1]) / (np.median(valid_disparities) * scale_factor)/100
                depth = (-0.7388+np.sqrt(0.7388**2-4*0.0478*(0.3123-depth)))/(2*0.0478)     # Adjusted depth

                # Filter objects based on a predefined depth (How far do you want to look)
                if depth < depth_distance:
                    object_data[object_id] = {"BoundingBox": (x1, y1, x2, y2), "depth": depth, "label": label, "last_seen": time.time()}
                    latest_depth = depth
                else:
                    object_data[object_id] = {"BoundingBox": (x1, y1, x2, y2), "depth": latest_depth, "label": label, "last_seen": time.time()}
            else:
                object_data[object_id] = {"BoundingBox": (x1, y1, x2, y2), "depth": None, "label": label, "last_seen": time.time()}

        return object_data
