# Running YOLO on both cameras and calculating depth using stereo vision
# Using YOLO's own trained model together with custom models.
# The custom models includes
# - Pallet detection
# - Pallet tunnel & pallet block detection

print("\nStarting...")

#################################################################
#################################################################
# -------------------------- Imports -------------------------- #
import time
import cv2

# Import
import Config
from Functions.CameraModelInit import init_cameras, set_processor, calculate_fov
from Functions.Disparity import disparity_calculation
from Functions.Detection import run_detection
from Functions.Depth import depth_calculation
from Functions.TunnelCenter import find_tunnel_center, find_tunnel_center_blocks
from Functions.CameraOffset import calculate_camera_offset
from Functions.Visualization import visualization, display
from Functions.resize_calibration import resize_calibration

print('\t\033[92mImport \u2714\033[0m')

# ------------------------ END Imports ------------------------ #
#################################################################
#################################################################


########################################################################
########################################################################
# --------------------------- MAIN LOGIC ----------------------------- #

## ---- Initializing ---- ##
camera_left, camera_right, frame_center = init_cameras(Config.LCameraID, Config.RCameraID, Config.brightness_value, Config.width, Config.height)
set_processor(Config.models, Config.processor, Config.fps)

# Calculate the cameras horizontal and vertical field of view.
HorVert_FoV = calculate_fov(Config.camera_propeties[2], Config.width, Config.height) 

# Resizing calibration data to match target resolution
mtx_new, map_new, Q_new = resize_calibration(Config.calib_data, (1920, 1080), (Config.width, Config.height))

# Initializing empty list
object_data = {}

## ---- MAIN LOOP ---- ##
frame_time = 1/Config.fps
print("Visualizing.. (press 'Q' to stop)")
while camera_left.isOpened() and camera_right.isOpened():
    _, frame_left = camera_left.read()
    _, frame_right = camera_right.read()
    
    # Calculate the disparity for the combined image
    disparity, disparitymap, rect = disparity_calculation(frame_left, frame_right, Config.stereoSGBM, map_new)

    # Run object and pallet detection
    all_detections = []
    for model, threshold in Config.models:
        detection = run_detection(frame_left, model, threshold, Config.target_class)
        all_detections.extend(detection)

    # Object Depth Calculations
    object_data = depth_calculation(all_detections, disparity, Config.camera_propeties, Config.scale_factor, Config.depth_distance, object_data)
    object_data = {k: v for k, v in object_data.items() if time.time() - v["last_seen"] < Config.remove_time}  # Remove objects that haven't been seen in 'remove_time' seconds
    
    # Find pallet tunnel center
    tunnel_center_points_blocks, is_inside_blocks= find_tunnel_center_blocks(all_detections)
    tunnel_center_points, is_inside_tunnel, tunnel_coords = find_tunnel_center(all_detections)

    # Calculate camera offset to the pallet tunnel center
    camera_offset, closest_tunnel_center, camera_angle_offset = calculate_camera_offset(frame_center, tunnel_center_points, tunnel_center_points_blocks, disparity, HorVert_FoV, Config.width, Config.height)

    # Visualize
    visualization(frame_left, object_data, tunnel_center_points, frame_center, camera_offset, closest_tunnel_center, tunnel_center_points_blocks, camera_angle_offset,
                  is_inside_tunnel, is_inside_blocks, tunnel_coords)

    # Display figures
    display(frame_left, _, disparitymap)

    # Breaks out of the loop when pressing 'Q' and stopping code
    if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord('q'):
        print("\033[91mStopping..\033[0m\n")
        break
# ------------------------ END MAIN LOGIC ------------------------ #
####################################################################
####################################################################


# Stop and Quit
camera_left.release(); camera_right.release(); cv2.destroyAllWindows()
