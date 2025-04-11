# --------------------------------------------------------------- #
# Made by Simon Tadros
# Master Thesis Project at Linköping University
# --------------------------------------------------------------- #
# Stereo Camera Object Detection and Depth Estimation
# The project combines stereo vision and YOLO-based object detection 
# to achieve depth estimation and object tracking.
# --------------------------------------------------------------- #

import time
import cv2
import Config
from Functions.CameraModelInit import init_cameras, set_processor, calculate_fov
from Functions.Disparity import disparity_calculation
from Functions.Detection import run_detection
from Functions.Depth import depth_calculation
from Functions.TunnelCenter import find_tunnel_center, find_tunnel_center_blocks
from Functions.CameraOffset import calculate_camera_offset
from Functions.Visualization import visualization, display
from Functions.resize_calibration import resize_calibration

def initialize_system():
    print("\n\033[93mInitializing system..\033[0m")
    camera_left, camera_right, frame_center = init_cameras(Config.LCameraID, Config.RCameraID, Config.brightness_value, Config.width, Config.height)
    set_processor(Config.models, Config.processor, Config.fps)

    # Calculate the cameras horizontal and vertical field of view.
    HorVert_FoV = calculate_fov(Config.camera_propeties[2], Config.width, Config.height) 

    # Resizing calibration data to match target resolution
    _, map_new, _ = resize_calibration(Config.calib_data, Config.original_calibdata_res, (Config.width, Config.height))

    print(f"\033[92mSystem initialized \u2714\033[0m")
    return camera_left, camera_right, frame_center, HorVert_FoV, map_new,


def MainLogic(camera_left, camera_right, frame_center, HorVert_FoV, map_new):
    print("Visualizing.. (press 'Q' to stop)")

    frame_time = 1/Config.fps
    object_data = {}

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

    # Stop and Quit
    camera_left.release(); camera_right.release(); cv2.destroyAllWindows()

def main():
    camera_left, camera_right, frame_center, HorVert_FoV, map_new = initialize_system()
    MainLogic(camera_left, camera_right, frame_center, HorVert_FoV, map_new)

if __name__ == "__main__":
    main()

