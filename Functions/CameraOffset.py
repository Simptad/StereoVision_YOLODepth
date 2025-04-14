# ------------------------------------------------------------------------
# This function calculates the camera angle offset from the frame center to the closest pallet tunnel.
# The tunnel center can either be calculated from the detection of 'tunnel' or 'blocks'.

# It returns the camera offset, the closest tunnel center, and the camera angle offset.



# ------------------------------------------------------------------------
# Calculates the camera offset from the frame center to the closest pallet tunnel
def calculate_camera_offset(frame_center, tunnel_center_points, tunnel_center_points_blocks, disparity, HorVert_FoV, width, height):
    horizontal_fov, vertical_fov = HorVert_FoV

    # Find the closest center point to the frame center
    combined_center_points = tunnel_center_points + tunnel_center_points_blocks
    if combined_center_points:
        closest_tunnel_center = min(combined_center_points,key=lambda p: (p[0] - frame_center[0])**2 + (p[1] - frame_center[1])**2)
        closest_disp = disparity[closest_tunnel_center[1], closest_tunnel_center[0]]
    else:
        closest_tunnel_center, closest_disp = None, 0

    # Filtering
    if 0 < closest_disp < 255: 
        valid_disp = closest_disp 
    else: 
        valid_disp = None

    if valid_disp and closest_tunnel_center:
        # Fetch cameras horizontal and vertical FoV
        offset_x_pixels = closest_tunnel_center[0] - frame_center[0]
        offset_y_pixels = closest_tunnel_center[1] - frame_center[1]
        offset_angle_x = offset_x_pixels*(horizontal_fov/width)
        offset_angle_y = offset_y_pixels*(vertical_fov/height)

        offset_length = (offset_x_pixels, offset_y_pixels)
        offset_angle = (round(offset_angle_x, 2), round(offset_angle_y, 2))
    else:
        offset_length, offset_angle = (None, None), (None, None)

    return offset_length, closest_tunnel_center, offset_angle