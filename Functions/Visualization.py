# ------------------------------------------------------------------------
# This script visualizes the bounding boxes, center points, and camera angle offset in the left camera frame.
# It also shows the disparity map.





# ------------------------------------------------------------------------
# Imports
import cv2

# Visualization colors
cross_size = 5; cross_width = 2         # Center cross dimensions
red = (0,0,255); green = (0,255,0); blue = (255,0,0)
white = (255,255,255); black = (0,0,0);                     
camera_offset_color = (255, 0, 255)

# Visualization
def visualization(frame_left, object_data, tunnel_center_points, frame_center, camera_offset, closest_tunnel_center, tunnel_center_points_blocks, angle_offset,
                  is_inside_tunnel, is_inside_blocks, tunnel_coords):
    
    # ------ Bounding box visualization ------ #
    tunnel_detected = False

    for tunnel_coord in tunnel_coords:
        x1t, y1t, x2t, y2t = tunnel_coord  # Extract each set of coordinates

        # Draws 'tunnel' if its found inside of a 'pallet'
        for obj in object_data.values():
            # Gets all object data
            label = obj["label"]
            bbox, depth = obj["BoundingBox"], obj["depth"]
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            if "tunnel" in label:
                center_xt, center_yt = (x1t + x2t) // 2, (y1t + y2t) // 2
                tunnel_detected = True

                # Checks if 'tunnel' is inside of pallet bounding box
                if is_inside_tunnel:
                    cv2.rectangle(frame_left, (x1t, y1t), (x2t, y2t), (0, 255, 0), 2)
                    cv2.circle(frame_left, (center_xt, center_yt), 4, (0, 0, 255), -1)
                    cv2.putText(frame_left, label, (x1t + 5, y1t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Top Label text

                    if obj["depth"] is not None:
                        depth_text = f"{obj['depth']:.2f}m"
                        cv2.putText(frame_left, depth_text, (x1t + 5, y2t + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Depth below boundingbox
                    
                    # Draw closest tunnel center circle with the color "camera_offset_color"
                    for mid_x, mid_y in tunnel_center_points:
                        color = camera_offset_color if closest_tunnel_center and (mid_x, mid_y) == closest_tunnel_center else blue
                        cv2.circle(frame_left, (mid_x, mid_y), 5, color, -1)
            
    # Draws 'block' if its found inside of a 'pallet' and "tunnel" is not found
    if not tunnel_detected:
        for obj in object_data.values():
            # Gets all object data
            label = obj["label"]
            bbox, depth = obj["BoundingBox"], obj["depth"]
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2 
            
            if "block" in label.lower():
                if is_inside_blocks:
                    cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame_left, (center_x, center_y), 4, (0, 0, 255), -1)
                    cv2.putText(frame_left, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Top Label text

                    if obj["depth"] is not None:
                        depth_text = f"{obj['depth']:.2f}m"
                        cv2.putText(frame_left, depth_text, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Depth below bounding box

                    # Draw center circle
                    for mid_x, mid_y in tunnel_center_points_blocks: 
                        color = camera_offset_color if closest_tunnel_center and (mid_x, mid_y) == closest_tunnel_center else red
                        cv2.circle(frame_left, (mid_x, mid_y), 5, color, -1)

    # Draws everything else
    for obj in object_data.values():
        label = obj["label"]
        bbox, depth = obj["BoundingBox"], obj["depth"]
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        if "block" not in label.lower() and "tunnel" not in label.lower():
            cv2.rectangle(frame_left, (x1, y1), (x2, y2), green, 2)
            cv2.circle(frame_left, (center_x, center_y), 4, red, -1)
            cv2.putText(frame_left, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)  # Top Label text
            cv2.putText(frame_left, label, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)  # Bottom Label text

            if depth is not None:
                depth_text = f"{obj['depth']:.2f}m"
                cv2.putText(frame_left, depth_text, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)  # Depth below bounding box
    # ------------------------------------------ #

    # ------ Draw 'x' in the center of the camera frame ------ #
    frame_center_x, frame_center_y = frame_center[0], frame_center[1]
    cv2.line(frame_left, (frame_center_x - cross_size, frame_center_y - cross_size), 
            (frame_center_x + cross_size, frame_center_y + cross_size), camera_offset_color, cross_width)
    cv2.line(frame_left, (frame_center_x - cross_size, frame_center_y + cross_size), 
            (frame_center_x + cross_size, frame_center_y - cross_size), camera_offset_color, cross_width)
    #---------------------------------------------------#

    # ------ Visualize camera offset to pallet center ------ #
    if camera_offset and closest_tunnel_center:
        angle_x, angle_y = angle_offset
        height, _, _ = frame_left.shape
        tunnel_center_x, tunnel_center_y = closest_tunnel_center

        # Start and end points for the angle offsets
        start_point_x = (frame_center_x, height)
        end_point_x = (tunnel_center_x, tunnel_center_y)
        start_point_y = (0, frame_center_y)
        end_point_y = (tunnel_center_x, tunnel_center_y)

        # Display the angle offset lines
        cv2.line(frame_left, start_point_x, end_point_x, camera_offset_color, 2)
        cv2.putText(frame_left, f"{angle_x} deg", (frame_center_x+10, height-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, camera_offset_color, 1, cv2.LINE_AA)
        
        cv2.line(frame_left, start_point_y, end_point_y, camera_offset_color, 2)
        cv2.putText(frame_left, f"{angle_y} deg", (10, frame_center_y+2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, camera_offset_color, 1, cv2.LINE_AA)

    #--------------------------------------------------------#

# Show camera feed
def display(frame_left, disparitymap):
        
        # Show camera feed
        cv2.imshow("Disparity Map", disparitymap)
        cv2.imshow("Left Camera - Object Detection + Depth", frame_left)
