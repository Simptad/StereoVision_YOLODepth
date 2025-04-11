# Checks if the center of "tunnel" or "block" is inside of pallet
def is_inside(center, detection):
    cx, cy = center
    for x1, y1, x2, y2, label, *_ in detection:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False

# Finds the tunnel center from the detection of pallet "tunnel"
def find_tunnel_center(detections):
    # Checks if the center of a "tunnel" or "block" is inside of a pallet bounding box
    tunnel_center_points = []
    is_inside_tunnel = []
    tunnel_coords = []

    detected_pallets = [d for d in detections if "pallet" in d[4].lower()]  # Find all pallet detections

    for x1, y1, x2, y2, label, *_ in detections:
        if "tunnel" in label.lower():
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            is_inside_tunnel = is_inside((mid_x, mid_y), detected_pallets)
            if is_inside_tunnel:
                tunnel_center_points.append((mid_x, mid_y))
                tunnel_coords.append((x1, y1, x2, y2))

    return tunnel_center_points, is_inside_tunnel, tunnel_coords

# Finds the tunnel center from the detection of pallet "blocks"
def find_tunnel_center_blocks(detections, y_tolerance=20):
    # Detects the pallet blocks and calculates the midpoint between them.
    # The midpoint should land in the center of the pallet tunnel.
    # IF only the left and right block are detected, the midpoint will land at the center block.

    # Find all block and pallet detections
    block_detections = [d for d in detections if "block" in d[4].lower()]
    detected_pallets = [d for d in detections if "pallet" in d[4].lower()]
    tunnel_center_points_blocks = []
    is_inside_blocks = []

    # Sort blocks by their y1 coordinate (top of the bounding box)
    block_detections.sort(key=lambda b: b[1])

    # Pair blocks based on y-level proximity
    for i, block1 in enumerate(block_detections):
        x1_1, y1_1, x2_1, y2_1, _, _ = block1
        mid_x1, mid_y1 = (x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2

        for j, block2 in enumerate(block_detections[i + 1:], start=i + 1):
            x1_2, y1_2, x2_2, y2_2, _, _ = block2
            mid_x2, mid_y2 = (x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2

            # Check if the blocks are at roughly the same y-level
            if abs(mid_y1 - mid_y2) <= y_tolerance:
                # Calculate the midpoint between the two blocks
                mid_x_blocks = (mid_x1 + mid_x2) // 2
                mid_y_blocks = (mid_y1 + mid_y2) // 2

                # Checks if center points are inside of 'pallet'
                is_inside_blocks = is_inside((mid_x_blocks, mid_y_blocks), detected_pallets)
                if is_inside_blocks:
                    tunnel_center_points_blocks.append((mid_x_blocks, mid_y_blocks))

    return tunnel_center_points_blocks, is_inside_blocks
