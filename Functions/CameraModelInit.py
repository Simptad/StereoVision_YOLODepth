import cv2
import math

# Initialize Cameras
def init_cameras(LCameraID, RCameraID, bvalue, width, height):
    
    print("\t\033[93mInitializing cameras..\033[0m")
    camera_left = cv2.VideoCapture(LCameraID, cv2.CAP_DSHOW)
    camera_right = cv2.VideoCapture(RCameraID, cv2.CAP_DSHOW)
    if not camera_left.isOpened():
        print("\t\t\033[91mError: Unable to open left camera.\033[0m")
        exit(1)
    elif not camera_right.isOpened():
        print("\t\t\033[91mError: Unable to open right camera.\033[0m")
        exit(1)
    else:
        camera_left.set(cv2.CAP_PROP_BRIGHTNESS, bvalue)
        camera_right.set(cv2.CAP_PROP_BRIGHTNESS, bvalue)
        camera_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"\t\tCamera resolution set to {width}x{height}.")
        print(f"\t\tBrightness set to {bvalue}.")
        print('\t\033[92mCameras initialized \u2714 \033[0m')

        # Read the first frame to calculate center of the left camera
        _, frame_left = camera_left.read()
        frame_center = (frame_left.shape[1] // 2, frame_left.shape[0] // 2)     # Calculate the center of the frame

    return camera_left, camera_right, frame_center

# Set processor for the models
def set_processor(models, processor, fps):
    if models is None:
        print("\t\t\033[91mError: No models provided.\033[0m")
        exit(1)

    for model, _ in models:
        if processor == 'GPU':
            model.to('cuda')
        else:
            model.to('cpu')
    print(f'\tRunning on {processor} at {fps} FPS.')

# Calculate the camera's horizontal and vertical field of view (in degrees)
def calculate_fov(diagonal_fov_deg, width, height):
    # Convert diagonal FoV to radians
    diagonal_fov_rad = math.radians(diagonal_fov_deg)

    # Aspect ratio normalization
    aspect_diag = math.sqrt(width**2 + height**2)
    a = width / aspect_diag
    b = height / aspect_diag

    # Calculate horizontal and vertical FoV in radians
    horizontal_fov_rad = 2 * math.atan(math.tan(diagonal_fov_rad / 2) * a)
    vertical_fov_rad = 2 * math.atan(math.tan(diagonal_fov_rad / 2) * b)

    # Convert back to degrees
    horizontal_fov_deg = math.degrees(horizontal_fov_rad)
    vertical_fov_deg = math.degrees(vertical_fov_rad)

    return (horizontal_fov_deg, vertical_fov_deg)