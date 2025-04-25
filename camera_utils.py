# camera_utils.py
import numpy as np
import logging
from typing import Union

# --- Constants ---
# Known object width (average male shoulder width)
KNOWN_PERSON_SHOULDER_WIDTH_M = 36 * 0.0254 # 30 inches to meters
# adjusted to be higher than average shoulder width because bounding box is too large. I could use keypoints to get a more accurate width.
# TODO: use keypoints to get a more accurate width

# --- Camera Parameters ---

# Tello EDU Camera Specs (from user info)
TELLO_VIDEO_WIDTH_PX = 1280
TELLO_SENSOR_WIDTH_MM = 2.77
TELLO_FOCAL_LENGTH_MM = 2.0
TELLO_FOCAL_LENGTH_PX = (TELLO_FOCAL_LENGTH_MM / TELLO_SENSOR_WIDTH_MM) * TELLO_VIDEO_WIDTH_PX
# Focal length (pixels) = (Focal length (mm) / Sensor width (mm)) * Image width (pixels)
logging.info(f"Tello Focal Length (pixels, horizontal): {TELLO_FOCAL_LENGTH_PX:.2f}")

# MacBook FaceTime HD Camera Specs (from user info)
MACBOOK_VIDEO_WIDTH_PX = 1280
MACBOOK_SENSOR_WIDTH_MM = 3.6
MACBOOK_FOCAL_LENGTH_MM = 2.0
MACBOOK_FOCAL_LENGTH_PX = (MACBOOK_FOCAL_LENGTH_MM / MACBOOK_SENSOR_WIDTH_MM) * MACBOOK_VIDEO_WIDTH_PX
logging.info(f"MacBook Focal Length (pixels, horizontal): {MACBOOK_FOCAL_LENGTH_PX:.2f}")

# Store parameters in a dictionary for easy lookup
CAMERA_PARAMS = {
    'tello': {
        'focal_length_px': TELLO_FOCAL_LENGTH_PX,
        'known_object_width_m': KNOWN_PERSON_SHOULDER_WIDTH_M,
    },
    'macbook': {
        'focal_length_px': MACBOOK_FOCAL_LENGTH_PX,
        'known_object_width_m': KNOWN_PERSON_SHOULDER_WIDTH_M,
    }
}

# --- Distance Calculation Function ---

def calculate_distance_from_width(pixel_width: int, camera_name: str) -> Union[float, None]:
    """
    Calculates the distance to an object based on its pixel width in the image.

    Uses the formula: Distance = (Known Object Width * Focal Length) / Pixel Width

    Args:
        pixel_width (int): The width of the object's bounding box in pixels.
        camera_name (str): The name of the camera ('tello' or 'macbook') to use its parameters.

    Returns:
        Union[float, None]: Estimated distance in meters, or None if calculation fails.
    """
    if camera_name not in CAMERA_PARAMS:
        logging.error(f"Unknown camera name: {camera_name}. Available: {list(CAMERA_PARAMS.keys())}")
        return None

    params = CAMERA_PARAMS[camera_name]
    focal_length_px = params['focal_length_px']
    known_width_m = params['known_object_width_m']

    if pixel_width <= 0:
        # logging.debug("Pixel width is zero or negative, cannot calculate distance.")
        return None

    try:
        distance_m = (known_width_m * focal_length_px) / pixel_width
        return distance_m
    except ZeroDivisionError:
        logging.warning("Division by zero error in distance calculation (pixel_width was likely zero).")
        return None
    except Exception as e:
        logging.error(f"Error calculating distance: {e}")
        return None 