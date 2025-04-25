import cv2
import time
import logging
import os
from dotenv import load_dotenv
import numpy as np # Needed for clamping

# Import project modules
from drone_handler import DroneHandler
from face_recognizer import FaceRecognizer
import db_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv() # Load environment variables from .env file

# Face recognition settings from environment or defaults
USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
MODEL_PACK_NAME = os.getenv('MODEL_PACK_NAME', 'buffalo_l')

# Time to run before landing (seconds)
FLIGHT_DURATION = 120 # Increased duration for testing centering

# Cooldown period (seconds) after recognizing someone before logging again
RECOGNITION_COOLDOWN = 5

# --- Drone Control Parameters ---
# Target face height as a ratio of the frame height
TARGET_FACE_HEIGHT_RATIO = 0.25 # Aim for face to be 25% of frame height
# Proportional gains for controller ( এগুলো পরিবর্তন করুন / Adjust these values)
KP_YAW = 0.25   # How strongly to react to horizontal error
KP_UD = 0.2     # How strongly to react to vertical error (REDUCED from 0.4)
KP_FB = 0.3     # How strongly to react to size error (forward/backward)
# Maximum control velocities (range: -100 to 100)
MAX_YAW_VEL = 60
MAX_UD_VEL = 70
MAX_FB_VEL = 50
# Dead zone for control errors (ignore small errors to prevent jitter)
ERROR_DEAD_ZONE_X = 15  # Pixels
ERROR_DEAD_ZONE_Y = 15  # Pixels
ERROR_DEAD_ZONE_H = 10 # Pixels (for height difference)
# Time to hover without face before searching (seconds)
SEARCH_TIMEOUT = 3.0
SEARCH_YAW_VEL = 25 # Slow rotation speed when searching

# --- Helper Functions ---
def draw_face_info(frame, faces_data, target_face_index):
    """Draws bounding boxes and recognition info on the frame.
       Highlights the target face.
    """
    for i, face_data in enumerate(faces_data):
        bbox = face_data['bbox']
        name = face_data.get('name') # Get name added during processing
        distance = face_data.get('distance')

        # Box color: Green if recognized, Red if unknown
        color = (0, 255, 0) if name else (0, 0, 255)
        thickness = 2
        # Highlight the target face
        if i == target_face_index:
            color = (255, 255, 0) # Cyan for target
            thickness = 3

        label = "Unknown"
        if name:
            label = f"{name} ({distance:.2f})"
        elif distance is not None: # Show distance even if unknown but match found
             label = f"Unknown ({distance:.2f})"

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

        # Prepare label text
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Put label background
        label_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
        cv2.rectangle(frame, (bbox[0], label_y - label_size[1] - base_line),
                      (bbox[0] + label_size[0], label_y + base_line),
                      color, cv2.FILLED)
        # Put label text
        cv2.putText(frame, label, (bbox[0], label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame

# --- Main Application Logic ---
def main():
    logging.info("--- Starting Drone Facial Recognition Application ---")

    # 1. Initialize Database
    logging.info("Initializing database...")
    if not db_utils.initialize_database():
        logging.error("Database initialization failed. Please check configuration and server status. Exiting.")
        return
    logging.info("Database ready.")

    # 2. Initialize Face Recognizer
    logging.info(f"Initializing face recognizer (GPU: {USE_GPU}, Model: {MODEL_PACK_NAME})...")
    face_recognizer = FaceRecognizer(use_gpu=USE_GPU, model_pack_name=MODEL_PACK_NAME)
    if face_recognizer.app is None:
        logging.error("Failed to initialize face recognizer. Check models and dependencies. Exiting.")
        return
    logging.info("Face recognizer ready.")

    # 3. Initialize Drone Handler
    logging.info("Initializing drone handler...")
    drone = DroneHandler()

    keep_running = True
    recognized_log = {} # Track last recognition time per person

    try:
        # 4. Connect to Drone
        if not drone.connect():
            logging.error("Failed to connect to drone. Exiting.")
            return

        # 5. Start Video Stream
        if not drone.start_stream():
            logging.error("Failed to start drone video stream. Exiting.")
            drone.disconnect()
            return

        # 6. Takeoff
        logging.info("Attempting takeoff...")
        if not drone.takeoff():
            logging.error("Drone takeoff failed. Landing and exiting.")
            drone.land() # Attempt landing just in case
            drone.disconnect()
            return
        logging.info(f"Drone airborne. Starting recognition and centering loop for {FLIGHT_DURATION} seconds.")

        start_time = time.time()
        frame_count = 0
        fps = 0
        last_face_seen_time = time.time() # Track when a face was last detected

        # 7. Main Loop (Control, Recognize, Display)
        while keep_running and (time.time() - start_time) < FLIGHT_DURATION:
            loop_start_time = time.time()

            # Get frame from drone
            frame = drone.get_frame()
            if frame is None:
                logging.warning("Failed to get frame from drone. Skipping cycle.")
                if drone.is_flying:
                     drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.1)
                continue

            # --- Face Detection and Recognition ---
            frame_height, frame_width, _ = frame.shape
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2
            target_face_height = frame_height * TARGET_FACE_HEIGHT_RATIO

            # Analyze frame for faces
            faces_data = face_recognizer.analyze_frame(frame)
            current_time = time.time()
            recognized_faces = []
            unknown_faces = []

            # Process and classify all detected faces
            if faces_data:
                last_face_seen_time = current_time # Reset timer
                for i, face in enumerate(faces_data):
                    face['original_index'] = i # Keep track of original index for display
                    bbox = face['bbox']
                    face['area'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    embedding = face.get('embedding')
                    name = None
                    distance = None
                    if embedding is not None:
                        name, distance = db_utils.find_similar_face(embedding)
                        face['name'] = name # Add recognition result directly to face data
                        face['distance'] = distance
                        if name:
                            recognized_faces.append(face)
                             # Log recognized person if cooldown period passed
                            last_seen = recognized_log.get(name, 0)
                            if current_time - last_seen > RECOGNITION_COOLDOWN:
                                logging.info(f"RECOGNIZED: {name} (Distance: {distance:.4f}) - Tracking")
                                recognized_log[name] = current_time
                        else:
                            unknown_faces.append(face)
                    else:
                        # If embedding failed, treat as unknown but don't add to unknown_faces list for targeting
                        logging.warning(f"Face {i} detected but embedding extraction failed.")
                        face['name'] = None
                        face['distance'] = None
            # else: logging.debug("No faces detected in this frame.")

            # --- Target Selection Logic ---
            target_face = None
            target_face_index = -1 # Original index for highlighting

            if recognized_faces:
                # Priority 1: Target the largest recognized face
                recognized_faces.sort(key=lambda x: x['area'], reverse=True)
                target_face = recognized_faces[0]
                target_face_index = target_face['original_index']
                logging.debug(f"Targeting recognized face: {target_face['name']}")
            elif unknown_faces:
                # Priority 2: Target the largest unknown face
                unknown_faces.sort(key=lambda x: x['area'], reverse=True)
                target_face = unknown_faces[0]
                target_face_index = target_face['original_index']
                logging.debug("Targeting largest unknown face.")
            # else: No faces detected, target_face remains None

            # --- Drone Control Logic ---
            yaw_velocity = 0
            ud_velocity = 0
            fb_velocity = 0

            if target_face and drone.is_flying:
                bbox = target_face['bbox']
                face_center_x = (bbox[0] + bbox[2]) // 2
                face_center_y = (bbox[1] + bbox[3]) // 2
                face_height = bbox[3] - bbox[1]

                error_x = face_center_x - frame_center_x
                error_y = frame_center_y - face_center_y
                error_fb = target_face_height - face_height

                if abs(error_x) > ERROR_DEAD_ZONE_X:
                    yaw_velocity = KP_YAW * error_x
                if abs(error_y) > ERROR_DEAD_ZONE_Y:
                    ud_velocity = KP_UD * error_y
                if abs(error_fb) > ERROR_DEAD_ZONE_H:
                    fb_velocity = KP_FB * error_fb

                yaw_velocity = int(np.clip(yaw_velocity, -MAX_YAW_VEL, MAX_YAW_VEL))
                ud_velocity = int(np.clip(ud_velocity, -MAX_UD_VEL, MAX_UD_VEL))
                fb_velocity = int(np.clip(fb_velocity, -MAX_FB_VEL, MAX_FB_VEL))

                logging.debug(f"Control: Target '{target_face.get('name', 'Unknown')}' Err(x:{error_x}, y:{error_y}, h:{error_fb}) -> Vel(yaw:{yaw_velocity}, ud:{ud_velocity}, fb:{fb_velocity})")
                drone.send_rc_control(0, fb_velocity, ud_velocity, yaw_velocity)

            elif drone.is_flying:
                # No target face selected (either no faces or only faces with failed embeddings)
                time_since_last_face = current_time - last_face_seen_time
                if time_since_last_face < SEARCH_TIMEOUT:
                    logging.debug("No targetable face detected, hovering...")
                    drone.send_rc_control(0, 0, 0, 0)
                else:
                    logging.debug(f"No targetable face detected for {time_since_last_face:.1f}s, searching...")
                    drone.send_rc_control(0, 0, 0, SEARCH_YAW_VEL)

            # --- Display Information ---
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - loop_start_time
            if elapsed_time > 0:
                 fps = 1.0 / elapsed_time

            # Draw info on the frame, highlighting the target
            display_frame = draw_face_info(frame.copy(), faces_data, target_face_index)

            # Draw frame center (already done in draw_face_info if needed, or keep here)
            # cv2.circle(display_frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)

            # Add FPS to display
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Drone Facial Recognition", display_frame)

            # Exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("'q' pressed. Landing and exiting.")
                keep_running = False
            elif key == ord('e'): # Emergency stop
                logging.critical("'e' pressed. EMERGENCY STOP!")
                drone.emergency_stop()
                keep_running = False

        # Check if loop exited due to duration limit
        if time.time() - start_time >= FLIGHT_DURATION:
            logging.info(f"Flight duration ({FLIGHT_DURATION}s) reached. Landing.")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Landing and exiting.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
        logging.critical("Attempting emergency stop due to unexpected error.")
        if 'drone' in locals() and drone.is_connected: # Check if drone object exists and connected
            drone.emergency_stop()
    finally:
        # 8. Cleanup
        logging.info("Cleaning up resources...")
        cv2.destroyAllWindows()
        if 'drone' in locals(): # Check if drone object was initialized
            if drone.is_flying: # Land if still flying
                logging.info("Ensuring drone is landed...")
                # Send zero velocity before landing
                drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                drone.land()
                time.sleep(5) # Wait for landing
            drone.disconnect()
        logging.info("--- Application Finished ---")

if __name__ == "__main__":
    main() 