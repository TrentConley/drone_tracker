import cv2
import time
import logging
import os
from dotenv import load_dotenv
import numpy as np # Needed for clamping
import camera_utils
from ultralytics import YOLO

# Import project modules
from drone_handler import DroneHandler
from face_recognizer import FaceRecognizer
import db_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Time to run before landing (seconds)
FLIGHT_DURATION = 120 # Increased duration for testing centering

# --- Drone Control Parameters ---
# Target face height as a ratio of the frame height (adjust based on testing for ~0.5m distance)
TARGET_FACE_HEIGHT_RATIO = 0.25 # Aim for face bbox height to be 25% of frame height
# Proportional gains for controller (TUNE THESE VALUES)
KP_YAW = 0.25   # How strongly to react to horizontal error
KP_UD = 0.3     # How strongly to react to vertical error
KP_FB = 0.4     # How strongly to react to size error (forward/backward)
# Maximum control velocities (range: -100 to 100)
MAX_YAW_VEL = 60
MAX_UD_VEL = 50 # Reduced max vertical speed for stability
MAX_FB_VEL = 40 # Reduced max forward/backward speed for safety
# Dead zone for control errors (ignore small errors to prevent jitter)
ERROR_DEAD_ZONE_X = 20  # Pixels from center X
ERROR_DEAD_ZONE_Y = 20  # Pixels from center Y
ERROR_DEAD_ZONE_H = 15  # Pixels difference from target height
# Time to hover without face before searching (seconds)
SEARCH_TIMEOUT = 3.0
SEARCH_YAW_VEL = 30 # Slow rotation speed when searching

load_dotenv()
USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
MODEL_PACK_NAME = os.getenv('MODEL_PACK_NAME', 'buffalo_l')

# --- Helper Functions ---
def draw_face_info(frame, faces_data, target_face_index):
    """Draws bounding boxes and recognition info on the frame.
       Highlights the target face.
    """
    for i, face_data in enumerate(faces_data):
        # Use original_index if available (added during processing), else use current index i
        current_index = face_data.get('original_index', i)
        bbox = face_data['bbox']
        name = face_data.get('name') # Get name added during processing
        distance = face_data.get('distance')

        # Box color: Green if recognized, Red if unknown
        color = (0, 255, 0) if name else (0, 0, 255)
        thickness = 2
        # Highlight the target face
        if current_index == target_face_index:
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
        logging.error("Database initialization failed. Exiting.")
        return
    logging.info("Database ready.")
    # 2. Initialize Face Recognizer
    logging.info(f"Initializing face recognizer (GPU: {USE_GPU}, Model: {MODEL_PACK_NAME})...")
    face_recognizer = FaceRecognizer(use_gpu=USE_GPU, model_pack_name=MODEL_PACK_NAME)
    if face_recognizer.app is None:
        logging.error("Failed to initialize face recognizer. Exiting.")
        return
    logging.info("Face recognizer ready.")
    # 3. Initialize YOLOv10 + ByteTrack model for person tracking
    model = YOLO('yolov10s.pt')
    logging.info('Loaded YOLOv10s for person tracking')

    # 3. Initialize Drone Handler
    logging.info("Initializing drone handler...")
    drone = DroneHandler()

    keep_running = True

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

        # Setup tracking state
        target_id = None
        target_name = None
        lost_frames = 0
        MAX_LOST_FRAMES = 30

        # 7. Main Loop (Person detection + recognition + distance control)
        while keep_running and (time.time() - start_time) < FLIGHT_DURATION:
            loop_start_time = time.time()

            # Get frame from drone
            frame = drone.get_frame()
            if frame is None:
                logging.warning("Failed to get frame from drone. Skipping cycle.")
                # Send hover command if frame fails to prevent drift
                if drone.is_flying:
                     drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.1) # Avoid busy-waiting
                continue

            # 1) Run ByteTrack person detection
            results = model.track(frame, tracker='bytetrack.yaml', classes=[0], persist=True, stream=False)
            if results:
                res = results[0]
                annotated = res.plot()
                boxes = res.boxes.xyxy.cpu().numpy()
                # Check if tracks exist before accessing id
                if res.boxes.id is not None:
                    ids = res.boxes.id.cpu().numpy().astype(int)
                else:
                    ids = np.empty((0,), dtype=int) # No tracks
            else:
                annotated = frame.copy()
                boxes = np.empty((0,4))
                ids = np.empty((0,), dtype=int)
            frame_height, frame_width, _ = frame.shape
            frame_center_x = frame_width // 2
            current_time = time.time()

            # 2) Lock onto a recognized face or track existing target
            if target_id is None:
                faces = face_recognizer.analyze_frame(frame)
                for f in faces:
                    emb = f.get('embedding')
                    if emb is None: continue
                    name, _ = db_utils.find_similar_face(emb)
                    if not name: continue
                    # containment match
                    fb = f['bbox']; fcx=(fb[0]+fb[2])/2; fcy=(fb[1]+fb[3])/2
                    for box, tid in zip(boxes, ids):
                        if box[0] <= fcx <= box[2] and box[1] <= fcy <= box[3]:
                            target_id, target_name = tid, name
                            lost_frames = 0
                            logging.info(f"Locked on {name} with track ID {tid}")
                            break
                    if target_id is not None: break
                # rotate to search if no lock yet
                if target_id is None and drone.is_flying:
                    drone.send_rc_control(0, 0, 0, SEARCH_YAW_VEL)
            else:
                # track and move towards 0.5m distance
                if target_id in ids and drone.is_flying:
                    lost_frames = 0
                    idx = np.where(ids == target_id)[0][0]
                    x1,y1,x2,y2 = boxes[idx].astype(int)
                    person_width = x2 - x1
                    dist_m = camera_utils.calculate_distance_from_width(person_width, 'tello')
                    err_x = ((x1+x2)//2) - frame_center_x
                    err_fb = dist_m - 0.5
                    yaw_vel = int(np.clip(KP_YAW * err_x, -MAX_YAW_VEL, MAX_YAW_VEL))
                    fb_vel = int(np.clip(KP_FB * err_fb, -MAX_FB_VEL, MAX_FB_VEL))
                    logging.info(f"Control -> yaw: {yaw_vel}, fb: {fb_vel}, dist: {dist_m:.2f}m")
                    drone.send_rc_control(0, fb_vel, 0, yaw_vel)
                    # highlight locked target
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),4)
                    cv2.putText(annotated, f"{target_name} {dist_m:.2f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                else:
                    lost_frames += 1
                    if lost_frames > MAX_LOST_FRAMES:
                        logging.info(f"Lost track of {target_name}, resetting.")
                        target_id = None
                if target_id is None and drone.is_flying:
                    drone.send_rc_control(0, 0, 0, SEARCH_YAW_VEL)

            # Display output
            cv2.imshow("Drone Facial Recognition", annotated)

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
                time.sleep(0.5) # Brief pause after zeroing velocity
                drone.land()
                time.sleep(5) # Wait for landing process
            drone.disconnect()
        logging.info("--- Application Finished ---")

if __name__ == "__main__":
    main() 