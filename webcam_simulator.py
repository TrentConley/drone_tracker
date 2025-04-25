import cv2
import time
import logging
import os
import numpy as np
from dotenv import load_dotenv
import camera_utils
from ultralytics import YOLO
from face_recognizer import FaceRecognizer
import db_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
MODEL_PACK_NAME = os.getenv('MODEL_PACK_NAME', 'buffalo_l')

# --- Drone Control Parameters (for simulation) ---
KP_YAW = 0.25
KP_FB = 0.4
MAX_YAW_VEL = 60
MAX_FB_VEL = 40
SEARCH_YAW_VEL = 30 # Slow rotation speed when searching
MAX_LOST_FRAMES = 30 # Frames before resetting target

# --- Main Simulation Logic ---
def main():
    logging.info("--- Starting Webcam Drone Simulator ---")

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

    # 4. Initialize Webcam
    camera_index = 0
    logging.info(f"Initializing webcam (index {camera_index}, aiming for built-in, not Continuity Camera)...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error("Failed to open webcam. Exiting.")
        return
    logging.info("Webcam ready.")

    keep_running = True

    # Setup tracking state
    target_id = None
    target_name = None
    lost_frames = 0

    try:
        # 5. Main Simulation Loop
        while keep_running:
            loop_start_time = time.time()

            # Get frame from webcam
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("Failed to get frame from webcam. Skipping cycle.")
                time.sleep(0.1)
                continue

            # 1) Run ByteTrack person detection
            # Use camera_utils to get camera profile (e.g., 'macbook' or define a new one)
            # You might need to adjust this based on your webcam
            camera_profile = 'macbook'
            try:
                # Verify the camera profile exists, though the distance func will also check
                if camera_profile not in camera_utils.CAMERA_PARAMS:
                    raise KeyError(f"Camera profile '{camera_profile}' not found in camera_utils.CAMERA_PARAMS")
            except KeyError:
                logging.error(f"Camera profile '{camera_profile}' not found in camera_utils. Exiting.")
                break

            results = model.track(frame, tracker='bytetrack.yaml', classes=[0], persist=True, stream=False, verbose=False) # Less verbose tracking

            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                res = results[0]
                annotated = res.plot() # Draw boxes from tracker
                boxes = res.boxes.xyxy.cpu().numpy()
                # Check if tracks exist before accessing id
                if res.boxes.id is not None:
                    ids = res.boxes.id.cpu().numpy().astype(int)
                else:
                    ids = np.empty((0,), dtype=int) # No tracks
            else:
                annotated = frame.copy() # Keep original frame if no results
                boxes = np.empty((0,4))
                ids = np.empty((0,), dtype=int)

            frame_height, frame_width, _ = frame.shape
            frame_center_x = frame_width // 2
            current_time = time.time()

            # Default simulated command
            sim_lr, sim_fb, sim_ud, sim_yaw = 0, 0, 0, 0

            # 2) Lock onto a recognized face or track existing target
            if target_id is None:
                # Run face recognition to find a target
                faces = face_recognizer.analyze_frame(frame)
                logging.debug(f"Searching: Found {len(faces)} faces. Found {len(ids)} person tracks.")
                for f in faces:
                    emb = f.get('embedding')
                    if emb is None: continue
                    name, dist = db_utils.find_similar_face(emb) # Use distance threshold from db_utils
                    if not name:
                        logging.debug(f"Face found, but no match (dist={dist:.2f}).")
                        continue

                    logging.debug(f"Recognized {name} (dist={dist:.2f}). Checking containment...")
                    # Containment match: Check if face center is inside any person box
                    fb = f['bbox']; fcx=(fb[0]+fb[2])/2; fcy=(fb[1]+fb[3])/2
                    for box, tid in zip(boxes, ids):
                        if box[0] <= fcx <= box[2] and box[1] <= fcy <= box[3]:
                            target_id, target_name = tid, name
                            lost_frames = 0
                            logging.info(f"Locked on {name} with track ID {tid}")
                            # Draw face box for confirmation during lock-on
                            cv2.rectangle(annotated, (fb[0], fb[1]), (fb[2], fb[3]), (255, 0, 255), 2)
                            break # Stop checking boxes for this face
                    if target_id is not None:
                        break # Stop checking other faces
                # If still no target, simulate searching
                if target_id is None:
                    sim_yaw = SEARCH_YAW_VEL
                    print(f"SIM CMD: Searching... (lr={sim_lr}, fb={sim_fb}, ud={sim_ud}, yaw={sim_yaw})")

            else:
                # Track existing target and simulate movement
                if target_id in ids:
                    lost_frames = 0
                    idx = np.where(ids == target_id)[0][0]
                    x1,y1,x2,y2 = boxes[idx].astype(int)
                    person_width_px = x2 - x1

                    # Calculate distance using camera utils
                    dist_m = camera_utils.calculate_distance_from_width(
                        person_width_px, camera_profile
                    )

                    if dist_m is not None:
                         # Compute control errors for simulation
                         err_x = ((x1+x2)//2) - frame_center_x
                         err_fb = dist_m - 0.5 # Target distance 0.5m

                         # Simulate control commands
                         sim_yaw = np.clip(KP_YAW * err_x, -MAX_YAW_VEL, MAX_YAW_VEL)
                         sim_fb = np.clip(KP_FB * err_fb, -MAX_FB_VEL, MAX_FB_VEL)

                         # Highlight locked target
                         cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),4)
                         cv2.putText(annotated, f"{target_name} {dist_m:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                         # Print with formatting for floats
                         print(f"SIM CMD: Tracking {target_name} - Dist={dist_m:.2f}m - Target=0.5m => (lr={sim_lr:.0f}, fb={sim_fb:.1f}, ud={sim_ud:.0f}, yaw={sim_yaw:.1f})")
                    else:
                         # Couldn't calculate distance, simulate hover/search
                         sim_yaw = SEARCH_YAW_VEL / 2 # Slower search if distance fails
                         print(f"SIM CMD: Tracking {target_name} - DISTANCE FAILED => Searching... (lr={sim_lr}, fb={sim_fb}, ud={sim_ud}, yaw={sim_yaw})")

                else:
                    # Target ID not found in current frame
                    lost_frames += 1
                    print(f"SIM CMD: Lost track of {target_name} (frame {lost_frames}/{MAX_LOST_FRAMES}). Hovering...")
                    # Simulate hover while lost briefly
                    sim_lr, sim_fb, sim_ud, sim_yaw = 0, 0, 0, 0
                    if lost_frames > MAX_LOST_FRAMES:
                        logging.info(f"Lost track of {target_name} for {MAX_LOST_FRAMES} frames, resetting target.")
                        target_id = None
                        target_name = None
                        # Simulate searching after reset
                        sim_yaw = SEARCH_YAW_VEL
                        print(f"SIM CMD: Resetting target. Searching... (lr={sim_lr}, fb={sim_fb}, ud={sim_ud}, yaw={sim_yaw})")


            # --- Display Information ---
            fps = 1.0 / (time.time() - loop_start_time)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (frame_width - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if target_id is not None:
                 cv2.putText(annotated, f"TARGET: {target_name} (ID: {target_id})", (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                  cv2.putText(annotated, "TARGET: None (Searching)", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            # Display the annotated frame
            cv2.imshow("Webcam Drone Simulator", annotated)

            # Exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("'q' pressed. Exiting simulator.")
                keep_running = False

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting simulator.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        # Cleanup
        logging.info("Cleaning up resources...")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        logging.info("--- Simulator Finished ---")

if __name__ == "__main__":
    main() 