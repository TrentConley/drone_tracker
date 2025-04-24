import cv2
import time
import logging
import os
from dotenv import load_dotenv

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

# Time to hover before landing (seconds)
HOVER_DURATION = 60 # Adjust as needed

# Cooldown period (seconds) after recognizing someone before logging again
RECOGNITION_COOLDOWN = 5

# --- Helper Functions ---
def draw_face_info(frame, faces_data, recognition_results):
    """Draws bounding boxes and recognition info on the frame."""
    for i, face_data in enumerate(faces_data):
        bbox = face_data['bbox']
        score = face_data['det_score']
        name = recognition_results.get(i, (None, None))[0] # Get name from results
        distance = recognition_results.get(i, (None, None))[1]

        # Box color: Green if recognized, Red if unknown/below threshold
        color = (0, 255, 0) if name else (0, 0, 255)
        label = "Unknown"
        if name:
            label = f"{name} ({distance:.2f})"
        else:
            # If distance is available (closest match but below threshold), show it
            if distance is not None:
                label = f"Unknown ({distance:.2f})"

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

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
        logging.info(f"Drone airborne. Hovering and starting recognition loop for {HOVER_DURATION} seconds.")

        start_time = time.time()
        frame_count = 0
        fps = 0

        # 7. Main Loop (Hover, Recognize, Display)
        while keep_running and (time.time() - start_time) < HOVER_DURATION:
            loop_start_time = time.time()

            # Get frame from drone
            frame = drone.get_frame()
            if frame is None:
                logging.warning("Failed to get frame from drone. Skipping cycle.")
                # Keep hovering command active even if frame fails
                if drone.is_flying:
                     drone.hover()
                time.sleep(0.1) # Avoid busy-waiting
                continue

            # Keep drone hovering
            if drone.is_flying:
                drone.hover()

            # Resize frame for faster processing (optional)
            # small_frame = cv2.resize(frame, (640, 480))
            small_frame = frame # Process original frame for now

            # Analyze frame for faces
            # Run analysis less frequently if needed for performance
            faces_data = face_recognizer.analyze_frame(small_frame)

            recognition_results = {} # Store results for this frame {index: (name, distance)}
            current_time = time.time()

            # Process detected faces
            if faces_data:
                logging.debug(f"Detected {len(faces_data)} faces.")
                for i, face in enumerate(faces_data):
                    embedding = face.get('embedding')
                    if embedding is not None:
                        # Find similar face in DB
                        name, distance = db_utils.find_similar_face(embedding)
                        recognition_results[i] = (name, distance)

                        # Log recognized person if cooldown period passed
                        if name:
                            last_seen = recognized_log.get(name, 0)
                            if current_time - last_seen > RECOGNITION_COOLDOWN:
                                logging.info(f"RECOGNIZED: {name} (Distance: {distance:.4f})")
                                recognized_log[name] = current_time
                        # else: logging.debug(f"Face {i} is unknown or below threshold (Dist: {distance})")
                    else:
                        logging.warning(f"Face {i} detected but embedding extraction failed.")
            # else: logging.debug("No faces detected in this frame.")

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - loop_start_time
            if elapsed_time > 0:
                 fps = 1.0 / elapsed_time

            # Draw info on the (original size) frame
            display_frame = draw_face_info(frame.copy(), faces_data, recognition_results)

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
        if time.time() - start_time >= HOVER_DURATION:
            logging.info(f"Hover duration ({HOVER_DURATION}s) reached. Landing.")

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
                drone.land()
                time.sleep(5) # Wait for landing
            drone.disconnect()
        logging.info("--- Application Finished ---")

if __name__ == "__main__":
    main() 