import cv2
import time
import logging
import os
import sys
from dotenv import load_dotenv

# Add project root to Python path to allow importing project modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from face_recognizer import FaceRecognizer
    import db_utils
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure this script is run from the project root directory or the path is configured correctly.")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv() # Load environment variables from .env file

# Face recognition settings from environment or defaults
USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
MODEL_PACK_NAME = os.getenv('MODEL_PACK_NAME', 'buffalo_l')

# --- Main Function ---
def main():
    logging.info("--- Starting Add Person Utility ---")

    # 1. Initialize Database
    logging.info("Initializing database connection...")
    if not db_utils.initialize_database():
        logging.error("Database initialization failed. Check .env configuration and server status. Exiting.")
        return
    logging.info("Database connection ready.")

    # 2. Initialize Face Recognizer
    logging.info(f"Initializing face recognizer (GPU: {USE_GPU}, Model: {MODEL_PACK_NAME})...")
    face_recognizer = FaceRecognizer(use_gpu=USE_GPU, model_pack_name=MODEL_PACK_NAME)
    if face_recognizer.app is None:
        logging.error("Failed to initialize face recognizer. Check models and dependencies. Exiting.")
        return
    logging.info("Face recognizer ready.")

    # 3. Initialize Video Capture
    logging.info("Initializing video capture (Camera Index 0)...")
    cap = cv2.VideoCapture(0) # 0 is usually the default built-in webcam
    if not cap.isOpened():
        logging.error("Cannot open camera. Ensure it's connected and not in use.")
        return
    logging.info("Camera opened successfully.")

    capture_face = False
    captured_embedding = None
    frame_to_save = None
    last_detection_time = 0
    detection_interval = 0.1 # Run detection every 0.1 seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Cannot read frame from camera stream. Exiting.")
            break

        current_time = time.time()
        display_frame = frame.copy()
        best_face_data = None # Store data of the face to be potentially saved

        # Run detection periodically
        if current_time - last_detection_time > detection_interval:
            last_detection_time = current_time
            faces_data = face_recognizer.analyze_frame(frame)

            if faces_data:
                # Find the largest face (often the most prominent/closest)
                largest_face_area = -1
                for face_data in faces_data:
                    bbox = face_data['bbox']
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > largest_face_area:
                        largest_face_area = area
                        best_face_data = face_data

                # Draw boxes for all faces
                for face_data in faces_data:
                     bbox = face_data['bbox']
                     color = (0, 0, 255) # Red for potential candidates
                     if face_data is best_face_data:
                         color = (0, 255, 0) # Green for the largest face
                         cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3) # Thicker border
                     else:
                         cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # else: # Optionally clear best_face_data if no faces are detected now
                # best_face_data = None


        # Add instruction text
        cv2.putText(display_frame, "Align face in green box.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 's' to save face, 'q' to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Add Person - Press "s" to save, "q" to quit', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            logging.info("'q' pressed. Exiting.")
            break
        elif key == ord('s'):
            if best_face_data and 'embedding' in best_face_data:
                logging.info("'s' pressed. Capturing face...")
                captured_embedding = best_face_data['embedding']
                frame_to_save = frame # Save the original frame at the moment of capture
                capture_face = True
                break # Exit loop to proceed with saving
            else:
                logging.warning("Cannot save: No face detected or embedding extraction failed. Try again.")

    # --- Save Face to DB ---
    if capture_face and captured_embedding is not None:
        # Keep showing the captured frame while asking for name
        cv2.putText(frame_to_save, "CAPTURED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Add Person - Captured Frame', frame_to_save)
        cv2.waitKey(500) # Display for a moment

        while True:
            person_name = input("Enter the name for the captured face (or type 'cancel'): ")
            if not person_name:
                print("Name cannot be empty.")
                continue
            if person_name.lower() == 'cancel':
                logging.info("Save operation cancelled by user.")
                break

            logging.info(f"Attempting to add '{person_name}' to the database...")
            success = db_utils.add_face(person_name, captured_embedding)

            if success:
                logging.info(f"Database operation for '{person_name}' completed successfully (added or updated).")
                print(f"Successfully added/updated '{person_name}' in the database.")
                # Optionally save the captured frame image
                # timestamp = time.strftime("%Y%m%d-%H%M%S")
                # img_filename = f"known_faces/{person_name}_{timestamp}.jpg"
                # os.makedirs("known_faces", exist_ok=True)
                # cv2.imwrite(img_filename, frame_to_save)
                # logging.info(f"Saved captured image to {img_filename}")
            else:
                logging.error(f"Database operation failed for '{person_name}'. Check previous logs.")
                print(f"Failed to add/update face for '{person_name}'. Check logs for details.")
                # Ask if user wants to retry or cancel
                retry = input("Retry adding? (y/n): ").lower()
                if retry != 'y':
                    logging.info("Save operation aborted after failure.")
                    break
                continue # Allow retry with potentially different name or after fixing DB issue

            break # Exit name input loop after success or non-retry failure

    # --- Cleanup ---
    logging.info("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    logging.info("--- Add Person Utility Finished ---")

if __name__ == "__main__":
    main() 