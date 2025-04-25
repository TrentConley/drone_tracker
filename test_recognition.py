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
    logging.info("--- Starting Recognition Test Utility ---")

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

    last_detection_time = 0
    detection_interval = 0.1 # Run detection every 0.1 seconds (adjust as needed)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Cannot read frame from camera stream. Exiting.")
            break

        current_time = time.time()
        display_frame = frame.copy()

        # Run detection periodically
        if current_time - last_detection_time > detection_interval:
            last_detection_time = current_time
            faces_data = face_recognizer.analyze_frame(frame)

            # Process detected faces for recognition
            for face_data in faces_data:
                name = None
                distance = None
                embedding = face_data.get('embedding')

                if embedding is not None:
                    name, distance = db_utils.find_similar_face(embedding)
                else:
                    logging.warning(f"Face detected but embedding extraction failed. BBox: {face_data['bbox']}")

                # --- Draw Bounding Box and Label ---
                bbox = face_data['bbox']
                color = (0, 0, 255) # Default: Red for unknown/failed embedding
                label = "Unknown"

                if name:
                    color = (0, 255, 0) # Green for recognized
                    label = f"{name} ({distance:.2f})"
                    logging.debug(f"Recognized: {name}, Distance: {distance:.2f}")
                elif distance is not None: # Match found but below threshold
                    label = f"Unknown ({distance:.2f})"
                    logging.debug(f"Unknown match, Distance: {distance:.2f}")
                # else: No match found or embedding failed, label remains "Unknown"

                # Draw bounding box
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                # Prepare label text background
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                label_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                cv2.rectangle(display_frame, (bbox[0], label_y - label_size[1] - base_line),
                              (bbox[0] + label_size[0], label_y + base_line),
                              color, cv2.FILLED)
                # Put label text
                cv2.putText(display_frame, label, (bbox[0], label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display FPS (optional)
        # Add FPS calculation if needed here

        # Display the resulting frame
        cv2.imshow('Recognition Test - Press "q" to quit', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logging.info("'q' pressed. Exiting.")
            break

    # --- Cleanup ---
    logging.info("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    logging.info("--- Recognition Test Utility Finished ---")

if __name__ == "__main__":
    main() 