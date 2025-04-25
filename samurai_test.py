import cv2
import torch
import numpy as np
import os
import sys
import time
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- SAMURAI/SAM 2 Setup ---
# Assuming SAMURAI/SAM 2 is installed in the environment
# Add the path to the SAMURAI repository if it's not automatically found
# Example:
# samurai_repo_path = '/path/to/your/samurai'
# sam2_repo_path = os.path.join(samurai_repo_path, 'sam2')
# if sam2_repo_path not in sys.path:
#     sys.path.append(sam2_repo_path)

try:
    from sam2.modeling import Sam
    from sam2.sam_predictor import SamPredictor
    from sam2.build_sam import build_sam2
    from sam2.utils.misc import get_model_path
except ImportError as e:
    logging.error(f"Error importing SAM 2 components: {e}")
    logging.error("Please ensure SAMURAI/SAM 2 is correctly installed in your Python environment.")
    logging.error("Make sure you have run 'pip install -e .' inside the 'sam2' directory of the SAMURAI repo.")
    sys.exit(1)

# Model configuration (adjust if using a different SAM 2 model)
MODEL_TYPE = "sam2_hiera_large_patch16" # Example model, check SAMURAI/SAM 2 for options
# Construct the path to the checkpoint relative to this script or using an absolute path
# Assumes checkpoints are downloaded into a 'checkpoints' subdir in the project root
project_root = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(project_root, 'checkpoints', get_model_path(MODEL_TYPE))

# Device selection (cuda, mps, or cpu)
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Functions ---
def load_sam_predictor(model_type, checkpoint_path, device):
    """Loads the SAM 2 model and predictor."""
    logging.info(f"Loading SAM 2 model: {model_type} from {checkpoint_path} onto {device}")
    try:
        if not os.path.exists(checkpoint_path):
             logging.error(f"Checkpoint file not found at: {checkpoint_path}")
             logging.error("Please ensure checkpoints are downloaded correctly using './download_ckpts.sh' in the 'checkpoints' directory.")
             return None

        sam_model = build_sam2(model_type, checkpoint_path).to(device)
        sam_model.eval() # Set model to evaluation mode
        predictor = SamPredictor(sam_model)
        logging.info("SAM 2 model loaded successfully.")
        return predictor
    except Exception as e:
        logging.error(f"Failed to load SAM 2 model: {e}", exc_info=True)
        return None

def draw_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Draws a translucent mask onto an image."""
    overlay = image.copy()
    overlay[mask] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# --- Main Function ---
def main():
    logging.info("--- Starting SAMURAI/SAM 2 Webcam Test ---")

    # 1. Load SAM Predictor
    predictor = load_sam_predictor(MODEL_TYPE, CHECKPOINT_PATH, DEVICE)
    if predictor is None:
        return

    # 2. Initialize Video Capture
    logging.info("Initializing video capture (Camera Index 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera.")
        return
    logging.info("Camera opened successfully.")

    # 3. Get Initial Frame and User Prompt (Bounding Box)
    input_bbox = None
    first_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Cannot read frame.")
            cap.release()
            return

        cv2.putText(frame, "Draw a box around the object to track", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press ENTER to confirm, C to cancel/redraw", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        roi = cv2.selectROI("Select Object to Track", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select Object to Track")

        if roi == (0, 0, 0, 0): # User cancelled (pressed C or Esc)
            logging.info("ROI selection cancelled. Retrying...")
            continue
        else:
            # Convert (x, y, w, h) to (x1, y1, x2, y2) format for SAM
            input_bbox = np.array([roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]])
            first_frame = frame.copy()
            logging.info(f"Initial bounding box selected: {input_bbox}")
            break

    # 4. Set Image in Predictor (Needed for subsequent predictions)
    logging.info("Setting initial frame in SAM predictor...")
    try:
         # Convert frame to RGB as SAM expects RGB
        predictor.set_image(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logging.error(f"Error setting image in predictor: {e}")
        cap.release()
        return

    # --- Tracking Loop (Using frame-by-frame SAM prediction) ---
    # Note: This uses SAM's basic prediction, not SAMURAI's full tracker logic
    #       It might lose the object easily without motion/memory components.
    last_bbox = input_bbox
    last_mask = None

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Cannot read frame from stream.")
            break

        start_time = time.time()
        display_frame = frame.copy()

        try:
            # Set the new frame in the predictor
            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Predict using the bounding box from the *last* frame as the prompt
            # We use the bounding box as the primary prompt here.
            # For more robust tracking, point prompts or mask prompts might be needed,
            # and potentially re-calculating the box from the previous mask.
            masks, scores, logits = predictor.predict(
                box=last_bbox[None, :], # Add batch dimension
                multimask_output=False # Get only the single best mask
            )

            mask = masks[0] # Get the single mask (batch dim, height, width)
            score = scores[0]
            last_mask = mask # Store for potential use or refinement

            # --- Update BBox for next frame (Simple approach: get bounding box of the new mask) ---
            # Find contours to get the bounding box of the segmented area
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Find the largest contour (in case of multiple small regions)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                last_bbox = np.array([x, y, x + w, y + h])
                logging.debug(f"Predicted mask score: {score:.3f}, New BBox: {last_bbox}")
            else:
                # Mask is empty or too small, track lost? Keep last box for now?
                # Or reset? For this test, we keep the last known box.
                logging.warning("No contour found from predicted mask. Keeping previous bounding box.")

            # Draw the mask on the display frame
            display_frame = draw_mask(display_frame, mask, color=(0, 255, 0)) # Green mask
            # Draw the updated bounding box
            cv2.rectangle(display_frame, (last_bbox[0], last_bbox[1]), (last_bbox[2], last_bbox[3]), (255, 0, 0), 2) # Blue box

        except Exception as e:
            logging.error(f"Error during SAM prediction: {e}", exc_info=True)
            # Continue without drawing if prediction fails

        end_time = time.time()
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('SAMURAI Test (Basic SAM Prediction) - Press "q" to quit', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logging.info("'q' pressed. Exiting.")
            break

    # --- Cleanup ---
    logging.info("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    logging.info("--- SAMURAI Test Utility Finished ---")

if __name__ == "__main__":
    main() 