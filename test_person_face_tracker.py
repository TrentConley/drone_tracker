import argparse
import os
import cv2
import logging
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from face_recognizer import FaceRecognizer
import db_utils

# --- Configuration ---
# Parse command-line arguments for debugging
parser = argparse.ArgumentParser(description='Test Person+Face Tracker')
parser.add_argument('--debug', action='store_true', help='enable debug logging')
args = parser.parse_args()
# Configure logging level based on --debug flag
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Utility Functions ---

def iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0

# --- Main Script ---
if __name__ == '__main__':
    load_dotenv()
    USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
    MODEL_PACK_NAME = os.getenv('MODEL_PACK_NAME', 'buffalo_l')

    # Initialize face recognizer
    face_recognizer = FaceRecognizer(use_gpu=USE_GPU, model_pack_name=MODEL_PACK_NAME)
    if face_recognizer.app is None:
        logging.error('Failed to initialize face recognizer.')
        exit(1)

    # Load YOLOv10s model for person detection + ByteTrack
    model = YOLO('yolov10s.pt')
    logging.info('Starting YOLOv10s + ByteTrack on camera index 0')

    # Target tracking state
    target_id = None
    target_name = None
    lost_frames = 0
    MAX_LOST_FRAMES = 30

    # Stream from camera, track only class 0=person, persist IDs
    results = model.track(
        source=0,
        tracker='bytetrack.yaml',
        classes=[0],       # only persons
        persist=True,
        stream=True,
        show=False         # we'll handle display ourselves
    )

    # Process each frame result
    for frame_idx, res in enumerate(results):
        logging.debug(f"Frame {frame_idx}: current target_id={target_id}, target_name={target_name}, lost_frames={lost_frames}")
        frame = res.orig_img  # BGR input frame

        # Get YOLO+ByteTrack results
        # Safe extraction: if no tracks/detections, res.boxes.id may be None
        if res.boxes.id is None:
            boxes = np.empty((0, 4))
            ids = np.empty((0,), dtype=int)
        else:
            boxes = res.boxes.xyxy.cpu().numpy()  # N×4
            ids = res.boxes.id.cpu().numpy().astype(int)    # N
        logging.debug(f"Person boxes/IDs in frame: {list(zip(boxes.astype(int), ids))}")

        annotated = res.plot()  # draw base detections

        # If we have already locked onto a person, skip face recognition
        if target_id is not None:
            logging.debug(f"[Locked] Verifying presence of ID {target_id} in current IDs")
            # check if the person is still tracked
            if target_id in ids:
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames > MAX_LOST_FRAMES:
                    logging.info(f"Lost track of '{target_name}' (ID {target_id}), resetting target.")
                    target_id = None
                    target_name = None
            # overlay target box if still present
            if target_id in ids:
                idx = np.where(ids == target_id)[0][0]
                x1, y1, x2, y2 = boxes[idx].astype(int)
                # Highlight locked-on target in bright green
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(annotated, target_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            # run face detection & recognition until we lock on
            faces = face_recognizer.analyze_frame(frame)
            # use face center-point containment to lock onto the recognized person
            for f in faces:
                emb = f.get('embedding')
                if emb is None:
                    logging.warning("Face without embedding, skipping")
                    continue
                name, dist = db_utils.find_similar_face(emb)
                f['name'] = name
                f['distance'] = dist
                if not name:
                    logging.debug(f"Recognized no match for face (best dist {dist:.2f}), continuing")
                    continue
                # compute face center
                fb = f['bbox']
                fcx = (fb[0] + fb[2]) / 2
                fcy = (fb[1] + fb[3]) / 2
                # find which person box contains the face center
                for box, tid in zip(boxes, ids):
                    if box[0] <= fcx <= box[2] and box[1] <= fcy <= box[3]:
                        target_id = tid
                        target_name = name
                        lost_frames = 0
                        logging.info(f"Locked on target '{target_name}' with track ID {target_id} via containment")
                        break
                if target_id is not None:
                    break
            # if no lock yet, we will continue recognition in next frame

        # Display
        cv2.imshow('Person+Face Tracker', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows() 