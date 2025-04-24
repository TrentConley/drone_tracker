# face_recognizer.py
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2 # Using OpenCV for image handling
import logging
import os
import sys # Import sys to check platform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceRecognizer:
    """
    Handles face detection and embedding extraction using InsightFace.
    Configurable for CPU or GPU execution (CUDA or MPS).
    """
    def __init__(self, det_thresh=0.5, model_pack_name='buffalo_l', use_gpu=False):
        """
        Initializes the FaceAnalysis app from InsightFace.

        Args:
            det_thresh (float): Detection confidence threshold.
            model_pack_name (str): Name of the model pack (e.g., 'buffalo_l', 'antelopev2').
            use_gpu (bool): Whether to attempt using GPU (CUDA on Linux/Windows, MPS on macOS).
        """
        self.app = None
        providers = ['CPUExecutionProvider']
        ctx_id = 0 # CPU context ID

        if use_gpu:
            try:
                import onnxruntime
                available_providers = onnxruntime.get_available_providers()
                logging.info(f"Available ONNXRuntime Providers: {available_providers}")

                # Check for MPS (CoreML) on macOS
                if sys.platform == "darwin":
                    if 'CoreMLExecutionProvider' in available_providers:
                        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                        # ctx_id remains 0 for CoreML according to some documentation,
                        # but InsightFace's prepare might handle it.
                        # Verify if specific ctx_id is needed for CoreML if issues arise.
                        logging.info("Configured to use CoreMLExecutionProvider (MPS) on macOS.")
                    else:
                        logging.warning("use_gpu=True on macOS but CoreMLExecutionProvider not found. Falling back to CPU.")
                        logging.warning("Ensure you have onnxruntime >= 1.10 (check compatibility).")

                # Check for CUDA on other platforms (Linux/Windows)
                elif 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    ctx_id = 0 # Use GPU device 0 by default for CUDA
                    logging.info("Configured to use CUDAExecutionProvider.")
                else:
                    logging.warning(f"use_gpu=True on {sys.platform} but CUDAExecutionProvider not found. Falling back to CPU.")
                    logging.warning("Ensure 'onnxruntime-gpu' is installed and CUDA is configured correctly.")

            except ImportError:
                 logging.warning("onnxruntime not found. Cannot check for GPU providers. Using CPU.")
            except Exception as e:
                 logging.warning(f"Error checking ONNXRuntime providers: {e}. Using CPU.")
        else:
            logging.info("Configured to use CPUExecutionProvider.")

        logging.info(f"Initializing InsightFace FaceAnalysis with model: {model_pack_name}, selected providers: {providers}")
        try:
            self.app = FaceAnalysis(name=model_pack_name,
                                    allowed_modules=['detection', 'recognition'],
                                    providers=providers)
            # Set det_thresh during prepare, not detection call
            self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh)
            logging.info("InsightFace models loaded successfully.")
        except FileNotFoundError:
            logging.error("Error initializing InsightFace: Model files not found.")
            logging.error(f"Please ensure models for '{model_pack_name}' are downloaded. They are typically stored in ~/.insightface/models/{model_pack_name}/")
            logging.error("InsightFace might attempt to download them automatically if internet is available.")
            self.app = None # Ensure app is None if models failed to load
        except Exception as e:
            logging.error(f"Error initializing InsightFace: {e}")
            logging.error("Ensure 'insightface' and 'onnxruntime' (or 'onnxruntime-gpu') are installed.")
            if 'CUDAExecutionProvider' in providers:
                logging.error("If using GPU (CUDA), ensure CUDA/cuDNN are installed and compatible with ONNXRuntime.")
            if 'CoreMLExecutionProvider' in providers:
                logging.error("If using GPU (MPS), ensure you are on macOS with a compatible ONNXRuntime version.")
            self.app = None

    def analyze_frame(self, frame: np.ndarray):
        """
        Detects faces in a frame and extracts their embeddings.

        Args:
            frame: The input image/frame (NumPy array in BGR format).

        Returns:
            A list of dictionaries, each containing:
            'bbox': Bounding box [x1, y1, x2, y2].
            'kps': Keypoints.
            'det_score': Detection score.
            'embedding': Normalized face embedding (NumPy array).
            Returns an empty list if no faces are detected or model failed.
        """
        if self.app is None:
            logging.error("FaceRecognizer not initialized successfully.")
            return []
        if frame is None or frame.size == 0:
            logging.warning("Received empty frame for analysis.")
            return []

        try:
            # Ensure frame is in BGR format (InsightFace standard)
            faces = self.app.get(frame)
            results = []
            for face in faces:
                if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                    results.append({
                        'bbox': face.bbox.astype(int), # [x1, y1, x2, y2]
                        'kps': face.kps,
                        'det_score': face.det_score,
                        'embedding': face.normed_embedding
                    })
                else:
                     logging.warning(f"Face detected (score: {face.det_score:.2f}) but no embedding extracted. BBox: {face.bbox}")
            return results
        except Exception as e:
            logging.error(f"Error during face analysis: {e}", exc_info=True) # Log traceback
            return []

    def get_single_embedding(self, frame: np.ndarray):
        """
        Analyzes a frame, returns the embedding of the most confident face.

        Args:
            frame: The input image/frame (NumPy array BGR).

        Returns:
            The embedding (NumPy array) of the most confident face, or None.
        """
        face_results = self.analyze_frame(frame)
        if not face_results:
            return None
        best_face = max(face_results, key=lambda x: x['det_score'])
        return best_face.get('embedding')

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    print("--- Face Recognizer Test ---")
    # Example: Initialize recognizer (try GPU if available, set use_gpu=True)
    recognizer = FaceRecognizer(use_gpu=False) # Set to True to test GPU

    if recognizer.app:
        print("Recognizer initialized.")
        # Create a dummy black image
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Test Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Test Frame", dummy_frame)
        cv2.waitKey(100) # Display for a short time

        print("Analyzing dummy frame...")
        results = recognizer.analyze_frame(dummy_frame)

        if results:
            print(f"Found {len(results)} faces (expected 0 in dummy frame).")
            for i, face in enumerate(results):
                print(f"  Face {i+1}: BBox={face['bbox']}, Score={face['det_score']:.2f}, EmbShape={face['embedding'].shape}")
        else:
            print("No faces detected in the dummy frame (as expected).")

        # Test with a sample image if available (replace 'path/to/your/image.jpg')
        test_image_path = 'test_face.jpg' # Put a test image here
        if os.path.exists(test_image_path):
            print(f"\nAnalyzing test image: {test_image_path}")
            test_frame = cv2.imread(test_image_path)
            if test_frame is not None:
                results = recognizer.analyze_frame(test_frame)
                if results:
                    print(f"Found {len(results)} faces in {test_image_path}.")
                    for i, face in enumerate(results):
                         print(f"  Face {i+1}: BBox={face['bbox']}, Score={face['det_score']:.2f}, EmbShape={face['embedding'].shape}")
                         # Draw bbox on image for visualization
                         bbox = face['bbox']
                         cv2.rectangle(test_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.imshow("Test Image with Detections", test_frame)
                    print("Displaying test image with detections. Press any key to close.")
                    cv2.waitKey(0)
                else:
                    print(f"No faces detected in {test_image_path}.")
                cv2.destroyWindow("Test Image with Detections") # Close specific window
            else:
                print(f"Could not read test image: {test_image_path}")
        else:
            print(f"\nTest image not found at '{test_image_path}'. Skipping image analysis test.")

        cv2.destroyAllWindows()
    else:
        print("Face Recognizer failed to initialize. Check logs.")

    print("--- Test Finished ---") 