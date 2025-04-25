# drone_face_recognition_transform.py
from transforms.api import transform, Input, Output, configure
import pandas as pd
from face_recognizer import FaceRecognizer
import base64
import cv2
import numpy as np

# Configure to run without dedicated executors (adjust profile as needed)
@configure(profile=["KUBERNETES_NO_EXECUTORS"])
@transform(
    output=Output("ri.foundry_dataset.YOUR_OUTPUT_DATASET_RID"),
    frame_stream=Input("ri.foundry_dataset.YOUR_INPUT_DATASET_RID")
)
def compute_face_recognition(output, frame_stream, ctx):
    """
    1. Pull frames from a Foundry dataset (frame_stream)
    2. Run InsightFace-based recognition per frame
    3. Write a DataFrame of results back to Foundry
    """
    # Initialize the FaceRecognizer (GPU or CPU)
    use_gpu = True  # or False
    model_pack_name = "buffalo_l"
    face_recognizer = FaceRecognizer(use_gpu=use_gpu, model_pack_name=model_pack_name)

    # Read input frames into a pandas DataFrame
    df = frame_stream.dataframe()

    def recognize_frame(row):
        # Assume input column 'image_base64' contains JPEG/PNG frame as base64 string
        img_data = base64.b64decode(row.image_base64)
        arr = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Run facial analysis
        faces = face_recognizer.analyze_frame(frame)
        # Simplify output: list of dicts with bbox and name
        simplified = []
        for f in faces:
            emb = f.get('embedding')
            name, _ = f.get('name', (None, None)) if 'name' in f else (None, None)
            bbox = f['bbox']
            simplified.append({
                'bbox': tuple(map(int, bbox)),
                'name': name,
                'distance': f.get('distance')
            })
        return simplified

    # Apply recognition to each row
    df['recognitions'] = df.apply(recognize_frame, axis=1)

    # Write results back to Foundry
    output.write_dataframe(df) 