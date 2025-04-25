import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv10 + ByteTrack Real-Time Tracking")
    parser.add_argument(
        "--model", type=str, default="yolov10s.pt",
        help="Path to the YOLOv10 model checkpoint"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source (RTSP URL, video file path, or camera index)"
    )
    parser.add_argument(
        "--tracker", type=str, default="bytetrack.yaml",
        help="Tracker configuration file (e.g. bytetrack.yaml)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IOU threshold for NMS"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the YOLOv10 model
    model = YOLO(args.model)
    
    # Run tracking with ByteTrack
    # stream=True returns a generator of Results objects
    # show=True will display annotated frames in a window
    print(f"Starting tracking: model={args.model}, source={args.source}, tracker={args.tracker}")
    results = model.track(
        source=args.source,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        stream=True,
        show=True
    )

    # Iterate through tracking results (streamed)
    for res in results:
        # res is a Results object; you can access res.boxes.xyxy, res.boxes.id, etc.
        # By default show=True has already drawn and displayed the frame.
        # If you need further processing, do it here.
        pass

if __name__ == "__main__":
    main() 