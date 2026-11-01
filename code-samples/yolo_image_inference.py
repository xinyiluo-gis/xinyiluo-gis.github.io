"""Run Ultralytics YOLO inference on an image, video, or directory."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Image, video, camera ID, or directory to process")
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.image_size,
        conf=args.confidence,
        stream=True,
        save=args.save,
    )

    for frame_index, result in enumerate(results):
        print(f"frame={frame_index} detections={len(result.boxes)}")


