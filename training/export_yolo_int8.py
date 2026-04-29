from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse export arguments kept compatible with common Ultralytics usage."""

    parser = argparse.ArgumentParser(description="Export a YOLO model to INT8 ONNX/OpenVINO artifacts.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained YOLO weights, e.g. runs/detect/train/weights/best.pt")
    parser.add_argument("--format", choices=["onnx", "openvino"], default="onnx", help="Export backend for CPU inference")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input image size")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization during export when supported")
    parser.add_argument("--output-dir", type=Path, default=Path("models/yolo"), help="Directory for exported artifacts")
    return parser.parse_args()


def main() -> None:
    """Log the exact export command to run once Ultralytics is installed."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Prepared YOLO export")
    logger.info("weights=%s format=%s imgsz=%s int8=%s output_dir=%s", args.weights, args.format, args.imgsz, args.int8, args.output_dir)
    logger.info("TODO: Install ultralytics and call YOLO(args.weights).export(format=args.format, imgsz=args.imgsz, int8=args.int8)")


if __name__ == "__main__":
    main()
