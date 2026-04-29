from __future__ import annotations

import logging

from app.ai.base import BaseWorker
from app.config import Settings
from app.core.events import ParkingEvent
from app.core.frame import Frame

logger = logging.getLogger(__name__)


class YOLOWorker(BaseWorker):
    """CPU-first YOLO worker with mock detections.

    TODO: Replace `detect` with ONNX Runtime or OpenVINO inference for YOLOv8/YOLO11
    nano models quantized to INT8.
    """

    def __init__(self, frame_queue: asyncio.Queue[Frame], event_queue: asyncio.Queue[dict[str, object]], settings: Settings) -> None:
        super().__init__("YOLOWorker", frame_queue, event_queue)
        self.settings = settings

    async def process_frame(self, frame: Frame) -> list[dict[str, object]]:
        detections = self.detect(frame)
        return [
            ParkingEvent.now(
                frame_id=frame.frame_id,
                type=detection["type"],
                bbox=detection["bbox"],
                confidence=detection["confidence"],
                image_crop=frame.crop_token(detection["bbox"]),
            ).to_payload()
            for detection in detections
        ]

    def detect(self, frame: Frame) -> list[dict[str, object]]:
        """Return mock car, slot, and fire detections."""

        logger.debug("Running mock YOLO detection on frame_id=%s", frame.frame_id)
        return [
            {"type": "car", "bbox": [40, 80, 220, 260], "confidence": 0.91},
            {"type": "slot", "bbox": [20, 60, 260, 300], "confidence": 0.88},
            {"type": "fire", "bbox": [430, 45, 510, 130], "confidence": 0.42},
        ]
