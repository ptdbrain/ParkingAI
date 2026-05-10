from __future__ import annotations

import logging

from app.ai.base import BaseWorker
from app.ai.plate_crop import PlateCropTask
from app.config import Settings
from app.core.events import ParkingEvent
from app.core.frame import Frame

logger = logging.getLogger(__name__)


class YOLOWorker(BaseWorker):
    """CPU-first YOLO worker with mock detections.

    TODO: Replace `detect` with ONNX Runtime or OpenVINO inference for YOLOv8/YOLO11
    nano models quantized to INT8.
    """

    def __init__(
        self,
        frame_queue: asyncio.Queue[Frame],
        event_queue: asyncio.Queue[dict[str, object]],
        settings: Settings,
        plate_crop_queue: asyncio.Queue[PlateCropTask] | None = None,
    ) -> None:
        super().__init__("YOLOWorker", frame_queue, event_queue)
        self.settings = settings
        self.plate_crop_queue = plate_crop_queue

    async def process_frame(self, frame: Frame) -> list[dict[str, object]]:
        detections = self.detect(frame)
        events: list[dict[str, object]] = []
        for detection in detections:
            crop = frame.crop_token(detection["bbox"])
            if detection["type"] == "license_plate":
                if self.plate_crop_queue is not None:
                    # Cắt ảnh biển số thật để gửi sang OCRWorker
                    image_data = frame.get_crop(detection["bbox"])
                    await self.plate_crop_queue.put(
                        PlateCropTask(
                            frame_id=frame.frame_id,
                            bbox=detection["bbox"],
                            confidence=detection["confidence"],
                            image_crop=crop,
                            image_data=image_data,
                        )
                    )
                continue

            events.append(
                ParkingEvent.now(
                    frame_id=frame.frame_id,
                    type=detection["type"],
                    bbox=detection["bbox"],
                    confidence=detection["confidence"],
                    image_crop=crop,
                ).to_payload()
            )
        return events

    def detect(self, frame: Frame) -> list[dict[str, object]]:
        """Return mock car, slot, fire, and license plate detections."""

        logger.debug("Running mock YOLO detection on frame_id=%s", frame.frame_id)
        return [
            {"type": "car", "bbox": [40, 80, 220, 260], "confidence": 0.91},
            {"type": "license_plate", "bbox": [70, 190, 165, 225], "confidence": 0.86},
            {"type": "slot", "bbox": [20, 60, 260, 300], "confidence": 0.88},
            {"type": "fire", "bbox": [430, 45, 510, 130], "confidence": 0.42},
        ]
