from __future__ import annotations

import logging

from app.ai.base import BaseWorker
from app.config import Settings
from app.core.events import ParkingEvent
from app.core.frame import Frame

logger = logging.getLogger(__name__)


class OCRWorker(BaseWorker):
    """License plate OCR worker.

    TODO: Replace mock OCR with PaddleOCR/EasyOCR or a compact ONNX recognizer.
    """

    def __init__(self, frame_queue: asyncio.Queue[Frame], event_queue: asyncio.Queue[dict[str, object]], settings: Settings) -> None:
        super().__init__("OCRWorker", frame_queue, event_queue)
        self.settings = settings

    async def process_frame(self, frame: Frame) -> list[dict[str, object]]:
        bbox = [70, 190, 165, 225]
        crop = frame.crop_token(bbox)
        plate_text = self.recognize_plate(crop)
        event = ParkingEvent.now(
            frame_id=frame.frame_id,
            type="car",
            bbox=bbox,
            confidence=0.84,
            image_crop=crop,
            plate_text=plate_text,
        )
        return [event.to_payload()]

    def recognize_plate(self, image_crop: str) -> str:
        """Return deterministic mock plate text from an image crop token."""

        logger.debug("Running mock OCR on crop=%s", image_crop)
        suffix = abs(hash(image_crop)) % 100000
        return f"51A{suffix:05d}"
