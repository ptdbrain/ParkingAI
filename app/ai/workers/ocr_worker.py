import asyncio
import logging
import torch
import numpy as np
from app.ai.base import BaseWorker
from app.ai.plate_crop import PlateCropTask
from app.config import Settings
from app.core.events import ParkingEvent
from models.ocr.easyocr import ANPR

logger = logging.getLogger(__name__)


class OCRWorker(BaseWorker):
    """License plate OCR worker using EasyOCR."""

    def __init__(
        self,
        plate_crop_queue: asyncio.Queue[PlateCropTask],
        event_queue: asyncio.Queue[dict[str, object]],
        settings: Settings,
    ) -> None:
        super().__init__("OCRWorker", plate_crop_queue, event_queue)
        self.settings = settings
        # Khởi tạo mô hình EasyOCR (ANPR)
        self.anpr = ANPR(gpu=torch.cuda.is_available())

    async def process_frame(self, crop_task: PlateCropTask) -> list[dict[str, object]]:
        if crop_task.image_data is not None:
            # Chạy OCR trên ảnh thực tế
            result = self.anpr.read_plate(crop_task.image_data)
            plate_text = result["plate"] or "UNKNOWN"
            confidence = result["confidence"]
        else:
            # Fallback nếu thiếu dữ liệu ảnh
            plate_text = "MISSING_DATA"
            confidence = 0.0

        event = ParkingEvent.now(
            frame_id=crop_task.frame_id,
            type="car",
            bbox=crop_task.bbox,
            confidence=crop_task.confidence,
            image_crop=crop_task.image_crop,
            plate_text=plate_text,
        )
        return [event.to_payload()]

