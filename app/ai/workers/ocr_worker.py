from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from app.ai.base import BaseWorker
from app.ai.plate_crop import PlateCropTask
from app.config import Settings
from app.core.events import ParkingEvent

logger = logging.getLogger(__name__)


class ANPREngine(Protocol):
    """Interface implemented by real OCR adapters."""

    def read_plate(self, image: object) -> dict[str, object]:
        """Return OCR text and confidence for a plate crop."""


class OCRWorker(BaseWorker):
    """License plate OCR worker with mock and real inference modes."""

    def __init__(
        self,
        plate_crop_queue: asyncio.Queue[PlateCropTask],
        event_queue: asyncio.Queue[dict[str, object]],
        settings: Settings,
    ) -> None:
        super().__init__("OCRWorker", plate_crop_queue, event_queue)
        self.settings = settings
        self.anpr: ANPREngine | None = self._load_anpr() if settings.inference_mode == "real" else None

    async def process_frame(self, crop_task: PlateCropTask) -> list[dict[str, object]]:
        if self.anpr is not None and crop_task.image_data is not None:
            result = self.anpr.read_plate(crop_task.image_data)
            plate_text = str(result.get("plate") or "UNKNOWN")
            ocr_confidence = float(result.get("confidence") or 0.0)
        elif self.settings.inference_mode == "mock":
            plate_text = "51A12345"
            ocr_confidence = 0.90
        else:
            plate_text = "MISSING_DATA"
            ocr_confidence = 0.0

        event = ParkingEvent.now(
            frame_id=crop_task.frame_id,
            type="car",
            bbox=crop_task.bbox,
            confidence=min(crop_task.confidence, ocr_confidence),
            image_crop=crop_task.image_crop,
            plate_text=plate_text,
        )
        return [event.to_payload()]

    def _load_anpr(self) -> ANPREngine:
        """Load EasyOCR only when real inference is explicitly requested."""

        try:
            import torch

            from models.ocr.easyocr import ANPR
        except ImportError as exc:
            raise RuntimeError(
                "Real OCR requires optional inference dependencies. "
                "Install requirements.txt or set PARKING_INFERENCE_MODE=mock."
            ) from exc

        return ANPR(gpu=torch.cuda.is_available())
