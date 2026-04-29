from __future__ import annotations

import logging

from app.ai.base import BaseWorker
from app.config import Settings
from app.core.events import ParkingEvent
from app.core.frame import Frame

logger = logging.getLogger(__name__)


class VLMWorker(BaseWorker):
    """Lightweight VLM semantic anomaly worker.

    TODO: Replace mock response with a quantized Moondream2/Florence-2 adapter.
    Run this worker only on sampled frames or detector-triggered crops on CPU.
    """

    def __init__(self, frame_queue: asyncio.Queue[Frame], event_queue: asyncio.Queue[dict[str, object]], settings: Settings) -> None:
        super().__init__("VLMWorker", frame_queue, event_queue)
        self.settings = settings
        self.prompt = "Detect parking anomalies, loitering, or parking outside lines."

    async def process_frame(self, frame: Frame) -> list[dict[str, object]]:
        result = self.infer(frame, self.prompt)
        event = ParkingEvent.now(
            frame_id=frame.frame_id,
            type="anomaly",
            bbox=result["bbox"],
            confidence=result["confidence"],
            image_crop=frame.crop_token(result["bbox"]),
        )
        payload = event.to_payload()
        payload["description"] = result["description"]
        return [payload]

    def infer(self, frame: Frame, prompt: str) -> dict[str, object]:
        """Return a mock semantic reasoning output."""

        logger.debug("Running mock VLM on frame_id=%s prompt=%s", frame.frame_id, prompt)
        return {
            "bbox": [260, 90, 390, 250],
            "confidence": 0.63,
            "description": "Vehicle appears slightly outside the parking line.",
        }
