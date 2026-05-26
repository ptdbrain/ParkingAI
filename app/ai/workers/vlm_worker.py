from __future__ import annotations

import asyncio
import logging

from app.ai.base import BaseWorker
from app.ai.vlm import VLMAdapter, create_vlm_adapter
from app.config import Settings
from app.core.events import ParkingEvent
from app.core.frame import Frame

logger = logging.getLogger(__name__)


class VLMWorker(BaseWorker):
    """Lightweight VLM semantic anomaly worker.

    The worker owns throttling and event formatting. Model-specific code lives
    behind a VLM adapter so the mock runtime and future quantized models share
    one contract.
    """

    def __init__(
        self,
        frame_queue: asyncio.Queue[Frame],
        event_queue: asyncio.Queue[dict[str, object]],
        settings: Settings,
        adapter: VLMAdapter | None = None,
    ) -> None:
        super().__init__("VLMWorker", frame_queue, event_queue)
        self.settings = settings
        self.adapter = adapter or create_vlm_adapter(settings)
        self.prompt = settings.vlm_prompt
        self.sample_interval = max(1, settings.vlm_frame_sample_interval)

    async def process_frame(self, frame: Frame) -> list[dict[str, object]]:
        if not self.settings.vlm_enabled:
            return []
        if frame.frame_id % self.sample_interval != 0:
            return []

        result = self.adapter.infer(frame, self.prompt)
        if result is None or result.confidence < self.settings.vlm_confidence_threshold:
            return []

        event = ParkingEvent.now(
            frame_id=frame.frame_id,
            type="anomaly",
            bbox=result.bbox,
            confidence=result.confidence,
            image_crop=frame.crop_token(result.bbox),
        )
        payload = event.to_payload()
        payload["description"] = result.description
        payload["model_backend"] = self.adapter.backend_name
        payload["vlm_label"] = result.label
        payload["vlm_prompt"] = self.prompt
        return [payload]
