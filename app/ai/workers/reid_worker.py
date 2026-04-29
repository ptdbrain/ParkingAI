from __future__ import annotations

import hashlib
import logging
import math

from app.ai.base import BaseWorker
from app.config import Settings
from app.core.events import ParkingEvent
from app.core.frame import Frame

logger = logging.getLogger(__name__)


class ResNet18EmbeddingSkeleton:
    """Small ResNet18-shaped embedding skeleton for future real checkpoints.

    The runnable mock path intentionally avoids importing PyTorch at module load
    time so old CPU-only laptops can start quickly.
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        self.embedding_dim = embedding_dim

    def load_checkpoint(self, path: str) -> None:
        """Load a future ResNet18 checkpoint.

        TODO: Import torch lazily here and attach torchvision.models.resnet18
        with a compact embedding projection head.
        """

        logger.info("ResNet18 checkpoint loading is not implemented yet: %s", path)


class ReIDWorker(BaseWorker):
    """Vehicle re-identification worker that emits embedding vectors."""

    def __init__(self, frame_queue: asyncio.Queue[Frame], event_queue: asyncio.Queue[dict[str, object]], settings: Settings) -> None:
        super().__init__("ReIDWorker", frame_queue, event_queue)
        self.settings = settings
        self.model = ResNet18EmbeddingSkeleton(settings.embedding_dim)

    async def process_frame(self, frame: Frame) -> list[dict[str, object]]:
        bbox = [40, 80, 220, 260]
        crop = frame.crop_token(bbox)
        embedding = self.extract_embedding(crop)
        event = ParkingEvent.now(
            frame_id=frame.frame_id,
            type="car",
            bbox=bbox,
            confidence=0.79,
            image_crop=crop,
            embedding=embedding,
        )
        return [event.to_payload()]

    def extract_embedding(self, image_crop: str) -> list[float]:
        """Return a normalized deterministic embedding for mock Re-ID."""

        logger.debug("Running mock ReID on crop=%s", image_crop)
        digest = hashlib.sha256(image_crop.encode("utf-8")).digest()
        values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(self.settings.embedding_dim)]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [round(value / norm, 6) for value in values]
