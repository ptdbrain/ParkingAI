from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from app.core.frame import Frame

logger = logging.getLogger(__name__)


class FakeFrameGenerator:
    """Generate deterministic frames for local testing and demos."""

    def __init__(self, width: int = 640, height: int = 360, source: str = "fake-camera") -> None:
        self.width = width
        self.height = height
        self.source = source
        self._frame_id = 0

    def next_frame(self) -> Frame:
        self._frame_id += 1
        return Frame.blank(self._frame_id, self.width, self.height, self.source)

    async def stream(self, fps: float = 5.0) -> AsyncIterator[Frame]:
        """Yield frames at a bounded rate to protect CPU-only devices."""

        delay = 1.0 / max(fps, 0.1)
        while True:
            frame = self.next_frame()
            logger.debug("Generated fake frame %s", frame.frame_id)
            yield frame
            await asyncio.sleep(delay)
