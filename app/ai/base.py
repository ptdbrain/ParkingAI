from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod

from app.core.frame import Frame

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """Base class for independent async inference workers."""

    def __init__(self, name: str, frame_queue: asyncio.Queue[Frame], event_queue: asyncio.Queue[dict[str, object]]) -> None:
        self.name = name
        self.frame_queue = frame_queue
        self.event_queue = event_queue
        self._running = True

    async def run(self) -> None:
        """Consume frames forever and publish structured events."""

        logger.info("%s started", self.name)
        while self._running:
            frame = await self.frame_queue.get()
            try:
                events = await self.process_frame(frame)
                for event in events:
                    await self.event_queue.put(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s failed to process frame_id=%s", self.name, frame.frame_id)
            finally:
                self.frame_queue.task_done()
        logger.info("%s stopped", self.name)

    def stop(self) -> None:
        """Signal a graceful worker stop."""

        self._running = False

    @abstractmethod
    async def process_frame(self, frame: Frame) -> list[dict[str, object]]:
        """Return zero or more event payloads for a frame."""

    async def close_queue(self) -> None:
        """Drain any pending work during shutdown."""

        with contextlib.suppress(asyncio.QueueEmpty):
            while True:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
