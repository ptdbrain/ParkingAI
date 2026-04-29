from __future__ import annotations

import asyncio
import contextlib
import logging

from app.ai.base import BaseWorker
from app.ai.workers.ocr_worker import OCRWorker
from app.ai.workers.reid_worker import ReIDWorker
from app.ai.workers.vlm_worker import VLMWorker
from app.ai.workers.yolo_worker import YOLOWorker
from app.config import Settings
from app.core.frame import Frame

logger = logging.getLogger(__name__)


class PipelineRuntime:
    """Owns queues and worker tasks for the AI inference layer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.frame_queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=settings.queue_max_size)
        self.event_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=settings.queue_max_size * 4)
        self._worker_queues: list[asyncio.Queue[Frame]] = [
            asyncio.Queue(maxsize=settings.queue_max_size) for _ in range(4)
        ]
        self._workers: list[BaseWorker] = [
            YOLOWorker(self._worker_queues[0], self.event_queue, settings),
            OCRWorker(self._worker_queues[1], self.event_queue, settings),
            ReIDWorker(self._worker_queues[2], self.event_queue, settings),
            VLMWorker(self._worker_queues[3], self.event_queue, settings),
        ]
        self._tasks: list[asyncio.Task[None]] = []
        self._router_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start router and independent AI workers."""

        if self._running:
            return
        self._running = True
        self._router_task = asyncio.create_task(self._route_frames(), name="frame-router")
        self._tasks = [asyncio.create_task(worker.run(), name=worker.name) for worker in self._workers]
        logger.info("Pipeline runtime started")

    async def submit_frame(self, frame: Frame) -> None:
        """Submit one frame to the shared ingress queue."""

        await self.frame_queue.put(frame)

    async def read_event(self, timeout: float | None = None) -> dict[str, object] | None:
        """Read one structured event from the pipeline."""

        try:
            if timeout is None:
                return await self.event_queue.get()
            return await asyncio.wait_for(self.event_queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    async def stop(self) -> None:
        """Cancel all background tasks."""

        self._running = False
        for worker in self._workers:
            worker.stop()
        tasks: list[asyncio.Task[None]] = [*self._tasks]
        if self._router_task is not None:
            tasks.append(self._router_task)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        logger.info("Pipeline runtime stopped")

    async def _route_frames(self) -> None:
        """Broadcast each ingress frame to every worker queue."""

        while self._running:
            frame = await self.frame_queue.get()
            try:
                for queue in self._worker_queues:
                    await queue.put(frame)
            finally:
                self.frame_queue.task_done()
