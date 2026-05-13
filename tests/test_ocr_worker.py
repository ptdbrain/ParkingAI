from __future__ import annotations

import asyncio

from app.ai.plate_crop import PlateCropTask
from app.ai.workers.ocr_worker import OCRWorker
from app.config import Settings


def test_mock_ocr_worker_does_not_require_easyocr_runtime() -> None:
    async def run_worker() -> list[dict[str, object]]:
        event_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        crop_queue: asyncio.Queue[PlateCropTask] = asyncio.Queue()
        worker = OCRWorker(
            plate_crop_queue=crop_queue,
            event_queue=event_queue,
            settings=Settings(inference_mode="mock"),
        )

        return await worker.process_frame(
            PlateCropTask(
                frame_id=5,
                bbox=[10, 20, 60, 40],
                confidence=0.86,
                image_crop="frame-5:crop-10-20-60-40",
            )
        )

    events = asyncio.run(run_worker())

    assert events[0]["type"] == "car"
    assert events[0]["plate_text"] == "51A12345"
    assert events[0]["image_crop"] == "frame-5:crop-10-20-60-40"
