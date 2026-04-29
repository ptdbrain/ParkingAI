from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.pipeline import PipelineRuntime
from app.config import get_settings
from app.core.camera import FakeFrameGenerator
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


async def main(frame_count: int = 3) -> None:
    """Run the mock AI pipeline without FastAPI or a database."""

    settings = get_settings()
    configure_logging(settings.log_level)
    runtime = PipelineRuntime(settings)
    generator = FakeFrameGenerator(width=settings.frame_width, height=settings.frame_height)
    await runtime.start()

    try:
        for _ in range(frame_count):
            await runtime.submit_frame(generator.next_frame())

        expected_events = frame_count * 6
        received = 0
        while received < expected_events:
            event = await runtime.read_event(timeout=2.0)
            if event is None:
                logger.warning("Timed out waiting for pipeline event")
                break
            print(_compact_event(event))
            received += 1
    finally:
        await runtime.stop()


def _compact_event(event: dict[str, object]) -> dict[str, object]:
    """Keep terminal demo output readable while preserving pipeline payloads."""

    compact = dict(event)
    embedding = compact.get("embedding")
    if isinstance(embedding, list):
        compact["embedding_dim"] = len(embedding)
        compact["embedding"] = embedding[:4]
    return compact


if __name__ == "__main__":
    asyncio.run(main())
