from __future__ import annotations

import asyncio

from app.ai.pipeline import PipelineRuntime
from app.config import Settings
from app.core.camera import FakeFrameGenerator


def test_mock_pipeline_produces_structured_events() -> None:
    async def run_pipeline() -> list[dict[str, object]]:
        settings = Settings(queue_max_size=8, frame_width=320, frame_height=180)
        runtime = PipelineRuntime(settings=settings)
        await runtime.start()

        generator = FakeFrameGenerator(width=320, height=180)
        await runtime.submit_frame(generator.next_frame())

        events: list[dict[str, object]] = []
        deadline = asyncio.get_running_loop().time() + 2.0
        required_types = {"car", "slot", "fire", "anomaly"}
        while {event["type"] for event in events} < required_types and asyncio.get_running_loop().time() < deadline:
            event = await runtime.read_event(timeout=0.5)
            if event is not None:
                events.append(event)

        await runtime.stop()
        return events

    events = asyncio.run(run_pipeline())

    assert events
    assert {event["type"] for event in events}.issuperset({"car", "slot", "fire", "anomaly"})
    for event in events:
        assert {"frame_id", "timestamp", "type", "bbox", "confidence"}.issubset(event)


def test_license_plate_detection_feeds_ocr_crop() -> None:
    async def run_pipeline() -> list[dict[str, object]]:
        settings = Settings(queue_max_size=8, frame_width=320, frame_height=180)
        runtime = PipelineRuntime(settings=settings)
        await runtime.start()

        generator = FakeFrameGenerator(width=320, height=180)
        await runtime.submit_frame(generator.next_frame())

        events: list[dict[str, object]] = []
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            event = await runtime.read_event(timeout=0.5)
            if event is not None:
                events.append(event)
            if any(event.get("plate_text") for event in events):
                break

        await runtime.stop()
        return events

    events = asyncio.run(run_pipeline())
    plate_events = [event for event in events if event.get("plate_text")]

    assert plate_events
    assert plate_events[0]["type"] == "car"
    assert plate_events[0]["bbox"] == [70, 190, 165, 225]
    assert plate_events[0]["image_crop"] == "frame-1:crop-70-190-165-225"
