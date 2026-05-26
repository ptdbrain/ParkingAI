from __future__ import annotations

import asyncio

from app.ai.vlm import VLMResult
from app.ai.workers.vlm_worker import VLMWorker
from app.config import Settings
from app.core.frame import Frame


class StaticVLMAdapter:
    backend_name = "unit-vlm"

    def __init__(self, result: VLMResult) -> None:
        self.result = result
        self.calls = 0
        self.last_prompt: str | None = None

    def infer(self, frame: Frame, prompt: str) -> VLMResult:
        self.calls += 1
        self.last_prompt = prompt
        return self.result


def run_vlm_once(
    adapter: StaticVLMAdapter,
    settings: Settings,
    *,
    frame_id: int = 1,
) -> list[dict[str, object]]:
    async def run() -> list[dict[str, object]]:
        worker = VLMWorker(asyncio.Queue(), asyncio.Queue(), settings, adapter=adapter)
        return await worker.process_frame(Frame.blank(frame_id=frame_id, width=320, height=180))

    return asyncio.run(run())


def test_vlm_worker_emits_semantic_anomaly_event_when_confident() -> None:
    adapter = StaticVLMAdapter(
        VLMResult(
            bbox=[10, 20, 30, 40],
            confidence=0.82,
            description="Vehicle is outside the marked parking box.",
        )
    )
    settings = Settings(
        vlm_confidence_threshold=0.5,
        vlm_frame_sample_interval=1,
        vlm_prompt="Check whether cars violate parking lines.",
    )

    events = run_vlm_once(adapter, settings)

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "anomaly"
    assert event["bbox"] == [10, 20, 30, 40]
    assert event["confidence"] == 0.82
    assert event["description"] == "Vehicle is outside the marked parking box."
    assert event["image_crop"] == "frame-1:crop-10-20-30-40"
    assert event["model_backend"] == "unit-vlm"
    assert event["vlm_prompt"] == "Check whether cars violate parking lines."
    assert adapter.last_prompt == "Check whether cars violate parking lines."


def test_vlm_worker_suppresses_low_confidence_result() -> None:
    adapter = StaticVLMAdapter(
        VLMResult(
            bbox=[10, 20, 30, 40],
            confidence=0.49,
            description="Weak anomaly candidate.",
        )
    )
    settings = Settings(vlm_confidence_threshold=0.5, vlm_frame_sample_interval=1)

    assert run_vlm_once(adapter, settings) == []
    assert adapter.calls == 1


def test_vlm_worker_skips_when_disabled() -> None:
    adapter = StaticVLMAdapter(
        VLMResult(
            bbox=[10, 20, 30, 40],
            confidence=0.99,
            description="Should not be evaluated.",
        )
    )
    settings = Settings(vlm_enabled=False, vlm_frame_sample_interval=1)

    assert run_vlm_once(adapter, settings) == []
    assert adapter.calls == 0


def test_vlm_worker_respects_frame_sample_interval() -> None:
    adapter = StaticVLMAdapter(
        VLMResult(
            bbox=[10, 20, 30, 40],
            confidence=0.99,
            description="Sampled anomaly.",
        )
    )
    settings = Settings(vlm_frame_sample_interval=5)

    assert run_vlm_once(adapter, settings, frame_id=4) == []
    assert adapter.calls == 0
    assert run_vlm_once(adapter, settings, frame_id=5)
    assert adapter.calls == 1
