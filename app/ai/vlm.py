from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.core.frame import Frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VLMResult:
    """Structured semantic result produced by a VLM adapter."""

    bbox: list[int]
    confidence: float
    description: str
    label: str = "anomaly"


class VLMAdapter(Protocol):
    """Adapter contract for local or quantized vision-language models."""

    backend_name: str

    def infer(self, frame: Frame, prompt: str) -> VLMResult | None:
        """Run semantic reasoning for one frame."""


class MockVLMAdapter:
    """Deterministic adapter for tests and CPU-only demos."""

    backend_name = "mock-vlm"

    def __init__(self, confidence: float = 0.63) -> None:
        self.confidence = confidence

    def infer(self, frame: Frame, prompt: str) -> VLMResult:
        logger.debug("Running mock VLM adapter on frame_id=%s prompt=%s", frame.frame_id, prompt)
        return VLMResult(
            bbox=_scaled_bbox(frame),
            confidence=self.confidence,
            description="Vehicle appears slightly outside the parking line.",
        )


def create_vlm_adapter(settings: Settings) -> VLMAdapter:
    """Create the configured VLM adapter.

    The runtime interface is intentionally ready for real Moondream/Florence
    adapters, while the default stays lightweight and runnable without model
    downloads.
    """

    backend = settings.vlm_backend.strip().lower()
    if backend == "mock":
        return MockVLMAdapter()
    msg = f"Unsupported VLM backend: {settings.vlm_backend!r}. Supported backend: 'mock'."
    raise ValueError(msg)


def _scaled_bbox(frame: Frame) -> list[int]:
    """Create a stable demo anomaly box scaled to the frame dimensions."""

    x1 = int(frame.width * 0.40)
    y1 = int(frame.height * 0.25)
    x2 = int(frame.width * 0.61)
    y2 = int(frame.height * 0.70)

    x1 = min(max(x1, 0), max(frame.width - 1, 0))
    y1 = min(max(y1, 0), max(frame.height - 1, 0))
    x2 = min(max(x2, x1 + 1), max(frame.width, x1 + 1))
    y2 = min(max(y2, y1 + 1), max(frame.height, y1 + 1))
    return [x1, y1, x2, y2]
