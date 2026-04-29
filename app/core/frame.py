from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class Frame:
    """Lightweight frame container passed through the AI queues."""

    frame_id: int
    timestamp: datetime
    width: int
    height: int
    data: bytes = field(repr=False)
    source: str = "mock-camera"

    def crop_token(self, bbox: list[int]) -> str:
        """Return a stable mock crop reference without storing image arrays."""

        return f"frame-{self.frame_id}:crop-{bbox[0]}-{bbox[1]}-{bbox[2]}-{bbox[3]}"

    @classmethod
    def blank(cls, frame_id: int, width: int, height: int, source: str = "mock-camera") -> "Frame":
        """Create a tiny mock frame payload suitable for CPU-only tests."""

        return cls(
            frame_id=frame_id,
            timestamp=datetime.now(UTC),
            width=width,
            height=height,
            data=f"mock-frame-{frame_id}".encode("utf-8"),
            source=source,
        )
