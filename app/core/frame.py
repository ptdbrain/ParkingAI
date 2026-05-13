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

    @classmethod
    def blank(cls, frame_id: int, width: int, height: int, source: str = "mock-camera") -> "Frame":
        """Create a deterministic blank BGR frame for tests and demos."""

        return cls(
            frame_id=frame_id,
            timestamp=datetime.now(UTC),
            width=width,
            height=height,
            data=bytes(width * height * 3),
            source=source,
        )

    def crop_token(self, bbox: list[int]) -> str:
        """Return a stable identifier for a crop without storing image bytes."""

        x1, y1, x2, y2 = [int(value) for value in bbox]
        return f"frame-{self.frame_id}:crop-{x1}-{y1}-{x2}-{y2}"

    def to_numpy(self) -> np.ndarray:
        """Convert raw bytes to a numpy BGR image."""
        import numpy as np

        expected_raw_size = self.width * self.height * 3
        if len(self.data) == expected_raw_size:
            return np.frombuffer(self.data, dtype=np.uint8).reshape((self.height, self.width, 3))

        try:
            import cv2
            nparr = np.frombuffer(self.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
        # Fallback: create a blank image if data is not valid image bytes
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def get_crop(self, bbox: list[int]) -> np.ndarray:
        """Extract a crop from the frame data."""
        img = self.to_numpy()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Clamp coordinates to image boundaries
        x1, x2 = max(0, x1), min(self.width, x2)
        y1, y2 = max(0, y1), min(self.height, y2)
        return img[y1:y2, x1:x2]
