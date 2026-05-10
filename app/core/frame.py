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

    def to_numpy(self) -> np.ndarray:
        """Convert raw bytes to a numpy BGR image."""
        try:
            import numpy as np
            import cv2
            nparr = np.frombuffer(self.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
        # Fallback: create a blank image if data is not valid image bytes
        import numpy as np
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def get_crop(self, bbox: list[int]) -> np.ndarray:
        """Extract a crop from the frame data."""
        img = self.to_numpy()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Clamp coordinates to image boundaries
        x1, x2 = max(0, x1), min(self.width, x2)
        y1, y2 = max(0, y1), min(self.height, y2)
        return img[y1:y2, x1:x2]

