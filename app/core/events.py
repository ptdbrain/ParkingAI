from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal


EventType = Literal["car", "slot", "fire", "anomaly"]
SUPPORTED_EVENT_TYPES = {"car", "slot", "fire", "anomaly"}


@dataclass(slots=True)
class ParkingEvent:
    """Unified event schema emitted by every AI worker."""

    frame_id: int
    timestamp: datetime
    type: EventType | str
    bbox: list[int]
    confidence: float
    image_crop: str | None = None
    plate_text: str | None = None
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if self.type not in SUPPORTED_EVENT_TYPES:
            raise ValueError(f"Unsupported event type: {self.type}")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly four integers")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    @classmethod
    def now(
        cls,
        *,
        frame_id: int,
        type: EventType,
        bbox: list[int],
        confidence: float,
        image_crop: str | None = None,
        plate_text: str | None = None,
        embedding: list[float] | None = None,
    ) -> "ParkingEvent":
        """Build an event with a UTC timestamp."""

        return cls(
            frame_id=frame_id,
            timestamp=datetime.now(UTC),
            type=type,
            bbox=bbox,
            confidence=confidence,
            image_crop=image_crop,
            plate_text=plate_text,
            embedding=embedding,
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize to the strict API/WebSocket payload format."""

        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.type,
            "bbox": self.bbox,
            "confidence": float(self.confidence),
            "image_crop": self.image_crop,
            "plate_text": self.plate_text,
            "embedding": self.embedding,
        }
