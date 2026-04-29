from __future__ import annotations

from datetime import UTC, datetime

from app.core.events import ParkingEvent


def test_parking_event_serializes_strict_schema() -> None:
    event = ParkingEvent(
        frame_id=7,
        timestamp=datetime.now(UTC),
        type="car",
        bbox=[1, 2, 30, 40],
        confidence=0.87,
        image_crop=None,
        plate_text="51A12345",
        embedding=[0.1, 0.2, 0.3],
    )

    payload = event.to_payload()

    assert payload["frame_id"] == 7
    assert payload["type"] == "car"
    assert payload["bbox"] == [1, 2, 30, 40]
    assert payload["confidence"] == 0.87
    assert payload["image_crop"] is None
    assert payload["plate_text"] == "51A12345"
    assert payload["embedding"] == [0.1, 0.2, 0.3]
    assert isinstance(payload["timestamp"], str)


def test_parking_event_rejects_invalid_type() -> None:
    try:
        ParkingEvent(
            frame_id=1,
            timestamp=datetime.now(UTC),
            type="unknown",
            bbox=[0, 0, 1, 1],
            confidence=0.5,
        )
    except ValueError as exc:
        assert "Unsupported event type" in str(exc)
    else:
        raise AssertionError("ParkingEvent accepted an invalid type")
