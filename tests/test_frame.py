from __future__ import annotations

from app.core.frame import Frame


def test_blank_frame_has_expected_shape_and_crop_token() -> None:
    frame = Frame.blank(frame_id=7, width=320, height=180, source="unit-camera")

    assert frame.frame_id == 7
    assert frame.width == 320
    assert frame.height == 180
    assert frame.source == "unit-camera"
    assert frame.crop_token([1, 2, 30, 40]) == "frame-7:crop-1-2-30-40"


def test_get_crop_clamps_bbox_to_frame_bounds() -> None:
    frame = Frame.blank(frame_id=1, width=8, height=6, source="unit-camera")

    crop = frame.get_crop([-10, -5, 20, 10])

    assert crop.shape == (6, 8, 3)
