from __future__ import annotations

import base64
import csv
from pathlib import Path

from scripts.build_ocr_crops import build_ocr_crops


PNG_2X2 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFElEQVR4nGNk+M8AB0wMDBgs"
    "AQAe5gICnCW9eAAAAABJRU5ErkJggg=="
)


def test_build_ocr_crops_writes_cropped_images_and_manifest(tmp_path: Path) -> None:
    source_image = tmp_path / "source.png"
    source_image.write_bytes(base64.b64decode(PNG_2X2))
    manifest = tmp_path / "ocr_manifest.csv"
    output_root = tmp_path / "ocr_crops"

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "image_path", "bbox_xywh", "text"])
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "unit",
                "split": "train",
                "image_path": str(source_image),
                "bbox_xywh": "0,0,1,1",
                "text": "51A12345",
            }
        )

    summary = build_ocr_crops(manifest_path=manifest, output_root=output_root)

    crop_manifest = output_root / "ocr_crops_manifest.csv"
    rows = list(csv.DictReader(crop_manifest.open(encoding="utf-8")))
    crop_path = Path(rows[0]["crop_path"])

    assert summary == {"written": 1, "skipped": 0}
    assert rows[0]["text"] == "51A12345"
    assert rows[0]["split"] == "train"
    assert crop_path.exists()
    assert crop_path.parent == output_root / "crops" / "train"
