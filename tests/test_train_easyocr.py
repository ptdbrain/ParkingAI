from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from training.train_easyocr import TrainConfig, OCRCropDataset, train_model


def _write_crop(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (64, 24), color=255)
    image.save(path)


def test_ocr_crop_dataset_reads_manifest_and_encodes_text(tmp_path: Path) -> None:
    train_crop = tmp_path / "crops" / "train" / "51A12345.png"
    _write_crop(train_crop)
    manifest = tmp_path / "ocr_crops_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "crop_path", "text"])
        writer.writeheader()
        writer.writerow({"split": "train", "crop_path": str(train_crop), "text": "51A-123.45"})

    dataset = OCRCropDataset(manifest_path=manifest, split="train", image_height=32, image_width=128)
    image, target = dataset[0]

    assert image.shape == (1, 32, 128)
    assert target.tolist() == dataset.encode_text("51A12345").tolist()


def test_train_model_writes_checkpoint_for_tiny_dataset(tmp_path: Path) -> None:
    train_crop = tmp_path / "crops" / "train" / "51A12345.png"
    valid_crop = tmp_path / "crops" / "val" / "59B67890.png"
    _write_crop(train_crop)
    _write_crop(valid_crop)

    manifest = tmp_path / "ocr_crops_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "crop_path", "text"])
        writer.writeheader()
        writer.writerow({"split": "train", "crop_path": str(train_crop), "text": "51A12345"})
        writer.writerow({"split": "val", "crop_path": str(valid_crop), "text": "59B67890"})

    output_dir = tmp_path / "models"
    checkpoint = train_model(
        TrainConfig(
            manifest=manifest,
            output_dir=output_dir,
            epochs=1,
            batch_size=1,
            image_width=64,
            image_height=24,
            hidden_size=8,
            learning_rate=0.001,
            device="cpu",
            max_train_samples=1,
            max_valid_samples=1,
        )
    )

    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["characters"] == OCRCropDataset.CHARACTERS
    assert payload["image_width"] == 64
