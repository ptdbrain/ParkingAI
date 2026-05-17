from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "dataset" / "processed" / "ocr" / "ocr_crops_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "ocr"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainConfig:
    manifest: Path = DEFAULT_MANIFEST
    output_dir: Path = DEFAULT_OUTPUT_DIR
    train_data: Path | None = None
    valid_data: Path | None = None
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    image_width: int = 160
    image_height: int = 32
    hidden_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    max_train_samples: int | None = None
    max_valid_samples: int | None = None


class OCRCropDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """License plate crop dataset backed by scripts/build_ocr_crops.py output."""

    CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(
        self,
        *,
        manifest_path: Path,
        split: str,
        image_height: int,
        image_width: int,
        data_dir: Path | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.split = split
        self.image_height = image_height
        self.image_width = image_width
        self.data_dir = data_dir.resolve() if data_dir is not None else None
        self.char_to_index = {char: index + 1 for index, char in enumerate(self.CHARACTERS)}
        self.rows = self._load_rows(max_samples)
        if not self.rows:
            raise ValueError(f"No usable OCR crops found for split={split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        crop_path, text = self.rows[index]
        with Image.open(crop_path) as image:
            image = image.convert("L").resize((self.image_width, self.image_height), Image.Resampling.BILINEAR)
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).float()
        tensor = pixels.reshape(1, self.image_height, self.image_width) / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor, self.encode_text(text)

    def encode_text(self, text: str) -> torch.Tensor:
        normalized = normalize_plate_text(text)
        try:
            values = [self.char_to_index[char] for char in normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported OCR character {exc.args[0]!r} in {text!r}") from exc
        return torch.tensor(values, dtype=torch.long)

    def _load_rows(self, max_samples: int | None) -> list[tuple[Path, str]]:
        rows: list[tuple[Path, str]] = []
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if (row.get("split") or "train") != self.split:
                    continue
                crop_path = Path(row["crop_path"]).resolve()
                text = normalize_plate_text(row.get("text", ""))
                if not crop_path.exists() or not text:
                    continue
                if self.data_dir is not None and not _is_relative_to(crop_path, self.data_dir):
                    continue
                rows.append((crop_path, text))
                if max_samples is not None and len(rows) >= max_samples:
                    break
        return rows


class CRNNOCRModel(nn.Module):
    """Small CRNN recognizer trained with CTC loss."""

    def __init__(self, num_classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.sequence = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.adaptive_avg_pool2d(features, (1, features.size(-1))).squeeze(2)
        sequence = features.permute(2, 0, 1)
        recurrent, _ = self.sequence(sequence)
        return self.classifier(recurrent).log_softmax(dim=2)


def normalize_plate_text(text: str) -> str:
    return "".join(char for char in text.upper() if char.isalnum())


def train_model(config: TrainConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if not config.manifest.exists():
        raise FileNotFoundError(
            f"OCR crop manifest not found: {config.manifest}. "
            "Run `python scripts/build_ocr_crops.py` before training."
        )

    device = _select_device(config.device)
    train_dataset = OCRCropDataset(
        manifest_path=config.manifest,
        split="train",
        image_height=config.image_height,
        image_width=config.image_width,
        data_dir=config.train_data,
        max_samples=config.max_train_samples,
    )
    valid_dataset = OCRCropDataset(
        manifest_path=config.manifest,
        split="val",
        image_height=config.image_height,
        image_width=config.image_width,
        data_dir=config.valid_data,
        max_samples=config.max_valid_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_ocr_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_ocr_batch,
    )

    model = CRNNOCRModel(num_classes=len(OCRCropDataset.CHARACTERS) + 1, hidden_size=config.hidden_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.output_dir / "crnn_ocr_best.pt"
    last_path = config.output_dir / "crnn_ocr_last.pt"
    best_valid_loss = float("inf")

    logger.info(
        "Starting OCR training: train=%s valid=%s epochs=%s device=%s",
        len(train_dataset),
        len(valid_dataset),
        config.epochs,
        device,
    )
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_loss = run_epoch(model, valid_loader, criterion, device)
        logger.info("epoch=%s train_loss=%.4f valid_loss=%.4f", epoch, train_loss, valid_loss)

        save_checkpoint(last_path, model, config, valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(best_path, model, config, valid_loss)

    return best_path


def run_epoch(
    model: CRNNOCRModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: nn.CTCLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    for images, targets, target_lengths, _texts in loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        logits = model(images)
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=logits.size(0),
            dtype=torch.long,
            device=device,
        )
        loss = criterion(logits, targets, input_lengths, target_lengths)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def collate_ocr_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    images = torch.stack([item[0] for item in batch])
    targets = torch.cat([item[1] for item in batch])
    target_lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
    texts = [_decode_indices(item[1]) for item in batch]
    return images, targets, target_lengths, texts


def save_checkpoint(path: Path, model: CRNNOCRModel, config: TrainConfig, valid_loss: float) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "characters": OCRCropDataset.CHARACTERS,
            "blank_index": 0,
            "image_width": config.image_width,
            "image_height": config.image_height,
            "hidden_size": config.hidden_size,
            "valid_loss": valid_loss,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a compact CRNN OCR model for license plates.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="CSV from scripts/build_ocr_crops.py")
    parser.add_argument("--train_data", type=Path, default=None, help="Optional train crop directory filter")
    parser.add_argument("--valid_data", type=Path, default=None, help="Optional validation crop directory filter")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = train_model(
        TrainConfig(
            manifest=args.manifest,
            output_dir=args.output_dir,
            train_data=args.train_data,
            valid_data=args.valid_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            image_width=args.image_width,
            image_height=args.image_height,
            hidden_size=args.hidden_size,
            num_workers=args.num_workers,
            device=args.device,
            max_train_samples=args.max_train_samples,
            max_valid_samples=args.max_valid_samples,
        )
    )
    print(f"best_checkpoint={checkpoint}")


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(device)


def _decode_indices(indices: torch.Tensor) -> str:
    characters = OCRCropDataset.CHARACTERS
    return "".join(characters[index - 1] for index in indices.tolist())


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
