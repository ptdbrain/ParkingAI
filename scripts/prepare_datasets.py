from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "datasets"
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
REPORT_DIR = DATASET_DIR / "analyzed"

DETECTION_CLASSES = ["car", "motorbike", "license_plate"]
CLASS_ALIASES = {
    "car": "car",
    "cars": "car",
    "motorbike": "motorbike",
    "motorcycles": "motorbike",
    "license plate": "license_plate",
    "license_plate": "license_plate",
    "license-plate": "license_plate",
    "license plate number": "license_plate",
    "biensoxehoi": "license_plate",
}
IGNORED_CATEGORIES = {"objects", "object", "car-motorbike-license_plate"}
SPLIT_ALIASES = {"valid": "val", "validation": "val", "val": "val", "train": "train", "test": "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class CocoSource:
    name: str
    split: str
    annotation_path: Path
    image_root: Path


@dataclass(slots=True)
class DatasetStats:
    name: str
    kind: str
    splits: dict[str, int] = field(default_factory=dict)
    annotations: int = 0
    categories: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def main() -> None:
    """Prepare raw, detection, and OCR datasets from mixed source archives."""

    _ensure_dirs()
    unzip_archives()
    coco_sources = discover_coco_sources()
    stats = analyze_sources(coco_sources)
    detection_counts, ocr_count = build_processed_datasets(coco_sources)
    write_reports(stats, detection_counts, ocr_count)
    logger.info("Dataset preparation finished")


def _ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def unzip_archives() -> None:
    """Extract each zip into a stable raw dataset folder."""

    for zip_path in sorted(SOURCE_DIR.glob("*.zip")):
        target = RAW_DIR / _safe_stem(zip_path.stem)
        marker = target / ".extract_complete"
        if marker.exists():
            logger.info("Skipping already extracted archive: %s", zip_path.name)
            continue
        target.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting %s -> %s", zip_path.name, target)
        with ZipFile(zip_path) as archive:
            archive.extractall(target)
        marker.write_text(zip_path.name + "\n", encoding="utf-8")


def discover_coco_sources() -> list[CocoSource]:
    """Find COCO annotation files in extracted archives and existing datasets."""

    sources: list[CocoSource] = []
    search_roots = [RAW_DIR, SOURCE_DIR]
    seen: set[Path] = set()
    for root in search_roots:
        for annotation_path in sorted(root.rglob("_annotations.coco.json")):
            resolved = annotation_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            split_name = SPLIT_ALIASES.get(annotation_path.parent.name.lower(), annotation_path.parent.name.lower())
            dataset_name = _dataset_name_for(annotation_path, root)
            sources.append(
                CocoSource(
                    name=dataset_name,
                    split=split_name,
                    annotation_path=annotation_path,
                    image_root=annotation_path.parent,
                )
            )
    logger.info("Discovered %s COCO annotation sources", len(sources))
    return sources


def analyze_sources(coco_sources: list[CocoSource]) -> list[DatasetStats]:
    """Classify datasets and collect class/image counts."""

    stats_by_name: dict[str, DatasetStats] = {}
    for source in coco_sources:
        data = _read_json(source.annotation_path)
        categories = [str(category.get("name", "")) for category in data.get("categories", [])]
        mapped = {_normalize_category(name) for name in categories}
        mapped.discard(None)
        kind = "detection"
        if mapped == {"license_plate"}:
            kind = "license_plate_detection"
        if "ocr" in source.name.lower():
            kind = "ocr_bbox_manifest"

        stats = stats_by_name.setdefault(source.name, DatasetStats(name=source.name, kind=kind))
        stats.splits[source.split] = stats.splits.get(source.split, 0) + len(data.get("images", []))
        stats.annotations += len(data.get("annotations", []))
        stats.categories = sorted(set([*stats.categories, *categories]))
        if any(name.strip().lower() in IGNORED_CATEGORIES for name in categories):
            stats.notes.append("contains wrapper categories ignored during YOLO conversion")

    for image_only in discover_image_only_sources():
        stats_by_name[image_only.name] = image_only

    return sorted(stats_by_name.values(), key=lambda item: item.name)


def discover_image_only_sources() -> list[DatasetStats]:
    """Classify raw image folders that do not contain annotations."""

    stats: list[DatasetStats] = []
    for folder in sorted(RAW_DIR.iterdir()):
        if not folder.is_dir() or list(folder.rglob("_annotations.coco.json")):
            continue
        image_count = sum(1 for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if image_count == 0:
            continue
        kind = "unlabeled_images"
        notes = ["not used for supervised YOLO/OCR training because no annotation file was found"]
        if "cartgmt" in folder.name.lower():
            kind = "vehicle_images_with_plate_in_filename"
            notes = ["useful later for Re-ID or OCR weak labels, but not used for bbox detection"]
        stats.append(DatasetStats(name=folder.name, kind=kind, splits={"unknown": image_count}, notes=notes))
    return stats


def build_processed_datasets(coco_sources: list[CocoSource]) -> tuple[dict[str, int], int]:
    """Create unified YOLO detection dataset and OCR manifest."""

    detection_root = PROCESSED_DIR / "detection"
    ocr_root = PROCESSED_DIR / "ocr"
    _reset_processed_subdirs(detection_root, ocr_root)

    class_to_id = {name: index for index, name in enumerate(DETECTION_CLASSES)}
    detection_counts = {split: 0 for split in ["train", "val", "test"]}
    ocr_rows: list[dict[str, str]] = []

    for source in coco_sources:
        split = SPLIT_ALIASES.get(source.split, source.split)
        if split not in detection_counts:
            split = "train"
        data = _read_json(source.annotation_path)
        categories = {category["id"]: str(category.get("name", "")) for category in data.get("categories", [])}
        images = {image["id"]: image for image in data.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, object]]] = {}
        for annotation in data.get("annotations", []):
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

        for image_id, image in images.items():
            converted_labels: list[str] = []
            source_image = source.image_root / image["file_name"]
            if not source_image.exists():
                logger.warning("Missing source image: %s", source_image)
                continue

            for annotation in annotations_by_image.get(image_id, []):
                category_name = categories.get(annotation["category_id"], "")
                normalized = _normalize_category(category_name)
                if normalized is None:
                    continue
                yolo_line = _coco_bbox_to_yolo(
                    class_id=class_to_id[normalized],
                    bbox=annotation["bbox"],
                    width=int(image["width"]),
                    height=int(image["height"]),
                )
                if yolo_line is not None:
                    converted_labels.append(yolo_line)
                if normalized == "license_plate":
                    plate_text = parse_plate_text(str(image["file_name"]))
                    if plate_text:
                        ocr_rows.append(
                            {
                                "dataset": source.name,
                                "split": split,
                                "image_path": str(source_image),
                                "bbox_xywh": ",".join(str(round(float(value), 2)) for value in annotation["bbox"]),
                                "text": plate_text,
                            }
                        )

            if converted_labels:
                output_name = _unique_image_name(source.name, split, source_image)
                output_image = detection_root / "images" / split / output_name
                output_label = detection_root / "labels" / split / (Path(output_name).stem + ".txt")
                output_image.parent.mkdir(parents=True, exist_ok=True)
                output_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, output_image)
                output_label.write_text("\n".join(converted_labels) + "\n", encoding="utf-8")
                detection_counts[split] += 1

    _write_detection_yaml(detection_root)
    _write_ocr_manifest(ocr_root, ocr_rows)
    return detection_counts, len(ocr_rows)


def _reset_processed_subdirs(*roots: Path) -> None:
    for root in roots:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)


def _write_detection_yaml(detection_root: Path) -> None:
    payload = {
        "path": str(detection_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(DETECTION_CLASSES)},
    }
    (detection_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (detection_root / "class_mapping.json").write_text(json.dumps(payload["names"], indent=2), encoding="utf-8")


def _write_ocr_manifest(ocr_root: Path, rows: list[dict[str, str]]) -> None:
    ocr_root.mkdir(parents=True, exist_ok=True)
    manifest = ocr_root / "ocr_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "image_path", "bbox_xywh", "text"])
        writer.writeheader()
        writer.writerows(rows)
    (ocr_root / "README.md").write_text(
        "OCR manifest uses source image path + license plate bbox + parsed plate text. "
        "Install Pillow/OpenCV later to materialize cropped plate images if needed.\n",
        encoding="utf-8",
    )


def write_reports(stats: list[DatasetStats], detection_counts: dict[str, int], ocr_count: int) -> None:
    """Write machine-readable and human-readable dataset reports."""

    summary = {
        "detection_classes": DETECTION_CLASSES,
        "detection_image_counts": detection_counts,
        "ocr_manifest_rows": ocr_count,
        "datasets": [
            {
                "name": item.name,
                "kind": item.kind,
                "splits": item.splits,
                "annotations": item.annotations,
                "categories": item.categories,
                "notes": sorted(set(item.notes)),
            }
            for item in stats
        ],
    }
    (REPORT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Dataset Analysis Report",
        "",
        "## Unified Detection Dataset",
        "",
        f"- Classes: {', '.join(DETECTION_CLASSES)}",
        f"- Train images: {detection_counts.get('train', 0)}",
        f"- Val images: {detection_counts.get('val', 0)}",
        f"- Test images: {detection_counts.get('test', 0)}",
        f"- YOLO config: `dataset/processed/detection/data.yaml`",
        "",
        "## OCR Dataset",
        "",
        f"- Manifest rows: {ocr_count}",
        "- Output: `dataset/processed/ocr/ocr_manifest.csv`",
        "- Rows contain source image path, license plate bbox, and plate text parsed from filenames.",
        "",
        "## Source Datasets",
        "",
    ]
    for item in stats:
        lines.append(f"### {item.name}")
        lines.append("")
        lines.append(f"- Type: {item.kind}")
        lines.append(f"- Splits/images: {item.splits}")
        lines.append(f"- Annotation count: {item.annotations}")
        if item.categories:
            lines.append(f"- Categories: {', '.join(item.categories)}")
        for note in sorted(set(item.notes)):
            lines.append(f"- Note: {note}")
        lines.append("")

    (REPORT_DIR / "dataset_report.md").write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_category(name: str) -> str | None:
    key = name.strip().lower().replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key)
    if key in IGNORED_CATEGORIES:
        return None
    return CLASS_ALIASES.get(key)


def _coco_bbox_to_yolo(class_id: int, bbox: list[float], width: int, height: int) -> str | None:
    if width <= 0 or height <= 0 or len(bbox) != 4:
        return None
    x, y, w, h = [float(value) for value in bbox]
    if w <= 0 or h <= 0:
        return None
    x_center = min(max((x + w / 2.0) / width, 0.0), 1.0)
    y_center = min(max((y + h / 2.0) / height, 0.0), 1.0)
    norm_w = min(max(w / width, 0.0), 1.0)
    norm_h = min(max(h / height, 0.0), 1.0)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def parse_plate_text(file_name: str) -> str | None:
    """Parse plate-like text from parking dataset filenames."""

    stem = Path(file_name).stem
    stem = stem.split("_jpg.rf.")[0]
    stem = stem.split(".rf.")[0]
    match = re.search(r"^[A-Za-z0-9]+_([^_]+)_(?:checkin|checkout|checkoutex)", stem, flags=re.IGNORECASE)
    if not match:
        return None
    text = re.sub(r"[^A-Za-z0-9]", "", match.group(1)).upper()
    if text in {"NULL", "NONE", ""}:
        return None
    if len(text) < 3:
        return None
    return text


def _unique_image_name(dataset_name: str, split: str, source_image: Path) -> str:
    digest = hashlib.sha1(str(source_image).encode("utf-8")).hexdigest()[:10]
    return f"{_safe_stem(dataset_name)}_{split}_{source_image.stem}_{digest}{source_image.suffix.lower()}"


def _dataset_name_for(annotation_path: Path, root: Path) -> str:
    relative = annotation_path.relative_to(root)
    if root == SOURCE_DIR:
        return "existing_datasets_root"
    return relative.parts[0]


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    main()
