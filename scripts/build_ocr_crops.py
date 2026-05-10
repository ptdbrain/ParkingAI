from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "dataset" / "processed" / "ocr" / "ocr_manifest.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "processed" / "ocr"

# hàm này đọc file CSV manifest chứa thông tin về các hình ảnh và bounding box của biển số xe, 
# sau đó cắt các phần hình ảnh chứa biển số xe dựa trên bounding box đó và lưu vào thư mục crops.
# Đồng thời, nó cũng tạo một file CSV mới ghi lại thông tin về các crop đã được tạo ra, 
# bao gồm đường dẫn đến crop và văn bản nhận dạng được từ OCR. 
# Hàm trả về một dictionary với số lượng crop đã được viết 
# và số lượng mục bị bỏ qua do lỗi hoặc thiếu dữ liệu.
def build_ocr_crops(manifest_path: Path = DEFAULT_MANIFEST, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, int]:
    """Crop license plate images listed in the OCR manifest."""

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for OCR crop generation. Install it with `pip install Pillow`.") from exc

    crop_root = output_root / "crops"
    output_manifest = output_root / "ocr_crops_manifest.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    # Đọc file CSV manifest và xử lý từng dòng để cắt ảnh và ghi thông tin vào file CSV mới
    with manifest_path.open(newline="", encoding="utf-8") as source, output_manifest.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source) # đọc file CSV và tạo một DictReader để truy cập dữ liệu theo tên cột
        fieldnames = ["dataset", "split", "image_path", "bbox_xywh", "text", "crop_path"] #các trường sẽ được ghi vào file CSV mới
        writer = csv.DictWriter(target, fieldnames=fieldnames) #tạo một DictWriter để ghi dữ liệu vào file CSV mới với các trường đã định nghĩa
        writer.writeheader()    #ghi tiêu đề cột vào file CSV mới

        for row in reader:
            image_path = Path(row["image_path"]) #lấy đường dẫn đến hình ảnh từ cột "image_path" của file CSV manifest
            bbox = _parse_bbox_xywh(row["bbox_xywh"]) #lấy tọa độ bounding box từ cột "bbox_xywh" của file CSV manifest
            split = row.get("split") or "train" #lấy thông tin về split (train/val/test) từ cột "split" của file CSV manifest, nếu không có thì mặc định là "train"
            text = row.get("text", "")
            if bbox is None or not image_path.exists() or not text:
                skipped += 1
                continue

            with Image.open(image_path) as image:
                crop_box = _clamp_xywh_to_box(bbox, image.width, image.height)
                if crop_box is None:
                    skipped += 1
                    continue
                crop = image.crop(crop_box)
                crop_name = _crop_name(row, image_path)
                crop_path = crop_root / split / crop_name
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path)

            writer.writerow(
                {
                    "dataset": row.get("dataset", ""),
                    "split": split,
                    "image_path": str(image_path),
                    "bbox_xywh": row["bbox_xywh"],
                    "text": text,
                    "crop_path": str(crop_path),
                }
            )
            written += 1

    return {"written": written, "skipped": skipped}


def _parse_bbox_xywh(value: str) -> tuple[float, float, float, float] | None:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        return None
    try:
        x, y, width, height = [float(part) for part in parts]
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _clamp_xywh_to_box( # hàm này nhận vào một bounding box được định dạng theo kiểu (x, y, width, height) và kích thước của hình ảnh (image_width, image_height).
    bbox: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[int, int, int, int] | None:
    x, y, width, height = bbox
    left = max(0, min(round(x), image_width))
    top = max(0, min(round(y), image_height))
    right = max(0, min(round(x + width), image_width))
    bottom = max(0, min(round(y + height), image_height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _crop_name(row: dict[str, str], image_path: Path) -> str: # hàm này tạo tên file cho crop được tạo ra dựa trên thông tin từ dòng của file CSV manifest và đường dẫn đến hình ảnh gốc.
    digest_input = "|".join(
        [row.get("dataset", ""), row.get("split", ""), str(image_path), row.get("bbox_xywh", "")]
    )
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
    text = "".join(char for char in row.get("text", "") if char.isalnum()).upper()
    stem = image_path.stem[:80]
    return f"{text}_{stem}_{digest}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop OCR license plate images from dataset/processed/ocr/ocr_manifest.csv"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Input OCR manifest CSV")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output OCR directory")
    args = parser.parse_args()

    summary = build_ocr_crops(manifest_path=args.manifest, output_root=args.output_root)
    print(f"written={summary['written']} skipped={summary['skipped']}")
    # cuối cùng, output của file này là một file CSV mới chứa thông tin về các crop đã được tạo ra, bao gồm đường dẫn đến crop và văn bản nhận dạng được từ OCR, 
    # cùng với số lượng crop đã được viết và số lượng mục bị bỏ qua do lỗi hoặc thiếu dữ liệu. 
    # Các crop sẽ được lưu trong thư mục "crops" bên trong thư mục output đã chỉ định.

if __name__ == "__main__":
    main()
