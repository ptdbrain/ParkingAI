import cv2
import numpy as np
import easyocr
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ANPR:
    """Automatic Number Plate Recognition using EasyOCR.
    
    Pipeline: Image Crop -> Preprocessing -> EasyOCR -> Postprocessing -> Plate Text
    """

    # Regex cho biển số xe Việt Nam: 51A-12345, 51A-123.45, 59P1-12345
    VN_PLATE_PATTERN = re.compile(
        r"(\d{2}[A-Z]\d?)[- .]?(\d{3,5}\.?\d{0,2})", re.IGNORECASE
    )

    def __init__(self, languages=None, gpu=True, model_storage_directory=None):
        if languages is None:
            languages = ["en"]
        
        # Tắt verbose để tránh spam log khi chạy pipeline
        self.reader = easyocr.Reader(
            languages,
            gpu=gpu,
            model_storage_directory=model_storage_directory,
            verbose=False
        )
        logger.info("ANPR initialized: languages=%s, gpu=%s", languages, gpu)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Chuẩn bị ảnh tốt nhất cho OCR."""
        h, w = image.shape[:2]
        if w < 200:
            scale = 200 / w
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Giảm nhiễu giữ cạnh
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        # Nhị phân hóa thích nghi
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def format_plate(self, raw_text: str) -> str | None:
        """Chuẩn hóa format biển số VN."""
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        match = self.VN_PLATE_PATTERN.search(cleaned)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return None

    def read_plate(self, image: np.ndarray) -> dict:
        """Đọc biển số từ ảnh crop."""
        processed = self.preprocess(image)
        
        # Thử với ảnh đã xử lý
        results = self.reader.readtext(processed, detail=1, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")
        
        if not results:
            # Fallback về ảnh gốc nếu không đọc được
            results = self.reader.readtext(image, detail=1)

        if not results:
            return {"plate": None, "confidence": 0.0, "raw": ""}

        combined_text = "".join(r[1] for r in results)
        avg_conf = sum(r[2] for r in results) / len(results)
        
        formatted = self.format_plate(combined_text)
        return {
            "plate": formatted or combined_text,
            "confidence": round(avg_conf, 4),
            "raw": combined_text
        }