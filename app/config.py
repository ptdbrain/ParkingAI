from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(slots=True)
class Settings:
    """Central application configuration.

    Defaults are CPU-first and mock-friendly. Override with environment
    variables when deploying on Ubuntu with MySQL and real model artifacts.
    """

    app_name: str = os.getenv("PARKING_APP_NAME", "Edge AI Parking System")
    environment: str = os.getenv("PARKING_ENVIRONMENT", "development")
    log_level: str = os.getenv("PARKING_LOG_LEVEL", "INFO")
    inference_mode: str = os.getenv("PARKING_INFERENCE_MODE", "mock")

    database_url: str = os.getenv(
        "PARKING_DATABASE_URL",
        "sqlite+aiosqlite:///./edge_parking_demo.db",
    )

    queue_max_size: int = _env_int("PARKING_QUEUE_MAX_SIZE", 32)
    frame_width: int = _env_int("PARKING_FRAME_WIDTH", 640)
    frame_height: int = _env_int("PARKING_FRAME_HEIGHT", 360)
    frame_rate_limit: float = _env_float("PARKING_FRAME_RATE_LIMIT", 5.0)

    yolo_model_path: Path = Path(os.getenv("YOLO_MODEL_PATH", PROJECT_ROOT / "models/yolo/yolo_mock.onnx"))
    ocr_model_path: Path = Path(os.getenv("OCR_MODEL_PATH", PROJECT_ROOT / "models/ocr/ocr_mock"))
    reid_model_path: Path = Path(os.getenv("REID_MODEL_PATH", PROJECT_ROOT / "models/reid/resnet18_mock.pt"))
    vlm_model_path: Path = Path(os.getenv("VLM_MODEL_PATH", PROJECT_ROOT / "models/vlm/vlm_mock"))

    yolo_confidence_threshold: float = _env_float("YOLO_CONFIDENCE_THRESHOLD", 0.35)
    fire_confidence_threshold: float = _env_float("FIRE_CONFIDENCE_THRESHOLD", 0.40)
    ocr_confidence_threshold: float = _env_float("OCR_CONFIDENCE_THRESHOLD", 0.60)
    reid_match_threshold: float = _env_float("REID_MATCH_THRESHOLD", 0.78)
    vlm_confidence_threshold: float = _env_float("VLM_CONFIDENCE_THRESHOLD", 0.50)

    embedding_dim: int = _env_int("REID_EMBEDDING_DIM", 512)
    faiss_index_path: Path = Path(os.getenv("FAISS_INDEX_PATH", PROJECT_ROOT / "database/faiss_vehicle.index"))

    websocket_heartbeat_seconds: float = _env_float("WEBSOCKET_HEARTBEAT_SECONDS", 15.0)


def get_settings() -> Settings:
    """Return a settings instance for dependency injection."""

    return Settings()
