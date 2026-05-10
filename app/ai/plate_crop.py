import numpy as np

@dataclass(slots=True)
class PlateCropTask:
    """Internal handoff from detection to OCR for one license plate crop."""

    frame_id: int
    bbox: list[int]
    confidence: float
    image_crop: str
    image_data: np.ndarray | None = None

