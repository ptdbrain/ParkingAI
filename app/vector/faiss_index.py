from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    faiss = None
    np = None


@dataclass(slots=True)
class SearchResult:
    vehicle_id: str
    score: float


class VehicleVectorIndex:
    """FAISS-backed vector index with a pure-Python fallback for demos."""

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension
        self._ids: list[str] = []
        self._fallback_vectors: list[list[float]] = []
        self._index = faiss.IndexFlatIP(dimension) if faiss and np else None

    def add(self, vehicle_id: str, embedding: list[float]) -> None:
        """Add or append a vehicle embedding."""

        self._validate_embedding(embedding)
        self._ids.append(vehicle_id)
        if self._index is not None and np is not None:
            vector = np.array([embedding], dtype="float32")
            self._index.add(vector)
        else:
            self._fallback_vectors.append(embedding)
        logger.debug("Added embedding for vehicle_id=%s", vehicle_id)

    def search(self, embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        """Return nearest vehicles by cosine similarity / inner product."""

        self._validate_embedding(embedding)
        if not self._ids:
            return []
        if self._index is not None and np is not None:
            scores, indices = self._index.search(np.array([embedding], dtype="float32"), top_k)
            return [
                SearchResult(vehicle_id=self._ids[int(index)], score=float(score))
                for score, index in zip(scores[0], indices[0], strict=False)
                if int(index) >= 0
            ]
        scored = [
            SearchResult(vehicle_id=vehicle_id, score=self._dot(embedding, stored))
            for vehicle_id, stored in zip(self._ids, self._fallback_vectors, strict=False)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _validate_embedding(self, embedding: list[float]) -> None:
        if len(embedding) != self.dimension:
            raise ValueError(f"Expected embedding dimension {self.dimension}, got {len(embedding)}")

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return float(sum(a * b for a, b in zip(left, right, strict=True)) / (math.sqrt(sum(a * a for a in left)) or 1.0))
