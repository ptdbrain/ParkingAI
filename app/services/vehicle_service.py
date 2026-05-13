from __future__ import annotations

from app.db.models import Vehicle
from app.db.repositories import VehicleRepository
from app.db.schemas import VehicleCreate
from app.services.errors import DuplicateResourceError


class VehicleService:
    """Business logic for vehicle registration and lookup."""

    def __init__(self, repository: VehicleRepository) -> None:
        self.repository = repository

    async def list_vehicles(self) -> list[Vehicle]:
        """Return all vehicles."""

        return await self.repository.list()

    async def register_vehicle(self, payload: VehicleCreate) -> Vehicle:
        """Register a new vehicle."""

        normalized = payload.model_copy(update={"plate_text": self._normalize_plate(payload.plate_text)})
        if await self.repository.get_by_plate(normalized.plate_text) is not None:
            raise DuplicateResourceError("vehicle", normalized.plate_text)
        return await self.repository.create(normalized)

    def _normalize_plate(self, plate_text: str) -> str:
        """Normalize plate text to the compact canonical form used for uniqueness."""

        return "".join(character for character in plate_text.upper() if character.isalnum())
