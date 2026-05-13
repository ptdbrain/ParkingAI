from __future__ import annotations

from app.db.models import ParkingSlot
from app.db.repositories import SlotRepository
from app.db.schemas import SlotCreate
from app.services.errors import DuplicateResourceError, ValidationError


SUPPORTED_SLOT_STATUSES = {"vacant", "occupied", "reserved", "disabled"}


class SlotService:
    """Business logic for parking slot state."""

    def __init__(self, repository: SlotRepository) -> None:
        self.repository = repository

    async def list_slots(self) -> list[ParkingSlot]:
        """Return all known parking slots."""

        return await self.repository.list()

    async def create_slot(self, payload: SlotCreate) -> ParkingSlot:
        """Create a parking slot."""

        normalized = payload.model_copy(
            update={
                "slot_code": payload.slot_code.strip().upper(),
                "status": payload.status.strip().lower(),
            }
        )
        if normalized.status not in SUPPORTED_SLOT_STATUSES:
            raise ValidationError(f"Unsupported slot status: {normalized.status}")
        if await self.repository.get_by_code(normalized.slot_code) is not None:
            raise DuplicateResourceError("slot", normalized.slot_code)
        return await self.repository.create(normalized)
