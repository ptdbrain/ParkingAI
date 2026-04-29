from __future__ import annotations

from app.db.models import ParkingSlot
from app.db.repositories import SlotRepository
from app.db.schemas import SlotCreate


class SlotService:
    """Business logic for parking slot state."""

    def __init__(self, repository: SlotRepository) -> None:
        self.repository = repository

    async def list_slots(self) -> list[ParkingSlot]:
        """Return all known parking slots."""

        return await self.repository.list()

    async def create_slot(self, payload: SlotCreate) -> ParkingSlot:
        """Create a parking slot.

        TODO: Validate slot geometry against the calibrated parking map.
        """

        return await self.repository.create(payload)
