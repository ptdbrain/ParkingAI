from __future__ import annotations

from app.db.models import Vehicle
from app.db.repositories import VehicleRepository
from app.db.schemas import VehicleCreate


class VehicleService:
    """Business logic for vehicle registration and lookup."""

    def __init__(self, repository: VehicleRepository) -> None:
        self.repository = repository

    async def list_vehicles(self) -> list[Vehicle]:
        """Return all vehicles."""

        return await self.repository.list()

    async def register_vehicle(self, payload: VehicleCreate) -> Vehicle:
        """Register a new vehicle.

        TODO: Add duplicate handling and resident ownership validation.
        """

        return await self.repository.create(payload)
