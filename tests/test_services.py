from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.db.schemas import SlotCreate, VehicleCreate
from app.services.errors import DuplicateResourceError, ValidationError
from app.services.slot_service import SlotService
from app.services.vehicle_service import VehicleService


@dataclass
class StoredVehicle:
    plate_text: str


class FakeVehicleRepository:
    def __init__(self, existing_plate: str | None = None) -> None:
        self.existing_plate = existing_plate
        self.created: VehicleCreate | None = None

    async def list(self) -> list[StoredVehicle]:
        return []

    async def get_by_plate(self, plate_text: str) -> StoredVehicle | None:
        if self.existing_plate == plate_text:
            return StoredVehicle(plate_text=plate_text)
        return None

    async def create(self, payload: VehicleCreate) -> VehicleCreate:
        self.created = payload
        return payload


@dataclass
class StoredSlot:
    slot_code: str


class FakeSlotRepository:
    def __init__(self, existing_slot: str | None = None) -> None:
        self.existing_slot = existing_slot
        self.created: SlotCreate | None = None

    async def list(self) -> list[StoredSlot]:
        return []

    async def get_by_code(self, slot_code: str) -> StoredSlot | None:
        if self.existing_slot == slot_code:
            return StoredSlot(slot_code=slot_code)
        return None

    async def create(self, payload: SlotCreate) -> SlotCreate:
        self.created = payload
        return payload


def test_vehicle_service_normalizes_plate_and_rejects_duplicates() -> None:
    async def run() -> None:
        duplicate_service = VehicleService(FakeVehicleRepository(existing_plate="51A12345"))

        try:
            await duplicate_service.register_vehicle(VehicleCreate(plate_text="51a-123.45"))
        except DuplicateResourceError as exc:
            assert exc.resource == "vehicle"
            assert exc.identifier == "51A12345"
        else:
            raise AssertionError("Duplicate vehicle was accepted")

        repository = FakeVehicleRepository()
        service = VehicleService(repository)
        await service.register_vehicle(VehicleCreate(plate_text="51a-123.45", owner_name="Resident"))

        assert repository.created is not None
        assert repository.created.plate_text == "51A12345"

    asyncio.run(run())


def test_slot_service_normalizes_slot_code_and_validates_status() -> None:
    async def run() -> None:
        duplicate_service = SlotService(FakeSlotRepository(existing_slot="A-01"))

        try:
            await duplicate_service.create_slot(SlotCreate(slot_code=" a-01 "))
        except DuplicateResourceError as exc:
            assert exc.resource == "slot"
            assert exc.identifier == "A-01"
        else:
            raise AssertionError("Duplicate slot was accepted")

        invalid_status_service = SlotService(FakeSlotRepository())
        try:
            await invalid_status_service.create_slot(SlotCreate(slot_code="B-01", status="broken"))
        except ValidationError as exc:
            assert "Unsupported slot status" in str(exc)
        else:
            raise AssertionError("Invalid slot status was accepted")

        repository = FakeSlotRepository()
        service = SlotService(repository)
        await service.create_slot(SlotCreate(slot_code=" b-02 ", status="occupied"))

        assert repository.created is not None
        assert repository.created.slot_code == "B-02"
        assert repository.created.status == "occupied"

    asyncio.run(run())
