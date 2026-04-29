from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.repositories import SlotRepository, VehicleRepository
from app.db.schemas import SlotCreate, VehicleCreate
from app.db.session import SessionLocal, init_db


async def main() -> None:
    """Seed demo vehicles and slots for local UI testing."""

    await init_db()
    async with SessionLocal() as session:
        vehicle_repo = VehicleRepository(session)
        slot_repo = SlotRepository(session)
        await vehicle_repo.create(VehicleCreate(plate_text="51A12345", owner_name="Demo Resident", brand="Toyota", color="white"))
        await slot_repo.create(SlotCreate(slot_code="A-01", status="occupied"))
        await slot_repo.create(SlotCreate(slot_code="A-02", status="vacant"))


if __name__ == "__main__":
    asyncio.run(main())
