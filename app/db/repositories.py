from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Camera, CameraRegion, ParkingSlot, Vehicle
from app.db.schemas import CameraCreate, SlotCreate, VehicleCreate


class VehicleRepository:
    """Async database operations for vehicles."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> list[Vehicle]:
        result = await self.session.execute(select(Vehicle).order_by(Vehicle.id))
        return list(result.scalars().all())

    async def get_by_plate(self, plate_text: str) -> Vehicle | None:
        result = await self.session.execute(select(Vehicle).where(Vehicle.plate_text == plate_text))
        return result.scalar_one_or_none()

    async def create(self, payload: VehicleCreate) -> Vehicle:
        vehicle = Vehicle(**payload.model_dump())
        self.session.add(vehicle)
        await self.session.commit()
        await self.session.refresh(vehicle)
        return vehicle


class SlotRepository:
    """Async database operations for parking slots."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> list[ParkingSlot]:
        result = await self.session.execute(select(ParkingSlot).order_by(ParkingSlot.slot_code))
        return list(result.scalars().all())

    async def get_by_code(self, slot_code: str) -> ParkingSlot | None:
        result = await self.session.execute(select(ParkingSlot).where(ParkingSlot.slot_code == slot_code))
        return result.scalar_one_or_none()

    async def create(self, payload: SlotCreate) -> ParkingSlot:
        slot = ParkingSlot(**payload.model_dump())
        self.session.add(slot)
        await self.session.commit()
        await self.session.refresh(slot)
        return slot


class CameraRepository:
    """Async database operations for camera profiles and overlay regions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> list[Camera]:
        result = await self.session.execute(
            select(Camera).options(selectinload(Camera.regions)).order_by(Camera.camera_id)
        )
        return list(result.scalars().all())

    async def get_by_camera_id(self, camera_id: str) -> Camera | None:
        result = await self.session.execute(
            select(Camera)
            .options(selectinload(Camera.regions))
            .where(Camera.camera_id == camera_id)
        )
        return result.scalar_one_or_none()

    async def create(self, payload: CameraCreate) -> Camera:
        data = payload.model_dump(exclude={"regions"})
        camera = Camera(**data)
        camera.regions = [CameraRegion(**region.model_dump()) for region in payload.regions]
        self.session.add(camera)
        await self.session.commit()
        await self.session.refresh(camera, attribute_names=["regions"])
        return camera
