from __future__ import annotations

import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.repositories import CameraRepository
from app.services.camera_service import CameraService


def test_camera_service_seeds_default_profiles() -> None:
    async def run() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_local = async_sessionmaker(engine, expire_on_commit=False)
        async with session_local() as session:
            service = CameraService(CameraRepository(session))
            cameras = await service.list_cameras()
            by_id = {camera.camera_id: camera for camera in cameras}

            assert set(by_id) == {"cam-01", "cam-02", "cam-03", "cam-slot"}
            assert len(by_id["cam-01"].regions) == 12
            assert len(by_id["cam-02"].regions) == 8
            assert len(by_id["cam-03"].regions) == 4
            assert by_id["cam-slot"].kind == "slot"

        await engine.dispose()

    asyncio.run(run())


def test_camera_service_returns_dedicated_slot_camera() -> None:
    async def run() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_local = async_sessionmaker(engine, expire_on_commit=False)
        async with session_local() as session:
            service = CameraService(CameraRepository(session))
            camera = await service.get_slot_camera("b-03")

            assert camera.camera_id == "cam-slot-b-03"
            assert camera.kind == "slot"
            assert camera.focus_slot_code == "B-03"
            assert camera.stream_url == "/cameras/by-slot/B-03/stream"
            assert len(camera.regions) == 1
            assert camera.regions[0].slot_code == "B-03"

        await engine.dispose()

    asyncio.run(run())
