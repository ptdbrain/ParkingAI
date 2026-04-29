from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.pipeline import PipelineRuntime
from app.db.repositories import SlotRepository, VehicleRepository
from app.db.session import get_db_session
from app.services.slot_service import SlotService
from app.services.vehicle_service import VehicleService


DbSession = Annotated[AsyncSession, Depends(get_db_session)]


def get_vehicle_service(session: DbSession) -> VehicleService:
    """Inject vehicle service with its repository."""

    return VehicleService(VehicleRepository(session))


def get_slot_service(session: DbSession) -> SlotService:
    """Inject slot service with its repository."""

    return SlotService(SlotRepository(session))


def get_pipeline_runtime(request: Request) -> PipelineRuntime:
    """Inject the process-level AI runtime."""

    return request.app.state.pipeline_runtime
