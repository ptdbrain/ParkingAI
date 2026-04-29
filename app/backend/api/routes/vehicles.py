from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.backend.api.deps import get_vehicle_service
from app.db.schemas import VehicleCreate, VehicleRead
from app.services.vehicle_service import VehicleService

router = APIRouter(prefix="/vehicles", tags=["vehicles"])


@router.get("", response_model=list[VehicleRead])
async def list_vehicles(
    service: Annotated[VehicleService, Depends(get_vehicle_service)],
) -> list[VehicleRead]:
    """List registered vehicles."""

    vehicles = await service.list_vehicles()
    return [VehicleRead.model_validate(vehicle) for vehicle in vehicles]


@router.post("", response_model=VehicleRead, status_code=status.HTTP_201_CREATED)
async def create_vehicle(
    payload: VehicleCreate,
    service: Annotated[VehicleService, Depends(get_vehicle_service)],
) -> VehicleRead:
    """Register a vehicle."""

    vehicle = await service.register_vehicle(payload)
    return VehicleRead.model_validate(vehicle)
