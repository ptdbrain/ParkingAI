from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class VehicleCreate(BaseModel):
    """Request body for registering a vehicle."""

    plate_text: str = Field(min_length=2, max_length=32)
    owner_name: str = Field(default="Unknown", max_length=128)
    brand: str | None = Field(default=None, max_length=64)
    color: str | None = Field(default=None, max_length=32)


class VehicleRead(BaseModel):
    """API response for a vehicle."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    plate_text: str
    owner_name: str
    brand: str | None
    color: str | None
    created_at: datetime


class SlotCreate(BaseModel):
    """Request body for creating a parking slot."""

    slot_code: str = Field(min_length=1, max_length=32)
    status: str = Field(default="vacant", max_length=32)


class SlotRead(BaseModel):
    """API response for a parking slot."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    slot_code: str
    status: str
    current_vehicle_id: int | None
    confidence: float
    updated_at: datetime
