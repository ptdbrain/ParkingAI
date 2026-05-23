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


class CameraRegionCreate(BaseModel):
    """Request body for a calibrated camera overlay region."""

    slot_code: str = Field(min_length=1, max_length=32)
    label: str = Field(min_length=1, max_length=32)
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    width: float = Field(gt=0, le=100)
    height: float = Field(gt=0, le=100)


class CameraRegionRead(BaseModel):
    """API response for a calibrated camera overlay region."""

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    slot_code: str
    label: str
    x: float
    y: float
    width: float
    height: float


class CameraCreate(BaseModel):
    """Request body for creating a camera profile."""

    camera_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    kind: str = Field(default="overview", max_length=32)
    stream_url: str | None = Field(default=None, max_length=512)
    status: str = Field(default="online", max_length=32)
    coverage: str = Field(default="All slots", max_length=128)
    focus_slot_code: str | None = Field(default=None, max_length=32)
    is_active: bool = True
    regions: list[CameraRegionCreate] = Field(default_factory=list)


class CameraRead(BaseModel):
    """API response for a camera profile and its overlay regions."""

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    camera_id: str
    name: str
    kind: str
    stream_url: str | None
    status: str
    coverage: str
    focus_slot_code: str | None
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None
    regions: list[CameraRegionRead] = Field(default_factory=list)


class CameraStreamRead(BaseModel):
    """Camera stream metadata for frontend video attachment."""

    camera_id: str
    stream_url: str | None
    status: str
    placeholder: bool
    message: str
