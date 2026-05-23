from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.backend.api.deps import get_camera_service
from app.db.schemas import CameraCreate, CameraRead, CameraStreamRead
from app.services.camera_service import CameraService
from app.services.errors import DuplicateResourceError, ValidationError

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=list[CameraRead])
async def list_cameras(
    service: Annotated[CameraService, Depends(get_camera_service)],
) -> list[CameraRead]:
    """List camera profiles and their calibrated overlay regions."""

    cameras = await service.list_cameras()
    return [CameraRead.model_validate(camera) for camera in cameras]


@router.post("", response_model=CameraRead, status_code=status.HTTP_201_CREATED)
async def create_camera(
    payload: CameraCreate,
    service: Annotated[CameraService, Depends(get_camera_service)],
) -> CameraRead:
    """Create a camera profile and overlay calibration."""

    try:
        camera = await service.create_camera(payload)
    except DuplicateResourceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return CameraRead.model_validate(camera)


@router.get("/by-slot/{slot_code}", response_model=CameraRead)
async def get_slot_camera(
    slot_code: str,
    service: Annotated[CameraService, Depends(get_camera_service)],
) -> CameraRead:
    """Return the dedicated camera view for one parking slot."""

    try:
        return await service.get_slot_camera(slot_code)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/by-slot/{slot_code}/stream", response_model=CameraStreamRead)
async def get_slot_stream(
    slot_code: str,
    service: Annotated[CameraService, Depends(get_camera_service)],
) -> CameraStreamRead:
    """Return stream metadata for a dedicated slot camera."""

    try:
        return await service.get_slot_stream_info(slot_code)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{camera_id}", response_model=CameraRead)
async def get_camera(
    camera_id: str,
    service: Annotated[CameraService, Depends(get_camera_service)],
) -> CameraRead:
    """Return one camera profile."""

    camera = await service.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return CameraRead.model_validate(camera)


@router.get("/{camera_id}/stream", response_model=CameraStreamRead)
async def get_camera_stream(
    camera_id: str,
    service: Annotated[CameraService, Depends(get_camera_service)],
) -> CameraStreamRead:
    """Return stream metadata for frontend video attachment."""

    stream = await service.get_stream_info(camera_id)
    if stream is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return stream
