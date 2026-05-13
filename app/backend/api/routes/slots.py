from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.backend.api.deps import get_slot_service
from app.db.schemas import SlotCreate, SlotRead
from app.services.errors import DuplicateResourceError, ValidationError
from app.services.slot_service import SlotService

router = APIRouter(prefix="/slots", tags=["slots"])


@router.get("", response_model=list[SlotRead])
async def list_slots(
    service: Annotated[SlotService, Depends(get_slot_service)],
) -> list[SlotRead]:
    """List parking slots."""

    slots = await service.list_slots()
    return [SlotRead.model_validate(slot) for slot in slots]


@router.post("", response_model=SlotRead, status_code=status.HTTP_201_CREATED)
async def create_slot(
    payload: SlotCreate,
    service: Annotated[SlotService, Depends(get_slot_service)],
) -> SlotRead:
    """Create a parking slot."""

    try:
        slot = await service.create_slot(payload)
    except DuplicateResourceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return SlotRead.model_validate(slot)
