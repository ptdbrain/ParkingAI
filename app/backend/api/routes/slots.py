from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.backend.api.deps import get_slot_service
from app.db.schemas import SlotCreate, SlotRead
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

    slot = await service.create_slot(payload)
    return SlotRead.model_validate(slot)
