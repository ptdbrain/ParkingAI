from __future__ import annotations

from datetime import UTC, datetime

from app.db.models import Camera
from app.db.repositories import CameraRepository
from app.db.schemas import (
    CameraCreate,
    CameraRead,
    CameraRegionCreate,
    CameraRegionRead,
    CameraStreamRead,
)
from app.services.errors import DuplicateResourceError, ValidationError

SUPPORTED_CAMERA_KINDS = {"overview", "entry", "row", "slot"}
SUPPORTED_CAMERA_STATUSES = {"online", "offline", "maintenance"}


DEFAULT_CAMERA_PAYLOADS = [
    CameraCreate(
        camera_id="cam-01",
        name="Camera 01 overview",
        kind="overview",
        stream_url="/cameras/cam-01/stream",
        status="online",
        coverage="A, B, C rows",
        regions=[
            CameraRegionCreate(slot_code="A-01", label="01", x=7, y=13, width=19, height=19),
            CameraRegionCreate(slot_code="A-02", label="02", x=29, y=13, width=19, height=19),
            CameraRegionCreate(slot_code="A-03", label="03", x=51, y=13, width=19, height=19),
            CameraRegionCreate(slot_code="A-04", label="04", x=73, y=13, width=19, height=19),
            CameraRegionCreate(slot_code="B-01", label="05", x=7, y=39, width=19, height=19),
            CameraRegionCreate(slot_code="B-02", label="06", x=29, y=39, width=19, height=19),
            CameraRegionCreate(slot_code="B-03", label="07", x=51, y=39, width=19, height=19),
            CameraRegionCreate(slot_code="B-04", label="08", x=73, y=39, width=19, height=19),
            CameraRegionCreate(slot_code="C-01", label="09", x=7, y=65, width=19, height=19),
            CameraRegionCreate(slot_code="C-02", label="10", x=29, y=65, width=19, height=19),
            CameraRegionCreate(slot_code="C-03", label="11", x=51, y=65, width=19, height=19),
            CameraRegionCreate(slot_code="C-04", label="12", x=73, y=65, width=19, height=19),
        ],
    ),
    CameraCreate(
        camera_id="cam-02",
        name="Camera 02 entry lane",
        kind="entry",
        stream_url="/cameras/cam-02/stream",
        status="online",
        coverage="A row, entry lane",
        regions=[
            CameraRegionCreate(slot_code="A-01", label="01", x=8, y=18, width=20, height=26),
            CameraRegionCreate(slot_code="A-02", label="02", x=31, y=18, width=20, height=26),
            CameraRegionCreate(slot_code="A-03", label="03", x=54, y=18, width=20, height=26),
            CameraRegionCreate(slot_code="A-04", label="04", x=77, y=18, width=16, height=26),
            CameraRegionCreate(slot_code="B-01", label="05", x=8, y=58, width=20, height=24),
            CameraRegionCreate(slot_code="B-02", label="06", x=31, y=58, width=20, height=24),
            CameraRegionCreate(slot_code="B-03", label="07", x=54, y=58, width=20, height=24),
            CameraRegionCreate(slot_code="B-04", label="08", x=77, y=58, width=16, height=24),
        ],
    ),
    CameraCreate(
        camera_id="cam-03",
        name="Camera 03 row B",
        kind="row",
        stream_url="/cameras/cam-03/stream",
        status="online",
        coverage="B row",
        regions=[
            CameraRegionCreate(slot_code="B-01", label="B1", x=9, y=21, width=18, height=55),
            CameraRegionCreate(slot_code="B-02", label="B2", x=31, y=21, width=18, height=55),
            CameraRegionCreate(slot_code="B-03", label="B3", x=53, y=21, width=18, height=55),
            CameraRegionCreate(slot_code="B-04", label="B4", x=75, y=21, width=16, height=55),
        ],
    ),
    CameraCreate(
        camera_id="cam-slot",
        name="Dedicated slot camera",
        kind="slot",
        stream_url="/cameras/by-slot/{slot_code}/stream",
        status="online",
        coverage="Single slot",
        regions=[],
    ),
]


class CameraService:
    """Business logic for camera profiles, streams, and overlay calibration."""

    def __init__(self, repository: CameraRepository) -> None:
        self.repository = repository

    async def list_cameras(self) -> list[Camera]:
        """Return camera profiles, creating local defaults on first use."""

        await self._ensure_default_cameras()
        return await self.repository.list()

    async def get_camera(self, camera_id: str) -> Camera | None:
        """Return one camera profile by public camera id."""

        await self._ensure_default_cameras()
        return await self.repository.get_by_camera_id(camera_id.strip().lower())

    async def create_camera(self, payload: CameraCreate) -> Camera:
        """Create a camera profile and its calibrated regions."""

        normalized = self._normalize_camera(payload)
        if await self.repository.get_by_camera_id(normalized.camera_id) is not None:
            raise DuplicateResourceError("camera", normalized.camera_id)
        return await self.repository.create(normalized)

    async def get_slot_camera(self, slot_code: str) -> CameraRead:
        """Return a virtual dedicated camera view for one parking slot."""

        await self._ensure_default_cameras()
        normalized_slot = slot_code.strip().upper()
        base = await self.repository.get_by_camera_id("cam-slot")
        if base is None:
            raise ValidationError("Default slot camera is not configured")
        now = datetime.now(UTC)
        return CameraRead(
            id=base.id,
            camera_id=f"cam-slot-{normalized_slot.lower()}",
            name=f"Dedicated slot camera {normalized_slot}",
            kind="slot",
            stream_url=f"/cameras/by-slot/{normalized_slot}/stream",
            status=base.status,
            coverage="Single slot",
            focus_slot_code=normalized_slot,
            is_active=base.is_active,
            created_at=now,
            updated_at=now,
            regions=[
                CameraRegionRead(
                    slot_code=normalized_slot,
                    label=normalized_slot,
                    x=18,
                    y=19,
                    width=64,
                    height=56,
                )
            ],
        )

    async def get_stream_info(self, camera_id: str) -> CameraStreamRead | None:
        """Return stream metadata for a camera profile."""

        camera = await self.get_camera(camera_id)
        if camera is None:
            return None
        return CameraStreamRead(
            camera_id=camera.camera_id,
            stream_url=camera.stream_url,
            status=camera.status,
            placeholder=camera.stream_url is None or camera.stream_url.startswith("/cameras/"),
            message="Attach an RTSP, MJPEG, WebRTC, or backend proxy URL to stream live video.",
        )

    async def get_slot_stream_info(self, slot_code: str) -> CameraStreamRead:
        """Return stream metadata for a dedicated slot camera."""

        slot_camera = await self.get_slot_camera(slot_code)
        return CameraStreamRead(
            camera_id=slot_camera.camera_id,
            stream_url=slot_camera.stream_url,
            status=slot_camera.status,
            placeholder=True,
            message="Dedicated slot stream endpoint is ready for a camera proxy implementation.",
        )

    async def _ensure_default_cameras(self) -> None:
        for payload in DEFAULT_CAMERA_PAYLOADS:
            if await self.repository.get_by_camera_id(payload.camera_id) is None:
                await self.repository.create(payload)

    def _normalize_camera(self, payload: CameraCreate) -> CameraCreate:
        camera_id = payload.camera_id.strip().lower()
        kind = payload.kind.strip().lower()
        status = payload.status.strip().lower()
        if kind not in SUPPORTED_CAMERA_KINDS:
            raise ValidationError(f"Unsupported camera kind: {kind}")
        if status not in SUPPORTED_CAMERA_STATUSES:
            raise ValidationError(f"Unsupported camera status: {status}")
        return payload.model_copy(
            update={
                "camera_id": camera_id,
                "name": payload.name.strip(),
                "kind": kind,
                "status": status,
                "coverage": payload.coverage.strip(),
                "focus_slot_code": payload.focus_slot_code.strip().upper() if payload.focus_slot_code else None,
                "regions": [
                    region.model_copy(
                        update={
                            "slot_code": region.slot_code.strip().upper(),
                            "label": region.label.strip(),
                        }
                    )
                    for region in payload.regions
                ],
            }
        )
