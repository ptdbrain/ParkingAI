from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Vehicle(Base):
    """Registered vehicle or currently observed parking vehicle."""

    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    plate_text: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    owner_name: Mapped[str] = mapped_column(String(128), default="Unknown")
    brand: Mapped[str | None] = mapped_column(String(64), nullable=True)
    color: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class ParkingSlot(Base):
    """Parking slot state shown on the dashboard."""

    __tablename__ = "parking_slots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    slot_code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="vacant")
    current_vehicle_id: Mapped[int | None] = mapped_column(ForeignKey("vehicles.id"), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    current_vehicle: Mapped[Vehicle | None] = relationship("Vehicle")
