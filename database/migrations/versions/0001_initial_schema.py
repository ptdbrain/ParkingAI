from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vehicles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("plate_text", sa.String(length=32), nullable=False),
        sa.Column("owner_name", sa.String(length=128), nullable=False),
        sa.Column("brand", sa.String(length=64), nullable=True),
        sa.Column("color", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_vehicles_id"), "vehicles", ["id"], unique=False)
    op.create_index(op.f("ix_vehicles_plate_text"), "vehicles", ["plate_text"], unique=True)

    op.create_table(
        "parking_slots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slot_code", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("current_vehicle_id", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["current_vehicle_id"], ["vehicles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_parking_slots_id"), "parking_slots", ["id"], unique=False)
    op.create_index(op.f("ix_parking_slots_slot_code"), "parking_slots", ["slot_code"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_parking_slots_slot_code"), table_name="parking_slots")
    op.drop_index(op.f("ix_parking_slots_id"), table_name="parking_slots")
    op.drop_table("parking_slots")
    op.drop_index(op.f("ix_vehicles_plate_text"), table_name="vehicles")
    op.drop_index(op.f("ix_vehicles_id"), table_name="vehicles")
    op.drop_table("vehicles")
