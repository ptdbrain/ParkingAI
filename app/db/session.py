from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings
from app.db.base import Base


def create_engine(settings: Settings | None = None) -> AsyncEngine:
    """Create an async SQLAlchemy engine.

    Use `mysql+asyncmy://user:pass@host:3306/dbname` for production MySQL.
    The default SQLite URL keeps the mock project runnable without services.
    """

    settings = settings or get_settings()
    return create_async_engine(settings.database_url, pool_pre_ping=True)


engine = create_engine()
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    """Create database tables for local demos.

    TODO: Replace this with Alembic migrations for production deployments.
    """

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency for async database sessions."""

    async with SessionLocal() as session:
        yield session
