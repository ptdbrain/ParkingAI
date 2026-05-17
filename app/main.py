from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai.pipeline import PipelineRuntime
from app.backend.api.routes import slots, vehicles, ws
from app.config import get_settings
from app.core.camera import FakeFrameGenerator
from app.db.session import init_db
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start database and AI runtime for the API process."""

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_db()

    runtime = PipelineRuntime(settings)
    await runtime.start()
    app.state.pipeline_runtime = runtime
    demo_task = asyncio.create_task(_feed_demo_frames(runtime, settings.frame_rate_limit), name="demo-frame-feed")
    logger.info("FastAPI application started")
    try:
        yield
    finally:
        demo_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await demo_task
        await runtime.stop()
        logger.info("FastAPI application stopped")


def create_app() -> FastAPI:
    """Application factory for ASGI servers and tests."""

    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(vehicles.router)
    app.include_router(slots.router)
    app.include_router(ws.router)
    return app


async def _feed_demo_frames(runtime: PipelineRuntime, fps: float) -> None:
    """Feed fake frames so WebSocket clients receive events immediately."""

    generator = FakeFrameGenerator()
    async for frame in generator.stream(fps=fps):
        await runtime.submit_frame(frame)


app = create_app()
