from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ai.pipeline import PipelineRuntime

logger = logging.getLogger(__name__)
router = APIRouter(tags=["live"])


@router.websocket("/ws/live")
async def live_events(websocket: WebSocket) -> None:
    """Stream AI events to dashboard clients."""

    await websocket.accept()
    runtime: PipelineRuntime = websocket.app.state.pipeline_runtime
    try:
        while True:
            event = await runtime.read_event(timeout=1.0)
            if event is None:
                await websocket.send_json({"type": "heartbeat"})
                continue
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("WebSocket live stream failed")
        await websocket.close(code=1011)
