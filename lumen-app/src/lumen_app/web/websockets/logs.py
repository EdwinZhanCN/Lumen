"""WebSocket endpoint for real-time log streaming."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lumen_app.utils.logger import get_logger
from lumen_app.web.core.state import app_state

logger = get_logger("lumen.web.ws.logs")
router = APIRouter()


@router.websocket("/logs")
async def log_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming logs."""
    await websocket.accept()
    logger.info("New log WebSocket connection")

    # Subscribe to logs
    log_queue = await app_state.subscribe_logs()

    try:
        # Send initial connection message
        await websocket.send_json(
            {
                "type": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Connected to log stream",
            }
        )

        while True:
            try:
                # Wait for log entry with timeout
                log_entry = await asyncio.wait_for(log_queue.get(), timeout=1.0)

                # Send to client
                await websocket.send_json(
                    {
                        "type": "log",
                        "timestamp": log_entry.get(
                            "timestamp", datetime.utcnow().isoformat()
                        ),
                        "level": log_entry.get("level", "INFO"),
                        "message": log_entry.get("message", ""),
                        "source": log_entry.get("source", "unknown"),
                    }
                )

            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

    except WebSocketDisconnect:
        logger.info("Log WebSocket disconnected")
    except Exception as e:
        logger.error(f"Log WebSocket error: {e}")
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
        except:
            pass
    finally:
        # Unsubscribe from logs
        await app_state.unsubscribe_logs(log_queue)
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/install/{task_id}")
async def install_progress_websocket(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for installation progress updates."""
    await websocket.accept()
    logger.info(f"New install progress WebSocket for task {task_id}")

    try:
        # Send initial status
        task = await app_state.get_task(task_id)
        if task:
            await websocket.send_json(
                {
                    "type": "status",
                    "task_id": task_id,
                    "task_type": task.type,
                    "status": task.status,
                    "progress": task.progress,
                    "message": task.message,
                }
            )

        # Poll for updates
        while True:
            await asyncio.sleep(1)

            task = await app_state.get_task(task_id)
            if not task:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Task {task_id} not found",
                    }
                )
                break

            await websocket.send_json(
                {
                    "type": "status",
                    "task_id": task_id,
                    "task_type": task.type,
                    "status": task.status,
                    "progress": task.progress,
                    "message": task.message,
                }
            )

            if task.status in ("completed", "failed", "cancelled"):
                await websocket.send_json(
                    {
                        "type": "complete",
                        "task_id": task_id,
                        "status": task.status,
                    }
                )
                break

    except WebSocketDisconnect:
        logger.info(f"Install progress WebSocket for task {task_id} disconnected")
    except Exception as e:
        logger.error(f"Install progress WebSocket error: {e}")
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
