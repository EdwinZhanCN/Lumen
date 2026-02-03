"""FastAPI application entry point for Lumen Web."""

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from lumen_app.utils.logger import get_logger

from .api.config import router as config_router
from .api.hardware import router as hardware_router
from .api.install import router as install_router
from .api.server import router as server_router
from .core.state import app_state
from .websockets.logs import router as logs_ws_router

logger = get_logger("lumen.web")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Lumen Web API")
    await app_state.initialize()
    yield
    logger.info("Shutting down Lumen Web API")
    await app_state.cleanup()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Lumen Web API",
        description="Web API for managing Lumen AI services",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(config_router, prefix="/api/v1/config", tags=["config"])
    app.include_router(hardware_router, prefix="/api/v1/hardware", tags=["hardware"])
    app.include_router(install_router, prefix="/api/v1/install", tags=["install"])
    app.include_router(server_router, prefix="/api/v1/server", tags=["server"])
    app.include_router(logs_ws_router, prefix="/ws", tags=["websocket"])

    # Static files for React frontend (in production)
    # app.mount("/", StaticFiles(directory="static", html=True), name="static")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "version": "0.1.0"}

    return app


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
):
    """Start the uvicorn server."""
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "lumen_app.web.main:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        factory=True,
    )


def start_app():
    """Entry point for lumen-web command."""
    import argparse

    parser = argparse.ArgumentParser(description="Lumen Web API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()
    start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    start_app()
