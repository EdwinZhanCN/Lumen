"""FastAPI application entry point for Lumen App."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware import Middleware

from lumen_app.utils.logger import get_logger

from .api.config import router as config_router
from .api.hardware import router as hardware_router
from .api.install import router as install_router
from .api.server import router as server_router
from .services.state import app_state
from .websockets.logs import router as logs_ws_router

logger = get_logger("lumen.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Lumen App API")
    await app_state.initialize()
    yield
    logger.info("Shutting down Lumen App API")
    await app_state.cleanup()


def get_webui_dist_paths() -> tuple[Path, Path, Path]:
    """Resolve the Web UI build directory paths."""
    module_dir = Path(__file__).resolve().parent
    packaged_dist = module_dir / "static"
    repo_dist = module_dir.parents[2] / "web-ui" / "dist"
    selected = packaged_dist if packaged_dist.exists() else repo_dist
    return selected, packaged_dist, repo_dist


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Lumen App API",
        description="Control plane API for managing Lumen AI services",
        version="0.1.0",
        lifespan=lifespan,
        middleware=[
            Middleware(
                cast(Any, CORSMiddleware),
                allow_origins=["*"],  # Configure for production
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        ],
    )

    # Include API routers
    app.include_router(config_router, prefix="/api/v1/config", tags=["config"])
    app.include_router(hardware_router, prefix="/api/v1/hardware", tags=["hardware"])
    app.include_router(install_router, prefix="/api/v1/install", tags=["install"])
    app.include_router(server_router, prefix="/api/v1/server", tags=["server"])
    app.include_router(logs_ws_router, prefix="/ws", tags=["websocket"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "version": "0.1.0"}

    # Static files for Web UI (production build)
    webui_dist, packaged_dist, repo_dist = get_webui_dist_paths()
    if webui_dist.exists():
        # Mount static assets
        app.mount(
            "/assets", StaticFiles(directory=str(webui_dist / "assets")), name="assets"
        )

        # Catch-all route for SPA - serve index.html for all frontend routes
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            """Serve index.html for all SPA routes."""
            file_path = webui_dist / full_path
            # Serve file if it exists (e.g., favicon.ico, robots.txt)
            if file_path.is_file():
                return FileResponse(file_path)
            # Otherwise serve index.html for SPA routing
            return FileResponse(webui_dist / "index.html")
    else:
        raise RuntimeError(
            "Web UI build not found. Looked in:\n"
            f"  - packaged: {packaged_dist}\n"
            f"  - repo: {repo_dist}\n"
            "Run the web-ui build to enable static hosting."
        )

    return app


def start_server(
    host: str = "0.0.0.0",
    port: int = 6658,
    reload: bool = False,
    workers: int = 1,
):
    """Start the uvicorn server."""
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "lumen_app.main:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        factory=True,
    )


def start_webui():
    """Entry point for lumen-webui command."""
    import argparse

    parser = argparse.ArgumentParser(description="Lumen Web UI Server")
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


def start_app():
    """Legacy entry point (kept for compatibility)."""
    start_webui()


if __name__ == "__main__":
    start_webui()
