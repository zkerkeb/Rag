"""FastAPI application entry point for the Enterprise RAG demo."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.routes import router as api_router
from src.api.ui_routes import router as ui_router
from src.core.config import get_settings
from src.enterprise.logging import setup_logging
from src.enterprise.monitoring import MetricsMiddleware

_BASE_DIR = Path(__file__).resolve().parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Enterprise RAG Demo",
        description="Retrieval-Augmented Generation API powered by Claude and ChromaDB",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics
    if settings.monitoring.prometheus_enabled:
        app.add_middleware(MetricsMiddleware)

    # Static files & templates
    static_dir = _BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routes
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(ui_router)

    return app


app = create_app()


def run() -> None:
    settings = get_settings().server
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
    )


if __name__ == "__main__":
    run()
