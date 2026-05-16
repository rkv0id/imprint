"""FastAPI application factory for imprint-server.

Usage:
  config = ServerConfig()
  registry = AgentRegistry(config)
  app = create_app(config, registry)

  # In production: pass to uvicorn
  uvicorn.run(app, host=config.host, port=config.port)

  # In tests: wrap with httpx.AsyncClient(transport=ASGITransport(app=app))

The factory pattern keeps the app testable -- no global state, no module-level
side effects. Registry startup/shutdown is wired to FastAPI's ASGI lifespan so
the store is connected before the first request and drained on shutdown.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from imprint_server.api.agents import router as agents_router
from imprint_server.api.sessions import router as sessions_router
from imprint_server.errors import ImprintError, imprint_error_handler

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig
    from imprint_server.registry import AgentRegistry


def create_app(config: ServerConfig, registry: AgentRegistry) -> FastAPI:
    """Create and configure the imprint-server FastAPI application.

    Does not start the registry -- that happens inside the lifespan on first
    request (ASGI startup event). Call this function once at process start and
    pass the returned app to uvicorn.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        await registry.startup()
        try:
            yield
        finally:
            await registry.shutdown()

    app = FastAPI(
        title="imprint-server",
        version="0.1.0",
        description="Networked memory service for AI agents.",
        lifespan=lifespan,
    )

    # Store on app.state for dependency injection in route handlers.
    app.state.config = config
    app.state.registry = registry

    # CORS -- permissive by default (IMPRINT_CORS_ORIGINS=* for local use).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # RFC 9457 error handler.
    app.add_exception_handler(ImprintError, imprint_error_handler)  # type: ignore[arg-type]

    # Routers.
    app.include_router(agents_router, prefix="/v1")
    app.include_router(sessions_router, prefix="/v1")

    return app
