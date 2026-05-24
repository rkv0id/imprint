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

Middleware stack (outermost to innermost, i.e. first added = last to run):
  CORS -> Auth -> RateLimit -> AccessLog -> RequestID
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from imprint_server.api.admin import router as admin_router
from imprint_server.api.agents import router as agents_router
from imprint_server.api.dashboard import router as dashboard_router
from imprint_server.api.health import router as health_router
from imprint_server.api.sessions import router as sessions_router
from imprint_server.auth import AuthMiddleware, maybe_generate_master_key
from imprint_server.errors import ImprintError, imprint_error_handler
from imprint_server.mcp.server import create_mcp_starlette_app
from imprint_server.middleware import AccessLogMiddleware, RequestIDMiddleware
from imprint_server.workers.metrics_refresh import MetricsRefresher
from imprint_server.workers.scheduler import Scheduler

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
        await maybe_generate_master_key(config, registry)
        scheduler = Scheduler(config, registry)
        scheduler.start()
        metrics_refresher: MetricsRefresher | None = None
        if config.metrics_extended:
            metrics_refresher = MetricsRefresher(config, registry)
            metrics_refresher.start()
        try:
            yield
        finally:
            if metrics_refresher is not None:
                await metrics_refresher.stop()
            await scheduler.stop()
            await registry.shutdown()

    _OPENAPI_TAGS = [
        {
            "name": "memory",
            "description": (
                "Core memory operations: observe exchanges, compile policies, "
                "list and search memories, correct behavior, apply learning signals."
            ),
        },
        {
            "name": "sessions",
            "description": (
                "Durable MemoryLoop sessions: open a session, observe within it, "
                "compile a tracked policy, and close with a learning signal."
            ),
        },
        {
            "name": "agents",
            "description": (
                "Agent administration: pre-configure agents, update config, "
                "drain and deregister, run scope consolidation."
            ),
        },
        {
            "name": "system",
            "description": "Health probes and Prometheus metrics.",
        },
    ]

    app = FastAPI(
        title="imprint-server",
        version="0.4.0",
        description=(
            "Networked behavioral memory service for AI agents. "
            "Stores, retrieves, and compiles per-user memory into agent policies."
        ),
        lifespan=lifespan,
        openapi_tags=_OPENAPI_TAGS,
    )

    # Store on app.state for dependency injection in route handlers.
    app.state.config = config
    app.state.registry = registry

    # Middleware is added in reverse execution order:
    # last added = outermost = first to run on the way in.

    # 1. RequestID (outermost -- runs first, so all downstream middleware
    #    can access request.state.request_id).
    app.add_middleware(RequestIDMiddleware)

    # 2. AccessLog -- logs after the response is fully formed.
    app.add_middleware(AccessLogMiddleware, config=config)

    # 3. RateLimit -- runs before auth so we can rate-limit by key or IP.
    if config.rate_limit_enabled:
        from imprint_server.middleware import RateLimitMiddleware

        app.add_middleware(
            RateLimitMiddleware,
            config=config,
            registry=registry,
        )

    # 4. Auth.
    app.add_middleware(AuthMiddleware, config=config, registry=registry)

    # 5. CORS (innermost of the middleware stack).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # RFC 9457 error handler.
    app.add_exception_handler(ImprintError, imprint_error_handler)  # type: ignore[arg-type]

    # Catch-all: log the full traceback so it appears in server logs even
    # when the default uvicorn handler would swallow it silently.
    _log = logging.getLogger("imprint_server.app")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> PlainTextResponse:  # pyright: ignore[reportUnusedFunction]
        tb = traceback.format_exc()
        _log.error(
            "Unhandled exception on %s %s -- %s: %s\n%s",
            request.method,
            request.url.path,
            type(exc).__name__,
            exc,
            tb,
        )
        return PlainTextResponse("Internal Server Error", status_code=500)

    # Routers.
    app.include_router(agents_router, prefix="/v1")
    app.include_router(sessions_router, prefix="/v1")
    app.include_router(admin_router, prefix="/v1")
    app.include_router(dashboard_router)
    app.include_router(health_router)

    # MCP SSE endpoint. Mounted as a sub-application so the SSE transport
    # handles its own routing internally. Only active when IMPRINT_MCP_AGENT_ID
    # is set. User identity is resolved per-connection from the Bearer token's
    # key.user_id field; falls back to IMPRINT_MCP_USER_ID when auth is disabled.
    if config.mcp_agent_id:
        app.mount("/mcp", create_mcp_starlette_app(config, registry))  # type: ignore[arg-type]

    return app
