"""Health and metrics endpoints.

/health/live   -- liveness probe: 200 while the process is up, no DB check.
/health/ready  -- readiness probe: 200 when DB + Redis (if configured) are reachable.
/health        -- alias for /health/ready, backward compatible.
/metrics       -- Prometheus exposition format.

Health and metrics endpoints are excluded from auth and rate limiting so
monitoring systems can always reach them.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from imprint_server.api.agents import ConfigDep, RegistryDep

router = APIRouter()


class LivenessResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"status": "ok"}}}

    status: str


class ReadinessResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "store": "postgres",
                "redis": "ok",
                "agents_loaded": 3,
                "db_ok": True,
            }
        }
    }

    status: str
    store: str
    redis: str
    agents_loaded: int
    db_ok: bool


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    operation_id="health_live",
    tags=["system"],
    summary="Liveness probe",
)
async def health_live() -> LivenessResponse:
    """Liveness probe: always 200 while the process is running.

    Does not check DB or Redis. Use this for container liveness probes
    that restart the container when it hangs.
    """
    return LivenessResponse(status="ok")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    operation_id="health_ready",
    tags=["system"],
    summary="Readiness probe",
)
async def health_ready(
    registry: RegistryDep,
    config: ConfigDep,
) -> ReadinessResponse:
    """Readiness probe: 200 when all configured backends are reachable.

    Checks DB connectivity and Redis connectivity (if IMPRINT_REDIS_URL is set).
    Returns 'degraded' when any backend is unreachable. Clients should check
    the 'status' field; the HTTP status is always 200.
    """
    db_ok = await _ping_db(registry, config)
    redis_status = await _ping_redis(registry, config)
    all_ok = db_ok and redis_status != "unavailable"
    return ReadinessResponse(
        status="ok" if all_ok else "degraded",
        store="postgres" if config.is_postgres else "sqlite",
        redis=redis_status,
        agents_loaded=registry.agent_count,
        db_ok=db_ok,
    )


@router.get(
    "/health",
    response_model=ReadinessResponse,
    operation_id="health",
    tags=["system"],
    summary="Health check (alias for /health/ready)",
)
async def health(
    registry: RegistryDep,
    config: ConfigDep,
) -> ReadinessResponse:
    """Alias for /health/ready. Preserved for backward compatibility."""
    return await health_ready(registry, config)


@router.get(
    "/metrics",
    operation_id="metrics",
    tags=["system"],
    summary="Prometheus exposition format metrics",
    response_class=Response,
)
async def metrics() -> Response:
    """Prometheus exposition format metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# -- Internal -----------------------------------------------------------------


async def _ping_db(registry: RegistryDep, config: ConfigDep) -> bool:
    try:
        if config.is_postgres:
            from imprint_server._pool import get_pg_pool

            await get_pg_pool(registry).fetchval("SELECT 1")
        else:
            import aiosqlite

            from imprint_server._utils import sqlite_file_path

            async with aiosqlite.connect(sqlite_file_path(config.store)) as conn:
                await conn.execute("SELECT 1")
        return True
    except Exception:
        return False


async def _ping_redis(registry: RegistryDep, config: ConfigDep) -> str:
    """Return 'ok', 'unavailable', or 'disabled'."""
    if not config.redis_enabled:
        return "disabled"
    redis = registry.redis
    if redis is None:
        return "disabled"
    reachable = await redis.ping()
    return "ok" if reachable else "unavailable"
