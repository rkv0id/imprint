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

from imprint_server.api.agents import ConfigDep, RegistryDep

router = APIRouter()


@router.get("/health/live")
async def health_live() -> dict[str, object]:
    """Liveness probe: always 200 while the process is running.

    Does not check DB or Redis. Use this for container liveness probes
    that restart the container when it hangs.
    """
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready(
    registry: RegistryDep,
    config: ConfigDep,
) -> dict[str, object]:
    """Readiness probe: 200 when all configured backends are reachable.

    Checks DB connectivity and Redis connectivity (if IMPRINT_REDIS_URL is set).
    Returns 'degraded' when any backend is unreachable. Clients should check
    the 'status' field; the HTTP status is always 200.
    """
    db_ok = await _ping_db(registry, config)
    redis_status = await _ping_redis(registry, config)
    all_ok = db_ok and redis_status != "unavailable"
    return {
        "status": "ok" if all_ok else "degraded",
        "store": "postgres" if config.is_postgres else "sqlite",
        "redis": redis_status,
        "agents_loaded": registry.agent_count,
        "db_ok": db_ok,
    }


@router.get("/health")
async def health(
    registry: RegistryDep,
    config: ConfigDep,
) -> dict[str, object]:
    """Alias for /health/ready. Preserved for backward compatibility."""
    return await health_ready(registry, config)


@router.get("/metrics")
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
