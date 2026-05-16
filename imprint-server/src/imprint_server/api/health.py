"""Health and metrics endpoints.

/health  -- DB connectivity check + server status. Does not require auth.
/metrics -- Prometheus exposition format. Does not require auth.

Both endpoints are excluded from auth middleware (step 6) so monitoring
systems can reach them without a key.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from imprint_server.api.agents import ConfigDep, RegistryDep

router = APIRouter()


@router.get("/health")
async def health(
    registry: RegistryDep,
    config: ConfigDep,
) -> dict[str, object]:
    """Return server health status and DB connectivity.

    Always returns 200 -- clients should check the 'status' field.
    'degraded' means the DB is unreachable; the server may still serve
    cached responses.
    """
    db_ok = await _ping_db(registry, config)
    return {
        "status": "ok" if db_ok else "degraded",
        "store": "postgres" if config.is_postgres else "sqlite",
        "agents_loaded": registry.agent_count,
        "db_ok": db_ok,
    }


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus exposition format metrics.

    All imprint_* metrics are defined in metrics.py and updated by
    route handlers inline.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# -- Internal -----------------------------------------------------------------


async def _ping_db(registry: RegistryDep, config: ConfigDep) -> bool:
    try:
        if config.is_postgres:
            from imprint.stores.postgres import PostgresMemoryStore

            pg_store: PostgresMemoryStore = registry.store  # type: ignore[assignment]
            await pg_store.pool.fetchval("SELECT 1")  # type: ignore[reportUnknownMemberType]
        else:
            import aiosqlite

            from imprint_server._utils import sqlite_file_path

            async with aiosqlite.connect(sqlite_file_path(config.store)) as conn:
                await conn.execute("SELECT 1")
        return True
    except Exception:
        return False
