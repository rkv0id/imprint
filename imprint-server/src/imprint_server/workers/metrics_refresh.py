"""Background metrics refresh for extended per-agent gauges.

Runs when IMPRINT_METRICS_EXTENDED=true. On each cycle it queries the DB
for active memory counts and reads the alpha tuner estimate from every
loaded agent instance, then updates the Prometheus gauges.

The refresh runs on a configurable interval (IMPRINT_METRICS_REFRESH_INTERVAL,
default 60s) and never blocks the hot path. Errors in individual agent
refreshes are logged and skipped -- a failing gauge update does not affect
serving traffic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from imprint_server.metrics import bandit_alpha_estimate, memories_active

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig
    from imprint_server.registry import AgentRegistry

log = logging.getLogger(__name__)


class MetricsRefresher:
    """Periodic background task that refreshes extended per-agent gauges."""

    def __init__(self, config: ServerConfig, registry: AgentRegistry) -> None:
        self._config = config
        self._registry = registry
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the refresh loop. Call from the ASGI lifespan startup."""
        self._task = asyncio.create_task(self._run(), name="metrics_refresh")
        log.info("metrics refresh started (interval=%ds)", self._config.metrics_refresh_interval)

    async def stop(self) -> None:
        """Cancel the refresh loop and wait for it to finish."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._config.metrics_refresh_interval)
            await self._refresh_all()

    async def _refresh_all(self) -> None:
        for agent_id in self._registry.agent_ids():
            try:
                await self._refresh_agent(agent_id)
            except Exception as exc:
                log.warning("metrics refresh failed for agent %s: %s", agent_id, exc)

    async def _refresh_agent(self, agent_id: str) -> None:
        # Use _instances directly -- we only refresh agents that are already
        # loaded. We do not trigger initialization of new agents here.
        imp = self._registry._instances.get(agent_id)  # type: ignore[attr-defined]
        if imp is None:
            return

        count = await _count_active_memories(self._config, self._registry, agent_id)
        memories_active.labels(agent_id=agent_id).set(count)

        alpha = imp.alpha_estimate
        bandit_alpha_estimate.labels(agent_id=agent_id).set(alpha)


async def _count_active_memories(
    config: ServerConfig,
    registry: AgentRegistry,
    agent_id: str,
) -> int:
    """Count active memories for an agent across all users via a direct DB query."""
    if config.is_postgres:
        from imprint_server._pool import get_pg_pool

        val = await get_pg_pool(registry).fetchval(
            "SELECT COUNT(*) FROM memories WHERE agent_id = $1 AND active = TRUE",
            agent_id,
        )
        return int(val) if val is not None else 0
    else:
        import aiosqlite

        from imprint_server._utils import sqlite_file_path

        path = sqlite_file_path(config.store)
        async with (
            aiosqlite.connect(path) as conn,
            conn.execute(
                "SELECT COUNT(*) FROM memories WHERE agent_id = ? AND active = 1",
                (agent_id,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0
