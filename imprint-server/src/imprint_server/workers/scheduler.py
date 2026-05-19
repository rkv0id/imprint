"""Asyncio-based maintenance scheduler for imprint-server.

Runs three recurring jobs inside the same process as the FastAPI server.
In Postgres mode, each job claims a row with SELECT FOR UPDATE SKIP LOCKED
before executing, so multiple uvicorn workers do not duplicate work. In
SQLite mode (single process), the locking step is skipped entirely.

Jobs:
  consolidate    -- prune decayed memories + scope consolidation per user.
                    Runs every IMPRINT_CONSOLIDATE_INTERVAL (default 24h).
  session_expiry -- close HTTP sessions past their expires_at.
                    Runs every 5 minutes.

Confusion-based early consolidation:
  enqueue_consolidate(config, registry, agent_id, user_id) can be called
  by route handlers to schedule an immediate consolidation for a specific
  user namespace. The observe route handler calls this when the recent
  contradiction rate exceeds IMPRINT_CONFUSION_THRESHOLD.

Population decay:
  After finalize_loop() in the session close and MCP end_session paths,
  the AgentRegistry's Imprint instance already holds the decay model that
  accumulates gradient updates from all users. Because the registry holds
  one instance per agent, population-calibrated decay is a consequence of
  the architecture, not a separate job.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from imprint_server.metrics import scheduler_job_total

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig
    from imprint_server.registry import AgentRegistry

log = logging.getLogger(__name__)

_SESSION_EXPIRY_INTERVAL = 300  # 5 minutes


# -- Public API ---------------------------------------------------------------


class Scheduler:
    """Wraps the recurring asyncio tasks for server-side maintenance."""

    def __init__(self, config: ServerConfig, registry: AgentRegistry) -> None:
        self._config = config
        self._registry = registry
        self._tasks: list[asyncio.Task[None]] = []
        self._worker_id = uuid.uuid4().hex[:8]

    def start(self) -> None:
        """Start all recurring tasks. Call from the ASGI lifespan startup."""
        self._tasks = [
            asyncio.create_task(self._run_consolidate_loop(), name="scheduler:consolidate"),
            asyncio.create_task(self._run_session_expiry_loop(), name="scheduler:session_expiry"),
        ]
        log.info("scheduler started (worker=%s)", self._worker_id)

    async def stop(self) -> None:
        """Cancel all tasks and wait for them to finish."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        log.info("scheduler stopped (worker=%s)", self._worker_id)

    # -- Consolidation loop ---------------------------------------------------

    async def _run_consolidate_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.consolidate_interval)
            await _run_job(
                self._config,
                self._registry,
                job_type="consolidate",
                worker_id=self._worker_id,
                fn=_consolidate_all,
            )

    # -- Session expiry loop --------------------------------------------------

    async def _run_session_expiry_loop(self) -> None:
        while True:
            await asyncio.sleep(_SESSION_EXPIRY_INTERVAL)
            await _run_job(
                self._config,
                self._registry,
                job_type="expire_sessions",
                worker_id=self._worker_id,
                fn=_expire_sessions,
            )


# -- Confusion-based early consolidation (called from route handlers) ---------


async def enqueue_consolidate(
    config: ServerConfig,
    registry: AgentRegistry,
    agent_id: str,
    user_id: str,
) -> None:
    """Immediately consolidate one user namespace.

    Called by the observe route handler when the recent contradiction rate
    exceeds IMPRINT_CONFUSION_THRESHOLD. Runs in a background task so the
    observe response is not delayed.
    """
    task = asyncio.create_task(
        _consolidate_user(config, registry, agent_id, user_id),
        name=f"scheduler:confusion:{agent_id}:{user_id}",
    )
    _ = task  # stored to satisfy RUF006; fire-and-forget by design


async def check_confusion_and_enqueue(
    config: ServerConfig,
    registry: AgentRegistry,
    agent_id: str,
    user_id: str,
) -> None:
    """Check the recent contradiction rate and enqueue consolidation if high.

    Reads the last IMPRINT_CONFUSION_WINDOW memory events for the user.
    If the correction/contradiction event ratio exceeds IMPRINT_CONFUSION_THRESHOLD,
    schedules an immediate consolidation.
    """
    try:
        imp = await registry.get(agent_id)
        events = await imp.list_events(user_id, limit=config.confusion_window)
        if not events:
            return
        contradiction_count = sum(
            1 for e in events if e.event_type in ("correction", "contradiction")
        )
        rate = contradiction_count / len(events)
        if rate >= config.confusion_threshold:
            log.debug(
                "confusion threshold exceeded (agent=%s user=%s rate=%.2f) enqueuing consolidation",
                agent_id,
                user_id,
                rate,
            )
            await enqueue_consolidate(config, registry, agent_id, user_id)
    except Exception as exc:
        log.warning("confusion check failed: %s", exc)


# -- Job execution ------------------------------------------------------------


async def _run_job(
    config: ServerConfig,
    registry: AgentRegistry,
    job_type: str,
    worker_id: str,
    fn: object,
) -> None:
    """Claim and execute a maintenance job, coordinated via the jobs table.

    In Postgres mode: INSERT a new pending job, then attempt SKIP LOCKED claim.
    Only the worker that claims the row executes the job. Others skip silently.
    In SQLite mode: run directly without locking (single-process invariant).
    """
    if config.is_postgres:
        await _run_postgres_job(config, registry, job_type, worker_id, fn)
    else:
        await _run_sqlite_job(config, registry, job_type, fn)


async def _run_postgres_job(
    config: ServerConfig,
    registry: AgentRegistry,
    job_type: str,
    worker_id: str,
    fn: object,
) -> None:
    from imprint.stores.postgres import PostgresMemoryStore

    pg_store: PostgresMemoryStore = registry.store  # type: ignore[assignment]
    pool = pg_store.pool
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.now(UTC)

    async with pool.acquire() as conn:  # type: ignore[reportUnknownMemberType]
        # Insert a new pending job.
        await conn.execute(  # type: ignore[reportUnknownMemberType]
            "INSERT INTO jobs (id, agent_id, job_type, status, priority, created_at)"
            " VALUES ($1, $2, $3, 'pending', 5, $4)",
            job_id,
            "__scheduler__",
            job_type,
            now,
        )
        # Attempt to claim it with SKIP LOCKED.
        row = await conn.fetchrow(  # type: ignore[reportUnknownMemberType]
            "SELECT id FROM jobs WHERE id = $1 AND status = 'pending' FOR UPDATE SKIP LOCKED",
            job_id,
        )
        if row is None:
            # Another worker claimed it -- skip.
            return
        # Mark as running.
        await conn.execute(  # type: ignore[reportUnknownMemberType]
            "UPDATE jobs SET status = 'running', locked_at = $1, locked_by = $2 WHERE id = $3",
            now,
            worker_id,
            job_id,
        )

    try:
        await fn(config, registry)  # type: ignore[operator]
        async with pool.acquire() as conn:  # type: ignore[reportUnknownMemberType]
            await conn.execute(  # type: ignore[reportUnknownMemberType]
                "UPDATE jobs SET status = 'done', completed_at = $1 WHERE id = $2",
                datetime.now(UTC),
                job_id,
            )
        scheduler_job_total.labels(job_type=job_type, status="success").inc()
        log.info("scheduler job done (type=%s worker=%s)", job_type, worker_id)
    except Exception as exc:
        async with pool.acquire() as conn:  # type: ignore[reportUnknownMemberType]
            await conn.execute(  # type: ignore[reportUnknownMemberType]
                "UPDATE jobs SET status = 'error', completed_at = $1, error = $2 WHERE id = $3",
                datetime.now(UTC),
                str(exc),
                job_id,
            )
        scheduler_job_total.labels(job_type=job_type, status="error").inc()
        log.error("scheduler job failed (type=%s): %s", job_type, exc)


async def _run_sqlite_job(
    config: ServerConfig,
    registry: AgentRegistry,
    job_type: str,
    fn: object,
) -> None:
    try:
        await fn(config, registry)  # type: ignore[operator]
        scheduler_job_total.labels(job_type=job_type, status="success").inc()
        log.info("scheduler job done (type=%s sqlite)", job_type)
    except Exception as exc:
        scheduler_job_total.labels(job_type=job_type, status="error").inc()
        log.error("scheduler job failed (type=%s): %s", job_type, exc)


# -- Job implementations ------------------------------------------------------


async def _consolidate_all(config: ServerConfig, registry: AgentRegistry) -> None:
    """Run consolidate() for every known (agent_id, user_id) pair."""
    for agent_id in registry.agent_ids():
        user_ids = await _list_user_ids(config, registry, agent_id)
        for user_id in user_ids:
            await _consolidate_user(config, registry, agent_id, user_id)


async def _consolidate_user(
    config: ServerConfig,
    registry: AgentRegistry,
    agent_id: str,
    user_id: str,
) -> None:
    try:
        imp = await registry.get(agent_id)
        async with await registry.get_op_lock(agent_id):
            pruned = await imp.consolidate(user_id)
        if pruned > 0:
            log.debug("consolidated %d memories (agent=%s user=%s)", pruned, agent_id, user_id)
    except Exception as exc:
        log.warning("consolidate failed (agent=%s user=%s): %s", agent_id, user_id, exc)


async def _expire_sessions(config: ServerConfig, registry: AgentRegistry) -> None:
    """Mark all expired open sessions as closed."""
    now = datetime.now(UTC)

    if config.is_postgres:
        from imprint.stores.postgres import PostgresMemoryStore

        pg_store: PostgresMemoryStore = registry.store  # type: ignore[assignment]
        result = await pg_store.pool.execute(  # type: ignore[reportUnknownMemberType]
            "UPDATE sessions SET closed_at = $1 WHERE closed_at IS NULL AND expires_at < $2",
            now,
            now,
        )
        expired = int(str(result).split()[-1])
    else:
        import aiosqlite

        from imprint_server._utils import sqlite_file_path

        path = sqlite_file_path(config.store)
        async with aiosqlite.connect(path) as conn:
            await conn.execute("PRAGMA busy_timeout=5000")
            cursor = await conn.execute(
                "UPDATE sessions SET closed_at = ? WHERE closed_at IS NULL AND expires_at < ?",
                (now.isoformat(), now.isoformat()),
            )
            await conn.commit()
            expired = cursor.rowcount

    if expired > 0:
        log.info("expired %d sessions", expired)


async def _list_user_ids(
    config: ServerConfig,
    registry: AgentRegistry,
    agent_id: str,
) -> list[str]:
    """Return distinct user_ids that have memories for an agent."""
    if config.is_postgres:
        from imprint.stores.postgres import PostgresMemoryStore

        pg_store: PostgresMemoryStore = registry.store  # type: ignore[assignment]
        rows = await pg_store.pool.fetch(  # type: ignore[reportUnknownMemberType]
            "SELECT DISTINCT user_id FROM memories WHERE agent_id = $1 AND active = TRUE",
            agent_id,
        )
        return [str(row["user_id"]) for row in rows]  # type: ignore[reportUnknownVariableType]
    else:
        import aiosqlite

        from imprint_server._utils import sqlite_file_path

        path = sqlite_file_path(config.store)
        async with (
            aiosqlite.connect(path) as conn,
            conn.execute(
                "SELECT DISTINCT user_id FROM memories WHERE agent_id = ? AND active = 1",
                (agent_id,),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [str(r[0]) for r in rows]  # type: ignore[reportUnknownVariableType]
