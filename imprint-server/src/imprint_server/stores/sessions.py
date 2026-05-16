"""CRUD for the sessions table.

Sessions persist MemoryLoop state across HTTP requests so the policy call
and the close call can happen in separate requests (or on different uvicorn
workers in a multi-worker Postgres deployment).

Session lifecycle:
  1. POST /sessions            -> create_session() -> session_id
  2. POST /sessions/{id}/policy -> update_session_policy() after get_policy()
  3. POST /sessions/{id}/close  -> get_session() + close_session()

The sessions row stores:
  retrieved_ids  JSON array of memory IDs from the last get_policy() call.
  alpha_used     The hybrid retrieval alpha from the last get_policy() call.
  context        The context string from the last get_policy() call.

All three are updated atomically on each policy call so that if policy is
called multiple times within a session, close() always uses the most recent
retrieval state (matching the library's MemoryLoop.retrieved_ids behaviour,
which is also overwritten on each get_policy() call).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig


@dataclass
class SessionRow:
    id: str
    agent_id: str
    user_id: str
    context: str | None
    retrieved_ids: list[str]
    alpha_used: float
    outcome: float | None
    correction: str | None
    opened_at: datetime
    expires_at: datetime
    closed_at: datetime | None


async def create_session(
    config: ServerConfig,
    *,
    agent_id: str,
    user_id: str,
    context: str | None,
) -> str:
    """Insert a new session row and return the session_id."""
    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=config.session_ttl)

    if config.is_postgres:
        # We need the pool. The registry is not available here, so this module
        # accepts the raw pool when called from route handlers that have registry.
        # This overload is handled by the route handler passing pg_store.pool.
        raise NotImplementedError(
            "Postgres create_session must be called via _pg_create_session(pool, ...)"
        )
    else:
        import aiosqlite

        from imprint_server._utils import sqlite_file_path

        path = sqlite_file_path(config.store)
        async with aiosqlite.connect(path) as conn:
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute(
                "INSERT INTO sessions"
                " (id, agent_id, user_id, context, retrieved_ids, alpha_used,"
                " opened_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    agent_id,
                    user_id,
                    context,
                    "[]",
                    0.0,
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            await conn.commit()
    return session_id


async def get_session(config: ServerConfig, session_id: str) -> SessionRow | None:
    """Return the session row or None if not found."""
    if config.is_postgres:
        raise NotImplementedError(
            "Postgres get_session must be called via _pg_get_session(pool, ...)"
        )
    else:
        import aiosqlite

        from imprint_server._utils import sqlite_file_path

        path = sqlite_file_path(config.store)
        async with aiosqlite.connect(path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_session(dict(row), postgres=False)


async def update_session_policy(
    config: ServerConfig,
    session_id: str,
    *,
    retrieved_ids: list[str],
    alpha_used: float,
    context: str | None,
) -> None:
    """Update retrieved_ids, alpha_used, context after a get_policy() call."""
    retrieved_json = json.dumps(retrieved_ids)

    if config.is_postgres:
        raise NotImplementedError("Postgres update_session_policy must be called via pg variant")
    else:
        import aiosqlite

        from imprint_server._utils import sqlite_file_path

        path = sqlite_file_path(config.store)
        async with aiosqlite.connect(path) as conn:
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute(
                "UPDATE sessions SET retrieved_ids = ?, alpha_used = ?, context = ? WHERE id = ?",
                (retrieved_json, alpha_used, context, session_id),
            )
            await conn.commit()


async def close_session(
    config: ServerConfig,
    session_id: str,
    *,
    outcome: float | None,
    correction: str | None,
) -> None:
    """Mark session as closed."""
    now = datetime.now(UTC)

    if config.is_postgres:
        raise NotImplementedError("Postgres close_session must be called via pg variant")
    else:
        import aiosqlite

        from imprint_server._utils import sqlite_file_path

        path = sqlite_file_path(config.store)
        async with aiosqlite.connect(path) as conn:
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute(
                "UPDATE sessions SET closed_at = ?, outcome = ?, correction = ? WHERE id = ?",
                (now.isoformat(), outcome, correction, session_id),
            )
            await conn.commit()


# -- Postgres variants (called from route handlers with pool) -----------------


async def pg_create_session(
    pool: object,
    *,
    agent_id: str,
    user_id: str,
    context: str | None,
    ttl: int,
) -> str:
    """Insert a session row using an asyncpg pool. Returns session_id."""
    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=ttl)
    await pool.execute(  # type: ignore[union-attr,reportUnknownMemberType]
        "INSERT INTO sessions"
        " (id, agent_id, user_id, context, retrieved_ids, alpha_used, opened_at, expires_at)"
        " VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        session_id,
        agent_id,
        user_id,
        context,
        "[]",
        0.0,
        now,
        expires_at,
    )
    return session_id


async def pg_get_session(pool: object, session_id: str) -> SessionRow | None:
    """Fetch a session row using an asyncpg pool."""
    row = await pool.fetchrow(  # type: ignore[union-attr,reportUnknownMemberType]
        "SELECT * FROM sessions WHERE id = $1", session_id
    )
    if row is None:
        return None
    return _row_to_session(dict(row), postgres=True)


async def pg_update_session_policy(
    pool: object,
    session_id: str,
    *,
    retrieved_ids: list[str],
    alpha_used: float,
    context: str | None,
) -> None:
    retrieved_json = json.dumps(retrieved_ids)
    await pool.execute(  # type: ignore[union-attr,reportUnknownMemberType]
        "UPDATE sessions SET retrieved_ids = $1, alpha_used = $2, context = $3 WHERE id = $4",
        retrieved_json,
        alpha_used,
        context,
        session_id,
    )


async def pg_close_session(
    pool: object,
    session_id: str,
    *,
    outcome: float | None,
    correction: str | None,
) -> None:
    now = datetime.now(UTC)
    await pool.execute(  # type: ignore[union-attr,reportUnknownMemberType]
        "UPDATE sessions SET closed_at = $1, outcome = $2, correction = $3 WHERE id = $4",
        now,
        outcome,
        correction,
        session_id,
    )


# -- Internal -----------------------------------------------------------------


def _row_to_session(row: dict[str, object], *, postgres: bool) -> SessionRow:
    def _dt(v: object) -> datetime:
        if isinstance(v, datetime):
            return v if v.tzinfo is not None else v.replace(tzinfo=UTC)
        return datetime.fromisoformat(str(v)).replace(tzinfo=UTC)

    def _dt_opt(v: object) -> datetime | None:
        return None if v is None else _dt(v)

    retrieved_raw = row.get("retrieved_ids") or "[]"
    retrieved: list[str] = json.loads(retrieved_raw)

    return SessionRow(
        id=str(row["id"]),
        agent_id=str(row["agent_id"]),
        user_id=str(row["user_id"]),
        context=row.get("context"),
        retrieved_ids=retrieved,
        alpha_used=float(row.get("alpha_used") or 0.0),
        outcome=row.get("outcome"),
        correction=row.get("correction"),
        opened_at=_dt(row["opened_at"]),
        expires_at=_dt(row["expires_at"]),
        closed_at=_dt_opt(row.get("closed_at")),
    )
