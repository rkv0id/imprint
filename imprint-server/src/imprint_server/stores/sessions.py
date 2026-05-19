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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imprint_server._pool import PgPool
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


# -- SQLite -------------------------------------------------------------------

_SQLITE_COLS = (
    "id, agent_id, user_id, context, retrieved_ids, alpha_used, "
    "outcome, correction, opened_at, expires_at, closed_at"
)


async def create_session(
    config: ServerConfig,
    *,
    agent_id: str,
    user_id: str,
    context: str | None,
) -> str:
    """Insert a new session row and return the session_id."""
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=config.session_ttl)

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
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(config.store)
    async with (
        aiosqlite.connect(path) as conn,
        conn.execute(
            f"SELECT {_SQLITE_COLS} FROM sessions WHERE id = ?",
            (session_id,),
        ) as cursor,
    ):
        raw = await cursor.fetchone()

    if raw is None:
        return None
    return _sqlite_row_to_session(raw)


async def update_session_policy(
    config: ServerConfig,
    session_id: str,
    *,
    retrieved_ids: list[str],
    alpha_used: float,
    context: str | None,
) -> None:
    """Update retrieved_ids, alpha_used, context after a get_policy() call."""
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(config.store)
    async with aiosqlite.connect(path) as conn:
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.execute(
            "UPDATE sessions SET retrieved_ids = ?, alpha_used = ?, context = ? WHERE id = ?",
            (json.dumps(retrieved_ids), alpha_used, context, session_id),
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
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(config.store)
    async with aiosqlite.connect(path) as conn:
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.execute(
            "UPDATE sessions SET closed_at = ?, outcome = ?, correction = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), outcome, correction, session_id),
        )
        await conn.commit()


def _sqlite_row_to_session(raw: Any) -> SessionRow:
    """Convert a raw aiosqlite row (positional) to a SessionRow.

    Column order must match _SQLITE_COLS:
      0: id, 1: agent_id, 2: user_id, 3: context, 4: retrieved_ids,
      5: alpha_used, 6: outcome, 7: correction,
      8: opened_at, 9: expires_at, 10: closed_at
    """

    def _dt(v: Any) -> datetime:
        return datetime.fromisoformat(str(v)).replace(tzinfo=UTC)

    def _dt_opt(v: Any) -> datetime | None:
        return None if v is None else _dt(v)

    return SessionRow(
        id=str(raw[0]),
        agent_id=str(raw[1]),
        user_id=str(raw[2]),
        context=str(raw[3]) if raw[3] is not None else None,
        retrieved_ids=json.loads(str(raw[4]) if raw[4] is not None else "[]"),
        alpha_used=float(raw[5]) if raw[5] is not None else 0.0,
        outcome=float(raw[6]) if raw[6] is not None else None,
        correction=str(raw[7]) if raw[7] is not None else None,
        opened_at=_dt(raw[8]),
        expires_at=_dt(raw[9]),
        closed_at=_dt_opt(raw[10]),
    )


# -- Postgres -----------------------------------------------------------------


async def pg_create_session(
    pool: PgPool,
    *,
    agent_id: str,
    user_id: str,
    context: str | None,
    ttl: int,
) -> str:
    """Insert a session row using a PgPool. Returns session_id."""
    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=ttl)
    await pool.execute(
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


async def pg_get_session(pool: PgPool, session_id: str) -> SessionRow | None:
    """Fetch a session row using a PgPool."""
    row = await pool.fetchrow(
        "SELECT id, agent_id, user_id, context, retrieved_ids, alpha_used,"
        " outcome, correction, opened_at, expires_at, closed_at"
        " FROM sessions WHERE id = $1",
        session_id,
    )
    if row is None:
        return None
    return _pg_row_to_session(row)


async def pg_update_session_policy(
    pool: PgPool,
    session_id: str,
    *,
    retrieved_ids: list[str],
    alpha_used: float,
    context: str | None,
) -> None:
    await pool.execute(
        "UPDATE sessions SET retrieved_ids = $1, alpha_used = $2, context = $3 WHERE id = $4",
        json.dumps(retrieved_ids),
        alpha_used,
        context,
        session_id,
    )


async def pg_close_session(
    pool: PgPool,
    session_id: str,
    *,
    outcome: float | None,
    correction: str | None,
) -> None:
    await pool.execute(
        "UPDATE sessions SET closed_at = $1, outcome = $2, correction = $3 WHERE id = $4",
        datetime.now(UTC),
        outcome,
        correction,
        session_id,
    )


def _pg_row_to_session(row: dict[str, Any]) -> SessionRow:
    def _dt(v: Any) -> datetime:
        if isinstance(v, datetime):
            return v if v.tzinfo is not None else v.replace(tzinfo=UTC)
        return datetime.fromisoformat(str(v)).replace(tzinfo=UTC)

    def _dt_opt(v: Any) -> datetime | None:
        return None if v is None else _dt(v)

    return SessionRow(
        id=str(row["id"]),
        agent_id=str(row["agent_id"]),
        user_id=str(row["user_id"]),
        context=str(row["context"]) if row.get("context") is not None else None,
        retrieved_ids=json.loads(str(row["retrieved_ids"]) if row.get("retrieved_ids") else "[]"),
        alpha_used=float(row["alpha_used"]) if row.get("alpha_used") is not None else 0.0,
        outcome=float(row["outcome"]) if row.get("outcome") is not None else None,
        correction=str(row["correction"]) if row.get("correction") is not None else None,
        opened_at=_dt(row["opened_at"]),
        expires_at=_dt(row["expires_at"]),
        closed_at=_dt_opt(row.get("closed_at")),
    )
