"""Server-specific database schema for imprint-server.

The library (imprint-mem) owns its own tables via MemoryStore.init_schema().
This module creates the four additional tables the server needs on top of that.
Call init_server_schema() once at startup, after the store is connected and the
library schema is initialized.

Tables:
  sessions      -- durable MemoryLoop state for HTTP sessions
  jobs          -- maintenance job queue (Postgres: SELECT FOR UPDATE SKIP LOCKED)
  api_keys      -- API key hashes for auth (when IMPRINT_AUTH_DISABLED=false)
  policy_events -- counterfactual log: every get_policy() call logged here

SQLite mode:  TEXT timestamps, INTEGER booleans, TEXT for JSON payloads.
              Opens a separate aiosqlite connection to the same file.
              Works correctly for file-based SQLite (WAL mode, concurrent readers).
              Not compatible with :memory: (each connection is a separate DB).

Postgres mode: TIMESTAMPTZ, BOOLEAN, JSONB native types.
               Uses the shared asyncpg pool from PostgresMemoryStore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imprint.protocols import MemoryStore

    from imprint_server.config import ServerConfig

# -- SQL: Postgres ------------------------------------------------------------

_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    context       TEXT,
    retrieved_ids TEXT,
    alpha_used    REAL DEFAULT 0.3,
    outcome       REAL,
    correction    TEXT,
    opened_at     TIMESTAMPTZ NOT NULL,
    expires_at    TIMESTAMPTZ NOT NULL,
    closed_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_user
    ON sessions(agent_id, user_id, closed_at);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    user_id       TEXT,
    job_type      TEXT NOT NULL,
    payload       JSONB,
    status        TEXT DEFAULT 'pending',
    priority      INT DEFAULT 5,
    created_at    TIMESTAMPTZ NOT NULL,
    locked_at     TIMESTAMPTZ,
    locked_by     TEXT,
    completed_at  TIMESTAMPTZ,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim
    ON jobs(status, priority DESC, created_at)
    WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS api_keys (
    key_hash      TEXT PRIMARY KEY,
    agent_id      TEXT,
    label         TEXT,
    created_at    TIMESTAMPTZ NOT NULL,
    expires_at    TIMESTAMPTZ,
    active        BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS policy_events (
    id            TEXT PRIMARY KEY,
    session_id    TEXT,
    agent_id      TEXT NOT NULL,
    user_id       TEXT,
    retrieved_ids TEXT NOT NULL,
    filtered_ids  TEXT NOT NULL,
    alpha_used    REAL NOT NULL,
    context_hash  TEXT,
    occurred_at   TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_policy_events_agent
    ON policy_events(agent_id, occurred_at DESC);
"""

# -- SQL: SQLite --------------------------------------------------------------

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    context       TEXT,
    retrieved_ids TEXT,
    alpha_used    REAL DEFAULT 0.3,
    outcome       REAL,
    correction    TEXT,
    opened_at     TEXT NOT NULL,
    expires_at    TEXT NOT NULL,
    closed_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_user
    ON sessions(agent_id, user_id, closed_at);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    user_id       TEXT,
    job_type      TEXT NOT NULL,
    payload       TEXT,
    status        TEXT DEFAULT 'pending',
    priority      INTEGER DEFAULT 5,
    created_at    TEXT NOT NULL,
    locked_at     TEXT,
    locked_by     TEXT,
    completed_at  TEXT,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim
    ON jobs(status, priority DESC, created_at)
    WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS api_keys (
    key_hash      TEXT PRIMARY KEY,
    agent_id      TEXT,
    label         TEXT,
    created_at    TEXT NOT NULL,
    expires_at    TEXT,
    active        INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS policy_events (
    id            TEXT PRIMARY KEY,
    session_id    TEXT,
    agent_id      TEXT NOT NULL,
    user_id       TEXT,
    retrieved_ids TEXT NOT NULL,
    filtered_ids  TEXT NOT NULL,
    alpha_used    REAL NOT NULL,
    context_hash  TEXT,
    occurred_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_policy_events_agent
    ON policy_events(agent_id, occurred_at DESC);
"""


# -- Public interface ---------------------------------------------------------


async def init_server_schema(config: ServerConfig, store: MemoryStore) -> None:
    """Create server-specific tables if they do not exist.

    Safe to call multiple times (CREATE TABLE IF NOT EXISTS + CREATE INDEX IF NOT
    EXISTS throughout). Call after MemoryStore.init_schema() so the library tables
    are already present.
    """
    if config.is_postgres:
        await _init_postgres(store)
    else:
        await _init_sqlite(config.store)


async def _init_postgres(store: MemoryStore) -> None:
    from imprint.stores.postgres import PostgresMemoryStore

    pg = store  # type: ignore[assignment]
    pg_store: PostgresMemoryStore = pg  # type: ignore[assignment]
    # Execute each statement individually -- asyncpg does not support multi-statement
    # strings in pool.execute(). Split on ";" and run each non-empty statement.
    for stmt in _POSTGRES_DDL.split(";"):
        stmt = stmt.strip()
        if stmt:
            await pg_store.pool.execute(stmt)  # type: ignore[reportUnknownMemberType]


async def _init_sqlite(store_url: str) -> None:
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(store_url)
    async with aiosqlite.connect(path) as conn:
        await conn.executescript(_SQLITE_DDL)
        await conn.commit()
