"""SQLite-backed storage for Imprint."""

from pathlib import Path

import aiosqlite

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    user_id         TEXT,
    type            TEXT NOT NULL,
    scope           TEXT NOT NULL,
    domain          TEXT,
    content         TEXT NOT NULL,
    applicability   TEXT,
    context_keys    TEXT,
    context_stats   TEXT,
    source          TEXT NOT NULL,
    stability       REAL NOT NULL DEFAULT 5.0,
    valid_from      TEXT NOT NULL,
    valid_until     TEXT,
    superseded_by   TEXT REFERENCES memories(id),
    pinned          INTEGER NOT NULL DEFAULT 0,
    active          INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_triggered  TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_agent_user
    ON memories(agent_id, user_id, active);

CREATE TABLE IF NOT EXISTS signals (
    id                TEXT PRIMARY KEY,
    agent_id          TEXT NOT NULL,
    user_id           TEXT,
    signal_type       TEXT NOT NULL,
    content           TEXT NOT NULL,
    prediction_delta  TEXT,
    context           TEXT,
    memory_id         TEXT REFERENCES memories(id),
    contradicted      INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id  TEXT NOT NULL REFERENCES memories(id),
    signal_id  TEXT NOT NULL REFERENCES signals(id),
    weight     REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (memory_id, signal_id)
);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn: aiosqlite.Connection | None = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Store is not connected; call connect() first")
        return self._conn

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(self.path)
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is None:
            return
        await self._conn.close()
        self._conn = None

    async def init_schema(self) -> None:
        await self.conn.executescript(_SCHEMA_SQL)
        await self.conn.commit()
