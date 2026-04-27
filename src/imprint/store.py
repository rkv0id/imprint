"""SQLite-backed storage for Imprint."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from imprint.types import (
    ContextStat,
    Memory,
    MemorySource,
    MemoryType,
    Signal,
)

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

_INSERT_MEMORY_SQL = """
INSERT INTO memories (
    id, agent_id, user_id, type, scope, domain, content,
    applicability, context_keys, context_stats, source, stability,
    valid_from, valid_until, superseded_by, pinned, active,
    created_at, updated_at, last_triggered
) VALUES (
    :id, :agent_id, :user_id, :type, :scope, :domain, :content,
    :applicability, :context_keys, :context_stats, :source, :stability,
    :valid_from, :valid_until, :superseded_by, :pinned, :active,
    :created_at, :updated_at, :last_triggered
)
"""

_INSERT_SIGNAL_SQL = """
INSERT INTO signals (
    id, agent_id, user_id, signal_type, content, prediction_delta,
    context, memory_id, contradicted, created_at
) VALUES (
    :id, :agent_id, :user_id, :signal_type, :content, :prediction_delta,
    :context, :memory_id, :contradicted, :created_at
)
"""


def _memory_to_params(m: Memory) -> dict[str, Any]:
    return {
        "id": m.id,
        "agent_id": m.agent_id,
        "user_id": m.user_id,
        "type": m.type.value,
        "scope": m.scope,
        "domain": m.domain,
        "content": m.content,
        "applicability": m.applicability,
        "context_keys": json.dumps(m.context_keys),
        "context_stats": json.dumps({k: v.model_dump() for k, v in m.context_stats.items()}),
        "source": m.source.value,
        "stability": m.stability,
        "valid_from": m.valid_from.isoformat(),
        "valid_until": m.valid_until.isoformat() if m.valid_until else None,
        "superseded_by": m.superseded_by,
        "pinned": int(m.pinned),
        "active": int(m.active),
        "created_at": m.created_at.isoformat(),
        "updated_at": m.updated_at.isoformat(),
        "last_triggered": m.last_triggered.isoformat() if m.last_triggered else None,
    }


def _row_to_memory(row: aiosqlite.Row) -> Memory:
    raw_stats: dict[str, dict[str, int]] = (
        json.loads(row["context_stats"]) if row["context_stats"] else {}
    )
    return Memory(
        id=row["id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        type=MemoryType(row["type"]),
        scope=row["scope"],
        domain=row["domain"],
        content=row["content"],
        applicability=row["applicability"],
        context_keys=json.loads(row["context_keys"]) if row["context_keys"] else [],
        context_stats={k: ContextStat(**v) for k, v in raw_stats.items()},
        source=MemorySource(row["source"]),
        stability=row["stability"],
        valid_from=datetime.fromisoformat(row["valid_from"]),
        valid_until=(datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None),
        superseded_by=row["superseded_by"],
        pinned=bool(row["pinned"]),
        active=bool(row["active"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        last_triggered=(
            datetime.fromisoformat(row["last_triggered"]) if row["last_triggered"] else None
        ),
    )


def _signal_to_params(s: Signal) -> dict[str, Any]:
    return {
        "id": s.id,
        "agent_id": s.agent_id,
        "user_id": s.user_id,
        "signal_type": s.signal_type.value,
        "content": s.content,
        "prediction_delta": s.prediction_delta,
        "context": s.context,
        "memory_id": s.memory_id,
        "contradicted": int(s.contradicted),
        "created_at": s.created_at.isoformat(),
    }


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
        self._conn.row_factory = aiosqlite.Row
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

    async def insert_memory(self, memory: Memory) -> None:
        await self.conn.execute(_INSERT_MEMORY_SQL, _memory_to_params(memory))
        await self.conn.commit()

    async def insert_signal(self, signal: Signal) -> None:
        await self.conn.execute(_INSERT_SIGNAL_SQL, _signal_to_params(signal))
        await self.conn.commit()

    async def link_signal_to_memory(
        self,
        *,
        memory_id: str,
        signal_id: str,
        weight: float = 1.0,
    ) -> None:
        await self.conn.execute(
            "INSERT INTO memory_sources (memory_id, signal_id, weight) VALUES (?, ?, ?)",
            (memory_id, signal_id, weight),
        )
        await self.conn.commit()

    async def list_memories(
        self,
        agent_id: str,
        user_id: str | None,
        *,
        memory_type: MemoryType | None = None,
        active_only: bool = True,
    ) -> list[Memory]:
        clauses = ["agent_id = :agent_id"]
        params: dict[str, Any] = {"agent_id": agent_id}

        if user_id is None:
            clauses.append("user_id IS NULL")
        else:
            clauses.append("user_id = :user_id")
            params["user_id"] = user_id

        if memory_type is not None:
            clauses.append("type = :type")
            params["type"] = memory_type.value

        if active_only:
            clauses.append("active = 1")

        sql = "SELECT * FROM memories WHERE " + " AND ".join(clauses) + " ORDER BY created_at"
        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_memory(row) for row in rows]
