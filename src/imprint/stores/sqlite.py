"""SQLite-backed storage and event logging for Imprint."""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import AwareDatetime

from imprint.types import (
    Memory,
    MemoryDiff,
    MemoryEvent,
    MemorySource,
    MemoryType,
    Signal,
    SignalType,
    SupersededPair,
)


@dataclass
class _AgentConfig:
    processing_mode: str | None
    agent_description: str | None
    scopes: list[str] | None
    alpha_tuner_state: str | None
    gradient_state: str | None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    user_id         TEXT,
    type            TEXT NOT NULL,
    scope           TEXT NOT NULL,
    content         TEXT NOT NULL,
    source          TEXT NOT NULL,
    stability       REAL NOT NULL DEFAULT 5.0,
    recall_count    INTEGER NOT NULL DEFAULT 0,
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

CREATE INDEX IF NOT EXISTS idx_memories_scope
    ON memories(agent_id, scope, active);

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

CREATE INDEX IF NOT EXISTS idx_signals_memory_id
    ON signals(memory_id);

CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id  TEXT NOT NULL REFERENCES memories(id),
    signal_id  TEXT NOT NULL REFERENCES signals(id),
    weight     REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (memory_id, signal_id)
);

CREATE TABLE IF NOT EXISTS compiled_policies (
    cache_key    TEXT PRIMARY KEY,
    agent_id     TEXT NOT NULL,
    user_id      TEXT,
    policy_text  TEXT NOT NULL,
    compiled_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_compiled_policies_agent_user
    ON compiled_policies(agent_id, user_id);

CREATE TABLE IF NOT EXISTS agent_config (
    agent_id           TEXT PRIMARY KEY,
    processing_mode    TEXT,
    agent_description  TEXT,
    scopes             TEXT,
    alpha_tuner_state  TEXT,
    gradient_state     TEXT,
    updated_at         TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize='unicode61 remove_diacritics 1'
);

CREATE TABLE IF NOT EXISTS memory_events (
    id           TEXT PRIMARY KEY,
    memory_id    TEXT NOT NULL REFERENCES memories(id),
    event_type   TEXT NOT NULL,
    occurred_at  TEXT NOT NULL,
    metadata     TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_events_memory
    ON memory_events(memory_id);

CREATE INDEX IF NOT EXISTS idx_memory_events_time
    ON memory_events(occurred_at);

CREATE TABLE IF NOT EXISTS scopes (
    agent_id    TEXT NOT NULL,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (agent_id, name)
);
"""

_INSERT_MEMORY_SQL = """
INSERT INTO memories (
    id, agent_id, user_id, type, scope, content,
    source, stability, recall_count,
    valid_from, valid_until, superseded_by, pinned, active,
    created_at, updated_at, last_triggered
) VALUES (
    :id, :agent_id, :user_id, :type, :scope, :content,
    :source, :stability, :recall_count,
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
        "content": m.content,
        "source": m.source.value,
        "stability": m.stability,
        "recall_count": m.recall_count,
        "valid_from": m.valid_from.isoformat(),
        "valid_until": m.valid_until.isoformat() if m.valid_until else None,
        "superseded_by": m.superseded_by,
        "pinned": int(m.pinned),
        "active": int(m.active),
        "created_at": m.created_at.isoformat(),
        "updated_at": m.updated_at.isoformat(),
        "last_triggered": m.last_triggered.isoformat() if m.last_triggered else None,
    }


def _row_to_memory(row: Any) -> Memory:
    return Memory(
        id=row["id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        type=MemoryType(row["type"]),
        scope=row["scope"],
        content=row["content"],
        source=MemorySource(row["source"]),
        stability=row["stability"],
        recall_count=row["recall_count"],
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


def _row_to_signal(row: Any) -> Signal:
    return Signal(
        id=row["id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        signal_type=SignalType(row["signal_type"]),
        content=row["content"],
        prediction_delta=row["prediction_delta"],
        context=row["context"],
        memory_id=row["memory_id"],
        contradicted=bool(row["contradicted"]),
        created_at=datetime.fromisoformat(row["created_at"]),
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


def _parse_iso_dt(value: str) -> datetime:
    """Parse an ISO datetime string, ensuring the result is timezone-aware."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class SQLiteMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn: aiosqlite.Connection | None = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("store is not connected; call connect() first")
        return self._conn

    async def connect(self) -> None:
        if self._conn is not None:
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        # WAL mode: concurrent reads, serialized writes, better write throughput.
        # synchronous=NORMAL: safe with WAL (last checkpoint is durable), ~3x faster
        # than FULL. busy_timeout: wait rather than fail immediately if another
        # connection holds a write lock -- prevents spurious errors on accidental
        # multi-process access (note: multi-process access is still not supported;
        # in-process state like feedback loops is not shared across processes).
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.execute("PRAGMA synchronous = NORMAL")
        await self._conn.execute("PRAGMA busy_timeout = 5000")
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is None:
            return
        await self._conn.close()
        self._conn = None

    async def init_schema(self) -> None:
        await self.conn.executescript(_SCHEMA_SQL)
        await self.conn.commit()
        try:
            await self.conn.execute(
                "ALTER TABLE memories ADD COLUMN recall_count INTEGER NOT NULL DEFAULT 0"
            )
            await self.conn.commit()
        except Exception:
            pass
        try:
            await self.conn.execute(
                "ALTER TABLE agent_config RENAME COLUMN detection_mode TO processing_mode"
            )
            await self.conn.commit()
        except Exception:
            pass
        try:
            await self.conn.execute("ALTER TABLE agent_config ADD COLUMN alpha_tuner_state TEXT")
            await self.conn.commit()
        except Exception:
            pass
        try:
            await self.conn.execute("ALTER TABLE agent_config ADD COLUMN gradient_state TEXT")
            await self.conn.commit()
        except Exception:
            pass
        # Remove columns that existed in pre-1.0 versions but are no longer used.
        for col in ("domain", "applicability", "context_keys", "context_stats"):
            try:
                await self.conn.execute(f"ALTER TABLE memories DROP COLUMN {col}")
                await self.conn.commit()
            except Exception:
                pass

    async def insert_memory(self, memory: Memory) -> None:
        await self.conn.execute(_INSERT_MEMORY_SQL, _memory_to_params(memory))
        await self.conn.execute(
            "INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)",
            (memory.id, memory.content),
        )
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
        scopes: list[str] | None = None,
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

        if scopes is not None:
            placeholders: list[str] = []
            for i, scope in enumerate(scopes):
                key = f"scope_{i}"
                params[key] = scope
                placeholders.append(f":{key}")
            if placeholders:
                clauses.append(f"(scope = 'global' OR scope IN ({', '.join(placeholders)}))")
            else:
                clauses.append("scope = 'global'")

        if active_only:
            clauses.append("active = 1")

        sql = "SELECT * FROM memories WHERE " + " AND ".join(clauses) + " ORDER BY created_at"
        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_memory(row) for row in rows]

    async def diff_memories(
        self,
        agent_id: str,
        user_id: str,
        since: AwareDatetime,
        until: AwareDatetime,
    ) -> MemoryDiff:
        since_s = since.isoformat()
        until_s = until.isoformat()

        # Memories created in the window that are still active.
        cursor = await self.conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND user_id = ?"
            " AND created_at >= ? AND created_at <= ? AND active = 1"
            " ORDER BY created_at",
            (agent_id, user_id, since_s, until_s),
        )
        added = [_row_to_memory(r) for r in await cursor.fetchall()]

        # Memories deactivated in the window with no replacement.
        cursor = await self.conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND user_id = ?"
            " AND updated_at >= ? AND updated_at <= ? AND active = 0"
            " AND superseded_by IS NULL ORDER BY updated_at",
            (agent_id, user_id, since_s, until_s),
        )
        deactivated = [_row_to_memory(r) for r in await cursor.fetchall()]

        # Memories superseded in the window -- fetch old memories first.
        cursor = await self.conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND user_id = ?"
            " AND updated_at >= ? AND updated_at <= ? AND active = 0"
            " AND superseded_by IS NOT NULL ORDER BY updated_at",
            (agent_id, user_id, since_s, until_s),
        )
        old_memories = [_row_to_memory(r) for r in await cursor.fetchall()]

        superseded: list[SupersededPair] = []
        if old_memories:
            ids = [m.superseded_by for m in old_memories if m.superseded_by]
            placeholders = ",".join("?" * len(ids))
            cursor = await self.conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
            )
            new_by_id = {_row_to_memory(r).id: _row_to_memory(r) for r in await cursor.fetchall()}
            for old in old_memories:
                if old.superseded_by and old.superseded_by in new_by_id:
                    superseded.append(SupersededPair(old=old, new=new_by_id[old.superseded_by]))

        return MemoryDiff(
            since=since, until=until, added=added, deactivated=deactivated, superseded=superseded
        )

    async def list_scopes(self, agent_id: str) -> list[str]:
        """Return all registered non-global scopes for this agent."""
        cursor = await self.conn.execute(
            "SELECT name FROM scopes WHERE agent_id = ? ORDER BY created_at",
            (agent_id,),
        )
        rows = await cursor.fetchall()
        return [row["name"] for row in rows]

    async def list_active_scopes_for_user(self, agent_id: str, user_id: str) -> list[str]:
        """Return distinct non-global scopes from active memories for this user.

        Used for scope routing during derivation and inference. Returns only
        scopes this user actually has memories in, keeping other users' scope
        names invisible during derivation.
        """
        cursor = await self.conn.execute(
            "SELECT DISTINCT scope FROM memories "
            "WHERE agent_id = ? AND user_id = ? AND active = 1 AND scope != 'global' "
            "ORDER BY scope",
            (agent_id, user_id),
        )
        rows = await cursor.fetchall()
        return [row["scope"] for row in rows]

    async def insert_scope(self, agent_id: str, name: str) -> None:
        """Register a new scope. No-ops if it already exists."""
        now = datetime.now(UTC).isoformat()
        await self.conn.execute(
            "INSERT OR IGNORE INTO scopes (agent_id, name, created_at) VALUES (?, ?, ?)",
            (agent_id, name, now),
        )
        await self.conn.commit()

    async def clear_scopes(self, agent_id: str) -> None:
        """Remove all registered scopes for this agent.

        Used when constructor-provided scopes override stored ones.
        """
        await self.conn.execute(
            "DELETE FROM scopes WHERE agent_id = ?",
            (agent_id,),
        )
        await self.conn.commit()

    async def rename_scope(self, agent_id: str, old_name: str, new_name: str) -> None:
        """Rename a scope and update all memories that reference it."""
        now = datetime.now(UTC).isoformat()
        await self.conn.execute(
            "INSERT OR IGNORE INTO scopes (agent_id, name, created_at) VALUES (?, ?, ?)",
            (agent_id, new_name, now),
        )
        await self.conn.execute(
            "UPDATE memories SET scope = ?, updated_at = ? WHERE agent_id = ? AND scope = ?",
            (new_name, now, agent_id, old_name),
        )
        await self.conn.execute(
            "DELETE FROM scopes WHERE agent_id = ? AND name = ?",
            (agent_id, old_name),
        )
        await self.conn.commit()

    async def merge_scopes(self, agent_id: str, from_scope: str, into_scope: str) -> None:
        """Move all memories from from_scope into into_scope, remove from_scope."""
        now = datetime.now(UTC).isoformat()
        await self.conn.execute(
            "UPDATE memories SET scope = ?, updated_at = ? WHERE agent_id = ? AND scope = ?",
            (into_scope, now, agent_id, from_scope),
        )
        await self.conn.execute(
            "DELETE FROM scopes WHERE agent_id = ? AND name = ?",
            (agent_id, from_scope),
        )
        await self.conn.commit()

    async def update_memory_scope(self, memory_id: str, new_scope: str) -> None:
        """Reassign one memory to a different scope."""
        now = datetime.now(UTC).isoformat()
        await self.conn.execute(
            "UPDATE memories SET scope = ?, updated_at = ? WHERE id = ?",
            (new_scope, now, memory_id),
        )
        await self.conn.commit()

    async def deactivate_memory(
        self,
        memory_id: str,
        *,
        superseded_by: str | None = None,
        valid_until: datetime | None = None,
    ) -> bool:
        now_iso = datetime.now(UTC).isoformat()
        cursor = await self.conn.execute(
            "UPDATE memories SET active = 0, superseded_by = :superseded_by, "
            "valid_until = :valid_until, updated_at = :updated_at WHERE id = :id AND active = 1",
            {
                "id": memory_id,
                "superseded_by": superseded_by,
                "valid_until": valid_until.isoformat() if valid_until else None,
                "updated_at": now_iso,
            },
        )
        found = cursor.rowcount > 0
        if found:
            await self.conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        await self.conn.commit()
        return found

    async def mark_signals_contradicted(self, memory_id: str) -> None:
        await self.conn.execute(
            "UPDATE signals SET contradicted = 1 WHERE id IN ("
            "SELECT signal_id FROM memory_sources WHERE memory_id = :m"
            ")",
            {"m": memory_id},
        )
        await self.conn.commit()

    async def get_cached_policy(self, cache_key: str) -> tuple[str, datetime] | None:
        cursor = await self.conn.execute(
            "SELECT policy_text, compiled_at FROM compiled_policies WHERE cache_key = :k",
            {"k": cache_key},
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return row["policy_text"], datetime.fromisoformat(row["compiled_at"])

    async def put_cached_policy(
        self,
        *,
        cache_key: str,
        agent_id: str,
        user_id: str | None,
        policy_text: str,
        compiled_at: datetime,
    ) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO compiled_policies "
            "(cache_key, agent_id, user_id, policy_text, compiled_at) "
            "VALUES (:k, :agent_id, :user_id, :text, :compiled_at)",
            {
                "k": cache_key,
                "agent_id": agent_id,
                "user_id": user_id,
                "text": policy_text,
                "compiled_at": compiled_at.isoformat(),
            },
        )
        await self.conn.commit()

    async def invalidate_cached_policies(self, agent_id: str, user_id: str | None) -> None:
        if user_id is None:
            await self.conn.execute(
                "DELETE FROM compiled_policies WHERE agent_id = :a AND user_id IS NULL",
                {"a": agent_id},
            )
        else:
            await self.conn.execute(
                "DELETE FROM compiled_policies WHERE agent_id = :a AND user_id = :u",
                {"a": agent_id, "u": user_id},
            )
        await self.conn.commit()

    async def search_fts(
        self,
        query: str,
        candidate_ids: set[str],
        limit: int = 200,
    ) -> list[tuple[str, float]]:
        """BM25 search over active memory content via FTS5.

        Returns (memory_id, rank) pairs ordered by relevance (best first).
        rank is the raw FTS5 rank value (negative; lower = more relevant).
        Only returns results whose memory_id is in candidate_ids.
        """
        if not query or not candidate_ids:
            return []
        # FTS5 MATCH treats the query as an expression. Apostrophes, quotes,
        # and parentheses break its syntax. Strip non-word chars, keep words.
        import re as _re

        safe_query = _re.sub(r"[^\w\s]", " ", query).strip()
        if not safe_query:
            return []
        cursor = await self.conn.execute(
            "SELECT memory_id, rank FROM memories_fts WHERE content MATCH ? ORDER BY rank LIMIT ?",
            (safe_query, limit),
        )
        rows = await cursor.fetchall()
        return [
            (row["memory_id"], float(row["rank"]))
            for row in rows
            if row["memory_id"] in candidate_ids
        ]

    async def put_alpha_tuner_state(self, agent_id: str, state: str) -> None:
        await self.conn.execute(
            "UPDATE agent_config SET alpha_tuner_state = ? WHERE agent_id = ?",
            (state, agent_id),
        )
        await self.conn.commit()

    async def put_gradient_state(self, agent_id: str, state: str) -> None:
        await self.conn.execute(
            "UPDATE agent_config SET gradient_state = ? WHERE agent_id = ?",
            (state, agent_id),
        )
        await self.conn.commit()

    async def get_agent_config(self, agent_id: str) -> _AgentConfig | None:
        cursor = await self.conn.execute(
            "SELECT processing_mode, agent_description, scopes, alpha_tuner_state, gradient_state "
            "FROM agent_config WHERE agent_id = :a",
            {"a": agent_id},
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _AgentConfig(
            processing_mode=row["processing_mode"],
            agent_description=row["agent_description"],
            scopes=json.loads(row["scopes"]) if row["scopes"] is not None else None,
            alpha_tuner_state=row["alpha_tuner_state"],
            gradient_state=row["gradient_state"],
        )

    async def put_agent_config(
        self,
        *,
        agent_id: str,
        processing_mode: str,
        agent_description: str | None,
        scopes: list[str],
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await self.conn.execute(
            "INSERT OR REPLACE INTO agent_config "
            "(agent_id, processing_mode, agent_description, scopes, updated_at) "
            "VALUES (:a, :dm, :desc, :scopes, :now)",
            {
                "a": agent_id,
                "dm": processing_mode,
                "desc": agent_description,
                "scopes": json.dumps(scopes),
                "now": now,
            },
        )
        await self.conn.commit()

    async def set_pinned(self, memory_id: str, pinned: bool) -> None:
        now_iso = datetime.now(UTC).isoformat()
        await self.conn.execute(
            "UPDATE memories SET pinned = :pinned, updated_at = :now WHERE id = :id",
            {"pinned": int(pinned), "now": now_iso, "id": memory_id},
        )
        await self.conn.commit()

    async def update_memory_stability(self, memory_id: str, stability: float) -> None:
        await self.conn.execute(
            "UPDATE memories SET stability = :s WHERE id = :id",
            {"s": stability, "id": memory_id},
        )
        await self.conn.commit()

    async def increment_recall_count(self, memory_id: str) -> None:
        now_iso = datetime.now(UTC).isoformat()
        await self.conn.execute(
            "UPDATE memories SET recall_count = recall_count + 1, "
            "last_triggered = :now WHERE id = :id",
            {"now": now_iso, "id": memory_id},
        )
        await self.conn.commit()

    async def increment_recall_count_batch(self, memory_ids: list[str]) -> None:
        """Increment recall_count and update last_triggered for a batch of memories.

        One query instead of N round-trips. Used by _apply_recall() after
        every get_policy() call.
        """
        if not memory_ids:
            return
        now_iso = datetime.now(UTC).isoformat()
        placeholders = ",".join("?" * len(memory_ids))
        await self.conn.execute(
            f"UPDATE memories SET recall_count = recall_count + 1, "
            f"last_triggered = ? WHERE id IN ({placeholders})",
            [now_iso, *memory_ids],
        )
        await self.conn.commit()

    async def list_events(
        self,
        agent_id: str,
        user_id: str | None,
        *,
        memory_id: str | None = None,
        limit: int = 50,
    ) -> list[MemoryEvent]:
        """Return events for a user's memories, newest first.

        If memory_id is given, returns events for that memory only.
        Otherwise returns events across all of the user's memories (joined
        through the memories table on agent_id and user_id).
        """
        if memory_id is not None:
            cursor = await self.conn.execute(
                "SELECT e.id, e.memory_id, e.event_type, e.occurred_at, e.metadata "
                "FROM memory_events e "
                "JOIN memories m ON m.id = e.memory_id "
                "WHERE e.memory_id = :memory_id "
                "  AND m.agent_id = :agent_id "
                "ORDER BY e.occurred_at DESC LIMIT :limit",
                {"memory_id": memory_id, "agent_id": agent_id, "limit": limit},
            )
        else:
            cursor = await self.conn.execute(
                "SELECT e.id, e.memory_id, e.event_type, e.occurred_at, e.metadata "
                "FROM memory_events e "
                "JOIN memories m ON m.id = e.memory_id "
                "WHERE m.agent_id = :agent_id AND m.user_id IS :user_id "
                "ORDER BY e.occurred_at DESC LIMIT :limit",
                {"agent_id": agent_id, "user_id": user_id, "limit": limit},
            )
        rows = await cursor.fetchall()
        return [
            MemoryEvent(
                id=row["id"],
                memory_id=row["memory_id"],
                event_type=row["event_type"],
                detail=json.loads(row["metadata"]) if row["metadata"] else None,
                occurred_at=_parse_iso_dt(row["occurred_at"]),
            )
            for row in rows
        ]

    async def delete_user_data(self, agent_id: str, user_id: str | None) -> None:
        """Hard delete all memories, signals, and events for an agent-user pair.

        Deletes in FK-safe order: memory_events and memory_sources first
        (reference memories), then memories_fts, then memories, then signals,
        then compiled_policies. All in one transaction.

        Does not touch the scopes table -- scopes are per-agent and shared
        across users. Call consolidate_scopes() afterwards if you want to
        prune scopes that have become empty.
        """
        if user_id is None:
            uid_clause = "IS NULL"
            params: dict[str, Any] = {"agent_id": agent_id}
        else:
            uid_clause = "= :user_id"
            params = {"agent_id": agent_id, "user_id": user_id}

        await self.conn.execute(
            f"DELETE FROM memory_events WHERE memory_id IN ("
            f"  SELECT id FROM memories WHERE agent_id = :agent_id AND user_id {uid_clause}"
            f")",
            params,
        )
        await self.conn.execute(
            f"DELETE FROM memory_sources WHERE memory_id IN ("
            f"  SELECT id FROM memories WHERE agent_id = :agent_id AND user_id {uid_clause}"
            f")",
            params,
        )
        await self.conn.execute(
            f"DELETE FROM memories_fts WHERE memory_id IN ("
            f"  SELECT id FROM memories WHERE agent_id = :agent_id AND user_id {uid_clause}"
            f")",
            params,
        )
        await self.conn.execute(
            f"DELETE FROM memories WHERE agent_id = :agent_id AND user_id {uid_clause}",
            params,
        )
        await self.conn.execute(
            f"DELETE FROM signals WHERE agent_id = :agent_id AND user_id {uid_clause}",
            params,
        )
        await self.conn.execute(
            f"DELETE FROM compiled_policies WHERE agent_id = :agent_id AND user_id {uid_clause}",
            params,
        )
        await self.conn.commit()

    async def get_memory_with_supersession(
        self,
        memory_id: str,
    ) -> tuple[Memory | None, Memory | None]:
        """Return (successor, predecessor) for the given memory_id.

        successor: the memory that replaced this one (via superseded_by field).
        predecessor: the most recent memory this one replaced.
        """
        cursor = await self.conn.execute("SELECT * FROM memories WHERE id = :id", {"id": memory_id})
        row = await cursor.fetchone()
        if row is None:
            return None, None

        target = _row_to_memory(row)

        successor: Memory | None = None
        if target.superseded_by:
            cursor = await self.conn.execute(
                "SELECT * FROM memories WHERE id = :id", {"id": target.superseded_by}
            )
            srow = await cursor.fetchone()
            if srow is not None:
                successor = _row_to_memory(srow)

        cursor = await self.conn.execute(
            "SELECT * FROM memories WHERE superseded_by = :id ORDER BY created_at DESC LIMIT 1",
            {"id": memory_id},
        )
        prow = await cursor.fetchone()
        predecessor: Memory | None = _row_to_memory(prow) if prow is not None else None

        return successor, predecessor

    async def get_memory(self, memory_id: str) -> Memory | None:
        cursor = await self.conn.execute("SELECT * FROM memories WHERE id = :id", {"id": memory_id})
        row = await cursor.fetchone()
        return _row_to_memory(row) if row is not None else None

    async def get_superseded_memories(self, memory_id: str) -> list[Memory]:
        """Return all memories that were superseded by the given memory_id."""
        cursor = await self.conn.execute(
            "SELECT * FROM memories WHERE superseded_by = :id", {"id": memory_id}
        )
        rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    def make_event_logger(self) -> "SQLiteEventLogger":
        """Return an EventLogger backed by this store's connection.

        Imprint.connect() calls this automatically when no explicit
        event_logger is provided.
        """
        return SQLiteEventLogger(self)

    async def get_creating_signal(self, memory_id: str) -> Signal | None:
        """Return the signal that created this memory via memory_sources, if any."""
        cursor = await self.conn.execute(
            "SELECT s.* FROM signals s "
            "JOIN memory_sources ms ON ms.signal_id = s.id "
            "WHERE ms.memory_id = :mid LIMIT 1",
            {"mid": memory_id},
        )
        row = await cursor.fetchone()
        return _row_to_signal(row) if row is not None else None


class SQLiteEventLogger:
    def __init__(self, store: SQLiteMemoryStore) -> None:
        self._store = store

    async def log(
        self,
        memory_id: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()
        await self._store.conn.execute(
            "INSERT INTO memory_events (id, memory_id, event_type, occurred_at, metadata) "
            "VALUES (:id, :memory_id, :event_type, :occurred_at, :metadata)",
            {
                "id": event_id,
                "memory_id": memory_id,
                "event_type": event_type,
                "occurred_at": now,
                "metadata": json.dumps(metadata) if metadata is not None else None,
            },
        )
        await self._store.conn.commit()


class NullEventLogger:
    async def log(
        self,
        memory_id: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass
