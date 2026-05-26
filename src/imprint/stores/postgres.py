"""Postgres-backed MemoryStore and VectorStore for Imprint.

Implements the MemoryStore protocol via asyncpg with connection pooling.
pgvector is supported via PostgresVectorStore (optional -- create it explicitly
and pass to Imprint(vector_store=...) when pgvector is installed in Postgres).

Divergences from SQLiteMemoryStore:
  - Datetimes stored as TIMESTAMPTZ; asyncpg returns datetime objects directly.
    _pg_row_to_memory does not call datetime.fromisoformat().
  - Booleans stored as BOOLEAN; asyncpg returns Python bool directly.
  - REAL -> DOUBLE PRECISION.
  - FTS via a TSVECTOR generated column + partial GIN index on active rows.
    websearch_to_tsquery handles raw natural language input. No separate
    FTS table -- no manual insert/delete needed on upsert or deactivate.
  - INSERT OR IGNORE -> INSERT ... ON CONFLICT DO NOTHING.
  - INSERT OR REPLACE -> INSERT ... ON CONFLICT (...) DO UPDATE SET ...
  - Multi-statement writes (rename_scope, merge_scopes) use explicit transactions.
  - asyncpg positional parameters: $1, $2, ... (not :name style).
  - list_memories scope filter uses scope = ANY($N::text[]) instead of
    expanding individual placeholders.
  - No migration shims -- init_schema creates a fresh schema only.
    Legacy SQLite migration code (ALTER TABLE fallbacks) is not ported.

Requires: pip install imprint-mem[postgres]
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import AwareDatetime

from imprint.stores.sqlite import _AgentConfig
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

if TYPE_CHECKING:
    import asyncpg

# -- Schema -------------------------------------------------------------------

# Each element is one DDL statement. They are executed sequentially in a single
# transaction during init_schema(). Order matters: tables before indexes,
# memories before signals (FK constraint on signals.memory_id).
#
# content_tsv: TSVECTOR GENERATED ALWAYS AS (...) STORED requires Postgres 12+.
# 'simple' dictionary: unicode tokenisation + lowercasing, no stemming, no
# stopwords. Closest match to SQLite FTS5 'unicode61 remove_diacritics 1'.
#
# Partial GIN index (WHERE active = TRUE) keeps the FTS index compact --
# deactivated memories are excluded automatically. The search_fts query already
# filters candidate_ids, so deactivated memories are never returned regardless,
# but the partial index avoids indexing them in the first place.

_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS memories (
        id              TEXT PRIMARY KEY,
        agent_id        TEXT NOT NULL,
        user_id         TEXT,
        type            TEXT NOT NULL,
        scope           TEXT NOT NULL,
        content         TEXT NOT NULL,
        source          TEXT NOT NULL,
        stability       DOUBLE PRECISION NOT NULL DEFAULT 5.0,
        recall_count    INTEGER NOT NULL DEFAULT 0,
        valid_from      TIMESTAMPTZ NOT NULL,
        valid_until     TIMESTAMPTZ,
        superseded_by   TEXT REFERENCES memories(id),
        pinned          BOOLEAN NOT NULL DEFAULT FALSE,
        active          BOOLEAN NOT NULL DEFAULT TRUE,
        created_at      TIMESTAMPTZ NOT NULL,
        updated_at      TIMESTAMPTZ NOT NULL,
        last_triggered  TIMESTAMPTZ,
        content_tsv     TSVECTOR GENERATED ALWAYS AS
                            (to_tsvector('simple', content)) STORED
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_agent_user ON memories(agent_id, user_id, active)",
    "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(agent_id, scope, active)",
    # Partial GIN: only active memories. Deactivated rows are never searched.
    (
        "CREATE INDEX IF NOT EXISTS idx_memories_fts"
        " ON memories USING GIN(content_tsv) WHERE active = TRUE"
    ),
    """
    CREATE TABLE IF NOT EXISTS signals (
        id                TEXT PRIMARY KEY,
        agent_id          TEXT NOT NULL,
        user_id           TEXT,
        signal_type       TEXT NOT NULL,
        content           TEXT NOT NULL,
        prediction_delta  TEXT,
        context           TEXT,
        memory_id         TEXT REFERENCES memories(id),
        contradicted      BOOLEAN NOT NULL DEFAULT FALSE,
        created_at        TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_signals_memory_id ON signals(memory_id)",
    """
    CREATE TABLE IF NOT EXISTS memory_sources (
        memory_id  TEXT NOT NULL REFERENCES memories(id),
        signal_id  TEXT NOT NULL REFERENCES signals(id),
        weight     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
        PRIMARY KEY (memory_id, signal_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS compiled_policies (
        cache_key    TEXT PRIMARY KEY,
        agent_id     TEXT NOT NULL,
        user_id      TEXT,
        policy_text  TEXT NOT NULL,
        compiled_at  TIMESTAMPTZ NOT NULL
    )
    """,
    (
        "CREATE INDEX IF NOT EXISTS idx_compiled_policies_agent_user"
        " ON compiled_policies(agent_id, user_id)"
    ),
    """
    CREATE TABLE IF NOT EXISTS agent_config (
        agent_id           TEXT PRIMARY KEY,
        processing_mode    TEXT,
        agent_description  TEXT,
        scopes             TEXT,
        alpha_tuner_state  TEXT,
        gradient_state     TEXT,
        updated_at         TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_events (
        id           TEXT PRIMARY KEY,
        memory_id    TEXT NOT NULL REFERENCES memories(id),
        event_type   TEXT NOT NULL,
        occurred_at  TIMESTAMPTZ NOT NULL,
        metadata     TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_events_memory ON memory_events(memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_events_time ON memory_events(occurred_at)",
    """
    CREATE TABLE IF NOT EXISTS scopes (
        agent_id    TEXT NOT NULL,
        name        TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (agent_id, name)
    )
    """,
]

# pgvector schema -- only applied when PostgresVectorStore.init_schema() is called.
_VECTOR_DDL: list[str] = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    # Dimension is interpolated at runtime: {dim} replaced before execution.
    (
        "CREATE TABLE IF NOT EXISTS memory_vectors ("
        "memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,"
        " embedding vector({dim}) NOT NULL)"
    ),
    # HNSW index (pgvector >= 0.5.0): works at any table size, good recall.
    # m=16, ef_construction=64 are pgvector defaults; explicit here for clarity.
    (
        "CREATE INDEX IF NOT EXISTS idx_memory_vectors_hnsw"
        " ON memory_vectors USING hnsw (embedding vector_cosine_ops)"
        " WITH (m=16, ef_construction=64)"
    ),
]

# -- Row deserializers --------------------------------------------------------
# asyncpg returns native Python types from Postgres:
#   TIMESTAMPTZ -> datetime (timezone-aware)
#   BOOLEAN     -> bool
#   DOUBLE PRECISION / INTEGER -> float / int
# No conversion needed. Field access by name via asyncpg.Record.


def _pg_row_to_memory(row: asyncpg.Record) -> Memory:
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
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        superseded_by=row["superseded_by"],
        pinned=row["pinned"],
        active=row["active"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_triggered=row["last_triggered"],
    )


def _pg_row_to_signal(row: asyncpg.Record) -> Signal:
    return Signal(
        id=row["id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        signal_type=SignalType(row["signal_type"]),
        content=row["content"],
        prediction_delta=row["prediction_delta"],
        context=row["context"],
        memory_id=row["memory_id"],
        contradicted=row["contradicted"],
        created_at=row["created_at"],
    )


# -- PostgresMemoryStore ------------------------------------------------------


class PostgresMemoryStore:
    """asyncpg-backed MemoryStore. Implements the MemoryStore protocol.

    Use make_event_logger() to obtain a PostgresEventLogger backed by the
    same connection pool. Imprint.connect() calls this automatically when
    no explicit event_logger is provided.

    Usage:
        store = PostgresMemoryStore("postgres://user:pass@host/db")
        # or via Imprint(store="postgres://user:pass@host/db")
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 2,
        max_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None

    @property
    def pool(self) -> asyncpg.Pool[asyncpg.Record]:
        if self._pool is None:
            raise RuntimeError("store is not connected; call connect() first")
        return self._pool

    async def connect(self) -> None:
        if self._pool is not None:
            return
        import asyncpg

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )

    async def close(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def init_schema(self) -> None:
        """Create all tables and indexes. Safe to call multiple times (IF NOT EXISTS)."""
        async with self.pool.acquire() as conn, conn.transaction():
            for stmt in _DDL:
                await conn.execute(stmt)

    def make_event_logger(self) -> PostgresEventLogger:
        """Return an EventLogger backed by this store's connection pool."""
        return PostgresEventLogger(self.pool)

    # -- MemoryStore protocol -------------------------------------------------

    async def insert_memory(self, memory: Memory) -> None:
        sql = """
            INSERT INTO memories (
                id, agent_id, user_id, type, scope, content, source,
                stability, recall_count, valid_from, valid_until,
                superseded_by, pinned, active, created_at, updated_at,
                last_triggered
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7,
                $8, $9, $10, $11,
                $12, $13, $14, $15, $16,
                $17
            )
        """
        await self.pool.execute(
            sql,
            memory.id,
            memory.agent_id,
            memory.user_id,
            memory.type.value,
            memory.scope,
            memory.content,
            memory.source.value,
            memory.stability,
            memory.recall_count,
            memory.valid_from,
            memory.valid_until,
            memory.superseded_by,
            memory.pinned,
            memory.active,
            memory.created_at,
            memory.updated_at,
            memory.last_triggered,
        )

    async def insert_signal(self, signal: Signal) -> None:
        sql = """
            INSERT INTO signals (
                id, agent_id, user_id, signal_type, content,
                prediction_delta, context, memory_id, contradicted, created_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10
            )
        """
        await self.pool.execute(
            sql,
            signal.id,
            signal.agent_id,
            signal.user_id,
            signal.signal_type.value,
            signal.content,
            signal.prediction_delta,
            signal.context,
            signal.memory_id,
            signal.contradicted,
            signal.created_at,
        )

    async def link_signal_to_memory(
        self,
        *,
        memory_id: str,
        signal_id: str,
        weight: float = 1.0,
    ) -> None:
        await self.pool.execute(
            "INSERT INTO memory_sources (memory_id, signal_id, weight) VALUES ($1, $2, $3)",
            memory_id,
            signal_id,
            weight,
        )

    async def list_memories(
        self,
        agent_id: str,
        user_id: str | None,
        *,
        memory_type: MemoryType | None = None,
        scopes: list[str] | None = None,
        active_only: bool = True,
    ) -> list[Memory]:
        conditions: list[str] = ["agent_id = $1"]
        params: list[Any] = [agent_id]
        idx = 2

        if user_id is None:
            conditions.append("user_id IS NULL")
        else:
            conditions.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1

        if memory_type is not None:
            conditions.append(f"type = ${idx}")
            params.append(memory_type.value)
            idx += 1

        if scopes is not None:
            if scopes:
                # Pass as array; ANY() avoids per-scope parameter expansion.
                conditions.append(f"(scope = 'global' OR scope = ANY(${idx}::text[]))")
                params.append(scopes)
                idx += 1
            else:
                conditions.append("scope = 'global'")

        if active_only:
            conditions.append("active = TRUE")

        sql = (
            "SELECT id, agent_id, user_id, type, scope, content, source, "
            "stability, recall_count, valid_from, valid_until, superseded_by, "
            "pinned, active, created_at, updated_at, last_triggered "
            "FROM memories WHERE " + " AND ".join(conditions) + " ORDER BY created_at"
        )
        rows = await self.pool.fetch(sql, *params)
        return [_pg_row_to_memory(r) for r in rows]

    async def diff_memories(
        self,
        agent_id: str,
        user_id: str,
        since: AwareDatetime,
        until: AwareDatetime,
    ) -> MemoryDiff:
        _cols = (
            "id, agent_id, user_id, type, scope, content, source, "
            "stability, recall_count, valid_from, valid_until, superseded_by, "
            "pinned, active, created_at, updated_at, last_triggered"
        )

        added_rows = await self.pool.fetch(
            f"SELECT {_cols} FROM memories WHERE agent_id = $1 AND user_id = $2"
            " AND created_at >= $3 AND created_at <= $4 AND active = TRUE"
            " ORDER BY created_at",
            agent_id,
            user_id,
            since,
            until,
        )
        added = [_pg_row_to_memory(r) for r in added_rows]

        deactivated_rows = await self.pool.fetch(
            f"SELECT {_cols} FROM memories WHERE agent_id = $1 AND user_id = $2"
            " AND updated_at >= $3 AND updated_at <= $4 AND active = FALSE"
            " AND superseded_by IS NULL ORDER BY updated_at",
            agent_id,
            user_id,
            since,
            until,
        )
        deactivated = [_pg_row_to_memory(r) for r in deactivated_rows]

        old_rows = await self.pool.fetch(
            f"SELECT {_cols} FROM memories WHERE agent_id = $1 AND user_id = $2"
            " AND updated_at >= $3 AND updated_at <= $4 AND active = FALSE"
            " AND superseded_by IS NOT NULL ORDER BY updated_at",
            agent_id,
            user_id,
            since,
            until,
        )
        old_memories = [_pg_row_to_memory(r) for r in old_rows]

        superseded: list[SupersededPair] = []
        if old_memories:
            ids = [m.superseded_by for m in old_memories if m.superseded_by]
            new_rows = await self.pool.fetch(
                f"SELECT {_cols} FROM memories WHERE id = ANY($1::text[])", ids
            )
            new_by_id = {_pg_row_to_memory(r).id: _pg_row_to_memory(r) for r in new_rows}
            for old in old_memories:
                if old.superseded_by and old.superseded_by in new_by_id:
                    superseded.append(SupersededPair(old=old, new=new_by_id[old.superseded_by]))

        return MemoryDiff(
            since=since, until=until, added=added, deactivated=deactivated, superseded=superseded
        )

    async def list_scopes(self, agent_id: str) -> list[str]:
        rows = await self.pool.fetch(
            "SELECT name FROM scopes WHERE agent_id = $1 ORDER BY created_at",
            agent_id,
        )
        return [r["name"] for r in rows]

    async def list_active_scopes_for_user(self, agent_id: str, user_id: str) -> list[str]:
        rows = await self.pool.fetch(
            "SELECT DISTINCT scope FROM memories "
            "WHERE agent_id = $1 AND user_id = $2 AND active = TRUE AND scope != 'global' "
            "ORDER BY scope",
            agent_id,
            user_id,
        )
        return [r["scope"] for r in rows]

    async def insert_scope(self, agent_id: str, name: str) -> None:
        now = datetime.now(UTC)
        await self.pool.execute(
            "INSERT INTO scopes (agent_id, name, created_at) VALUES ($1, $2, $3)"
            " ON CONFLICT DO NOTHING",
            agent_id,
            name,
            now,
        )

    async def clear_scopes(self, agent_id: str) -> None:
        await self.pool.execute(
            "DELETE FROM scopes WHERE agent_id = $1",
            agent_id,
        )

    async def rename_scope(self, agent_id: str, old_name: str, new_name: str) -> None:
        now = datetime.now(UTC)
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO scopes (agent_id, name, created_at) VALUES ($1, $2, $3)"
                " ON CONFLICT DO NOTHING",
                agent_id,
                new_name,
                now,
            )
            await conn.execute(
                "UPDATE memories SET scope = $1, updated_at = $2"
                " WHERE agent_id = $3 AND scope = $4",
                new_name,
                now,
                agent_id,
                old_name,
            )
            await conn.execute(
                "DELETE FROM scopes WHERE agent_id = $1 AND name = $2",
                agent_id,
                old_name,
            )

    async def merge_scopes(self, agent_id: str, from_scope: str, into_scope: str) -> None:
        now = datetime.now(UTC)
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "UPDATE memories SET scope = $1, updated_at = $2"
                " WHERE agent_id = $3 AND scope = $4",
                into_scope,
                now,
                agent_id,
                from_scope,
            )
            await conn.execute(
                "DELETE FROM scopes WHERE agent_id = $1 AND name = $2",
                agent_id,
                from_scope,
            )

    async def update_memory_scope(self, memory_id: str, new_scope: str) -> None:
        now = datetime.now(UTC)
        await self.pool.execute(
            "UPDATE memories SET scope = $1, updated_at = $2 WHERE id = $3",
            new_scope,
            now,
            memory_id,
        )

    async def deactivate_memory(
        self,
        memory_id: str,
        *,
        superseded_by: str | None = None,
        valid_until: datetime | None = None,
    ) -> bool:
        now = datetime.now(UTC)
        status = await self.pool.execute(
            "UPDATE memories SET active = FALSE, superseded_by = $1,"
            " valid_until = $2, updated_at = $3"
            " WHERE id = $4 AND active = TRUE",
            superseded_by,
            valid_until,
            now,
            memory_id,
        )
        # asyncpg execute returns a string like "UPDATE 1" or "UPDATE 0".
        # asyncpg execute() returns "UPDATE N". Parse N to detect whether
        # the WHERE clause matched. endswith("1") would misfire on "UPDATE 11".
        return int(status.split()[-1]) > 0

    async def mark_signals_contradicted(self, memory_id: str) -> None:
        await self.pool.execute(
            "UPDATE signals SET contradicted = TRUE WHERE id IN ("
            " SELECT signal_id FROM memory_sources WHERE memory_id = $1"
            ")",
            memory_id,
        )

    async def get_cached_policy(self, cache_key: str) -> tuple[str, datetime] | None:
        row = await self.pool.fetchrow(
            "SELECT policy_text, compiled_at FROM compiled_policies WHERE cache_key = $1",
            cache_key,
        )
        if row is None:
            return None
        return row["policy_text"], row["compiled_at"]

    async def put_cached_policy(
        self,
        *,
        cache_key: str,
        agent_id: str,
        user_id: str | None,
        policy_text: str,
        compiled_at: datetime,
    ) -> None:
        await self.pool.execute(
            "INSERT INTO compiled_policies (cache_key, agent_id, user_id, policy_text, compiled_at)"
            " VALUES ($1, $2, $3, $4, $5)"
            " ON CONFLICT (cache_key) DO UPDATE SET"
            " policy_text = EXCLUDED.policy_text, compiled_at = EXCLUDED.compiled_at",
            cache_key,
            agent_id,
            user_id,
            policy_text,
            compiled_at,
        )

    async def invalidate_cached_policies(self, agent_id: str, user_id: str | None) -> None:
        if user_id is None:
            await self.pool.execute(
                "DELETE FROM compiled_policies WHERE agent_id = $1 AND user_id IS NULL",
                agent_id,
            )
        else:
            await self.pool.execute(
                "DELETE FROM compiled_policies WHERE agent_id = $1 AND user_id = $2",
                agent_id,
                user_id,
            )

    async def update_memory_stability(self, memory_id: str, stability: float) -> None:
        # Does NOT update updated_at. Stability is a learning signal, not a
        # content change. updated_at is used in the policy cache key, so
        # touching it here would silently bust the cache on every get_policy call.
        # See DECISIONS.md: "update_memory_stability does not touch updated_at".
        await self.pool.execute(
            "UPDATE memories SET stability = $1 WHERE id = $2",
            stability,
            memory_id,
        )

    async def set_pinned(self, memory_id: str, pinned: bool) -> None:
        now = datetime.now(UTC)
        await self.pool.execute(
            "UPDATE memories SET pinned = $1, updated_at = $2 WHERE id = $3",
            pinned,
            now,
            memory_id,
        )

    async def increment_recall_count(self, memory_id: str) -> None:
        now = datetime.now(UTC)
        await self.pool.execute(
            "UPDATE memories SET recall_count = recall_count + 1, last_triggered = $1"
            " WHERE id = $2",
            now,
            memory_id,
        )

    async def increment_recall_count_batch(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = datetime.now(UTC)
        await self.pool.execute(
            "UPDATE memories SET recall_count = recall_count + 1, last_triggered = $1"
            " WHERE id = ANY($2::text[])",
            now,
            memory_ids,
        )

    async def search_fts(
        self,
        query: str,
        candidate_ids: set[str],
        limit: int = 200,
    ) -> list[tuple[str, float]]:
        """BM25-equivalent full-text search via Postgres tsvector + ts_rank.

        Uses websearch_to_tsquery which handles raw natural language input
        gracefully (including multi-word phrases, quoted strings, negation).
        Unlike FTS5 MATCH, it does not fail on special characters.

        Returns (memory_id, rank) pairs ordered by descending relevance.
        Only returns results whose memory_id is in candidate_ids.
        The rank value is positive (higher = more relevant); callers
        in _hybrid_retrieve use only the ordering, not the value.
        """
        if not query or not candidate_ids:
            return []
        ids = list(candidate_ids)
        rows = await self.pool.fetch(
            "SELECT id, ts_rank(content_tsv, websearch_to_tsquery('simple', $1)) AS rank"
            " FROM memories"
            " WHERE active = TRUE AND id = ANY($2::text[])"
            " AND content_tsv @@ websearch_to_tsquery('simple', $1)"
            " ORDER BY rank DESC LIMIT $3",
            query,
            ids,
            limit,
        )
        return [(r["id"], float(r["rank"])) for r in rows]

    async def put_alpha_tuner_state(self, agent_id: str, state: str) -> None:
        await self.pool.execute(
            "UPDATE agent_config SET alpha_tuner_state = $1 WHERE agent_id = $2",
            state,
            agent_id,
        )

    async def put_gradient_state(self, agent_id: str, state: str) -> None:
        await self.pool.execute(
            "UPDATE agent_config SET gradient_state = $1 WHERE agent_id = $2",
            state,
            agent_id,
        )

    async def get_agent_config(self, agent_id: str) -> _AgentConfig | None:
        row = await self.pool.fetchrow(
            "SELECT processing_mode, agent_description, scopes,"
            " alpha_tuner_state, gradient_state"
            " FROM agent_config WHERE agent_id = $1",
            agent_id,
        )
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
        now = datetime.now(UTC)
        await self.pool.execute(
            "INSERT INTO agent_config"
            " (agent_id, processing_mode, agent_description, scopes, updated_at)"
            " VALUES ($1, $2, $3, $4, $5)"
            " ON CONFLICT (agent_id) DO UPDATE SET"
            " processing_mode = EXCLUDED.processing_mode,"
            " agent_description = EXCLUDED.agent_description,"
            " scopes = EXCLUDED.scopes,"
            " updated_at = EXCLUDED.updated_at",
            agent_id,
            processing_mode,
            agent_description,
            json.dumps(scopes),
            now,
        )

    async def list_events(
        self,
        agent_id: str,
        user_id: str | None,
        *,
        memory_id: str | None = None,
        limit: int = 50,
    ) -> list[MemoryEvent]:
        if memory_id is not None:
            rows = await self.pool.fetch(
                "SELECT e.id, e.memory_id, e.event_type, e.occurred_at, e.metadata"
                " FROM memory_events e"
                " JOIN memories m ON m.id = e.memory_id"
                " WHERE e.memory_id = $1 AND m.agent_id = $2"
                " ORDER BY e.occurred_at DESC LIMIT $3",
                memory_id,
                agent_id,
                limit,
            )
        else:
            rows = await self.pool.fetch(
                "SELECT e.id, e.memory_id, e.event_type, e.occurred_at, e.metadata"
                " FROM memory_events e"
                " JOIN memories m ON m.id = e.memory_id"
                " WHERE m.agent_id = $1 AND m.user_id IS NOT DISTINCT FROM $2"
                " ORDER BY e.occurred_at DESC LIMIT $3",
                agent_id,
                user_id,
                limit,
            )
        return [
            MemoryEvent(
                id=r["id"],
                memory_id=r["memory_id"],
                event_type=r["event_type"],
                detail=json.loads(r["metadata"]) if r["metadata"] else None,
                # asyncpg returns timezone-aware datetimes from TIMESTAMPTZ.
                occurred_at=r["occurred_at"],
            )
            for r in rows
        ]

    async def delete_user_data(self, agent_id: str, user_id: str | None) -> None:
        """Hard delete all memories, signals, and events for an agent-user pair.

        Postgres cascades memory_vectors deletes via ON DELETE CASCADE on the
        FK from memory_vectors.memory_id -> memories.id. No FTS table to clean
        up -- the tsvector is a generated column on the memories row itself.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            if user_id is None:
                sub = "SELECT id FROM memories WHERE agent_id = $1 AND user_id IS NULL"
                await conn.execute(
                    f"DELETE FROM memory_events WHERE memory_id IN ({sub})",
                    agent_id,
                )
                await conn.execute(
                    f"DELETE FROM memory_sources WHERE memory_id IN ({sub})",
                    agent_id,
                )
                await conn.execute(
                    "DELETE FROM memories WHERE agent_id = $1 AND user_id IS NULL",
                    agent_id,
                )
                await conn.execute(
                    "DELETE FROM signals WHERE agent_id = $1 AND user_id IS NULL",
                    agent_id,
                )
                await conn.execute(
                    "DELETE FROM compiled_policies WHERE agent_id = $1 AND user_id IS NULL",
                    agent_id,
                )
            else:
                sub = "SELECT id FROM memories WHERE agent_id = $1 AND user_id = $2"
                await conn.execute(
                    f"DELETE FROM memory_events WHERE memory_id IN ({sub})",
                    agent_id,
                    user_id,
                )
                await conn.execute(
                    f"DELETE FROM memory_sources WHERE memory_id IN ({sub})",
                    agent_id,
                    user_id,
                )
                await conn.execute(
                    "DELETE FROM memories WHERE agent_id = $1 AND user_id = $2",
                    agent_id,
                    user_id,
                )
                await conn.execute(
                    "DELETE FROM signals WHERE agent_id = $1 AND user_id = $2",
                    agent_id,
                    user_id,
                )
                await conn.execute(
                    "DELETE FROM compiled_policies WHERE agent_id = $1 AND user_id = $2",
                    agent_id,
                    user_id,
                )

    async def get_memory_with_supersession(
        self,
        memory_id: str,
    ) -> tuple[Memory | None, Memory | None]:
        row = await self.pool.fetchrow(
            "SELECT id, agent_id, user_id, type, scope, content, source, "
            "stability, recall_count, valid_from, valid_until, superseded_by, "
            "pinned, active, created_at, updated_at, last_triggered "
            "FROM memories WHERE id = $1",
            memory_id,
        )
        if row is None:
            return None, None

        target = _pg_row_to_memory(row)

        successor: Memory | None = None
        if target.superseded_by:
            srow = await self.pool.fetchrow(
                "SELECT id, agent_id, user_id, type, scope, content, source, "
                "stability, recall_count, valid_from, valid_until, superseded_by, "
                "pinned, active, created_at, updated_at, last_triggered "
                "FROM memories WHERE id = $1",
                target.superseded_by,
            )
            if srow is not None:
                successor = _pg_row_to_memory(srow)

        prow = await self.pool.fetchrow(
            "SELECT id, agent_id, user_id, type, scope, content, source, "
            "stability, recall_count, valid_from, valid_until, superseded_by, "
            "pinned, active, created_at, updated_at, last_triggered "
            "FROM memories WHERE superseded_by = $1 ORDER BY created_at DESC LIMIT 1",
            memory_id,
        )
        predecessor: Memory | None = _pg_row_to_memory(prow) if prow is not None else None

        return successor, predecessor

    async def get_memory(self, memory_id: str) -> Memory | None:
        row = await self.pool.fetchrow(
            "SELECT id, agent_id, user_id, type, scope, content, source, "
            "stability, recall_count, valid_from, valid_until, superseded_by, "
            "pinned, active, created_at, updated_at, last_triggered "
            "FROM memories WHERE id = $1",
            memory_id,
        )
        return _pg_row_to_memory(row) if row is not None else None

    async def get_creating_signal(self, memory_id: str) -> Signal | None:
        row = await self.pool.fetchrow(
            "SELECT s.id, s.agent_id, s.user_id, s.signal_type, s.content,"
            " s.prediction_delta, s.context, s.memory_id, s.contradicted, s.created_at"
            " FROM signals s"
            " JOIN memory_sources ms ON ms.signal_id = s.id"
            " WHERE ms.memory_id = $1 LIMIT 1",
            memory_id,
        )
        return _pg_row_to_signal(row) if row is not None else None

    async def get_superseded_memories(self, memory_id: str) -> list[Memory]:
        rows = await self.pool.fetch(
            "SELECT id, agent_id, user_id, type, scope, content, source, "
            "stability, recall_count, valid_from, valid_until, superseded_by, "
            "pinned, active, created_at, updated_at, last_triggered "
            "FROM memories WHERE superseded_by = $1",
            memory_id,
        )
        return [_pg_row_to_memory(r) for r in rows]


# -- PostgresEventLogger ------------------------------------------------------


class PostgresEventLogger:
    """EventLogger backed by a Postgres connection pool.

    Obtained via PostgresMemoryStore.make_event_logger(). Not part of the
    MemoryStore protocol -- it is a factory concern only.
    """

    def __init__(self, pool: asyncpg.Pool[asyncpg.Record]) -> None:
        self._pool = pool

    async def log(
        self,
        memory_id: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)
        await self._pool.execute(
            "INSERT INTO memory_events (id, memory_id, event_type, occurred_at, metadata)"
            " VALUES ($1, $2, $3, $4, $5)",
            event_id,
            memory_id,
            event_type,
            now,
            json.dumps(metadata) if metadata is not None else None,
        )


# -- PostgresVectorStore ------------------------------------------------------


class PostgresVectorStore:
    """pgvector-backed VectorStore using cosine distance.

    Requires pgvector extension installed in the Postgres instance.
    Create explicitly and pass to Imprint(vector_store=..., embedder=...).
    Shares the connection pool with PostgresMemoryStore.

    Vectors are passed to Postgres as a formatted string and cast with ::vector.
    This avoids requiring the pgvector Python package while still using the
    native pgvector type.

    Distance metric: cosine (operator <=>). Returns values in [0, 2] but
    typically [0, 1] for normalized embeddings with non-negative similarity.
    The convention matches what _prefilter_candidates expects: similarity =
    1.0 - distance, threshold comparisons use the [0, 1] range.

    Usage:
        store = PostgresMemoryStore("postgres://...")
        vectors = PostgresVectorStore(store.pool, dim=1024)
        await vectors.init_schema()
        imp = Imprint(agent_id="...", store=store, vector_store=vectors, ...)
    """

    def __init__(self, pool: asyncpg.Pool[asyncpg.Record], dim: int) -> None:
        self._pool = pool
        self._dim = dim

    async def init_schema(self) -> None:
        """Create vector extension, table, and HNSW index.

        Must be called after PostgresMemoryStore.init_schema() because the
        memory_vectors table has a FK to memories(id).

        HNSW index requires pgvector >= 0.5.0. If the Postgres instance has
        an older pgvector, drop the index creation and use exact search.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            for stmt in _VECTOR_DDL:
                await conn.execute(stmt.replace("{dim}", str(self._dim)))

    @staticmethod
    def _serialize(embedding: list[float]) -> str:
        """Format a Python list as a pgvector literal '[x,y,z,...]'."""
        return "[" + ",".join(str(x) for x in embedding) + "]"

    async def upsert(self, id: str, embedding: list[float]) -> None:
        if len(embedding) != self._dim:
            raise ValueError(f"embedding dim {len(embedding)} does not match store dim {self._dim}")
        vec = self._serialize(embedding)
        await self._pool.execute(
            "INSERT INTO memory_vectors (memory_id, embedding) VALUES ($1, $2::vector)"
            " ON CONFLICT (memory_id) DO UPDATE SET embedding = EXCLUDED.embedding",
            id,
            vec,
        )

    async def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return (memory_id, distance) pairs ordered by ascending cosine distance.

        Distance is in [0, 2]; lower = more similar. Callers interpret
        similarity = 1.0 - distance.
        """
        vec = self._serialize(embedding)
        rows = await self._pool.fetch(
            "SELECT memory_id, embedding <=> $1::vector AS distance"
            " FROM memory_vectors"
            " ORDER BY embedding <=> $1::vector LIMIT $2",
            vec,
            top_k,
        )
        return [(r["memory_id"], float(r["distance"])) for r in rows]

    async def delete(self, id: str) -> None:
        await self._pool.execute(
            "DELETE FROM memory_vectors WHERE memory_id = $1",
            id,
        )

    async def delete_batch(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._pool.execute(
            "DELETE FROM memory_vectors WHERE memory_id = ANY($1::text[])",
            ids,
        )
