"""Turso/libSQL-backed MemoryStore for Imprint.

Requires: pip install imprint-mem[turso]

URL forms accepted by TursoMemoryStore:
  libsql://name.turso.io       -- remote Turso database (wss)
  https://name.turso.io        -- remote Turso database (http)
  file:local.db                -- local libSQL file (testing / embedded replica)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from imprint.store import (
    _INSERT_MEMORY_SQL,
    _INSERT_SIGNAL_SQL,
    _SCHEMA_SQL,
    _AgentConfig,
    _memory_to_params,
    _row_to_memory,
    _signal_to_params,
)
from imprint.types import Memory, MemoryType, Signal


def _split_schema(sql: str) -> list[str]:
    """Split a multi-statement SQL string into individual statements."""
    stmts: list[str] = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        current.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(current).strip().rstrip(";").strip()
            if stmt:
                stmts.append(stmt)
            current = []
    if current:
        stmt = "\n".join(current).strip().rstrip(";").strip()
        if stmt:
            stmts.append(stmt)
    return stmts


class TursoMemoryStore:
    """Turso/libSQL-backed implementation of the MemoryStore protocol.

    Identical SQL surface to SQLiteMemoryStore; adapted for the libsql_client
    async API. FTS5 full-text search is supported. sqlite-vec is not available
    on remote Turso -- use SQLiteVecStore against a local embedded replica if
    dense vector retrieval is needed alongside Turso.

    Requires: pip install imprint-mem[turso]
    """

    def __init__(self, url: str, *, auth_token: str | None = None) -> None:
        self.url = url
        self.auth_token = auth_token
        self._client: Any = None

    async def connect(self) -> None:
        if self._client is not None:
            return
        try:
            import libsql_client  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "libsql-client is required for TursoMemoryStore; "
                "install it with: pip install imprint-mem[turso]"
            ) from e
        kwargs: dict[str, Any] = {"url": self.url}
        if self.auth_token is not None:
            kwargs["auth_token"] = self.auth_token
        self._client = libsql_client.create_client(**kwargs)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def init_schema(self) -> None:
        import contextlib

        import libsql_client  # type: ignore[import-untyped]

        stmts = _split_schema(_SCHEMA_SQL)
        await self._client.batch([libsql_client.Statement(s) for s in stmts])
        for migration in (
            "ALTER TABLE agent_config RENAME COLUMN detection_mode TO processing_mode",
            "ALTER TABLE agent_config ADD COLUMN alpha_tuner_state TEXT",
            "ALTER TABLE agent_config ADD COLUMN gradient_state TEXT",
        ):
            with contextlib.suppress(Exception):
                await self._client.execute(migration)

    async def insert_memory(self, memory: Memory) -> None:
        import libsql_client  # type: ignore[import-untyped]

        params = _memory_to_params(memory)
        await self._client.batch(
            [
                libsql_client.Statement(_INSERT_MEMORY_SQL, params),
                libsql_client.Statement(
                    "INSERT INTO memories_fts(memory_id, content) VALUES (:id, :content)",
                    {"id": memory.id, "content": memory.content},
                ),
            ]
        )

    async def insert_signal(self, signal: Signal) -> None:
        await self._client.execute(_INSERT_SIGNAL_SQL, _signal_to_params(signal))

    async def link_signal_to_memory(
        self, *, memory_id: str, signal_id: str, weight: float = 1.0
    ) -> None:
        await self._client.execute(
            "INSERT OR IGNORE INTO memory_sources(memory_id, signal_id, weight) "
            "VALUES (:memory_id, :signal_id, :weight)",
            {"memory_id": memory_id, "signal_id": signal_id, "weight": weight},
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
        conditions = ["agent_id = :agent_id"]
        params: dict[str, Any] = {"agent_id": agent_id}
        if user_id is None:
            conditions.append("user_id IS NULL")
        else:
            conditions.append("user_id = :user_id")
            params["user_id"] = user_id
        if active_only:
            conditions.append("active = 1")
        if memory_type is not None:
            conditions.append("type = :type")
            params["type"] = memory_type.value
        if scopes is not None:
            placeholders = ",".join(f":s{i}" for i in range(len(scopes)))
            conditions.append(f"(scope = 'global' OR scope IN ({placeholders}))")
            for i, s in enumerate(scopes):
                params[f"s{i}"] = s
        sql = "SELECT * FROM memories WHERE " + " AND ".join(conditions)
        result = await self._client.execute(sql, params)
        return [_row_to_memory(row) for row in result.rows]

    async def deactivate_memory(
        self,
        memory_id: str,
        *,
        superseded_by: str | None = None,
        valid_until: datetime | None = None,
    ) -> bool:
        import libsql_client  # type: ignore[import-untyped]

        now_iso = datetime.now(UTC).isoformat()
        rss = await self._client.batch(
            [
                libsql_client.Statement(
                    "UPDATE memories SET active = 0, superseded_by = :superseded_by, "
                    "valid_until = :valid_until, updated_at = :updated_at "
                    "WHERE id = :id AND active = 1",
                    {
                        "id": memory_id,
                        "superseded_by": superseded_by,
                        "valid_until": valid_until.isoformat() if valid_until else None,
                        "updated_at": now_iso,
                    },
                ),
                libsql_client.Statement(
                    "DELETE FROM memories_fts WHERE memory_id = :id",
                    {"id": memory_id},
                ),
            ]
        )
        return rss[0].rows_affected > 0

    async def mark_signals_contradicted(self, memory_id: str) -> None:
        await self._client.execute(
            "UPDATE signals SET contradicted = 1 "
            "WHERE id IN ("
            "  SELECT signal_id FROM memory_sources WHERE memory_id = :memory_id"
            ")",
            {"memory_id": memory_id},
        )

    async def get_cached_policy(self, cache_key: str) -> tuple[str, datetime] | None:
        result = await self._client.execute(
            "SELECT policy_text, compiled_at FROM compiled_policies WHERE cache_key = :key",
            {"key": cache_key},
        )
        if not result.rows:
            return None
        row = result.rows[0]
        return (row["policy_text"], datetime.fromisoformat(row["compiled_at"]))

    async def put_cached_policy(
        self,
        *,
        cache_key: str,
        agent_id: str,
        user_id: str | None,
        policy_text: str,
        compiled_at: datetime,
    ) -> None:
        await self._client.execute(
            "INSERT OR REPLACE INTO compiled_policies "
            "(cache_key, agent_id, user_id, policy_text, compiled_at) "
            "VALUES (:key, :agent_id, :user_id, :text, :compiled_at)",
            {
                "key": cache_key,
                "agent_id": agent_id,
                "user_id": user_id,
                "text": policy_text,
                "compiled_at": compiled_at.isoformat(),
            },
        )

    async def invalidate_cached_policies(self, agent_id: str, user_id: str | None) -> None:
        await self._client.execute(
            "DELETE FROM compiled_policies "
            "WHERE agent_id = :agent_id AND (user_id = :user_id OR user_id IS NULL)",
            {"agent_id": agent_id, "user_id": user_id},
        )

    async def search_fts(
        self,
        query: str,
        candidate_ids: set[str],
        limit: int = 200,
    ) -> list[tuple[str, float]]:
        if not query or not candidate_ids:
            return []
        result = await self._client.execute(
            "SELECT memory_id, rank FROM memories_fts WHERE content MATCH :q "
            "ORDER BY rank LIMIT :lim",
            {"q": query, "lim": limit},
        )
        return [
            (row["memory_id"], float(row["rank"]))
            for row in result.rows
            if row["memory_id"] in candidate_ids
        ]

    async def put_alpha_tuner_state(self, agent_id: str, state: str) -> None:
        await self._client.execute(
            "UPDATE agent_config SET alpha_tuner_state = :state WHERE agent_id = :id",
            {"state": state, "id": agent_id},
        )

    async def put_gradient_state(self, agent_id: str, state: str) -> None:
        await self._client.execute(
            "UPDATE agent_config SET gradient_state = :state WHERE agent_id = :id",
            {"state": state, "id": agent_id},
        )

    async def get_agent_config(self, agent_id: str) -> _AgentConfig | None:
        result = await self._client.execute(
            "SELECT processing_mode, agent_description, scopes, "
            "alpha_tuner_state, gradient_state "
            "FROM agent_config WHERE agent_id = :id",
            {"id": agent_id},
        )
        if not result.rows:
            return None
        row = result.rows[0]
        return _AgentConfig(
            processing_mode=row["processing_mode"],
            agent_description=row["agent_description"],
            scopes=json.loads(row["scopes"]) if row["scopes"] else None,
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
        now_iso = datetime.now(UTC).isoformat()
        await self._client.execute(
            "INSERT OR REPLACE INTO agent_config "
            "(agent_id, processing_mode, agent_description, scopes, updated_at) "
            "VALUES (:id, :mode, :desc, :scopes, :now)",
            {
                "id": agent_id,
                "mode": processing_mode,
                "desc": agent_description,
                "scopes": json.dumps(scopes),
                "now": now_iso,
            },
        )

    async def set_pinned(self, memory_id: str, pinned: bool) -> None:
        now_iso = datetime.now(UTC).isoformat()
        await self._client.execute(
            "UPDATE memories SET pinned = :pinned, updated_at = :now WHERE id = :id",
            {"pinned": int(pinned), "now": now_iso, "id": memory_id},
        )

    async def update_memory_stability(self, memory_id: str, stability: float) -> None:
        now_iso = datetime.now(UTC).isoformat()
        await self._client.execute(
            "UPDATE memories SET stability = :s, updated_at = :now WHERE id = :id",
            {"s": stability, "id": memory_id, "now": now_iso},
        )

    async def increment_recall_count(self, memory_id: str) -> None:
        now_iso = datetime.now(UTC).isoformat()
        await self._client.execute(
            "UPDATE memories SET recall_count = recall_count + 1, "
            "last_triggered = :now WHERE id = :id",
            {"now": now_iso, "id": memory_id},
        )
