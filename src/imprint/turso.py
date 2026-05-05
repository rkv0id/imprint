"""Turso/sqld-backed MemoryStore for Imprint.

Implements the MemoryStore protocol by calling sqld's hrana-over-HTTP
API using httpx. No Rust extension required -- plain HTTP JSON calls,
works on any Python version without a C or Rust build toolchain.

Requires: pip install imprint-mem[turso]

URL formats accepted:
  http://host:port       -- local sqld, no TLS (just turso-dev)
  https://host:port      -- remote sqld with TLS
  libsql://name.turso.io -- Turso cloud (converted to https://)
  ws://host:port         -- converted to http://
  wss://host:port        -- converted to https://
"""

from __future__ import annotations

import contextlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from imprint.store import (
    _INSERT_MEMORY_SQL,
    _INSERT_SIGNAL_SQL,
    _SCHEMA_SQL,
    _AgentConfig,
    _memory_to_params,
    _row_to_memory,
    _row_to_signal,
    _signal_to_params,
)
from imprint.types import Memory, MemoryType, Signal


def _normalize_url(url: str) -> str:
    """Convert any supported URL scheme to an http/https base URL."""
    if url.startswith("libsql://"):
        return "https://" + url[len("libsql://") :]
    if url.startswith("wss://"):
        return "https://" + url[len("wss://") :]
    if url.startswith("ws://"):
        return "http://" + url[len("ws://") :]
    return url


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


def _to_positional(sql: str, params: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """Convert :name style SQL + dict params to ? style + list."""
    if not params:
        return sql, []
    values: list[Any] = []

    def _replace(m: re.Match[str]) -> str:
        values.append(params[m.group(1)])
        return "?"

    converted = re.sub(r":(\w+)", _replace, sql)
    return converted, values


def _encode_value(v: Any) -> dict[str, Any]:
    """Encode a Python value as a hrana argument object.

    hrana v2 value types:
      null    -> {"type": "null"}
      integer -> {"type": "integer", "value": "<string>"}  -- string repr
      float   -> {"type": "float", "value": <number>}      -- JSON number, NOT string
      text    -> {"type": "text", "value": "<string>"}
      blob    -> {"type": "blob", "base64": "<string>"}
    """
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": "1" if v else "0"}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}  # JSON number, not string
    if isinstance(v, bytes):
        import base64

        return {"type": "blob", "base64": base64.b64encode(v).decode()}
    return {"type": "text", "value": str(v)}


def _decode_value(v: dict[str, Any]) -> Any:
    """Decode a hrana value object to a Python value."""
    t = v.get("type")
    if t == "null":
        return None
    if t == "integer":
        return int(v["value"])
    if t == "float":
        return float(v["value"])
    if t == "blob":
        import base64

        return base64.b64decode(v["base64"])
    return v.get("value")  # text and unknown types


class TursoMemoryStore:
    """Turso/sqld-backed implementation of the MemoryStore protocol.

    Calls the sqld hrana-over-HTTP JSON API using httpx. Works with
    self-hosted sqld (via just turso-dev) and Turso cloud. FTS5 is
    supported. sqlite-vec is not available on remote Turso.

    Requires: pip install imprint-mem[turso]
    """

    def __init__(self, url: str, *, auth_token: str | None = None) -> None:
        self.url = url
        self.auth_token = auth_token
        self._client: Any = None
        self._baton: str | None = None

    async def connect(self) -> None:
        if self._client is not None:
            return
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "httpx is required for TursoMemoryStore; "
                "install it with: pip install imprint-mem[turso]"
            ) from e
        headers: dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        self._client = httpx.AsyncClient(
            base_url=_normalize_url(self.url).rstrip("/"),
            headers=headers,
            timeout=30.0,
        )
        self._baton = None

    async def close(self) -> None:
        if self._client is not None:
            client = self._client
            baton = self._baton
            self._client = None
            self._baton = None
            # Release the sqld connection immediately rather than waiting for timeout.
            if baton:
                with contextlib.suppress(Exception):
                    await client.post(
                        "/v2/pipeline",
                        json={"baton": baton, "requests": [{"type": "close"}]},
                    )
            with contextlib.suppress(Exception):
                await client.aclose()

    async def _pipeline(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """POST /v2/pipeline, track the baton, and return the results list.

        The baton is sqld's connection handle. Reusing it across requests
        keeps the same database connection open instead of creating a new
        one per call (which causes DbCreateTimeout under load).

        Retries with backoff on 429 in case sqld needs a moment to recover.
        """
        import asyncio

        response: Any = None
        for attempt in range(3):
            body = {"baton": self._baton, "requests": requests}
            response = await self._client.post("/v2/pipeline", json=body)
            if response.status_code == 429:
                self._baton = None  # reset stale baton on throttle
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            self._baton = data.get("baton")  # update for next call
            return data["results"]  # type: ignore[no-any-return]
        response.raise_for_status()  # exhausted retries
        return []  # unreachable, satisfies type checker

    async def _execute(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], int]:
        """Execute one SQL statement. Returns (rows, affected_row_count)."""
        converted, values = _to_positional(sql, params)
        args = [_encode_value(v) for v in values]
        results = await self._pipeline(
            [{"type": "execute", "stmt": {"sql": converted, "args": args}}]
        )
        result = results[0]
        if result["type"] == "error":
            raise RuntimeError(result["error"]["message"])
        r = result["response"]["result"]
        cols = [c["name"] for c in r["cols"]]
        rows = [dict(zip(cols, [_decode_value(v) for v in row], strict=False)) for row in r["rows"]]
        return rows, r.get("affected_row_count", 0)

    async def _q(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read query and return rows as dicts."""
        rows, _ = await self._execute(sql, params)
        return rows

    async def _w(self, sql: str, params: dict[str, Any] | None = None) -> int:
        """Execute a write statement and return affected row count."""
        _, count = await self._execute(sql, params)
        return count

    async def _batch(self, stmts: list[tuple[str, dict[str, Any] | None]]) -> list[int]:
        """Execute multiple write statements atomically.

        Uses the hrana v2 "batch" request type which wraps all steps in a
        single transaction. begin/end are hrana v3 -- not valid in /v2/pipeline.
        The batch step condition is null (unconditional); sqld rolls back the
        whole transaction if any step errors.
        """
        steps = []
        for sql, params in stmts:
            converted, values = _to_positional(sql, params)
            steps.append(
                {
                    "condition": None,
                    "stmt": {
                        "sql": converted,
                        "args": [_encode_value(v) for v in values],
                    },
                }
            )
        results = await self._pipeline([{"type": "batch", "batch": {"steps": steps}}])
        result = results[0]
        if result["type"] == "error":
            raise RuntimeError(result["error"]["message"])
        batch_result = result["response"]["result"]
        counts: list[int] = []
        for i, step_result in enumerate(batch_result["step_results"]):
            err = batch_result["step_errors"][i]
            if err is not None:
                raise RuntimeError(err["message"])
            counts.append(
                step_result.get("affected_row_count", 0) if step_result is not None else 0
            )
        return counts

    async def init_schema(self) -> None:
        # Send all CREATE TABLE/INDEX statements as a single batch. They all
        # use IF NOT EXISTS so a pre-existing schema is safe. One HTTP request
        # instead of one-per-statement dramatically reduces connection churn.
        schema_stmts = _split_schema(_SCHEMA_SQL)
        steps: list[dict[str, Any]] = [
            {"condition": None, "stmt": {"sql": s, "args": []}} for s in schema_stmts
        ]
        with contextlib.suppress(Exception):
            await self._pipeline([{"type": "batch", "batch": {"steps": steps}}])
        # Migrations may fail on existing schemas -- suppress individually.
        for migration in (
            "ALTER TABLE agent_config RENAME COLUMN detection_mode TO processing_mode",
            "ALTER TABLE agent_config ADD COLUMN alpha_tuner_state TEXT",
            "ALTER TABLE agent_config ADD COLUMN gradient_state TEXT",
        ):
            with contextlib.suppress(Exception):
                await self._execute(migration)

    async def insert_memory(self, memory: Memory) -> None:
        await self._batch(
            [
                (_INSERT_MEMORY_SQL, _memory_to_params(memory)),
                (
                    "INSERT INTO memories_fts(memory_id, content) VALUES (:id, :content)",
                    {"id": memory.id, "content": memory.content},
                ),
            ]
        )

    async def insert_signal(self, signal: Signal) -> None:
        await self._w(_INSERT_SIGNAL_SQL, _signal_to_params(signal))

    async def link_signal_to_memory(
        self, *, memory_id: str, signal_id: str, weight: float = 1.0
    ) -> None:
        await self._w(
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
        rows = await self._q(sql, params)
        return [_row_to_memory(r) for r in rows]

    async def list_scopes(self, agent_id: str) -> list[str]:
        rows = await self._q(
            "SELECT name FROM scopes WHERE agent_id = :agent_id ORDER BY created_at",
            {"agent_id": agent_id},
        )
        return [str(r["name"]) for r in rows]

    async def insert_scope(self, agent_id: str, name: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self._w(
            "INSERT OR IGNORE INTO scopes (agent_id, name, created_at) "
            "VALUES (:agent_id, :name, :now)",
            {"agent_id": agent_id, "name": name, "now": now},
        )

    async def clear_scopes(self, agent_id: str) -> None:
        await self._w(
            "DELETE FROM scopes WHERE agent_id = :agent_id",
            {"agent_id": agent_id},
        )

    async def rename_scope(self, agent_id: str, old_name: str, new_name: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self._batch(
            [
                (
                    "INSERT OR IGNORE INTO scopes (agent_id, name, created_at) "
                    "VALUES (:agent_id, :name, :now)",
                    {"agent_id": agent_id, "name": new_name, "now": now},
                ),
                (
                    "UPDATE memories SET scope = :new, updated_at = :now "
                    "WHERE agent_id = :agent_id AND scope = :old",
                    {"new": new_name, "now": now, "agent_id": agent_id, "old": old_name},
                ),
                (
                    "DELETE FROM scopes WHERE agent_id = :agent_id AND name = :old",
                    {"agent_id": agent_id, "old": old_name},
                ),
            ]
        )

    async def merge_scopes(self, agent_id: str, from_scope: str, into_scope: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self._batch(
            [
                (
                    "UPDATE memories SET scope = :into, updated_at = :now "
                    "WHERE agent_id = :agent_id AND scope = :frm",
                    {"into": into_scope, "now": now, "agent_id": agent_id, "frm": from_scope},
                ),
                (
                    "DELETE FROM scopes WHERE agent_id = :agent_id AND name = :frm",
                    {"agent_id": agent_id, "frm": from_scope},
                ),
            ]
        )

    async def update_memory_scope(self, memory_id: str, new_scope: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self._w(
            "UPDATE memories SET scope = :scope, updated_at = :now WHERE id = :id",
            {"scope": new_scope, "now": now, "id": memory_id},
        )

    async def deactivate_memory(
        self,
        memory_id: str,
        *,
        superseded_by: str | None = None,
        valid_until: datetime | None = None,
    ) -> bool:
        now_iso = datetime.now(UTC).isoformat()
        counts = await self._batch(
            [
                (
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
                (
                    "DELETE FROM memories_fts WHERE memory_id = :id",
                    {"id": memory_id},
                ),
            ]
        )
        return counts[0] > 0

    async def mark_signals_contradicted(self, memory_id: str) -> None:
        await self._w(
            "UPDATE signals SET contradicted = 1 "
            "WHERE id IN ("
            "  SELECT signal_id FROM memory_sources WHERE memory_id = :memory_id"
            ")",
            {"memory_id": memory_id},
        )

    async def get_cached_policy(self, cache_key: str) -> tuple[str, datetime] | None:
        rows = await self._q(
            "SELECT policy_text, compiled_at FROM compiled_policies WHERE cache_key = :key",
            {"key": cache_key},
        )
        if not rows:
            return None
        row = rows[0]
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
        await self._w(
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
        await self._w(
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
        rows = await self._q(
            "SELECT memory_id, rank FROM memories_fts WHERE content MATCH :q "
            "ORDER BY rank LIMIT :lim",
            {"q": query, "lim": limit},
        )
        return [(r["memory_id"], float(r["rank"])) for r in rows if r["memory_id"] in candidate_ids]

    async def get_agent_config(self, agent_id: str) -> _AgentConfig | None:
        rows = await self._q(
            "SELECT processing_mode, agent_description, scopes, "
            "alpha_tuner_state, gradient_state "
            "FROM agent_config WHERE agent_id = :id",
            {"id": agent_id},
        )
        if not rows:
            return None
        row = rows[0]
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
        await self._w(
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

    async def put_alpha_tuner_state(self, agent_id: str, state: str) -> None:
        await self._w(
            "UPDATE agent_config SET alpha_tuner_state = :state WHERE agent_id = :id",
            {"state": state, "id": agent_id},
        )

    async def put_gradient_state(self, agent_id: str, state: str) -> None:
        await self._w(
            "UPDATE agent_config SET gradient_state = :state WHERE agent_id = :id",
            {"state": state, "id": agent_id},
        )

    async def set_pinned(self, memory_id: str, pinned: bool) -> None:
        now_iso = datetime.now(UTC).isoformat()
        await self._w(
            "UPDATE memories SET pinned = :pinned, updated_at = :now WHERE id = :id",
            {"pinned": int(pinned), "now": now_iso, "id": memory_id},
        )

    async def update_memory_stability(self, memory_id: str, stability: float) -> None:
        now_iso = datetime.now(UTC).isoformat()
        await self._w(
            "UPDATE memories SET stability = :s, updated_at = :now WHERE id = :id",
            {"s": stability, "id": memory_id, "now": now_iso},
        )

    async def increment_recall_count(self, memory_id: str) -> None:
        now_iso = datetime.now(UTC).isoformat()
        await self._w(
            "UPDATE memories SET recall_count = recall_count + 1, "
            "last_triggered = :now WHERE id = :id",
            {"now": now_iso, "id": memory_id},
        )

    async def get_memory(self, memory_id: str) -> Memory | None:
        rows = await self._q("SELECT * FROM memories WHERE id = :id", {"id": memory_id})
        return _row_to_memory(rows[0]) if rows else None

    async def get_creating_signal(self, memory_id: str) -> Signal | None:
        rows = await self._q(
            "SELECT s.* FROM signals s "
            "JOIN memory_sources ms ON ms.signal_id = s.id "
            "WHERE ms.memory_id = :mid LIMIT 1",
            {"mid": memory_id},
        )
        return _row_to_signal(rows[0]) if rows else None

    async def get_superseded_memories(self, memory_id: str) -> list[Memory]:
        rows = await self._q("SELECT * FROM memories WHERE superseded_by = :id", {"id": memory_id})
        return [_row_to_memory(r) for r in rows]

    async def get_memory_with_supersession(
        self,
        memory_id: str,
    ) -> tuple[Memory | None, Memory | None]:
        rows = await self._q("SELECT * FROM memories WHERE id = :id", {"id": memory_id})
        if not rows:
            return None, None

        target = _row_to_memory(rows[0])

        successor: Memory | None = None
        if target.superseded_by:
            sr = await self._q(
                "SELECT * FROM memories WHERE id = :id", {"id": target.superseded_by}
            )
            if sr:
                successor = _row_to_memory(sr[0])

        pr = await self._q(
            "SELECT * FROM memories WHERE superseded_by = :id ORDER BY created_at DESC LIMIT 1",
            {"id": memory_id},
        )
        predecessor: Memory | None = _row_to_memory(pr[0]) if pr else None

        return successor, predecessor

    async def list_events(
        self,
        agent_id: str,
        user_id: str | None,
        *,
        memory_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if memory_id is not None:
            rows = await self._q(
                "SELECT e.memory_id, e.event_type, e.occurred_at, e.metadata "
                "FROM memory_events e "
                "JOIN memories m ON m.id = e.memory_id "
                "WHERE e.memory_id = :memory_id AND m.agent_id = :agent_id "
                "ORDER BY e.occurred_at DESC LIMIT :limit",
                {"memory_id": memory_id, "agent_id": agent_id, "limit": limit},
            )
        else:
            rows = await self._q(
                "SELECT e.memory_id, e.event_type, e.occurred_at, e.metadata "
                "FROM memory_events e "
                "JOIN memories m ON m.id = e.memory_id "
                "WHERE m.agent_id = :agent_id AND m.user_id IS :user_id "
                "ORDER BY e.occurred_at DESC LIMIT :limit",
                {"agent_id": agent_id, "user_id": user_id, "limit": limit},
            )
        return [
            {
                "memory_id": r["memory_id"],
                "event_type": r["event_type"],
                "detail": json.loads(r["metadata"]) if r["metadata"] else None,
                "occurred_at": r["occurred_at"],
            }
            for r in rows
        ]
