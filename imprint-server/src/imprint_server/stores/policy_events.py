"""CRUD for the policy_events counterfactual log table.

Every get_policy() call logs here: which memories were retrieved, which were
dropped by scope filter or token budget, the alpha weight used, and a
privacy-safe hash of the context string. This is the data foundation for
Phase 2 learning improvements (per-context alpha analysis).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imprint.types import Memory

    from imprint_server.config import ServerConfig
    from imprint_server.registry import AgentRegistry

_INSERT = """
INSERT INTO policy_events
    (id, session_id, agent_id, user_id, retrieved_ids, filtered_ids,
     alpha_used, context_hash, occurred_at)
VALUES
    {placeholders}
"""


async def log_policy_event(
    *,
    registry: AgentRegistry,
    config: ServerConfig,
    agent_id: str,
    user_id: str,
    session_id: str | None,
    retrieved_memories: list[Memory],
    filtered_memories: list[Memory],
    alpha_used: float,
    context: str | None,
) -> None:
    """Insert one row into policy_events. Fire-and-forget; errors are swallowed.

    alpha_used is 0.0 for sessionless policy calls (no MemoryLoop to carry the
    actual value). Session-based calls (step 4) will pass the real loop.alpha_used.
    """
    with contextlib.suppress(Exception):
        await _insert(
            registry=registry,
            config=config,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            retrieved_memories=retrieved_memories,
            filtered_memories=filtered_memories,
            alpha_used=alpha_used,
            context=context,
        )


async def _insert(
    *,
    registry: AgentRegistry,
    config: ServerConfig,
    agent_id: str,
    user_id: str,
    session_id: str | None,
    retrieved_memories: list[Memory],
    filtered_memories: list[Memory],
    alpha_used: float,
    context: str | None,
) -> None:
    event_id = f"pe_{uuid.uuid4().hex[:12]}"
    now = datetime.now(UTC)
    retrieved_ids = json.dumps([m.id for m in retrieved_memories])
    filtered_ids = json.dumps([m.id for m in filtered_memories])
    context_hash = hashlib.sha256(context.encode()).hexdigest() if context else None

    if config.is_postgres:
        from imprint_server._pool import get_pg_pool

        pool = get_pg_pool(registry)
        await pool.execute(
            "INSERT INTO policy_events"
            " (id, session_id, agent_id, user_id, retrieved_ids, filtered_ids,"
            " alpha_used, context_hash, occurred_at)"
            " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            event_id,
            session_id,
            agent_id,
            user_id,
            retrieved_ids,
            filtered_ids,
            alpha_used,
            context_hash,
            now,
        )
    else:
        import aiosqlite

        from imprint_server._utils import sqlite_file_path

        path = sqlite_file_path(config.store)
        async with aiosqlite.connect(path) as conn:
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute(
                "INSERT INTO policy_events"
                " (id, session_id, agent_id, user_id, retrieved_ids, filtered_ids,"
                " alpha_used, context_hash, occurred_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    session_id,
                    agent_id,
                    user_id,
                    retrieved_ids,
                    filtered_ids,
                    alpha_used,
                    context_hash,
                    now.isoformat(),
                ),
            )
            await conn.commit()
