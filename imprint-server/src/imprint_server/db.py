"""Server-specific database schema management for imprint-server.

The library (imprint-mem) owns its own tables via MemoryStore.init_schema().
This module manages the five additional tables the server needs on top of that,
using the versioned migration runner in migrate.py.

Tables owned by imprint-server:
  sessions       -- durable MemoryLoop state for HTTP sessions
  jobs           -- maintenance job queue (SELECT FOR UPDATE SKIP LOCKED)
  api_keys       -- API key hashes for auth (when IMPRINT_AUTH_DISABLED=false)
  policy_events  -- counterfactual log: every get_policy() call logged here
  agent_ext_config -- per-agent extended configuration (dynamic_scopes, etc.)

The schema_migrations table is also created and managed here (bootstrapped
by the migration runner before any migrations are applied).

Call init_server_schema() once at startup, after the store is connected and
the library schema is initialized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imprint.protocols import MemoryStore

    from imprint_server.config import ServerConfig
    from imprint_server.registry import AgentRegistry


async def init_server_schema(
    config: ServerConfig,
    store: MemoryStore,
    registry: AgentRegistry | None = None,
) -> None:
    """Apply all pending schema migrations for the server tables.

    Safe to call multiple times -- already-applied migrations are verified
    by checksum and skipped. Raises RuntimeError if a shipped migration file
    has been modified after being applied.

    When registry is None (e.g. called from tests that bypass the registry),
    the Postgres path requires the registry for pool access and will raise.
    For SQLite, registry is not needed.
    """
    from imprint_server.migrate import apply_pending

    if registry is None and not config.is_postgres:
        # SQLite path: migrate.py only needs the store URL from config.
        class _MinimalRegistry:
            pass

        await apply_pending(config, _MinimalRegistry())  # type: ignore[arg-type]
    elif registry is not None:
        await apply_pending(config, registry)
    else:
        raise RuntimeError("init_server_schema requires a registry when using Postgres")


# -- Per-agent extended config helpers ----------------------------------------


async def get_agent_dynamic_scopes(config: ServerConfig, store: MemoryStore, agent_id: str) -> bool:
    """Read dynamic_scopes for agent_id from agent_ext_config. Returns False if not set."""
    if config.is_postgres:
        from imprint.stores.postgres import PostgresMemoryStore

        from imprint_server._pool import PgPool

        pg_store: PostgresMemoryStore = store  # type: ignore[assignment]
        pool = pg_store.pool  # type: ignore[reportUnknownMemberType]
        pg_pool = PgPool(pool)
        row = await pg_pool.fetchrow(
            "SELECT dynamic_scopes FROM agent_ext_config WHERE agent_id = $1", agent_id
        )
        if row is None:
            return False
        return bool(row["dynamic_scopes"])
    else:
        from imprint.stores.sqlite import SQLiteMemoryStore

        sq_store: SQLiteMemoryStore = store  # type: ignore[assignment]
        cursor = await sq_store.conn.execute(
            "SELECT dynamic_scopes FROM agent_ext_config WHERE agent_id = ?", (agent_id,)
        )
        row_sq = await cursor.fetchone()
        if row_sq is None:
            return False
        return bool(row_sq[0])


async def set_agent_dynamic_scopes(
    config: ServerConfig, store: MemoryStore, agent_id: str, dynamic_scopes: bool
) -> None:
    """Upsert dynamic_scopes for agent_id in agent_ext_config."""
    if config.is_postgres:
        from imprint.stores.postgres import PostgresMemoryStore

        from imprint_server._pool import PgPool

        pg_store: PostgresMemoryStore = store  # type: ignore[assignment]
        pool = pg_store.pool  # type: ignore[reportUnknownMemberType]
        pg_pool = PgPool(pool)
        await pg_pool.execute(
            "INSERT INTO agent_ext_config (agent_id, dynamic_scopes) VALUES ($1, $2)"
            " ON CONFLICT (agent_id) DO UPDATE SET dynamic_scopes = EXCLUDED.dynamic_scopes",
            agent_id,
            dynamic_scopes,
        )
    else:
        from imprint.stores.sqlite import SQLiteMemoryStore

        sq_store: SQLiteMemoryStore = store  # type: ignore[assignment]
        await sq_store.conn.execute(
            "INSERT OR REPLACE INTO agent_ext_config (agent_id, dynamic_scopes) VALUES (?, ?)",
            (agent_id, int(dynamic_scopes)),
        )
        await sq_store.conn.commit()
