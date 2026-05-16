"""AgentRegistry: pool of live Imprint instances for imprint-server.

One Imprint instance per agent_id, shared across all users of that agent.
Agent-level state (gradient decay model, alpha tuner, scope vocabulary) is
naturally shared and aggregated across users by this single-instance design.
Population-calibrated FSRS decay is a consequence: all user feedback flows
through the same decay model, which the library persists to agent_config
after every finalize_loop(). New requests for an agent load the accumulated
state in _sync_agent_config().

Two locks per agent:
  _init_lock  -- prevents duplicate lazy initialization (internal to get())
  _op_lock    -- held by route handlers around observe() calls to prevent
                 concurrent scope consolidation races (exposed via get_op_lock())

Config reload:
  Agent configuration (processing_mode, scopes, agent_description) is written
  to the DB when an agent is first initialized. Admin PATCH /v1/agents/{id}/config
  updates the DB and calls registry.reload_config(agent_id), which re-runs
  _sync_agent_config() on the live instance. No TTL polling.

Default mode seeding:
  When a new agent_id has no DB config, get() writes config.default_mode to the
  DB before constructing the Imprint instance. This ensures IMPRINT_DEFAULT_MODE
  is meaningful: new agents get the server's configured default, and admin PATCH
  can override it without touching the server config.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from imprint import Imprint

if TYPE_CHECKING:
    from imprint.protocols import MemoryStore

    from imprint_server.config import ServerConfig

_VALID_MODES: frozenset[str] = frozenset({"frugal", "balanced", "eager"})


class AgentRegistry:
    """Pool of live Imprint instances keyed by agent_id.

    Usage:
      registry = AgentRegistry(config)
      await registry.startup()
      try:
          imp = await registry.get("my-agent")
          async with registry.get_op_lock("my-agent"):
              await imp.observe(...)
      finally:
          await registry.shutdown()
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._store: MemoryStore | None = None
        self._instances: dict[str, Imprint] = {}
        self._init_locks: dict[str, asyncio.Lock] = {}
        self._op_locks: dict[str, asyncio.Lock] = {}

    # -- Lifecycle ------------------------------------------------------------

    async def startup(self) -> None:
        """Create and connect the shared store, initialize all schemas.

        Must be called before any other method. Idempotent if called twice
        (init_schema and init_server_schema are both IF NOT EXISTS).
        """
        from imprint_server.db import init_server_schema

        if self._config.is_postgres:
            from imprint.stores.postgres import PostgresMemoryStore

            self._store = PostgresMemoryStore(
                self._config.store,
                min_size=self._config.pool_min,
                max_size=self._config.pool_max,
            )
        else:
            from imprint.stores.sqlite import SQLiteMemoryStore

            from imprint_server._utils import sqlite_file_path

            self._store = SQLiteMemoryStore(sqlite_file_path(self._config.store))

        await self._store.connect()
        await self._store.init_schema()
        await init_server_schema(self._config, self._store)

    async def shutdown(self) -> None:
        """Drain all pending background tasks and close the shared store."""
        await self.drain_all()
        if self._store is not None:
            await self._store.close()
            self._store = None

    # -- Store access ---------------------------------------------------------

    @property
    def store(self) -> MemoryStore:
        """The shared MemoryStore. Raises if startup() has not been called."""
        if self._store is None:
            raise RuntimeError(
                "AgentRegistry.store accessed before startup() -- call startup() first"
            )
        return self._store

    # -- Instance management --------------------------------------------------

    async def get(self, agent_id: str) -> Imprint:
        """Return the live Imprint instance for agent_id, initializing lazily.

        Thread-safe: uses a per-agent init lock to prevent duplicate initialization
        when two concurrent requests arrive for the same agent_id simultaneously.
        The lock is held only during the init sequence -- subsequent calls for the
        same agent_id return immediately without acquiring anything.
        """
        # Fast path: already initialized.
        if agent_id in self._instances:
            return self._instances[agent_id]

        # Slow path: acquire per-agent init lock.
        if agent_id not in self._init_locks:
            self._init_locks[agent_id] = asyncio.Lock()

        async with self._init_locks[agent_id]:
            # Double-check inside lock in case another coroutine finished first.
            if agent_id in self._instances:
                return self._instances[agent_id]

            imp = await self._create_instance(agent_id)
            self._instances[agent_id] = imp
            return imp

    async def get_op_lock(self, agent_id: str) -> asyncio.Lock:
        """Return the operation lock for agent_id.

        Route handlers acquire this around observe() calls to prevent concurrent
        scope consolidation races. The lock is created on first access.
        """
        if agent_id not in self._op_locks:
            self._op_locks[agent_id] = asyncio.Lock()
        return self._op_locks[agent_id]

    async def reload_config(self, agent_id: str) -> None:
        """Re-read DB config and apply to the live instance.

        Called by admin PATCH /v1/agents/{agent_id}/config after the DB row is
        updated. No-op if the agent has not been initialized yet (first request
        will pick up the new config automatically).
        """
        if agent_id not in self._instances:
            return
        imp = self._instances[agent_id]
        await imp._sync_agent_config()  # type: ignore[attr-defined]

    async def deregister(self, agent_id: str) -> None:
        """Drain and remove an agent from the registry.

        No-op if the agent is not initialized. The agent_config row in the DB
        is untouched -- a new request will re-initialize the agent from existing
        DB config. Does not close the Imprint instance because it uses the shared
        store (_owns_store=False).
        """
        if agent_id not in self._instances:
            return
        imp = self._instances.pop(agent_id)
        await imp.drain()

    async def drain_all(self) -> None:
        """Await all pending background learning tasks across all instances."""
        if not self._instances:
            return
        await asyncio.gather(
            *(imp.drain() for imp in self._instances.values()),
            return_exceptions=True,
        )

    # -- Internal -------------------------------------------------------------

    async def _create_instance(self, agent_id: str) -> Imprint:
        """Construct, seed, and connect a new Imprint instance for agent_id.

        Pre-populates the DB config for brand-new agents so IMPRINT_DEFAULT_MODE
        is honoured. Existing agents load their config from the DB as-is.
        """
        store = self.store

        # Seed DB config for new agents with server defaults so that
        # _sync_agent_config() picks up config.default_mode instead of the
        # library's hardcoded "balanced" fallback.
        existing = await store.get_agent_config(agent_id)
        if existing is None:
            await store.put_agent_config(
                agent_id=agent_id,
                processing_mode=self._config.default_mode,
                agent_description=None,
                scopes=[],
            )

        # Construct with processing_mode=None so the DB row controls the value.
        # _sync_agent_config() (called inside connect()) reads from DB.
        imp = Imprint(
            agent_id=agent_id,
            model=self._config.default_model,
            store=store,
            processing_mode=None,
        )
        # connect() runs init_schema() (no-op -- already done) and _sync_agent_config().
        await imp.connect()
        return imp

    @property
    def agent_count(self) -> int:
        """Number of initialized agent instances."""
        return len(self._instances)

    def agent_ids(self) -> list[str]:
        """Sorted list of initialized agent IDs. Used in health checks."""
        return sorted(self._instances.keys())


def make_registry(config: ServerConfig) -> AgentRegistry:
    """Convenience factory. Returns an unstarted AgentRegistry."""
    return AgentRegistry(config)
