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
  Agent configuration (processing_mode, scopes, agent_description,
  dynamic_scopes) is written to the DB when an agent is first initialized.
  Admin PATCH /v1/agents/{id}/config updates the DB and calls
  registry.reload_config(agent_id), which re-runs _sync_agent_config() on
  the live instance and updates _dynamic_scopes directly. No TTL polling.

Default mode seeding:
  When a new agent_id has no DB config, get() writes config.default_mode to
  the DB before constructing the Imprint instance. This ensures
  IMPRINT_DEFAULT_MODE is meaningful: new agents get the server's configured
  default, and admin PATCH can override it without touching the server config.

Embedder / vector store / decay model:
  Built once in startup() and shared across all agent instances.
  Embedder and vector store are wired into every Imprint constructor call.
  BanditAlphaTuner is automatically enabled when a vector store is present.
  FSRSGradientDecay is used when IMPRINT_DECAY_MODEL=gradient.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from imprint import Imprint
from imprint.decay import FSRSStaticDecay

if TYPE_CHECKING:
    from imprint.protocols import DecayModel, Embedder, MemoryStore, VectorStore

    from imprint_server.config import ServerConfig
    from imprint_server.redis import RateLimiter, RedisClient

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
        self._embedder: Embedder | None = None
        self._vector_store: VectorStore | None = None
        self._decay_model: DecayModel = FSRSStaticDecay()
        self._redis: RedisClient | None = None
        self._rate_limiter: RateLimiter | None = None
        self._instances: dict[str, Imprint] = {}
        self._init_locks: dict[str, asyncio.Lock] = {}
        self._op_locks: dict[str, asyncio.Lock] = {}

    # -- Lifecycle ------------------------------------------------------------

    async def startup(self) -> None:
        """Create and connect the shared store, build embedder/vector_store/decay.

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
        await init_server_schema(self._config, self._store, registry=self)

        # Build shared embedder.
        if self._config.embedder == "voyage":
            from imprint.providers.voyage import VoyageEmbedder

            self._embedder = VoyageEmbedder(model=self._config.embedder_model)
        elif self._config.embedder == "openai":
            from imprint.providers.openai import OpenAIEmbedder

            self._embedder = OpenAIEmbedder(model=self._config.embedder_model)

        # Build shared vector store (shares the memory store connection/pool).
        if self._config.vector_store == "sqlite-vec":
            from imprint.stores.sqlite import SQLiteMemoryStore
            from imprint.stores.vector import SQLiteVecStore

            sq_store: SQLiteMemoryStore = self._store  # type: ignore[assignment]
            self._vector_store = SQLiteVecStore(sq_store.conn, dim=self._config.embedder_dim)
        elif self._config.vector_store == "postgres":
            from imprint.stores.postgres import PostgresMemoryStore, PostgresVectorStore

            pg_store: PostgresMemoryStore = self._store  # type: ignore[assignment]
            self._vector_store = PostgresVectorStore(
                pg_store.pool,  # type: ignore[reportUnknownMemberType]
                dim=self._config.embedder_dim,
            )
            await self._vector_store.init_schema()

        # Build decay model.
        if self._config.decay_model == "gradient":
            from imprint.online import FSRSGradientDecay

            self._decay_model = FSRSGradientDecay()
        else:
            self._decay_model = FSRSStaticDecay()

        # Build Redis client and rate limiter.
        if self._config.redis_enabled:
            from imprint_server.redis import RedisClient

            self._redis = RedisClient(self._config.redis_url)
            await self._redis.connect()

        if self._config.rate_limit_enabled:
            from imprint_server.redis import RateLimiter

            self._rate_limiter = RateLimiter(self._config, self._redis)

    async def shutdown(self) -> None:
        """Drain pending background tasks (with timeout) and close connections."""
        import asyncio

        try:
            await asyncio.wait_for(self.drain_all(), timeout=self._config.drain_timeout)
        except TimeoutError:
            import logging

            logging.getLogger("imprint.server").warning(
                "drain_all() timed out after %ds -- some background tasks were abandoned",
                self._config.drain_timeout,
            )

        if self._redis is not None:
            await self._redis.close()
            self._redis = None

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

        # Re-read dynamic_scopes from agent_ext_config and apply directly.
        from imprint_server.db import get_agent_dynamic_scopes

        dynamic_scopes = await get_agent_dynamic_scopes(self._config, self.store, agent_id)
        imp._dynamic_scopes = dynamic_scopes  # type: ignore[attr-defined]

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
        from imprint.retrieval import BanditAlphaTuner

        from imprint_server.db import get_agent_dynamic_scopes

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

        # Read per-agent dynamic_scopes from agent_ext_config (default False).
        dynamic_scopes = await get_agent_dynamic_scopes(self._config, store, agent_id)

        # Enable BanditAlphaTuner only when a vector store is configured.
        alpha_tuner = BanditAlphaTuner() if self._vector_store is not None else None

        # Construct with processing_mode=None so the DB row controls the value.
        # _sync_agent_config() (called inside connect()) reads from DB.
        from imprint.stores.postgres import PostgresMemoryStore
        from imprint.stores.sqlite import SQLiteMemoryStore

        if isinstance(store, (SQLiteMemoryStore, PostgresMemoryStore)):
            event_logger = store.make_event_logger()
        else:
            event_logger = None

        imp = Imprint(
            agent_id=agent_id,
            model=self._config.default_model,
            store=store,
            processing_mode=None,
            embedder=self._embedder,
            vector_store=self._vector_store,
            alpha_tuner=alpha_tuner,
            decay_model=self._decay_model,
            dynamic_scopes=dynamic_scopes,
            event_logger=event_logger,
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

    @property
    def config(self) -> ServerConfig:
        """The server configuration this registry was created with."""
        return self._config

    @property
    def redis(self) -> RedisClient | None:
        """The shared RedisClient, or None if Redis is not configured."""
        return self._redis

    @property
    def rate_limiter(self) -> RateLimiter | None:
        """The shared RateLimiter, or None if rate limiting is disabled."""
        return self._rate_limiter


def make_registry(config: ServerConfig) -> AgentRegistry:
    """Convenience factory. Returns an unstarted AgentRegistry."""
    return AgentRegistry(config)
