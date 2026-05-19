"""Typed wrapper around asyncpg pool and connection objects.

asyncpg does not ship type stubs, so accessing pool methods directly
produces unknown-type cascades throughout the codebase. This module
wraps the pool and connection objects with a fully-typed API that
converts asyncpg Record objects to dict[str, Any] immediately.

All type: ignore comments are contained here. Code outside this module
that uses PgPool and PgConnection has full type coverage.

Usage:
  pool = get_pg_pool(registry)
  row = await pool.fetchrow("SELECT ...", value)
  rows = await pool.fetch("SELECT ...", value)

  async with pool.acquire() as conn:
      await conn.execute("INSERT ...", value)
      row = await conn.fetchrow("SELECT ...", value)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imprint_server.registry import AgentRegistry


class PgConnection:
    """Typed wrapper around an asyncpg connection (acquired from a pool)."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def execute(self, query: str, *args: Any) -> str:
        result: str = await self._conn.execute(query, *args)  # type: ignore[union-attr]
        return result

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        row = await self._conn.fetchrow(query, *args)  # type: ignore[union-attr]
        if row is None:
            return None
        return dict(row)  # type: ignore[call-overload]


class PgPool:
    """Typed wrapper around asyncpg.Pool.

    Converts asyncpg Record objects to dict[str, Any] on every fetch
    so callers always work with plain, typed Python dicts.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def execute(self, query: str, *args: Any) -> str:
        result: str = await self._pool.execute(query, *args)  # type: ignore[union-attr]
        return result

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(query, *args)  # type: ignore[union-attr]
        return [dict(row) for row in rows]  # type: ignore[call-overload]

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(query, *args)  # type: ignore[union-attr]
        if row is None:
            return None
        return dict(row)  # type: ignore[call-overload]

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self._pool.fetchval(query, *args)  # type: ignore[union-attr]

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[PgConnection, None]:
        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            yield PgConnection(conn)


def get_pg_pool(registry: AgentRegistry) -> PgPool:
    """Extract and wrap the asyncpg pool from the registry's Postgres store.

    Only call this when config.is_postgres is True. The type: ignore is
    intentional -- the cast is correct by construction (registry.store is
    PostgresMemoryStore when Postgres is configured).
    """
    from imprint.stores.postgres import PostgresMemoryStore

    pg_store: PostgresMemoryStore = registry.store  # type: ignore[assignment]
    return PgPool(pg_store.pool)
