"""CRUD for the api_keys table.

Keys are never stored in plaintext. Only the SHA-256 hex digest lives in the
DB. The raw key is shown once at creation time and never again.

Key format: sk-imp- followed by 64 hex characters (32 random bytes).
Full key string length: 71 characters.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig


KEY_PREFIX = "sk-imp-"
_RANDOM_BYTES = 32


@dataclass
class ApiKeyRow:
    key_hash: str
    agent_id: str | None
    label: str | None
    created_at: datetime
    expires_at: datetime | None
    active: bool


def generate_raw_key() -> str:
    """Generate a new raw API key. Call once; store the hash, not this."""
    return KEY_PREFIX + secrets.token_hex(_RANDOM_BYTES)


def hash_key(raw_key: str) -> str:
    """SHA-256 hex digest of a raw key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def insert_key(
    config: ServerConfig,
    *,
    raw_key: str,
    agent_id: str | None = None,
    label: str | None = None,
    expires_at: datetime | None = None,
) -> ApiKeyRow:
    """Insert a new API key row. Returns the row (without the raw key)."""
    key_hash = hash_key(raw_key)
    now = datetime.now(UTC)
    row = ApiKeyRow(
        key_hash=key_hash,
        agent_id=agent_id,
        label=label,
        created_at=now,
        expires_at=expires_at,
        active=True,
    )
    if config.is_postgres:
        await _pg_insert(config, row)
    else:
        await _sqlite_insert(config, row)
    return row


async def lookup_key(config: ServerConfig, raw_key: str) -> ApiKeyRow | None:
    """Return the key row if the raw key hashes to a known active key, else None."""
    key_hash = hash_key(raw_key)
    if config.is_postgres:
        return await _pg_lookup(config, key_hash)
    return await _sqlite_lookup(config, key_hash)


async def list_keys(config: ServerConfig) -> list[ApiKeyRow]:
    """Return all key rows (hashes only, never raw keys)."""
    if config.is_postgres:
        return await _pg_list(config)
    return await _sqlite_list(config)


async def revoke_key(config: ServerConfig, key_hash: str) -> bool:
    """Set active=False for the given hash. Returns True if the row existed."""
    if config.is_postgres:
        return await _pg_revoke(config, key_hash)
    return await _sqlite_revoke(config, key_hash)


async def count_active_keys(config: ServerConfig) -> int:
    """Count active keys. Used on startup to decide whether to auto-generate."""
    if config.is_postgres:
        return await _pg_count(config)
    return await _sqlite_count(config)


# -- SQLite -------------------------------------------------------------------


async def _sqlite_insert(config: ServerConfig, row: ApiKeyRow) -> None:
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(config.store)
    async with aiosqlite.connect(path) as conn:
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.execute(
            "INSERT INTO api_keys (key_hash, agent_id, label, created_at, expires_at, active)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                row.key_hash,
                row.agent_id,
                row.label,
                row.created_at.isoformat(),
                row.expires_at.isoformat() if row.expires_at else None,
                1 if row.active else 0,
            ),
        )
        await conn.commit()


async def _sqlite_lookup(config: ServerConfig, key_hash: str) -> ApiKeyRow | None:
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(config.store)
    async with aiosqlite.connect(path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ? AND active = 1",
            (key_hash,),
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    return _sqlite_row(dict(row))  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]


async def _sqlite_list(config: ServerConfig) -> list[ApiKeyRow]:
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(config.store)
    async with aiosqlite.connect(path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()
    return [_sqlite_row(dict(r)) for r in rows]  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]


async def _sqlite_revoke(config: ServerConfig, key_hash: str) -> bool:
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(config.store)
    async with aiosqlite.connect(path) as conn:
        await conn.execute("PRAGMA busy_timeout=5000")
        cursor = await conn.execute(
            "UPDATE api_keys SET active = 0 WHERE key_hash = ?", (key_hash,)
        )
        await conn.commit()
        return cursor.rowcount > 0


async def _sqlite_count(config: ServerConfig) -> int:
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(config.store)
    async with (
        aiosqlite.connect(path) as conn,
        conn.execute("SELECT COUNT(*) FROM api_keys WHERE active = 1") as cursor,
    ):
        row = await cursor.fetchone()
    return int(row[0]) if row else 0  # type: ignore[reportUnknownMemberType]


def _sqlite_row(row: dict[str, object]) -> ApiKeyRow:  # type: ignore[reportUnknownParameterType]
    def _dt(v: object) -> datetime:
        return datetime.fromisoformat(str(v)).replace(tzinfo=UTC)

    def _dt_opt(v: object) -> datetime | None:
        return None if v is None else _dt(v)

    return ApiKeyRow(
        key_hash=str(row["key_hash"]),
        agent_id=str(row["agent_id"]) if row["agent_id"] is not None else None,
        label=str(row["label"]) if row["label"] is not None else None,
        created_at=_dt(row["created_at"]),
        expires_at=_dt_opt(row.get("expires_at")),
        active=bool(row["active"]),
    )


# -- Postgres -----------------------------------------------------------------


async def _pg_insert(config: ServerConfig, row: ApiKeyRow) -> None:
    from imprint_server.registry import AgentRegistry  # noqa: F401 -- imported at runtime

    # Pool is not directly accessible here without the registry.
    # This function is called from auth.py lifespan which has the registry.
    # Raise to force callers to use pg_insert_with_pool instead.
    raise NotImplementedError("Use pg_insert_with_pool(pool, row) from lifespan/routes")


async def pg_insert_with_pool(pool: object, row: ApiKeyRow) -> None:
    await pool.execute(  # type: ignore[union-attr,reportUnknownMemberType]
        "INSERT INTO api_keys (key_hash, agent_id, label, created_at, expires_at, active)"
        " VALUES ($1, $2, $3, $4, $5, $6)",
        row.key_hash,
        row.agent_id,
        row.label,
        row.created_at,
        row.expires_at,
        row.active,
    )


async def pg_lookup_with_pool(pool: object, key_hash: str) -> ApiKeyRow | None:
    row = await pool.fetchrow(  # type: ignore[union-attr,reportUnknownMemberType]
        "SELECT * FROM api_keys WHERE key_hash = $1 AND active = TRUE", key_hash
    )
    if row is None:
        return None
    return _pg_row(dict(row))


async def pg_list_with_pool(pool: object) -> list[ApiKeyRow]:
    rows = await pool.fetch(  # type: ignore[union-attr,reportUnknownMemberType]
        "SELECT * FROM api_keys ORDER BY created_at DESC"
    )
    return [_pg_row(dict(r)) for r in rows]


async def pg_revoke_with_pool(pool: object, key_hash: str) -> bool:
    result = await pool.execute(  # type: ignore[union-attr,reportUnknownMemberType]
        "UPDATE api_keys SET active = FALSE WHERE key_hash = $1", key_hash
    )
    return int(result.split()[-1]) > 0


async def pg_count_with_pool(pool: object) -> int:
    val = await pool.fetchval(  # type: ignore[union-attr,reportUnknownMemberType]
        "SELECT COUNT(*) FROM api_keys WHERE active = TRUE"
    )
    return int(val)


async def _pg_lookup(config: ServerConfig, key_hash: str) -> ApiKeyRow | None:
    raise NotImplementedError("Use pg_lookup_with_pool from route handlers")


async def _pg_list(config: ServerConfig) -> list[ApiKeyRow]:
    raise NotImplementedError("Use pg_list_with_pool from route handlers")


async def _pg_revoke(config: ServerConfig, key_hash: str) -> bool:
    raise NotImplementedError("Use pg_revoke_with_pool from route handlers")


async def _pg_count(config: ServerConfig) -> int:
    raise NotImplementedError("Use pg_count_with_pool from route handlers")


def _pg_row(row: dict[str, object]) -> ApiKeyRow:
    def _ensure_aware(v: object) -> datetime:
        if isinstance(v, datetime):
            return v if v.tzinfo is not None else v.replace(tzinfo=UTC)
        return datetime.fromisoformat(str(v)).replace(tzinfo=UTC)

    def _opt(v: object) -> datetime | None:
        return None if v is None else _ensure_aware(v)

    return ApiKeyRow(
        key_hash=str(row["key_hash"]),
        agent_id=str(row["agent_id"]) if row["agent_id"] is not None else None,
        label=str(row["label"]) if row["label"] is not None else None,
        created_at=_ensure_aware(row["created_at"]),
        expires_at=_opt(row.get("expires_at")),
        active=bool(row["active"]),
    )
