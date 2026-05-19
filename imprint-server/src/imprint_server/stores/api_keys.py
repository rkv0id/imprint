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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imprint_server._pool import PgPool
    from imprint_server.config import ServerConfig


KEY_PREFIX = "sk-imp-"
_RANDOM_BYTES = 32

_SQLITE_COLS = "key_hash, agent_id, label, created_at, expires_at, active"


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


# -- SQLite -------------------------------------------------------------------


async def insert_key(
    config: ServerConfig,
    *,
    raw_key: str,
    agent_id: str | None = None,
    label: str | None = None,
    expires_at: datetime | None = None,
) -> ApiKeyRow:
    """Insert a new API key row. Returns the row (without the raw key)."""
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

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
    return row


async def lookup_key(config: ServerConfig, raw_key: str) -> ApiKeyRow | None:
    """Return the key row if the raw key hashes to a known active key, else None."""
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    key_hash = hash_key(raw_key)
    path = sqlite_file_path(config.store)
    async with (
        aiosqlite.connect(path) as conn,
        conn.execute(
            f"SELECT {_SQLITE_COLS} FROM api_keys WHERE key_hash = ? AND active = 1",
            (key_hash,),
        ) as cursor,
    ):
        raw = await cursor.fetchone()
    if raw is None:
        return None
    return _sqlite_row_to_key(raw)


async def list_keys(config: ServerConfig) -> list[ApiKeyRow]:
    """Return all key rows (hashes only, never raw keys)."""
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(config.store)
    async with (
        aiosqlite.connect(path) as conn,
        conn.execute(f"SELECT {_SQLITE_COLS} FROM api_keys ORDER BY created_at DESC") as cursor,
    ):
        rows = await cursor.fetchall()
    return [_sqlite_row_to_key(r) for r in rows]


async def revoke_key(config: ServerConfig, key_hash: str) -> bool:
    """Set active=False for the given hash. Returns True if the row existed."""
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


async def count_active_keys(config: ServerConfig) -> int:
    """Count active keys. Used on startup to decide whether to auto-generate."""
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(config.store)
    async with (
        aiosqlite.connect(path) as conn,
        conn.execute("SELECT COUNT(*) FROM api_keys WHERE active = 1") as cursor,
    ):
        raw = await cursor.fetchone()
    return int(raw[0]) if raw is not None else 0


def _sqlite_row_to_key(raw: Any) -> ApiKeyRow:
    """Convert a raw aiosqlite row (positional) to an ApiKeyRow.

    Column order must match _SQLITE_COLS:
      0: key_hash, 1: agent_id, 2: label, 3: created_at, 4: expires_at, 5: active
    """

    def _dt(v: Any) -> datetime:
        return datetime.fromisoformat(str(v)).replace(tzinfo=UTC)

    return ApiKeyRow(
        key_hash=str(raw[0]),
        agent_id=str(raw[1]) if raw[1] is not None else None,
        label=str(raw[2]) if raw[2] is not None else None,
        created_at=_dt(raw[3]),
        expires_at=_dt(raw[4]) if raw[4] is not None else None,
        active=bool(raw[5]),
    )


# -- Postgres -----------------------------------------------------------------


async def pg_insert_with_pool(pool: PgPool, row: ApiKeyRow) -> None:
    await pool.execute(
        "INSERT INTO api_keys (key_hash, agent_id, label, created_at, expires_at, active)"
        " VALUES ($1, $2, $3, $4, $5, $6)",
        row.key_hash,
        row.agent_id,
        row.label,
        row.created_at,
        row.expires_at,
        row.active,
    )


async def pg_lookup_with_pool(pool: PgPool, key_hash: str) -> ApiKeyRow | None:
    row = await pool.fetchrow(
        "SELECT key_hash, agent_id, label, created_at, expires_at, active"
        " FROM api_keys WHERE key_hash = $1 AND active = TRUE",
        key_hash,
    )
    if row is None:
        return None
    return _pg_row_to_key(row)


async def pg_list_with_pool(pool: PgPool) -> list[ApiKeyRow]:
    rows = await pool.fetch(
        "SELECT key_hash, agent_id, label, created_at, expires_at, active"
        " FROM api_keys ORDER BY created_at DESC"
    )
    return [_pg_row_to_key(row) for row in rows]


async def pg_revoke_with_pool(pool: PgPool, key_hash: str) -> bool:
    result = await pool.execute("UPDATE api_keys SET active = FALSE WHERE key_hash = $1", key_hash)
    return int(result.split()[-1]) > 0


async def pg_count_with_pool(pool: PgPool) -> int:
    val = await pool.fetchval("SELECT COUNT(*) FROM api_keys WHERE active = TRUE")
    return int(val) if val is not None else 0


def _pg_row_to_key(row: dict[str, Any]) -> ApiKeyRow:
    def _dt(v: Any) -> datetime:
        if isinstance(v, datetime):
            return v if v.tzinfo is not None else v.replace(tzinfo=UTC)
        return datetime.fromisoformat(str(v)).replace(tzinfo=UTC)

    def _dt_opt(v: Any) -> datetime | None:
        return None if v is None else _dt(v)

    return ApiKeyRow(
        key_hash=str(row["key_hash"]),
        agent_id=str(row["agent_id"]) if row["agent_id"] is not None else None,
        label=str(row["label"]) if row["label"] is not None else None,
        created_at=_dt(row["created_at"]),
        expires_at=_dt_opt(row.get("expires_at")),
        active=bool(row["active"]),
    )
