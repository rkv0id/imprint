"""Schema migration runner for imprint-server.

Migrations are numbered SQL files bundled in imprint_server/migrations/:
  {version:04d}_{description}.postgres.sql
  {version:04d}_{description}.sqlite.sql

The schema_migrations table tracks applied versions and SHA-256 checksums
of the migration files. On each run:
  1. schema_migrations is bootstrapped if it does not exist.
  2. Applied migration checksums are verified against the current files.
     A mismatch means a shipped migration was edited after being applied --
     this is treated as an integrity error and raises immediately.
  3. Unapplied migrations are applied in version order and recorded.

SQLite limitation: ALTER TABLE ... ADD COLUMN does not support IF NOT EXISTS.
The runner catches OperationalError("duplicate column") and skips silently.
This makes column-addition migrations idempotent on SQLite without requiring
conditional SQL syntax.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig
    from imprint_server.registry import AgentRegistry

log = logging.getLogger(__name__)

# Migration files live alongside this module in the migrations/ subdirectory.
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Bootstrap DDL for the migration tracking table itself.
# Never versioned -- the runner creates this before reading any versions.
_BOOTSTRAP_SQLITE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    checksum    TEXT NOT NULL,
    applied_at  TEXT NOT NULL
);
"""

_BOOTSTRAP_POSTGRES = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    checksum    TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL
);
"""


@dataclass
class MigrationResult:
    applied: list[int] = field(default_factory=list[int])
    verified: list[int] = field(default_factory=list[int])

    @property
    def up_to_date(self) -> bool:
        return len(self.applied) == 0


# -- Public entry point -------------------------------------------------------


async def apply_pending(config: ServerConfig, registry: AgentRegistry) -> MigrationResult:
    """Apply all pending migrations and verify applied ones.

    Called at server startup (via init_server_schema) and by the migrate CLI
    command. Safe to call multiple times -- applied migrations are skipped
    after checksum verification.

    Raises RuntimeError if a previously applied migration file has been
    modified (checksum mismatch).
    """
    if config.is_postgres:
        return await _apply_postgres(registry)
    else:
        return await _apply_sqlite(config.store)


# -- File discovery -----------------------------------------------------------


def _file_checksum(path: Path) -> str:
    """SHA-256 hex digest of a migration file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_migrations(backend: str) -> list[tuple[int, str, Path]]:
    """Return (version, description, path) for all migration files of the given backend.

    Files must be named {version:04d}_{description}.{backend}.sql.
    Returned list is sorted ascending by version.
    """
    suffix = f".{backend}.sql"
    results: list[tuple[int, str, Path]] = []

    for entry in _MIGRATIONS_DIR.iterdir():
        if not entry.name.endswith(suffix):
            continue
        version_str = entry.name.split("_")[0]
        try:
            version = int(version_str)
        except ValueError:
            continue
        description = entry.name[len(version_str) + 1 : -len(suffix)]
        results.append((version, description, entry))

    results.sort(key=lambda t: t[0])
    return results


# -- SQLite runner ------------------------------------------------------------


async def _apply_sqlite(store_url: str) -> MigrationResult:
    import aiosqlite

    from imprint_server._utils import sqlite_file_path

    path = sqlite_file_path(store_url)
    result = MigrationResult()
    migrations = _find_migrations("sqlite")

    async with aiosqlite.connect(path) as conn:
        await conn.executescript(_BOOTSTRAP_SQLITE)
        await conn.commit()

        applied = await _sqlite_applied_versions(conn)
        _verify_checksums(applied, migrations)

        for version, description, migration_path in migrations:
            checksum = _file_checksum(migration_path)
            if version in applied:
                result.verified.append(version)
                continue

            log.info("applying migration %04d (%s)", version, description)
            sql = migration_path.read_text(encoding="utf-8")
            for stmt in _split_statements(sql):
                try:
                    await conn.execute(stmt)
                except Exception as exc:
                    # SQLite raises OperationalError("duplicate column name: X")
                    # when ADD COLUMN is used without IF NOT EXISTS.
                    if "duplicate column" in str(exc).lower():
                        log.debug("migration %04d: skipping duplicate column statement", version)
                        continue
                    raise

            await conn.execute(
                "INSERT INTO schema_migrations (version, checksum, applied_at) VALUES (?, ?, ?)",
                (version, checksum, datetime.now(UTC).isoformat()),
            )
            await conn.commit()
            result.applied.append(version)
            log.info("migration %04d applied", version)

    return result


async def _sqlite_applied_versions(conn: object) -> dict[int, str]:
    """Return {version: checksum} for all rows in schema_migrations."""
    import aiosqlite

    c: aiosqlite.Connection = conn  # type: ignore[assignment]
    async with c.execute("SELECT version, checksum FROM schema_migrations") as cursor:
        rows = await cursor.fetchall()
    return {int(row[0]): str(row[1]) for row in rows}


# -- Postgres runner ----------------------------------------------------------


async def _apply_postgres(registry: AgentRegistry) -> MigrationResult:
    from imprint_server._pool import get_pg_pool

    pool = get_pg_pool(registry)
    result = MigrationResult()
    migrations = _find_migrations("postgres")

    for stmt in _split_statements(_BOOTSTRAP_POSTGRES):
        await pool.execute(stmt)

    applied = await _postgres_applied_versions(pool)
    _verify_checksums(applied, migrations)

    for version, description, migration_path in migrations:
        checksum = _file_checksum(migration_path)
        if version in applied:
            result.verified.append(version)
            continue

        log.info("applying migration %04d (%s)", version, description)
        sql = migration_path.read_text(encoding="utf-8")
        for stmt in _split_statements(sql):
            await pool.execute(stmt)

        await pool.execute(
            "INSERT INTO schema_migrations (version, checksum, applied_at) VALUES ($1, $2, $3)",
            version,
            checksum,
            datetime.now(UTC),
        )
        result.applied.append(version)
        log.info("migration %04d applied", version)

    return result


async def _postgres_applied_versions(pool: object) -> dict[int, str]:
    from imprint_server._pool import PgPool

    p: PgPool = pool  # type: ignore[assignment]
    rows = await p.fetch("SELECT version, checksum FROM schema_migrations")
    return {int(row["version"]): str(row["checksum"]) for row in rows}


# -- Shared helpers -----------------------------------------------------------


def _split_statements(sql: str) -> list[str]:
    """Split a SQL string on semicolons, stripping comments and blank lines."""
    stmts: list[str] = []
    for raw in sql.split(";"):
        lines: list[str] = [
            line for line in raw.splitlines() if line.strip() and not line.strip().startswith("--")
        ]
        stmt = "\n".join(lines).strip()
        if stmt:
            stmts.append(stmt)
    return stmts


def _verify_checksums(
    applied: dict[int, str],
    migrations: list[tuple[int, str, Path]],
) -> None:
    """Raise RuntimeError if any applied migration file has been modified."""
    migration_map = {v: (desc, path) for v, desc, path in migrations}

    for version, stored_checksum in applied.items():
        if version not in migration_map:
            log.warning("migration %04d was applied but its file no longer exists", version)
            continue

        _, path = migration_map[version]
        current_checksum = _file_checksum(path)
        if current_checksum != stored_checksum:
            raise RuntimeError(
                f"Migration {version:04d} checksum mismatch -- "
                f"the file was modified after being applied.\n"
                f"  stored:  {stored_checksum}\n"
                f"  current: {current_checksum}\n"
                f"Editing shipped migration files corrupts the schema history. "
                f"Create a new migration instead."
            )
