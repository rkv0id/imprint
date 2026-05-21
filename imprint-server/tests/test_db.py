"""Tests for db.init_server_schema and the migration runner.

Uses pytest tmp_path for an isolated temp database per test. No mocking --
the point is to verify the actual DDL executes correctly and is idempotent,
and that the migration system tracks versions and checksums correctly.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from imprint_server.config import ServerConfig
from imprint_server.db import init_server_schema


@pytest.fixture()
def sqlite_config(tmp_path: Path) -> ServerConfig:
    db_path = tmp_path / "test_server.db"
    return ServerConfig(store=f"sqlite:///{db_path}")


def _db_path(config: ServerConfig, tmp_path: Path) -> str:
    return str(tmp_path / "test_server.db")


async def _table_names(path: str) -> set[str]:
    """Return the set of user-created table names in the SQLite file."""
    async with (
        aiosqlite.connect(path) as conn,
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ) as cursor,
    ):
        rows = await cursor.fetchall()
    return {row[0] for row in rows}


async def _index_names(path: str) -> set[str]:
    """Return the set of user-created index names in the SQLite file."""
    async with (
        aiosqlite.connect(path) as conn,
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        ) as cursor,
    ):
        rows = await cursor.fetchall()
    return {row[0] for row in rows}


async def _column_names(path: str, table: str) -> set[str]:
    """Return column names for a table in the SQLite file."""
    async with (
        aiosqlite.connect(path) as conn,
        conn.execute(f"PRAGMA table_info({table})") as cursor,
    ):
        rows = await cursor.fetchall()
    return {row[1] for row in rows}


async def _applied_versions(path: str) -> list[int]:
    """Return applied migration versions from schema_migrations, sorted."""
    async with (
        aiosqlite.connect(path) as conn,
        conn.execute("SELECT version FROM schema_migrations ORDER BY version") as cursor,
    ):
        rows = await cursor.fetchall()
    return [row[0] for row in rows]


# -- Idempotency --------------------------------------------------------------


async def test_init_server_schema_sqlite_is_idempotent(
    sqlite_config: ServerConfig,
) -> None:
    """Calling init_server_schema twice must not raise."""
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]


# -- Table existence ----------------------------------------------------------


async def test_sessions_table_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    assert "sessions" in await _table_names(_db_path(sqlite_config, tmp_path))


async def test_jobs_table_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    assert "jobs" in await _table_names(_db_path(sqlite_config, tmp_path))


async def test_api_keys_table_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    assert "api_keys" in await _table_names(_db_path(sqlite_config, tmp_path))


async def test_policy_events_table_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    assert "policy_events" in await _table_names(_db_path(sqlite_config, tmp_path))


async def test_all_four_tables_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    tables = await _table_names(_db_path(sqlite_config, tmp_path))
    assert {"sessions", "jobs", "api_keys", "policy_events"} <= tables


# -- Indexes ------------------------------------------------------------------


async def test_jobs_claim_index_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    assert "idx_jobs_claim" in await _index_names(_db_path(sqlite_config, tmp_path))


async def test_sessions_index_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    assert "idx_sessions_agent_user" in await _index_names(_db_path(sqlite_config, tmp_path))


async def test_policy_events_index_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    assert "idx_policy_events_agent" in await _index_names(_db_path(sqlite_config, tmp_path))


# -- Column spot-checks -------------------------------------------------------


async def test_sessions_has_required_columns(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    cols = await _column_names(_db_path(sqlite_config, tmp_path), "sessions")
    expected = {
        "id",
        "agent_id",
        "user_id",
        "context",
        "retrieved_ids",
        "alpha_used",
        "outcome",
        "correction",
        "opened_at",
        "expires_at",
        "closed_at",
    }
    assert expected <= cols


async def test_jobs_has_required_columns(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    cols = await _column_names(_db_path(sqlite_config, tmp_path), "jobs")
    expected = {
        "id",
        "agent_id",
        "user_id",
        "job_type",
        "payload",
        "status",
        "priority",
        "created_at",
        "locked_at",
        "locked_by",
        "completed_at",
        "error",
    }
    assert expected <= cols


async def test_api_keys_has_required_columns(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    """api_keys must include user_id (added in migration 0002)."""
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    cols = await _column_names(_db_path(sqlite_config, tmp_path), "api_keys")
    expected = {"key_hash", "agent_id", "user_id", "label", "created_at", "expires_at", "active"}
    assert expected <= cols


async def test_policy_events_has_required_columns(
    sqlite_config: ServerConfig, tmp_path: Path
) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    cols = await _column_names(_db_path(sqlite_config, tmp_path), "policy_events")
    expected = {
        "id",
        "session_id",
        "agent_id",
        "user_id",
        "retrieved_ids",
        "filtered_ids",
        "alpha_used",
        "context_hash",
        "occurred_at",
    }
    assert expected <= cols


async def test_agent_ext_config_table_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    assert "agent_ext_config" in await _table_names(_db_path(sqlite_config, tmp_path))


async def test_agent_ext_config_has_required_columns(
    sqlite_config: ServerConfig, tmp_path: Path
) -> None:
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    cols = await _column_names(_db_path(sqlite_config, tmp_path), "agent_ext_config")
    assert {"agent_id", "dynamic_scopes"} <= cols


# -- Migration tracking -------------------------------------------------------


async def test_schema_migrations_table_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    """schema_migrations tracking table must exist after init."""
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    assert "schema_migrations" in await _table_names(_db_path(sqlite_config, tmp_path))


async def test_all_migrations_recorded(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    """Both migration versions must be recorded in schema_migrations."""
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    versions = await _applied_versions(_db_path(sqlite_config, tmp_path))
    assert versions == [1, 2]


async def test_migration_checksums_stored(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    """Each applied migration must have a non-empty checksum stored."""
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    db = _db_path(sqlite_config, tmp_path)
    async with (
        aiosqlite.connect(db) as conn,
        conn.execute("SELECT version, checksum FROM schema_migrations") as cursor,
    ):
        rows = list(await cursor.fetchall())
    assert len(rows) == 2
    for row in rows:
        assert row[1] and len(str(row[1])) == 64  # SHA-256 hex is 64 chars


async def test_idempotent_run_does_not_duplicate_versions(
    sqlite_config: ServerConfig, tmp_path: Path
) -> None:
    """Running init_server_schema twice must not insert duplicate migration rows."""
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    versions = await _applied_versions(_db_path(sqlite_config, tmp_path))
    assert versions == [1, 2]  # no duplicates


async def test_checksum_mismatch_raises(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    """Modifying a migration file after applying it must raise RuntimeError."""
    from imprint_server.migrate import _apply_sqlite

    db = _db_path(sqlite_config, tmp_path)

    # Apply migrations normally first.
    await _apply_sqlite(sqlite_config.store)

    # Corrupt the stored checksum for migration 1.
    async with aiosqlite.connect(db) as conn:
        await conn.execute("UPDATE schema_migrations SET checksum = 'deadbeef' WHERE version = 1")
        await conn.commit()

    # Next run must raise.
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        await _apply_sqlite(sqlite_config.store)


async def test_new_migration_applied_after_existing(
    sqlite_config: ServerConfig, tmp_path: Path
) -> None:
    """Simulates a deployment upgrade: first run applies 0001, second applies 0002."""
    from imprint_server.migrate import _apply_sqlite

    result = await _apply_sqlite(sqlite_config.store)
    assert 1 in result.applied
    assert 2 in result.applied

    # Second run: both already applied, none new.
    result2 = await _apply_sqlite(sqlite_config.store)
    assert result2.applied == []
    assert set(result2.verified) == {1, 2}


# -- Dynamic scopes helpers ---------------------------------------------------


async def test_get_agent_dynamic_scopes_returns_false_when_not_set(
    sqlite_config: ServerConfig,
) -> None:
    from imprint.stores.sqlite import SQLiteMemoryStore

    from imprint_server._utils import sqlite_file_path
    from imprint_server.db import get_agent_dynamic_scopes

    store = SQLiteMemoryStore(sqlite_file_path(sqlite_config.store))
    await store.connect()
    await store.init_schema()
    await init_server_schema(sqlite_config, store)
    try:
        result = await get_agent_dynamic_scopes(sqlite_config, store, "agent-a")
        assert result is False
    finally:
        await store.close()


async def test_set_and_get_agent_dynamic_scopes_true(
    sqlite_config: ServerConfig,
) -> None:
    from imprint.stores.sqlite import SQLiteMemoryStore

    from imprint_server._utils import sqlite_file_path
    from imprint_server.db import get_agent_dynamic_scopes, set_agent_dynamic_scopes

    store = SQLiteMemoryStore(sqlite_file_path(sqlite_config.store))
    await store.connect()
    await store.init_schema()
    await init_server_schema(sqlite_config, store)
    try:
        await set_agent_dynamic_scopes(sqlite_config, store, "agent-b", True)
        result = await get_agent_dynamic_scopes(sqlite_config, store, "agent-b")
        assert result is True
    finally:
        await store.close()


async def test_set_agent_dynamic_scopes_upserts(
    sqlite_config: ServerConfig,
) -> None:
    from imprint.stores.sqlite import SQLiteMemoryStore

    from imprint_server._utils import sqlite_file_path
    from imprint_server.db import get_agent_dynamic_scopes, set_agent_dynamic_scopes

    store = SQLiteMemoryStore(sqlite_file_path(sqlite_config.store))
    await store.connect()
    await store.init_schema()
    await init_server_schema(sqlite_config, store)
    try:
        await set_agent_dynamic_scopes(sqlite_config, store, "agent-c", True)
        await set_agent_dynamic_scopes(sqlite_config, store, "agent-c", False)
        result = await get_agent_dynamic_scopes(sqlite_config, store, "agent-c")
        assert result is False
    finally:
        await store.close()
