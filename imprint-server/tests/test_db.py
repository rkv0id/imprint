"""Tests for db.init_server_schema against a real SQLite file.

Uses pytest tmp_path for an isolated temp database per test. No mocking -- the
point is to verify the actual DDL executes correctly and is idempotent.
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


# -- Idempotency --------------------------------------------------------------


async def _column_names(path: str, table: str) -> set[str]:
    """Return column names for a table in the SQLite file."""
    async with (
        aiosqlite.connect(path) as conn,
        conn.execute(f"PRAGMA table_info({table})") as cursor,
    ):
        rows = await cursor.fetchall()
    return {row[1] for row in rows}


async def test_init_server_schema_sqlite_is_idempotent(
    sqlite_config: ServerConfig,
) -> None:
    """Calling init_server_schema twice must not raise."""
    # We skip the library schema (no MemoryStore needed for server tables).
    # Pass a dummy store -- init_server_schema dispatches on config.is_postgres
    # and for SQLite only uses the store_url from config, not the store object.
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]


# -- Table existence ----------------------------------------------------------


async def test_sessions_table_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    tables = await _table_names(db_path)
    assert "sessions" in tables


async def test_jobs_table_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    tables = await _table_names(db_path)
    assert "jobs" in tables


async def test_api_keys_table_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    tables = await _table_names(db_path)
    assert "api_keys" in tables


async def test_policy_events_table_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    tables = await _table_names(db_path)
    assert "policy_events" in tables


async def test_all_four_tables_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    tables = await _table_names(db_path)
    assert {"sessions", "jobs", "api_keys", "policy_events"} <= tables


# -- Indexes ------------------------------------------------------------------


async def test_jobs_claim_index_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    indexes = await _index_names(db_path)
    assert "idx_jobs_claim" in indexes


async def test_sessions_index_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    indexes = await _index_names(db_path)
    assert "idx_sessions_agent_user" in indexes


async def test_policy_events_index_created(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    indexes = await _index_names(db_path)
    assert "idx_policy_events_agent" in indexes


# -- Column spot-checks -------------------------------------------------------


async def test_sessions_has_required_columns(sqlite_config: ServerConfig, tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    cols = await _column_names(db_path, "sessions")
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
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    cols = await _column_names(db_path, "jobs")
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
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    cols = await _column_names(db_path, "api_keys")
    expected = {"key_hash", "agent_id", "label", "created_at", "expires_at", "active"}
    assert expected <= cols


async def test_policy_events_has_required_columns(
    sqlite_config: ServerConfig, tmp_path: Path
) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    cols = await _column_names(db_path, "policy_events")
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
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    tables = await _table_names(db_path)
    assert "agent_ext_config" in tables


async def test_agent_ext_config_has_required_columns(
    sqlite_config: ServerConfig, tmp_path: Path
) -> None:
    db_path = str(tmp_path / "test_server.db")
    await init_server_schema(sqlite_config, None)  # type: ignore[arg-type]
    cols = await _column_names(db_path, "agent_ext_config")
    assert {"agent_id", "dynamic_scopes"} <= cols


# -- Dynamic scopes helpers ---------------------------------------------------


async def test_get_agent_dynamic_scopes_returns_false_when_not_set(
    sqlite_config: ServerConfig,
) -> None:
    """get_agent_dynamic_scopes returns False when no row exists."""
    from imprint.stores.sqlite import SQLiteMemoryStore

    from imprint_server._utils import sqlite_file_path
    from imprint_server.db import get_agent_dynamic_scopes, init_server_schema

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
    """set_agent_dynamic_scopes persists True and get reads it back."""
    from imprint.stores.sqlite import SQLiteMemoryStore

    from imprint_server._utils import sqlite_file_path
    from imprint_server.db import (
        get_agent_dynamic_scopes,
        init_server_schema,
        set_agent_dynamic_scopes,
    )

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
    """Calling set_agent_dynamic_scopes twice with different values updates the row."""
    from imprint.stores.sqlite import SQLiteMemoryStore

    from imprint_server._utils import sqlite_file_path
    from imprint_server.db import (
        get_agent_dynamic_scopes,
        init_server_schema,
        set_agent_dynamic_scopes,
    )

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
