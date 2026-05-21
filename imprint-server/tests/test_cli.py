"""Tests for the imprint-server CLI.

Uses typer.testing.CliRunner to invoke commands against a real SQLite store.
The `serve` command is not tested here -- it blocks on uvicorn and is covered
by the integration tests that use create_app() directly.
"""

from __future__ import annotations

# Strip ANSI escape codes before asserting on output content.
# Typer/rich outputs color codes when running in a terminal-like environment
# (e.g. GitHub Actions), which breaks plain string membership tests.
import re as _re
from pathlib import Path

from typer.testing import CliRunner

from imprint_server.cli import app


def _plain(output: str) -> str:
    return _re.sub(r"\x1b\[[0-9;]*m", "", output)


runner = CliRunner()


# -- Help (wiring smoke tests) ------------------------------------------------


def test_help_root() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.output
    assert "migrate" in result.output
    assert "keys" in result.output


def test_help_serve() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in _plain(result.output)
    assert "--port" in _plain(result.output)
    assert "--workers" in _plain(result.output)


def test_help_migrate() -> None:
    result = runner.invoke(app, ["migrate", "--help"])
    assert result.exit_code == 0


def test_help_keys() -> None:
    result = runner.invoke(app, ["keys", "--help"])
    assert result.exit_code == 0
    assert "create" in result.output
    assert "list" in result.output
    assert "revoke" in result.output


def test_help_keys_create() -> None:
    result = runner.invoke(app, ["keys", "create", "--help"])
    assert result.exit_code == 0
    assert "--label" in _plain(result.output)
    assert "--agent" in _plain(result.output)
    assert "--user" in _plain(result.output)


def test_keys_create_with_user(tmp_path: Path) -> None:
    import os

    db_path = str(tmp_path / "keys_user.db")
    env = {**os.environ, "IMPRINT_STORE": f"sqlite:///{db_path}"}
    runner.invoke(app, ["migrate"], env=env)

    result = runner.invoke(
        app,
        ["keys", "create", "--agent", "my-agent", "--user", "alice", "--label", "alice-key"],
        env=env,
    )
    assert result.exit_code == 0, result.output
    assert "alice" in result.output
    assert "my-agent" in result.output


def test_keys_list_shows_user(tmp_path: Path) -> None:
    import os

    db_path = str(tmp_path / "keys_list_user.db")
    env = {**os.environ, "IMPRINT_STORE": f"sqlite:///{db_path}"}
    runner.invoke(app, ["migrate"], env=env)
    runner.invoke(
        app,
        ["keys", "create", "--agent", "my-agent", "--user", "bob", "--label", "bob-key"],
        env=env,
    )

    result = runner.invoke(app, ["keys", "list"], env=env)
    assert result.exit_code == 0
    assert "bob" in result.output


def test_help_keys_revoke() -> None:
    result = runner.invoke(app, ["keys", "revoke", "--help"])
    assert result.exit_code == 0


# -- migrate ------------------------------------------------------------------


def test_migrate_sqlite(tmp_path: Path) -> None:
    """migrate must apply all migrations, report versions applied, and exit 0."""
    import asyncio
    import os

    import aiosqlite

    db_path = str(tmp_path / "migrate_test.db")
    env = {**os.environ, "IMPRINT_STORE": f"sqlite:///{db_path}"}
    result = runner.invoke(app, ["migrate"], env=env)
    assert result.exit_code == 0, result.output
    assert "0001" in result.output
    assert "0002" in result.output
    assert "complete" in result.output.lower() or "up to date" in result.output.lower()

    # Verify server tables and schema_migrations were created.
    async def _check() -> set[str]:
        async with (
            aiosqlite.connect(db_path) as conn,
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor,
        ):
            return {row[0] for row in await cursor.fetchall()}

    tables = asyncio.run(_check())
    assert {"sessions", "jobs", "api_keys", "policy_events", "schema_migrations"} <= tables


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    """Running migrate twice must not raise and second run must report up to date."""
    import os

    db_path = str(tmp_path / "idempotent.db")
    env = {**os.environ, "IMPRINT_STORE": f"sqlite:///{db_path}"}
    r1 = runner.invoke(app, ["migrate"], env=env)
    r2 = runner.invoke(app, ["migrate"], env=env)
    assert r1.exit_code == 0
    assert r2.exit_code == 0
    assert "up to date" in r2.output.lower()


# -- keys create / list / revoke ----------------------------------------------


def test_keys_create_prints_key(tmp_path: Path) -> None:
    import os

    db_path = str(tmp_path / "keys_test.db")
    env = {**os.environ, "IMPRINT_STORE": f"sqlite:///{db_path}"}

    # Migrate first so the api_keys table exists.
    runner.invoke(app, ["migrate"], env=env)

    result = runner.invoke(app, ["keys", "create", "--label", "test-key"], env=env)
    assert result.exit_code == 0, result.output
    assert "sk-imp-" in result.output
    assert "test-key" in result.output


def test_keys_create_scoped(tmp_path: Path) -> None:
    import os

    db_path = str(tmp_path / "keys_scoped.db")
    env = {**os.environ, "IMPRINT_STORE": f"sqlite:///{db_path}"}
    runner.invoke(app, ["migrate"], env=env)

    result = runner.invoke(
        app, ["keys", "create", "--agent", "my-agent", "--label", "scoped"], env=env
    )
    assert result.exit_code == 0
    assert "my-agent" in result.output


def test_keys_list_empty(tmp_path: Path) -> None:
    import os

    db_path = str(tmp_path / "keys_list_empty.db")
    env = {**os.environ, "IMPRINT_STORE": f"sqlite:///{db_path}"}
    runner.invoke(app, ["migrate"], env=env)

    result = runner.invoke(app, ["keys", "list"], env=env)
    assert result.exit_code == 0
    assert "No API keys found" in result.output


def test_keys_list_shows_created_key(tmp_path: Path) -> None:
    import os

    db_path = str(tmp_path / "keys_list_show.db")
    env = {**os.environ, "IMPRINT_STORE": f"sqlite:///{db_path}"}
    runner.invoke(app, ["migrate"], env=env)
    runner.invoke(app, ["keys", "create", "--label", "listed-key"], env=env)

    result = runner.invoke(app, ["keys", "list"], env=env)
    assert result.exit_code == 0
    assert "listed-key" in result.output
    assert "yes" in result.output  # active column


def test_keys_revoke_existing_key(tmp_path: Path) -> None:
    import os

    from imprint_server.stores.api_keys import hash_key

    db_path = str(tmp_path / "keys_revoke.db")
    env = {**os.environ, "IMPRINT_STORE": f"sqlite:///{db_path}"}
    runner.invoke(app, ["migrate"], env=env)

    create_result = runner.invoke(app, ["keys", "create", "--label", "to-revoke"], env=env)
    assert create_result.exit_code == 0

    # Extract the raw key from output and compute its hash.
    raw_key = next(
        line.strip()
        for line in create_result.output.splitlines()
        if line.strip().startswith("sk-imp-")
    )
    key_hash = hash_key(raw_key)

    revoke_result = runner.invoke(app, ["keys", "revoke", key_hash], env=env)
    assert revoke_result.exit_code == 0
    assert "revoked" in revoke_result.output

    # Key should now show as inactive in list.
    list_result = runner.invoke(app, ["keys", "list"], env=env)
    assert "no" in list_result.output


def test_keys_revoke_nonexistent_exits_1(tmp_path: Path) -> None:
    import os

    db_path = str(tmp_path / "keys_revoke_missing.db")
    env = {**os.environ, "IMPRINT_STORE": f"sqlite:///{db_path}"}
    runner.invoke(app, ["migrate"], env=env)

    result = runner.invoke(app, ["keys", "revoke", "a" * 64], env=env)
    assert result.exit_code == 1
