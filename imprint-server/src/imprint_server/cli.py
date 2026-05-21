"""Command-line interface for imprint-server.

Entry point: imprint-server (declared in pyproject.toml)

Commands:
  imprint-server serve               start the HTTP server
  imprint-server migrate             run schema migrations only (no server)
  imprint-server keys create         generate and store a new API key
  imprint-server keys list           list all API keys (hashes + labels)
  imprint-server keys revoke <hash>  revoke a key by its SHA-256 hash
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
import uvicorn

from imprint_server.config import ServerConfig

app = typer.Typer(
    name="imprint-server",
    help="Networked memory service for AI agents.",
    no_args_is_help=True,
)

keys_app = typer.Typer(
    name="keys",
    help="Manage API keys.",
    no_args_is_help=True,
)
app.add_typer(keys_app)


# -- serve --------------------------------------------------------------------


@app.command()
def serve(
    host: Annotated[str, typer.Option(envvar="IMPRINT_HOST", help="Bind address.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(envvar="IMPRINT_PORT", help="Bind port.")] = 8000,
    workers: Annotated[
        int, typer.Option(envvar="IMPRINT_WORKERS", help="Uvicorn worker count.")
    ] = 1,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload (development only).")] = False,
) -> None:
    """Start the imprint-server HTTP server."""
    _load_file_secrets()
    from imprint_server.app import create_app
    from imprint_server.registry import AgentRegistry

    config = ServerConfig(host=host, port=port, workers=workers)

    if config.is_postgres and workers > 1:
        # Multi-worker: pass the app factory string to uvicorn so each worker
        # creates its own app instance with its own asyncpg connection pool.
        uvicorn.run(
            "imprint_server.cli:_make_app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level="info",
        )
    else:
        # Single-worker or SQLite: create the app directly so we can pass
        # the registry instance (avoids re-parsing config in a factory string).
        registry = AgentRegistry(config)
        application = create_app(config, registry)
        uvicorn.run(
            application,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )


def _make_app() -> object:
    """App factory for multi-worker uvicorn deployments.

    Called once per worker process. Each worker gets its own registry and
    asyncpg connection pool. Only used when IMPRINT_WORKERS > 1 and Postgres
    is configured.
    """
    _load_file_secrets()
    from imprint_server.app import create_app
    from imprint_server.registry import AgentRegistry

    config = ServerConfig()
    registry = AgentRegistry(config)
    return create_app(config, registry)


# -- migrate ------------------------------------------------------------------


@app.command()
def migrate() -> None:
    """Run schema migrations without starting the server.

    Initializes both the imprint-mem library schema and the imprint-server
    schema (sessions, jobs, api_keys, policy_events tables). Safe to run
    multiple times -- all DDL statements use CREATE TABLE IF NOT EXISTS.

    Useful for:
      - First-time setup before starting the server
      - CI/CD pipelines that need the schema created before running tests
      - Verifying the database connection is healthy
    """

    async def _run() -> None:
        from imprint_server.db import init_server_schema
        from imprint_server.registry import AgentRegistry

        config = ServerConfig()
        registry = AgentRegistry(config)

        typer.echo(f"Connecting to store: {_redact(config.store)}")
        await registry.startup()
        try:
            await init_server_schema(config, registry.store)
        finally:
            await registry.shutdown()
        typer.echo("Schema migration complete.")

    _load_file_secrets()
    asyncio.run(_run())


# -- keys ---------------------------------------------------------------------


@keys_app.command("create")
def keys_create(
    label: Annotated[
        str | None, typer.Option("--label", "-l", help="Human-readable label for this key.")
    ] = None,
    agent_id: Annotated[
        str | None,
        typer.Option("--agent", "-a", help="Scope this key to a single agent ID."),
    ] = None,
    user_id: Annotated[
        str | None,
        typer.Option("--user", "-u", help="Bind this key to a specific user namespace."),
    ] = None,
) -> None:
    """Generate a new API key and store its hash.

    The raw key is printed once and never stored. Copy it before closing
    this terminal. Use --agent to scope to one agent; use --user to bind to
    a specific user namespace (required for multi-user MCP deployments).
    """

    async def _run() -> None:
        from datetime import UTC, datetime

        from imprint_server._pool import get_pg_pool
        from imprint_server.registry import AgentRegistry
        from imprint_server.stores.api_keys import (
            ApiKeyRow,
            generate_raw_key,
            hash_key,
            insert_key,
            pg_insert_with_pool,
        )

        config = ServerConfig()
        registry = AgentRegistry(config)
        await registry.startup()

        raw_key = generate_raw_key()

        try:
            if config.is_postgres:
                pool = get_pg_pool(registry)
                row = ApiKeyRow(
                    key_hash=hash_key(raw_key),
                    agent_id=agent_id,
                    user_id=user_id,
                    label=label,
                    created_at=datetime.now(UTC),
                    expires_at=None,
                    active=True,
                )
                await pg_insert_with_pool(pool, row)
            else:
                await insert_key(
                    config, raw_key=raw_key, agent_id=agent_id, user_id=user_id, label=label
                )
        finally:
            await registry.shutdown()

        typer.echo("")
        typer.echo("=================================================================")
        typer.echo("New API key created. Copy it now -- it will not be shown again.")
        typer.echo("")
        typer.echo(f"  {raw_key}")
        typer.echo("")
        if agent_id:
            typer.echo(f"  Scoped to agent: {agent_id}")
        else:
            typer.echo("  Scope: master (all agents)")
        if user_id:
            typer.echo(f"  Bound to user:   {user_id}")
        if label:
            typer.echo(f"  Label: {label}")
        typer.echo("=================================================================")
        typer.echo("")

    _load_file_secrets()
    asyncio.run(_run())


@keys_app.command("list")
def keys_list() -> None:
    """List all API keys (key hashes and labels, never raw keys)."""

    async def _run() -> None:
        from imprint_server._pool import get_pg_pool
        from imprint_server.registry import AgentRegistry
        from imprint_server.stores.api_keys import list_keys, pg_list_with_pool

        config = ServerConfig()
        registry = AgentRegistry(config)
        await registry.startup()

        try:
            if config.is_postgres:
                pool = get_pg_pool(registry)
                rows = await pg_list_with_pool(pool)
            else:
                rows = await list_keys(config)
        finally:
            await registry.shutdown()

        if not rows:
            typer.echo("No API keys found.")
            return

        typer.echo(f"\n{'HASH':>16}  {'ACTIVE':6}  {'AGENT':20}  {'USER':20}  {'LABEL'}")
        typer.echo("-" * 90)
        for row in rows:
            short_hash = row.key_hash[:16]
            active = "yes" if row.active else "no"
            agent = row.agent_id or "(master)"
            user = row.user_id or ""
            lbl = row.label or ""
            typer.echo(f"{short_hash}  {active:6}  {agent:20}  {user:20}  {lbl}")
        typer.echo("")

    _load_file_secrets()
    asyncio.run(_run())


@keys_app.command("revoke")
def keys_revoke(
    key_hash: Annotated[str, typer.Argument(help="SHA-256 hash of the key to revoke.")],
) -> None:
    """Revoke an API key by its hash. The key is deactivated immediately."""

    async def _run() -> None:
        from imprint_server._pool import get_pg_pool
        from imprint_server.registry import AgentRegistry
        from imprint_server.stores.api_keys import pg_revoke_with_pool, revoke_key

        config = ServerConfig()
        registry = AgentRegistry(config)
        await registry.startup()

        try:
            if config.is_postgres:
                pool = get_pg_pool(registry)
                found = await pg_revoke_with_pool(pool, key_hash)
            else:
                found = await revoke_key(config, key_hash)
        finally:
            await registry.shutdown()

        if found:
            typer.echo(f"Key {key_hash[:16]}... revoked.")
        else:
            typer.echo(f"Key {key_hash[:16]}... not found.", err=True)
            raise typer.Exit(1)

    _load_file_secrets()
    asyncio.run(_run())


# -- Helpers ------------------------------------------------------------------


def _redact(store_url: str) -> str:
    """Replace password in a Postgres URL with *** for safe logging."""
    if "://" not in store_url or not store_url.startswith("postgres"):
        return store_url
    try:
        from urllib.parse import urlparse, urlunparse

        p = urlparse(store_url)
        redacted = p._replace(netloc=p.netloc.replace(f":{p.password}@", ":***@"))
        return urlunparse(redacted)
    except Exception:
        return store_url


# Env vars that support a _FILE variant for Docker secrets.
# Each entry maps the _FILE var name to the target env var name.
_FILE_VARS: dict[str, str] = {
    "ANTHROPIC_API_KEY_FILE": "ANTHROPIC_API_KEY",
    "VOYAGE_API_KEY_FILE": "VOYAGE_API_KEY",
    "OPENAI_API_KEY_FILE": "OPENAI_API_KEY",
    "IMPRINT_REDIS_URL_FILE": "IMPRINT_REDIS_URL",
}


def _load_file_secrets() -> None:
    """Read *_FILE env vars and populate the corresponding base env vars.

    Follows the Docker secrets convention: if ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic,
    read the file content and set ANTHROPIC_API_KEY to that value. Raises clearly
    if a _FILE path is set but the file does not exist or cannot be read.

    Called once at the top of every CLI command before ServerConfig is instantiated,
    so pydantic-settings picks up the populated env vars correctly.
    """
    import os

    for file_var, target_var in _FILE_VARS.items():
        path = os.environ.get(file_var)
        if not path:
            continue
        try:
            with open(path) as fh:
                value = fh.read().strip()
        except OSError as exc:
            raise SystemExit(
                f"[imprint-server] {file_var}={path!r} is set but the file could not be read: {exc}"
            ) from exc
        if not value:
            raise SystemExit(f"[imprint-server] {file_var}={path!r} is set but the file is empty.")
        os.environ[target_var] = value
