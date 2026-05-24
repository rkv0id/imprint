"""API key authentication middleware for imprint-server.

Enabled when IMPRINT_AUTH_DISABLED=false. Disabled by default.

Every request except /health and /metrics must carry:
  Authorization: Bearer sk-imp-<64 hex chars>

The middleware hashes the raw key with SHA-256 and looks it up in the
api_keys table. If found, active, not expired, and agent_id scope matches
the path agent_id (or the key is a master key with NULL agent_id), the
request proceeds. Otherwise 401 or 403.

Agent_id scoping:
  api_keys.agent_id = NULL   master key, valid for all agents
  api_keys.agent_id = "foo"  only valid for paths under /v1/agents/foo/...

Auto-generate:
  On first auth-enabled startup with no active keys in the DB, a master key
  is generated and printed once to stdout. The raw key is never stored.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from imprint_server.stores.api_keys import (
    ApiKeyRow,
    generate_raw_key,
    hash_key,
    lookup_api_key,
)

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig
    from imprint_server.registry import AgentRegistry

# Paths that bypass auth entirely.
_AUTH_EXEMPT = frozenset({"/health", "/metrics", "/admin"})

# Path prefix that carries an agent_id segment we can scope-check.
_AGENTS_PREFIX = "/v1/agents/"


def _problem(status: int, title: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": "about:blank", "title": title, "status": status, "detail": detail},
        headers={"Content-Type": "application/problem+json"},
    )


def _extract_agent_id_from_path(path: str) -> str | None:
    """Return the agent_id segment from /v1/agents/{agent_id}/... or None."""
    if not path.startswith(_AGENTS_PREFIX):
        return None
    rest = path[len(_AGENTS_PREFIX) :]
    if not rest:
        return None
    return rest.split("/")[0] or None


def _key_is_valid(row: ApiKeyRow) -> bool:
    if not row.active:
        return False
    return row.expires_at is None or row.expires_at >= datetime.now(UTC)


def _key_authorizes_path(row: ApiKeyRow, path: str, mcp_agent_id: str) -> bool:
    """Return True if this key may access the requested path."""
    if row.agent_id is None:
        return True  # master key
    # MCP paths don't carry an agent_id segment. Check against server MCP config.
    if path.startswith("/mcp"):
        return row.agent_id == mcp_agent_id
    path_agent = _extract_agent_id_from_path(path)
    return path_agent == row.agent_id


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate API keys on every non-exempt request."""

    def __init__(self, app: object, config: ServerConfig, registry: AgentRegistry) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._config = config
        self._registry = registry

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self._config.auth_disabled:
            return await call_next(request)

        if request.url.path in _AUTH_EXEMPT:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _problem(401, "Unauthorized", "Missing or malformed Authorization header.")

        raw_key = auth_header[len("Bearer ") :]

        row = await lookup_api_key(self._config, self._registry, raw_key)
        if row is None or not _key_is_valid(row):
            return _problem(401, "Unauthorized", "Invalid or expired API key.")

        if not _key_authorizes_path(row, request.url.path, self._config.mcp_agent_id):
            return _problem(
                403,
                "Forbidden",
                "This key is not authorized for the requested agent.",
            )

        return await call_next(request)


async def maybe_generate_master_key(config: ServerConfig, registry: AgentRegistry) -> None:
    """If auth is enabled and no active keys exist, auto-generate a master key.

    Called once during ASGI lifespan startup, after registry.startup().
    The raw key is printed to stdout and never stored.
    """
    if config.auth_disabled:
        return

    raw_key = await _create_first_key_if_needed(config, registry)
    if raw_key is None:
        return

    print(
        "\n"
        "=================================================================\n"
        "imprint-server: no API keys found -- auto-generated a master key.\n"
        "Copy it now. It will not be shown again.\n"
        "\n"
        f"  {raw_key}\n"
        "\n"
        "Use: Authorization: Bearer <key>\n"
        "=================================================================\n",
        file=sys.stdout,
        flush=True,
    )


async def _create_first_key_if_needed(config: ServerConfig, registry: AgentRegistry) -> str | None:
    """Create a master key if none exist. Returns the raw key or None."""
    if config.is_postgres:
        from datetime import UTC, datetime

        from imprint_server._pool import get_pg_pool
        from imprint_server.stores.api_keys import (
            ApiKeyRow,
            pg_count_with_pool,
            pg_insert_with_pool,
        )

        pool = get_pg_pool(registry)
        if await pg_count_with_pool(pool) > 0:
            return None
        key = generate_raw_key()
        row = ApiKeyRow(
            key_hash=hash_key(key),
            agent_id=None,
            label="auto-generated master key",
            created_at=datetime.now(UTC),
            expires_at=None,
            active=True,
        )
        await pg_insert_with_pool(pool, row)
        return key

    from imprint_server.stores.api_keys import count_active_keys, insert_key

    if await count_active_keys(config) > 0:
        return None
    key = generate_raw_key()
    await insert_key(config, raw_key=key, label="auto-generated master key")
    return key
