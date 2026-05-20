# imprint-server

Networked memory service for AI agents, built on [imprint-mem](https://pypi.org/project/imprint-mem/).

Exposes the full imprint-mem API over HTTP/REST and MCP SSE. Supports SQLite
for local development and Postgres + Redis for production multi-worker deployments.

## Install

```sh
pip install imprint-server
```

## Quick start

```sh
# SQLite (local, zero infrastructure)
imprint-server serve

# Postgres (production)
IMPRINT_STORE=postgres://user:pass@host/db \
IMPRINT_AUTH_DISABLED=false \
imprint-server serve
```

On first start with auth enabled and an empty key store, the server generates
and prints a master API key. Copy it -- it is not stored.

## Docker

```sh
# Postgres + imprint-server (production)
docker compose -f imprint-server/docker-compose.yml up

# Full stack: Postgres + Redis + imprint-server + Voyage embedder + pgvector
# (rate limiting, distributed policy cache, semantic search)
docker compose -f imprint-server/docker-compose.live.yml up --build --wait
```

The server image ships with all embedder extras pre-installed (`voyage`, `openai`,
`postgres`, `vector`, `online`, `redis`). Swap embedder and vector store via env vars
-- no image rebuild required.

### Docker secrets

To avoid passing API keys as env vars, use the `_FILE` variant:

```sh
ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key
VOYAGE_API_KEY_FILE=/run/secrets/voyage_key
IMPRINT_REDIS_URL_FILE=/run/secrets/redis_url
```

The file content is read at startup and used as the value of the base variable.

## MCP (Claude Code, Cursor, Continue)

Set two environment variables before starting:

```sh
IMPRINT_MCP_AGENT_ID=my-agent \
IMPRINT_MCP_USER_ID=my-user \
imprint-server serve
```

Then add `http://localhost:8000/mcp/sse` as an MCP server in your client.
Eight tools are available: `imprint_begin_session`, `imprint_get_policy`,
`imprint_observe`, `imprint_recall`, `imprint_direct`, `imprint_end_session`,
`imprint_correct`, `imprint_reinforce`.

## Python client

```python
from imprint.client import ImprintClient

# Requires: pip install imprint-mem[client]

async with ImprintClient("http://localhost:8000", api_key="sk-imp-...") as client:
    # Compile a behavioral policy for a user.
    policy = await client.get_policy("my-agent", "user-1")
    print(policy.text)

    # Record a turn.
    await client.observe("my-agent", "user-1",
        agent_output="Here is a bullet list.",
        user_response="No bullet points please.")

    # Search memories semantically (falls back to list order without embedder):
    memories = await client.search_memories("my-agent", "user-1", "formatting style")

    # Paginate memories when the list is large:
    page = await client.paginate_memories("my-agent", "user-1", limit=50)
    while page.has_more:
        page = await client.paginate_memories("my-agent", "user-1",
            limit=50, cursor=page.next_cursor)

    # Store a correction and apply a negative learning signal:
    await client.correct("my-agent", "user-1", "No bullet points please.")

    # Pin a memory so it is never dropped by token budget truncation:
    await client.pin_memory("my-agent", memories[0].id)

    # Soft-deactivate a single memory (reversible, unlike forget()):
    await client.deactivate_memory("my-agent", "user-1", memories[0].id)

    # Session-scoped usage (enables learning signal on close):
    async with client.session("my-agent", "user-1", context="coding") as sess:
        policy = await sess.get_policy()
        await sess.observe("output", "response")
        sess.set_outcome(0.9)

    # Or apply signals directly:
    sess_id = await client.open_session("my-agent", "user-1")
    await client.reinforce("my-agent", "user-1", session_id=sess_id)

# Agent-scoped shortcut (avoids repeating agent_id):
agent = client.agent("my-agent")
page = await agent.paginate_memories("user-1", limit=50)
```

## CLI

```sh
imprint-server serve             # start the HTTP server
imprint-server migrate           # run schema migrations only (no server)
imprint-server keys create       # generate a new API key
imprint-server keys list         # list all keys (hashes + labels)
imprint-server keys revoke HASH  # revoke a key by its SHA-256 hash
```

## REST API

```
POST   /v1/agents/{agent_id}/observe
POST   /v1/agents/{agent_id}/policy
GET    /v1/agents/{agent_id}/memories/{user_id}?limit=N&cursor=C
GET    /v1/agents/{agent_id}/memories/{user_id}/search?q=text&limit=N
DELETE /v1/agents/{agent_id}/memories/{user_id}
DELETE /v1/agents/{agent_id}/memories/{user_id}/{memory_id}
POST   /v1/agents/{agent_id}/memories/{memory_id}/pin
POST   /v1/agents/{agent_id}/memories/{user_id}/consolidate
POST   /v1/agents/{agent_id}/memories/{user_id}/directions
POST   /v1/agents/{agent_id}/correct/{user_id}
POST   /v1/agents/{agent_id}/reinforce/{user_id}
GET    /v1/agents/{agent_id}/events/{user_id}?limit=N&cursor=C
GET    /v1/agents/{agent_id}/health/{user_id}

GET    /v1/memories/{memory_id}/lineage

POST   /v1/agents/{agent_id}/sessions
POST   /v1/agents/{agent_id}/sessions/{id}/observe
POST   /v1/agents/{agent_id}/sessions/{id}/policy
POST   /v1/agents/{agent_id}/sessions/{id}/close

GET    /v1/agents
POST   /v1/agents
GET    /v1/agents/{agent_id}
PATCH  /v1/agents/{agent_id}/config
DELETE /v1/agents/{agent_id}
POST   /v1/agents/{agent_id}/scopes/consolidate

GET    /health          (alias for /health/ready)
GET    /health/live     (liveness probe -- no DB check)
GET    /health/ready    (readiness probe -- checks DB + Redis)
GET    /metrics
GET    /mcp/sse
```

Paginated endpoints return `{items: [...], next_cursor: str | null}` when
`limit` is provided, or a plain list for backward compatibility.

All errors use RFC 9457 `application/problem+json` with `type`, `title`,
`status`, and `detail` fields. Every response includes `X-Request-ID`.

## Configuration

All settings via environment variables (prefix `IMPRINT_`). See
[.env.example](.env.example) for the full list with defaults and comments.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_STORE` | `sqlite:///~/.imprint/imprint.db` | SQLite path or Postgres URL |
| `IMPRINT_AUTH_DISABLED` | `true` | Set `false` to require API keys |
| `IMPRINT_DEFAULT_MODE` | `balanced` | `frugal`, `balanced`, or `eager` |
| `IMPRINT_EMBEDDER` | `none` | `none`, `voyage`, or `openai` |
| `IMPRINT_EMBEDDER_MODEL` | `voyage-3` | Model string for the chosen embedder |
| `IMPRINT_EMBEDDER_DIM` | `1024` | Output dimension (must match vector store) |
| `IMPRINT_VECTOR_STORE` | `none` | `none`, `sqlite-vec`, or `postgres` |
| `IMPRINT_DECAY_MODEL` | `static` | `static` or `gradient` (requires `[online]`) |
| `IMPRINT_REDIS_URL` | `` | Redis URL -- enables rate limiting and distributed policy cache |
| `IMPRINT_CACHE_TTL` | `3600` | Policy cache TTL in seconds (when Redis is configured) |
| `IMPRINT_RATE_LIMIT_ENABLED` | `false` | Enable sliding window rate limiting |
| `IMPRINT_RATE_LIMIT_REQUESTS` | `100` | Requests per window per key/IP |
| `IMPRINT_RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |
| `IMPRINT_DRAIN_TIMEOUT` | `30` | Seconds to wait for background tasks on SIGTERM |
| `IMPRINT_MCP_AGENT_ID` | `` | Agent ID for the MCP endpoint |
| `IMPRINT_MCP_USER_ID` | `` | User namespace for the MCP endpoint |
| `IMPRINT_PORT` | `8000` | Bind port |
| `IMPRINT_WORKERS` | `1` | Uvicorn worker count (Postgres only for >1) |

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```sh
just server-dev                   # serve with SQLite, auth disabled
just server-mcp-dev               # serve with MCP enabled
just server-check                 # lint, typecheck, test
just server-live-test             # live retrieval tests (requires VOYAGE_API_KEY)
just server-integration-test      # Postgres tests via Docker Compose
just server-redis-test            # Redis rate limit tests via Docker
just test-all                     # full suite: library + server + Postgres + Redis (requires Docker)
just live-all                     # all live tests across library + server (requires API keys)
```
