# imprint-server

[![PyPI](https://img.shields.io/pypi/v/imprint-server?color=0d9488&label=imprint-server)](https://pypi.org/project/imprint-server/)
[![License](https://img.shields.io/badge/license-Apache%202.0-0d9488)](../LICENSE)

Networked behavioral memory service for AI agents, built on [imprint-mem](https://pypi.org/project/imprint-mem/).

Exposes the full imprint-mem API over HTTP/REST and MCP SSE. Supports SQLite
for local development and Postgres + Redis for production multi-worker deployments.

## Install

```sh
pip install imprint-server
```

## Quick start

```sh
# SQLite (local, zero infrastructure -- auth disabled by default)
imprint-server serve

# Postgres (production)
IMPRINT_STORE=postgres://user:pass@host/db \
IMPRINT_AUTH_DISABLED=false \
imprint-server serve
```

On first start with auth enabled and an empty key store, the server generates
and prints a master API key. Copy it -- it is not stored.

## Demo

Seed a running server with demo data and open the admin dashboard:

```sh
just demo
```

This starts the server, seeds four agents (frugal, balanced, frugal, eager),
stores behavioral memories for multiple users, runs sessions with outcome
signals to exercise the online learning path, and prints stability readouts
showing FSRS decay in action.

## Docker

```sh
# Postgres + imprint-server
docker compose -f imprint-server/docker-compose.yml up

# Full stack: Postgres + Redis + imprint-server + Voyage embedder + pgvector
docker compose -f imprint-server/docker-compose.live.yml up --build --wait
```

The server image ships with all extras pre-installed. Swap embedder and vector
store via env vars -- no rebuild required.

### Docker secrets

```sh
ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key
VOYAGE_API_KEY_FILE=/run/secrets/voyage_key
IMPRINT_REDIS_URL_FILE=/run/secrets/redis_url
```

## Admin dashboard

A read-only admin dashboard is served at `/admin`. No build step, no external
dependencies -- a single self-contained HTML/CSS/JS file.

Panels: Overview, Agents, Memory Browser (stability bars + health stats),
Events (memory event log per agent/user), API Keys (stats + key list).

Auto-refreshes every 10 seconds. Auth prompt on first load when auth is enabled.

## MCP (Claude Code, Cursor, Continue)

```sh
IMPRINT_MCP_AGENT_ID=my-agent \
IMPRINT_MCP_USER_ID=me \
imprint-server serve
```

Add `http://localhost:8000/mcp/sse` as an MCP server in your client.

Eight tools: `imprint_begin_session`, `imprint_get_policy`, `imprint_observe`,
`imprint_recall`, `imprint_direct`, `imprint_end_session`, `imprint_correct`,
`imprint_reinforce`.

### Multi-user MCP

With auth enabled, user identity is resolved per-connection from the Bearer
token. Create a user-bound key:

```sh
imprint-server keys create --label "alice-key" --agent my-agent --user alice
# or via REST:
curl -X POST http://localhost:8000/v1/keys \
  -H "Authorization: Bearer $MASTER_KEY" \
  -d '{"label": "alice-key", "agent_id": "my-agent", "user_id": "alice"}'
```

Each user connects with their own key. Every tool call is automatically scoped
to their memory namespace -- no code change required on the client side.

## API keys

Keys are managed via CLI or REST. The raw key is shown once at creation and
never stored.

```sh
# CLI
imprint-server keys create --label "ci-key"
imprint-server keys create --label "alice" --agent my-agent --user alice
imprint-server keys list
imprint-server keys revoke HASH_PREFIX

# REST (requires master key when auth is enabled)
curl -X POST http://localhost:8000/v1/keys \
  -H "Authorization: Bearer $MASTER_KEY" \
  -d '{"label": "ci-key", "agent_id": "my-agent"}'

curl -X GET http://localhost:8000/v1/keys \
  -H "Authorization: Bearer $MASTER_KEY"

curl -X DELETE http://localhost:8000/v1/keys/a1b2c3d4e5f6a1b2 \
  -H "Authorization: Bearer $MASTER_KEY"
```

## Python client

```python
from imprint.client import ImprintClient  # pip install imprint-mem[client]

async with ImprintClient("http://localhost:8000", api_key="sk-imp-...") as client:
    policy = await client.get_policy("my-agent", "alice")
    await client.observe("my-agent", "alice",
        agent_output="Here is a list.",
        user_response="Prose please.")

    memories = await client.search_memories("my-agent", "alice", "formatting style")
    page = await client.paginate_memories("my-agent", "alice", limit=50)

    # Session with outcome signal (online learning path)
    async with client.session("my-agent", "alice", context="code review") as sess:
        policy = await sess.get_policy()
        await sess.observe("output", "response")
        sess.set_outcome(0.9)  # 0=correction, 0.5=neutral, 1=ideal
```

## CLI

```sh
imprint-server serve             # start the HTTP server
imprint-server migrate           # run schema migrations only
imprint-server keys create       # generate a new API key
imprint-server keys list         # list all keys (hashes + labels)
imprint-server keys revoke HASH  # revoke a key by its SHA-256 hash prefix
```

## REST API

```
# Memory operations
POST   /v1/agents/{agent_id}/observe
POST   /v1/agents/{agent_id}/observe/batch        # up to 100 items, partial failure
POST   /v1/agents/{agent_id}/policy
GET    /v1/agents/{agent_id}/memories/{user_id}?limit=N&cursor=C
GET    /v1/agents/{agent_id}/memories/{user_id}/search?q=text&limit=N
GET    /v1/agents/{agent_id}/memories/{user_id}/diff?since=ISO&until=ISO
DELETE /v1/agents/{agent_id}/memories/{user_id}
DELETE /v1/agents/{agent_id}/memories/{user_id}/{memory_id}
POST   /v1/agents/{agent_id}/memories/{memory_id}/pin
POST   /v1/agents/{agent_id}/memories/{user_id}/consolidate
POST   /v1/agents/{agent_id}/memories/{user_id}/directions
POST   /v1/agents/{agent_id}/correct/{user_id}
POST   /v1/agents/{agent_id}/reinforce/{user_id}
GET    /v1/agents/{agent_id}/events/{user_id}
GET    /v1/agents/{agent_id}/health/{user_id}
GET    /v1/memories/{memory_id}/lineage

# Sessions (online learning signal path)
POST   /v1/agents/{agent_id}/sessions
POST   /v1/agents/{agent_id}/sessions/{id}/observe
POST   /v1/agents/{agent_id}/sessions/{id}/policy
POST   /v1/agents/{agent_id}/sessions/{id}/close  # {"outcome": 0.9}

# Agent administration
GET    /v1/agents
POST   /v1/agents
GET    /v1/agents/{agent_id}
PATCH  /v1/agents/{agent_id}/config
DELETE /v1/agents/{agent_id}
POST   /v1/agents/{agent_id}/scopes/consolidate

# API keys (REST)
GET    /v1/keys
POST   /v1/keys
DELETE /v1/keys/{key_hash}

# System
GET    /health          (alias for /health/ready)
GET    /health/live     (liveness probe)
GET    /health/ready    (readiness probe -- checks DB + Redis)
GET    /metrics         (Prometheus exposition format)
GET    /admin           (read-only admin dashboard)
GET    /mcp/sse         (MCP SSE endpoint)
```

Paginated endpoints return `{items: [...], next_cursor: str | null}` when
`limit` is provided, or a plain list without it.

All errors use RFC 9457 `application/problem+json`. Every response includes
`X-Request-ID`.

## Prometheus metrics

Key metrics exposed at `/metrics`:

| Metric | Description |
|---|---|
| `imprint_observe_latency_seconds` | observe() latency histogram |
| `imprint_policy_latency_seconds` | get_policy() latency, labelled by cache hit |
| `imprint_policy_cache_hits_total` | Redis cache hits |
| `imprint_policy_cache_misses_total` | Redis cache misses |
| `imprint_consolidation_pruned_total` | Memories pruned by consolidation |

With `IMPRINT_METRICS_EXTENDED=true`:

| Metric | Description |
|---|---|
| `imprint_memories_active` | Active memory count per agent |
| `imprint_bandit_alpha_estimate` | BanditAlphaTuner alpha estimate per agent |

## Configuration

All settings via `IMPRINT_` environment variables. See `.env.example` for the
full list with defaults and comments.

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_STORE` | `sqlite:///~/.imprint/imprint.db` | SQLite path or Postgres URL |
| `IMPRINT_AUTH_DISABLED` | `true` | Set `false` to require API keys |
| `IMPRINT_DEFAULT_MODE` | `balanced` | `frugal`, `balanced`, or `eager` |
| `IMPRINT_EMBEDDER` | `none` | `none`, `voyage`, or `openai` |
| `IMPRINT_VECTOR_STORE` | `none` | `none`, `sqlite-vec`, or `postgres` |
| `IMPRINT_DECAY_MODEL` | `static` | `static` or `gradient` (requires `[online]`) |
| `IMPRINT_REDIS_URL` | `` | Redis URL -- enables rate limiting + distributed cache |
| `IMPRINT_RATE_LIMIT_ENABLED` | `false` | Enable sliding window rate limiting |
| `IMPRINT_METRICS_EXTENDED` | `false` | Expose per-agent memory count + bandit alpha gauges |
| `IMPRINT_DRAIN_TIMEOUT` | `30` | Seconds to wait for background tasks on SIGTERM |
| `IMPRINT_MCP_AGENT_ID` | `` | Agent ID for the MCP endpoint |
| `IMPRINT_MCP_USER_ID` | `` | Default user namespace for MCP (auth disabled) |
| `IMPRINT_PORT` | `8000` | Bind port |
| `IMPRINT_WORKERS` | `1` | Uvicorn worker count (Postgres only for >1) |

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```sh
just server-dev                   # serve with SQLite, auth disabled
just server-mcp-dev               # serve with MCP enabled
just server-check                 # lint, typecheck, test
just server-integration-test      # Postgres tests via Docker Compose
just server-redis-test            # Redis rate limit tests via Docker
just test-all                     # full suite: library + server + Postgres + Redis + compose
just live-all                     # all live tests across library + server
just demo                         # start server + seed demo data + open admin dashboard
just docs-serve                   # preview the documentation site locally
```
