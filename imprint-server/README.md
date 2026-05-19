# imprint-server

Networked memory service for AI agents, built on [imprint-mem](https://pypi.org/project/imprint-mem/).

Exposes the full imprint-mem API over HTTP/REST and MCP SSE. Supports SQLite
for local development and Postgres for production multi-worker deployments.

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
# Start Postgres + imprint-server via Docker Compose
docker compose -f imprint-server/docker-compose.yml up

# Or build and run the image directly
docker build -t imprint-server -f imprint-server/Dockerfile .
docker run -p 8000:8000 imprint-server
```

## MCP (Claude Code, Cursor, Continue)

Set two environment variables before starting:

```sh
IMPRINT_MCP_AGENT_ID=my-agent \
IMPRINT_MCP_USER_ID=my-user \
imprint-server serve
```

Then add `http://localhost:8000/mcp/sse` as an MCP server in your client.
Six tools are available: `imprint_begin_session`, `imprint_get_policy`,
`imprint_observe`, `imprint_recall`, `imprint_direct`, `imprint_end_session`.

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

    # Session-scoped usage (enables learning signal on close):
    async with client.session("my-agent", "user-1", context="coding") as sess:
        policy = await sess.get_policy()
        await sess.observe("output", "response")
        sess.set_outcome(0.9)

# Agent-scoped shortcut (avoids repeating agent_id):
agent = client.agent("my-agent")
policy = await agent.get_policy("user-1")
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
GET    /v1/agents/{agent_id}/memories/{user_id}
DELETE /v1/agents/{agent_id}/memories/{user_id}
POST   /v1/agents/{agent_id}/memories/{user_id}/consolidate
POST   /v1/agents/{agent_id}/memories/{user_id}/directions
GET    /v1/agents/{agent_id}/events/{user_id}
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

GET    /health
GET    /metrics
GET    /mcp/sse
```

## Configuration

All settings via environment variables (prefix `IMPRINT_`). See
[.env.example](.env.example) for the full list with defaults and comments.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_STORE` | `sqlite:///~/.imprint/imprint.db` | SQLite path or Postgres URL |
| `IMPRINT_AUTH_DISABLED` | `true` | Set `false` to require API keys |
| `IMPRINT_DEFAULT_MODE` | `balanced` | `frugal`, `balanced`, or `eager` |
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
just server-integration-test      # Postgres tests via Docker Compose
just test-all                     # full suite: library + server + Postgres
```
