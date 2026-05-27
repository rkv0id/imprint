# Server overview

imprint-server wraps imprint-mem in a production-grade HTTP service.

## Install and start

```sh
pip install imprint-server

# SQLite (local, zero infrastructure)
imprint-server serve

# Postgres (production)
IMPRINT_STORE=postgres://user:pass@host/db imprint-server serve
```

On first start with auth enabled and an empty key store, the server prints a
generated master API key. Copy it -- it is not stored anywhere.

## What the server adds

| Feature | Library | Server |
|---|:---:|:---:|
| HTTP/REST API | | ✓ |
| MCP SSE endpoint | | ✓ |
| Multi-agent via REST | | ✓ |
| Multi-user MCP (per-connection identity) | | ✓ |
| API key auth | | ✓ |
| Rate limiting (Redis-backed) | | ✓ |
| Distributed policy cache (Redis) | | ✓ |
| Admin dashboard at /admin | | ✓ |
| Prometheus metrics | | ✓ |
| Scheduled consolidation | | ✓ |
| Background job queue | | ✓ |
| Docker image | | ✓ |

## Quick demo

Seed a running server with demo data and open the admin dashboard:

```sh
# One-shot: starts the server, seeds data, opens dashboard URL
just demo

# Or manually:
just server-dev        # terminal 1 -- starts server with auth disabled
just demo-seed         # terminal 2 -- seeds agents, memories, sessions, keys
```

Open `http://localhost:8000/admin` to explore the dashboard.

## Architecture

```
HTTP client / MCP client
        |
   AuthMiddleware
        |
   RateLimitMiddleware (Redis, optional)
        |
   Router (/v1/agents/*, /health, /metrics, /admin)
        |
   AgentRegistry (per-agent Imprint instances)
        |
   MemoryStore (SQLite or Postgres)
        |
   VectorStore (SQLite-vec or pgvector, optional)
```

The `AgentRegistry` holds one `Imprint` instance per agent ID, initialized
on first access. Agents are configured via `POST /v1/agents` or automatically
on first observe/policy call.

## CLI

```sh
imprint-server serve             # start the HTTP server
imprint-server migrate           # run schema migrations only
imprint-server keys create       # generate a new API key
imprint-server keys list         # list all keys (hashes + labels)
imprint-server keys revoke HASH  # revoke a key by hash prefix
```

## Python client

```python
from imprint.client import ImprintClient  # pip install imprint-mem[client]

async with ImprintClient("http://localhost:8000", api_key="sk-imp-...") as client:
    policy = await client.get_policy("my-agent", "user-1")
    await client.observe("my-agent", "user-1",
        agent_output="...", user_response="...")
```

See [API reference](../library/api.md#imprintclient-imprint-memclient) for the
full client API.
