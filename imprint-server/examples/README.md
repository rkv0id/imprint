# imprint-server examples

Four examples showing how to use imprint-server as a deployed memory service,
from a local dev demo to a full production stack.

All examples assume a running imprint-server instance. The fastest way to start one:

```sh
just server-dev   # SQLite, auth disabled, port 8000, auto-reload
```

## Overview

| Example | Requires server | API keys needed | Teaches |
|---|---|---|---|
| seed_demo.py | yes (port 8000) | none | Populates demo data, open /admin to inspect |
| with_server_client.py | yes (port 8000) | none (frugal mode) | ImprintClient, paginate_memories, sessions, correct, reinforce |
| with_server_and_pydantic_ai.py | yes (port 8000) | ANTHROPIC | PydanticAI + ImprintClient, multi-service pattern |
| with_production_server.py | yes (port 18001, full stack) | none (server has keys) | Full stack: Voyage, Redis cache, Postgres, gradient decay, pagination |

---

## seed_demo.py

Populates a running server with demo agents, behavioral memories, and API keys.
Open `http://localhost:8000/admin` after running to explore the dashboard.

```sh
# Start the server (one terminal):
just server-dev

# Seed demo data (another terminal):
just demo-seed

# Or: start server + seed + open dashboard in one command:
just demo
```

---

## with_server_client.py

A customer support agent that learns per-user communication preferences via
the typed HTTP client (`ImprintClient`). The server runs in frugal mode so
no LLM API key is needed. Demonstrates `get_policy`, `observe`, `correct`,
`reinforce`, `paginate_memories`, and `search_memories`.

```sh
# Start the server (one terminal):
just server-dev

# Run the example (another terminal):
uv run python imprint-server/examples/with_server_client.py
```

---

## with_server_and_pydantic_ai.py

The multi-service architecture: a PydanticAI agent in one process, imprint-server
as the memory backend over HTTP. Tools are defined manually wrapping `ImprintClient`
rather than using `make_pydantic_ai_tools` (which takes a local `Imprint` instance).
This is the production pattern for separate agent and memory deployments.

```sh
# Start the server in balanced mode (one terminal):
cd imprint-server && IMPRINT_DEFAULT_MODE=balanced uv run imprint-server serve

# Run the example (another terminal):
export ANTHROPIC_API_KEY=sk-ant-...
uv run python imprint-server/examples/with_server_and_pydantic_ai.py
```

---

## with_production_server.py

Shows the complete feature set working through the HTTP client against the full
production stack: Voyage AI semantic search, LLM policy compilation, Redis cache,
Postgres persistent store, and gradient decay learning from session outcomes.

```sh
# Start the full stack:
docker compose -f imprint-server/docker-compose.live.yml up --build --wait

# Or via just (starts, runs the example, then tears down):
just run-production-example

# Run the example against the running stack:
uv run python imprint-server/examples/with_production_server.py
```
