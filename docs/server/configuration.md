# Configuration

All settings are environment variables with the `IMPRINT_` prefix.
Copy `.env.example` from the repo for a complete template with comments.

## Store

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_STORE` | `sqlite:///~/.imprint/imprint.db` | SQLite path or Postgres URL |
| `IMPRINT_POOL_MIN` | `1` | Postgres connection pool minimum size |
| `IMPRINT_POOL_MAX` | `10` | Postgres connection pool maximum size |

```sh
# SQLite
IMPRINT_STORE=sqlite:///./data/imprint.db

# Postgres
IMPRINT_STORE=postgres://user:pass@host:5432/dbname
```

## Agent behavior

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_DEFAULT_MODE` | `balanced` | Default processing mode: `frugal`, `balanced`, `eager` |
| `IMPRINT_MODEL` | `anthropic:claude-haiku-4-5-20251001` | pydantic-ai model string |
| `IMPRINT_MAX_INPUT_TOKENS` | `8000` | Token budget for memory retrieval |
| `IMPRINT_MAX_OUTPUT_TOKENS` | `3000` | Token budget for policy compilation |

## Auth

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_AUTH_DISABLED` | `true` | Set `false` to require API keys |

When auth is enabled, all `/v1/*` endpoints require `Authorization: Bearer <key>`.
Health and metrics endpoints are always exempt. The admin dashboard page loads
without auth; its JS handles the token for API calls.

Keys are managed via the CLI or the REST API:

```sh
# CLI
imprint-server keys create --label "production"

# REST
curl -X POST http://localhost:8000/v1/keys \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"label": "ci-key"}'
```

## Embedder and vector store

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_EMBEDDER` | `none` | `none`, `voyage`, or `openai` |
| `IMPRINT_EMBEDDER_MODEL` | `voyage-3` | Model string for the chosen embedder |
| `IMPRINT_EMBEDDER_DIM` | `1024` | Output dimension (must match vector store) |
| `IMPRINT_VECTOR_STORE` | `none` | `none`, `sqlite-vec`, or `postgres` |

```sh
IMPRINT_EMBEDDER=voyage
IMPRINT_EMBEDDER_MODEL=voyage-3
IMPRINT_EMBEDDER_DIM=1024
IMPRINT_VECTOR_STORE=postgres
VOYAGE_API_KEY=pa-...
```

## Redis

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_REDIS_URL` | `` | Redis URL. Enables rate limiting and distributed policy cache. |
| `IMPRINT_CACHE_TTL` | `3600` | Policy cache TTL in seconds |

```sh
IMPRINT_REDIS_URL=redis://localhost:6379/0
```

## Rate limiting

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_RATE_LIMIT_ENABLED` | `false` | Enable sliding window rate limiting |
| `IMPRINT_RATE_LIMIT_REQUESTS` | `100` | Requests per window per key/IP |
| `IMPRINT_RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |

Rate limiting requires Redis. Health and metrics endpoints are always exempt.

## MCP

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_MCP_AGENT_ID` | `` | Agent ID for the MCP SSE endpoint |
| `IMPRINT_MCP_USER_ID` | `` | Default user namespace for MCP (auth disabled) |

Both must be set to mount the MCP endpoint. With auth enabled, user identity
is resolved per-connection from the Bearer token's `user_id` field.

## Server

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_HOST` | `0.0.0.0` | Bind address |
| `IMPRINT_PORT` | `8000` | Bind port |
| `IMPRINT_WORKERS` | `1` | Uvicorn worker count (Postgres only for > 1) |
| `IMPRINT_LOG_LEVEL` | `info` | Log level |
| `IMPRINT_DRAIN_TIMEOUT` | `30` | Seconds to wait for background tasks on SIGTERM |

## Metrics

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_METRICS_EXTENDED` | `false` | Expose per-agent memory count and bandit alpha gauges |
| `IMPRINT_METRICS_REFRESH_INTERVAL` | `60` | Seconds between extended gauge refreshes |

Extended metrics require a periodic DB query. Enable for dashboards and alerting:

```sh
IMPRINT_METRICS_EXTENDED=true
IMPRINT_METRICS_REFRESH_INTERVAL=30
```

## Online decay

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_DECAY_MODEL` | `static` | `static` or `gradient` (requires `[online]`) |

## Docker secrets

To avoid passing secrets as env vars, use `_FILE` variants:

```sh
ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key
VOYAGE_API_KEY_FILE=/run/secrets/voyage_key
IMPRINT_REDIS_URL_FILE=/run/secrets/redis_url
```

The file content is read once at startup and used as the variable value.
