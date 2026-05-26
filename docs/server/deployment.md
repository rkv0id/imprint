# Deployment

## SQLite (single-node, local)

Zero infrastructure. Good for development, single-user tools, and low-traffic
single-instance deployments.

```sh
IMPRINT_STORE=sqlite:///./data/imprint.db \
IMPRINT_AUTH_DISABLED=false \
imprint-server serve
```

## Postgres (multi-node, production)

Requires `pgvector/pgvector:pg16` (ships with the pgvector extension):

```yaml
# docker-compose.yml (included in the repo)
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: imprint
      POSTGRES_USER: imprint
      POSTGRES_PASSWORD: imprint

  imprint-server:
    image: ghcr.io/rkv0id/imprint-server:latest
    environment:
      IMPRINT_STORE: postgres://imprint:imprint@postgres/imprint
      IMPRINT_AUTH_DISABLED: "false"
    depends_on:
      postgres:
        condition: service_healthy
```

```sh
docker compose -f imprint-server/docker-compose.yml up
```

## Full production stack

Postgres + Redis + imprint-server in one compose file:

```sh
docker compose -f imprint-server/docker-compose.live.yml up --build --wait
```

This stack enables:

- Postgres + pgvector storage
- Redis distributed policy cache and rate limiting
- Voyage AI embedder (set `VOYAGE_API_KEY`)
- Prometheus metrics (extended gauges enabled)
- Multi-worker mode (`IMPRINT_WORKERS=4`)

## Docker image

```sh
docker pull ghcr.io/rkv0id/imprint-server:latest

docker run \
  -e IMPRINT_STORE=postgres://... \
  -e IMPRINT_AUTH_DISABLED=false \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -p 8000:8000 \
  ghcr.io/rkv0id/imprint-server:latest
```

The image includes all extras: `postgres`, `vector`, `voyage`, `openai`,
`online`, `redis`. Swap embedder and vector store via env vars -- no rebuild.

## Schema migrations

Migrations run automatically on server startup. To run them separately
(useful in CD pipelines before starting the server):

```sh
imprint-server migrate
```

The migration system uses versioned SQL files with SHA-256 checksum
verification. Modifying a shipped migration file after it has been applied
raises a `RuntimeError` at startup.

## Health probes

Use for container liveness/readiness:

```yaml
healthcheck:
  test: ["CMD", "curl", "-sf", "http://localhost:8000/health/live"]
  interval: 10s
  timeout: 5s
  retries: 3
```

| Endpoint | Checks | Use for |
|---|---|---|
| `/health/live` | Process is running | Container liveness probe |
| `/health/ready` | DB + Redis reachable | Container readiness probe, load balancer |

## Prometheus metrics

```
GET /metrics
```

Standard process and Python metrics plus imprint-specific counters:

| Metric | Type | Description |
|---|---|---|
| `imprint_observe_total` | counter | Total observe() calls by agent and mode |
| `imprint_observe_errors_total` | counter | Failed observe() calls |
| `imprint_observe_latency_seconds` | histogram | observe() latency |
| `imprint_policy_total` | counter | Total get_policy() calls |
| `imprint_policy_errors_total` | counter | Failed get_policy() calls |
| `imprint_policy_latency_seconds` | histogram | get_policy() latency, labelled by cache hit |
| `imprint_policy_cache_hits_total` | counter | Redis cache hits |
| `imprint_policy_cache_misses_total` | counter | Redis cache misses |
| `imprint_policy_memories_retrieved` | histogram | Memories retrieved per policy call |
| `imprint_policy_memories_dropped` | histogram | Memories dropped by token budget |
| `imprint_redis_invalidations_total` | counter | Redis cache invalidations |
| `imprint_consolidation_pruned_total` | counter | Memories pruned by consolidation |
| `imprint_scheduler_job_total` | counter | Background scheduler jobs run |

Extended gauges (set `IMPRINT_METRICS_EXTENDED=true`):

| Metric | Type | Description |
|---|---|---|
| `imprint_memories_active` | gauge | Active memory count by agent |
| `imprint_bandit_alpha_estimate` | gauge | BanditAlphaTuner alpha estimate by agent |

## Multi-worker deployment

Multiple workers share state via Postgres. SQLite does not support multiple writers.

```sh
IMPRINT_WORKERS=4 imprint-server serve
```

!!! warning
    Always use Postgres with `IMPRINT_WORKERS > 1`. SQLite with multiple
    workers will produce write errors and data corruption.

## Secret management

Use `_FILE` variants to pass secrets from files (Docker secrets, Kubernetes secrets):

```sh
ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key
VOYAGE_API_KEY_FILE=/run/secrets/voyage_key
IMPRINT_STORE_FILE=/run/secrets/database_url
IMPRINT_REDIS_URL_FILE=/run/secrets/redis_url
```

See [Production checklist](../guides/production.md) for a full pre-deploy checklist.
