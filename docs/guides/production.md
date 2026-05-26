# Production checklist

Pre-deploy checklist for running imprint-server in production.

## Infrastructure

- [ ] Postgres with pgvector extension (`pgvector/pgvector:pg16`)
- [ ] Redis for distributed policy cache and rate limiting
- [ ] Reverse proxy (nginx, Caddy, or cloud LB) handling TLS termination
- [ ] Health probe at `/health/ready` for readiness, `/health/live` for liveness

## Configuration

- [ ] `IMPRINT_STORE` set to Postgres URL
- [ ] `IMPRINT_AUTH_DISABLED=false` -- auth enabled
- [ ] `IMPRINT_RATE_LIMIT_ENABLED=true` with Redis configured
- [ ] `IMPRINT_WORKERS` set to CPU count (Postgres only; never > 1 with SQLite)
- [ ] `IMPRINT_DRAIN_TIMEOUT` sized for your background task latency (default 30s)
- [ ] `IMPRINT_METRICS_EXTENDED=true` if using Prometheus alerting on memory counts

## Secrets

- [ ] All API keys passed via `_FILE` variants (not inline env vars)
- [ ] `ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key`
- [ ] `IMPRINT_STORE_FILE=/run/secrets/database_url` (if using file-based secrets)
- [ ] At least one master API key created before disabling auth (`imprint-server keys create`)

## Auth and keys

- [ ] Master key created and stored securely (shown once at creation)
- [ ] Per-agent keys created for automated clients
- [ ] Per-user keys created for multi-user MCP deployments
- [ ] Key rotation plan documented

## Schema

- [ ] Run `imprint-server migrate` in CD pipeline before starting the server
- [ ] Never modify shipped migration files after they have been applied
- [ ] Migration checksums verified on every deploy (automatic on startup)

## Monitoring

- [ ] `/metrics` scraped by Prometheus
- [ ] Alert on `imprint_observe_errors_total` rate spike
- [ ] Alert on `imprint_policy_errors_total` rate spike
- [ ] Alert on `imprint_policy_latency_seconds{p99}` exceeding SLA
- [ ] Alert on `/health/ready` returning degraded status
- [ ] Extended metrics enabled if alerting on memory counts: `IMPRINT_METRICS_EXTENDED=true`

## Embedder (if using vector retrieval)

- [ ] `IMPRINT_EMBEDDER=voyage` or `openai`
- [ ] `IMPRINT_EMBEDDER_DIM` matches the configured model's output dimension
- [ ] `IMPRINT_VECTOR_STORE=postgres` with pgvector
- [ ] `VOYAGE_API_KEY` or `OPENAI_API_KEY` set

## Decay and learning

- [ ] Decide between `static` and `gradient` decay: `IMPRINT_DECAY_MODEL=gradient` for production deployments that benefit from learned parameters
- [ ] `imprint-mem[online]` installed if using gradient decay
- [ ] Review `prune_threshold` for your memory volume -- lower threshold = more aggressive pruning

## Backup

- [ ] Postgres backup schedule configured (memories and keys are in Postgres)
- [ ] Backup tested (restore drill at least quarterly)
- [ ] SQLite WAL checkpointing if running SQLite in production (not recommended)

## Example production compose

```yaml
services:
  imprint-server:
    image: ghcr.io/rkv0id/imprint-server:latest
    environment:
      IMPRINT_STORE: postgres://imprint:${PG_PASSWORD}@postgres/imprint
      IMPRINT_AUTH_DISABLED: "false"
      IMPRINT_REDIS_URL: redis://redis:6379/0
      IMPRINT_RATE_LIMIT_ENABLED: "true"
      IMPRINT_WORKERS: "4"
      IMPRINT_METRICS_EXTENDED: "true"
      IMPRINT_EMBEDDER: voyage
      IMPRINT_VECTOR_STORE: postgres
      ANTHROPIC_API_KEY_FILE: /run/secrets/anthropic_key
      VOYAGE_API_KEY_FILE: /run/secrets/voyage_key
    secrets:
      - anthropic_key
      - voyage_key
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/health/live"]
      interval: 10s
      timeout: 5s
      retries: 3
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

secrets:
  anthropic_key:
    file: ./secrets/anthropic_key.txt
  voyage_key:
    file: ./secrets/voyage_key.txt
```
