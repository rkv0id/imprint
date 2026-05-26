# REST API

Base URL: `http://localhost:8000` (default)

Auth: `Authorization: Bearer <key>` on all `/v1/*` endpoints when auth is enabled.

The full OpenAPI spec is served at `/openapi.json`. An interactive UI is
available at `/docs` (Swagger) and `/redoc`.

## Memory operations

### Observe

```
POST /v1/agents/{agent_id}/observe
```

Record an agent-user exchange or a set of explicit directions.

```json
{
  "user_id": "alice",
  "agent_output": "Here is a bullet list.",
  "user_response": "Please use prose."
}
```

Or with explicit directions:

```json
{
  "user_id": "alice",
  "directions": ["Always write in prose.", "Be concise."]
}
```

### Batch observe

```
POST /v1/agents/{agent_id}/observe/batch
```

Observe up to 100 items in a single request. Items are processed sequentially
under a single lock acquisition. Partial failures are reported per-item.

```json
{
  "items": [
    {"user_id": "alice", "directions": ["be concise"]},
    {"user_id": "alice", "agent_output": "list.", "user_response": "prose please."}
  ]
}
```

Response:

```json
{
  "processed": 2,
  "failed": 0,
  "results": [
    {"index": 0, "ok": true, "error": null},
    {"index": 1, "ok": true, "error": null}
  ]
}
```

### Get policy

```
POST /v1/agents/{agent_id}/policy
```

Compile and return a behavioral policy for a user namespace.

```json
{
  "user_id": "alice",
  "context": "reviewing a Python PR",
  "scopes": ["style", "global"],
  "existing_instructions": "You are a helpful reviewer."
}
```

Response:

```json
{
  "policy_text": "Write feedback in prose, not bullet points.",
  "memory_count": 3,
  "dropped_count": 0,
  "compiled_at": "2025-04-01T12:00:00+00:00",
  "memory_ids": ["mem_abc123", "mem_def456", "mem_ghi789"]
}
```

### List memories

```
GET /v1/agents/{agent_id}/memories/{user_id}?scopes=style,global&limit=50&cursor=...
```

Without `limit`: returns a plain list. With `limit`: returns a paginated envelope.

```json
{
  "items": [...],
  "next_cursor": "eyJjcmVhdGVkX2F0IjoiMjAyNS0wNC0wMVQxMjowMDowMCswMDowMCJ9"
}
```

### Search memories

```
GET /v1/agents/{agent_id}/memories/{user_id}/search?q=formatting+preferences&limit=20
```

Semantic search. Falls back to list order when no embedder is configured.

### Memory diff

```
GET /v1/agents/{agent_id}/memories/{user_id}/diff?since=2025-04-01T00:00:00Z&until=2025-04-15T12:00:00Z
```

`since` is required. `until` defaults to now. Timestamps must be ISO 8601 with timezone.

```json
{
  "since": "2025-04-01T00:00:00+00:00",
  "until": "2025-04-15T12:00:00+00:00",
  "added": [...],
  "deactivated": [...],
  "superseded": [{"old": {...}, "new": {...}}],
  "summary": {"added": 3, "deactivated": 1, "superseded": 2}
}
```

### Memory health

```
GET /v1/agents/{agent_id}/health/{user_id}
```

Aggregate memory statistics for a user namespace.

### Events

```
GET /v1/agents/{agent_id}/events/{user_id}?memory_id=mem_abc&limit=50
```

Memory events (create, recall, deactivate) for a user namespace.

### Lineage

```
GET /v1/memories/{memory_id}/lineage
```

Full creation and mutation history of one memory: origin signal, supersession chain, events.

### Forget user

```
DELETE /v1/agents/{agent_id}/memories/{user_id}
```

Hard delete all memories, signals, and events for a user. Irreversible.

### Deactivate memory

```
DELETE /v1/agents/{agent_id}/memories/{user_id}/{memory_id}
```

Soft-deactivate one memory. It is excluded from future retrievals but stays in
the store for lineage tracking.

### Pin memory

```
POST /v1/agents/{agent_id}/memories/{memory_id}/pin
```

Pin a memory so it is never dropped by token budget truncation.

### Consolidate

```
POST /v1/agents/{agent_id}/memories/{user_id}/consolidate?prune_threshold=0.5
```

Prune decayed memories and run scope consolidation.

### Observe directions

```
POST /v1/agents/{agent_id}/memories/{user_id}/directions
```

Store explicit behavioral directions without signal detection.

### Correct

```
POST /v1/agents/{agent_id}/correct/{user_id}
```

Store a correction as a memory and apply a negative learning signal.

### Reinforce

```
POST /v1/agents/{agent_id}/reinforce/{user_id}
```

Apply a positive learning signal for a closed session.

## Sessions

```
POST /v1/agents/{agent_id}/sessions              # open a session
POST /v1/agents/{agent_id}/sessions/{id}/observe # observe within a session
POST /v1/agents/{agent_id}/sessions/{id}/policy  # get policy within a session
POST /v1/agents/{agent_id}/sessions/{id}/close   # close with an outcome
```

Sessions track retrieval state and enable outcome-based learning signals.
Close with `{"outcome": 0.9}` to apply a positive signal, or use
`POST /v1/agents/{agent_id}/reinforce/{user_id}` for post-hoc signaling.

## Agent administration

```
GET    /v1/agents                       # list all initialized agents
POST   /v1/agents                       # pre-configure an agent
GET    /v1/agents/{agent_id}            # get agent config
PATCH  /v1/agents/{agent_id}/config     # update agent config
DELETE /v1/agents/{agent_id}            # drain and deregister
POST   /v1/agents/{agent_id}/scopes/consolidate  # consolidate scope vocabulary
```

## API keys

```
GET    /v1/keys                # list all keys (hashes only, never raw)
POST   /v1/keys                # create a new key (raw key returned once)
DELETE /v1/keys/{key_hash}     # revoke by 16-char hash prefix
```

Create a key:

```json
{"label": "ci-key", "agent_id": "my-agent", "user_id": "alice"}
```

Response includes `raw_key` -- shown once, never stored:

```json
{
  "raw_key": "sk-imp-abc123...",
  "key_hash": "a1b2c3d4e5f6a1b2",
  "label": "ci-key",
  "agent_id": "my-agent",
  "user_id": "alice",
  "created_at": "2025-04-01T12:00:00+00:00"
}
```

## System

```
GET /health          # alias for /health/ready
GET /health/live     # liveness probe (always 200 while process runs)
GET /health/ready    # readiness probe (checks DB + Redis)
GET /metrics         # Prometheus exposition format
GET /admin           # admin dashboard (HTML)
```

## Error format

All errors use [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) `application/problem+json`:

```json
{
  "type": "about:blank",
  "title": "Not Found",
  "status": 404,
  "detail": "memory 'mem_abc123' not found"
}
```

Every response includes `X-Request-ID` for correlation.
