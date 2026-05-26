# Multi-user MCP

By default, the MCP endpoint uses a single user namespace set by
`IMPRINT_MCP_USER_ID`. For multi-user deployments, each user connects
with their own API key and gets an isolated memory namespace automatically.

## Setup

**1. Enable auth and set the agent ID:**

```sh
IMPRINT_MCP_AGENT_ID=my-agent \
IMPRINT_AUTH_DISABLED=false \
imprint-server serve
```

Note: `IMPRINT_MCP_USER_ID` is NOT set. User identity comes from the API key.

**2. Create user-bound keys:**

```sh
# CLI
imprint-server keys create --label "alice-key" \
  --agent my-agent --user alice

imprint-server keys create --label "bob-key" \
  --agent my-agent --user bob
```

Or via REST:

```sh
curl -X POST http://localhost:8000/v1/keys \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"label": "alice-key", "agent_id": "my-agent", "user_id": "alice"}'
```

**3. Each user connects with their key:**

```json
{
  "mcpServers": {
    "imprint": {
      "url": "http://your-server:8000/mcp/sse",
      "headers": {
        "Authorization": "Bearer sk-imp-alice..."
      }
    }
  }
}
```

Every tool call is automatically scoped to alice's memory namespace. Bob's
key routes all calls to bob's namespace. No code change on the agent side.

## How it works

`_MCPUserMiddleware` wraps the FastMCP Starlette app. On every HTTP request
to `/mcp/*`:

1. Reads the `Authorization: Bearer <key>` header
2. Looks up the key in the database
3. If the key has a `user_id`, sets the per-request `_mcp_user_id` ContextVar
4. Tool handlers read the ContextVar to determine which user namespace to use

Master keys (no `user_id`) leave the ContextVar unset. Tool handlers raise
a clear error if called with a master key -- master keys are for admin use,
not for per-user MCP access.

## Key types

| Key type | user_id | agent_id | Use for |
|---|---|---|---|
| Master | none | none | Admin access, creating other keys |
| Agent-scoped | none | set | Programmatic access to one agent |
| User-bound | set | set | Multi-user MCP access |

## Rotating keys

Revoke a key and create a replacement without downtime:

```sh
# Create new key for alice
NEW_KEY=$(curl -s -X POST http://localhost:8000/v1/keys \
  -H "Authorization: Bearer $MASTER_KEY" \
  -d '{"label": "alice-key-v2", "agent_id": "my-agent", "user_id": "alice"}' \
  | jq -r .raw_key)

# Update alice's client config, then revoke the old key
curl -X DELETE http://localhost:8000/v1/keys/$OLD_HASH \
  -H "Authorization: Bearer $MASTER_KEY"
```

Memories persist through key rotation -- they are bound to `user_id`, not
to the key.

## Listing all keys

```sh
curl http://localhost:8000/v1/keys \
  -H "Authorization: Bearer $MASTER_KEY"
```

Returns all keys with hashes, labels, agent scope, user binding, and status.
Raw keys are never returned -- they are stored only as SHA-256 hashes.
