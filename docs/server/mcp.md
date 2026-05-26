# MCP integration

imprint-server exposes an MCP SSE endpoint at `/mcp/sse`. Connect Claude Code,
Cursor, Continue, or any MCP-compatible client.

## Start the server with MCP

```sh
IMPRINT_MCP_AGENT_ID=my-agent \
IMPRINT_MCP_USER_ID=my-user \
imprint-server serve
```

Both variables must be set for the `/mcp/sse` endpoint to be mounted.

## Add as an MCP server

In Claude Code:

```json
{
  "mcpServers": {
    "imprint": {
      "url": "http://localhost:8000/mcp/sse"
    }
  }
}
```

In Cursor (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "imprint": {
      "url": "http://localhost:8000/mcp/sse"
    }
  }
}
```

## Available tools

Eight tools are registered:

| Tool | Description |
|---|---|
| `imprint_begin_session` | Open a memory session. Returns a `session_id` for tracking. |
| `imprint_get_policy` | Compile and return the behavioral policy for the current user. |
| `imprint_observe` | Record an agent-user exchange. Stores a memory when a signal is detected. |
| `imprint_recall` | Search memories by semantic similarity. |
| `imprint_direct` | Store explicit behavioral directions without signal detection. |
| `imprint_end_session` | Close a session with an outcome signal (0=correction, 1=ideal). |
| `imprint_correct` | Store a correction and apply a negative learning signal. |
| `imprint_reinforce` | Apply a positive learning signal to a closed session. |

## Typical usage pattern

```
1. Call imprint_begin_session at the start of each conversation.
2. Call imprint_get_policy to load behavioral instructions.
3. Inject policy.text into the system prompt.
4. After each user turn, call imprint_observe.
5. At end of conversation, call imprint_end_session with an outcome.
```

## Multi-user MCP

With auth enabled, user identity is resolved per-connection from the Bearer
token rather than from `IMPRINT_MCP_USER_ID`. This lets multiple users share
a single MCP endpoint with isolated memory namespaces.

Create a user-bound key:

```sh
imprint-server keys create --label "alice-key" \
  --agent my-agent --user alice
```

Or via REST:

```sh
curl -X POST http://localhost:8000/v1/keys \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"label": "alice-key", "agent_id": "my-agent", "user_id": "alice"}'
```

Alice's client connects with her personal key:

```
Authorization: Bearer sk-imp-alice...
```

Every tool call is automatically scoped to `alice`'s memory namespace. No
code change required on the client side.

See [Multi-user MCP guide](../guides/multi-user-mcp.md) for a full walkthrough.

## just recipes

```sh
just server-mcp-dev               # start server with MCP enabled (SQLite, auth disabled)
just server-mcp-dev agent=X user=Y  # custom agent and user
```
