# Admin dashboard

A read-only admin dashboard is served at `/admin`. It is a self-contained HTML
page with no build step and no external dependencies beyond Google Fonts CDN.

## Access

```sh
# Auth disabled (default in development)
open http://localhost:8000/admin

# Auth enabled -- the dashboard prompts for an API key on first load.
# Enter a master key. It is stored in sessionStorage and cleared when the tab closes.
open http://localhost:8000/admin
```

The `/admin` route is exempt from `AuthMiddleware`. The API calls made by the
dashboard JS are not -- they carry the Bearer token stored in sessionStorage.

## Panels

### Overview

System health at a glance: store type, Redis status, DB connectivity, agent
count. A table of initialized agents with their processing mode and scope
configuration. Auto-refreshes every 10 seconds.

### Agents

Full agent registry table with agent_id, processing mode, description, scopes,
and dynamic scope status.

### Memory Browser

Select an agent and enter a user ID to browse that user's memory namespace.
Shows all active memories with:

- Content (truncated, full content on hover)
- Memory type pill (rule, preference, fact, etc.)
- Scope
- Stability bar (visual indicator 0-1)
- Recall count
- Pinned / active status

Also shows aggregate memory health stats: total, active, pinned, average recall count.

### Events

Select an agent and user to view the memory event log. Shows event type, memory
ID, and detail for each event. Events include: create, recall, deactivate,
supersede.

### API Keys

All active API keys with hash, label, agent scope, user binding, status, and
creation date. Stat cards show total active, master, scoped, and user-bound counts.

## Auto-refresh

The sidebar shows a countdown timer. Refresh is on by default (10 second
interval). Click the toggle to pause. Health data refreshes on every tick;
memory and event data refresh only when their panel is active.

## Demo

The fastest way to see the dashboard with real data:

```sh
just demo
```

This starts the server, seeds demo data (4 agents, 10 sessions, 10 API keys,
behavioral memories for multiple users), and prints the dashboard URL.
