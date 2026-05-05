"""
with_turso.py -- TursoMemoryStore as a drop-in replacement for SQLite.

The entire observe/get_policy API is identical to the SQLite examples.
Changing one constructor argument moves storage to a remote sqld instance.
This is the foundation for multi-instance deployments where multiple
service replicas share the same memory store.

Demonstrates:
  - TursoMemoryStore setup (local sqld via Docker)
  - That the API surface is identical to SQLiteMemoryStore
  - A complete observe -> get_policy cycle against a real remote store

Requirements:
  pip install imprint-mem[turso]
  export ANTHROPIC_API_KEY=sk-ant-...
  export TURSO_DATABASE_URL=http://127.0.0.1:8080  (see setup below)

Setup -- start a local sqld server via Docker:
  docker run --rm -p 8080:8080 ghcr.io/tursodatabase/libsql-server:latest

For Turso cloud instead of local sqld:
  export TURSO_DATABASE_URL=libsql://your-db.turso.io
  export TURSO_AUTH_TOKEN=your-token

Usage:
  python examples/with_turso.py
"""

import asyncio
import os

from imprint import Imprint, TursoMemoryStore


def _require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise SystemExit(
            f"Missing required environment variable: {name}\n"
            "See the module docstring for setup instructions."
        )
    return val


async def main() -> None:
    db_url = _require_env("TURSO_DATABASE_URL")
    auth_token = os.environ.get("TURSO_AUTH_TOKEN") or None  # optional for local sqld

    # TursoMemoryStore calls sqld's hrana-over-HTTP API using httpx.
    # No Rust extension, no cmake -- just HTTP JSON calls.
    # URL formats: http://, https://, libsql://, ws://, wss://
    store = TursoMemoryStore(db_url, auth_token=auth_token)

    imprint = Imprint(
        agent_id="turso_assistant",
        store=store,
        processing_mode="frugal",
    )
    await imprint.connect()

    print("=== TursoMemoryStore -- Remote Storage ===\n")
    print(f"Connected to: {db_url}\n")

    user_id = "remote_user"

    # Everything from here is identical to the SQLite examples.
    # Swap TursoMemoryStore for SQLiteMemoryStore and nothing else changes.

    await imprint.observe(
        user_id=user_id,
        agent_output="I can help you with that. Would you like me to list the options?",
        user_response=(
            "Please don't ask me if I want a list. Just give me the information directly."
        ),
    )

    await imprint.observe(
        user_id=user_id,
        agent_output="Here are the three options: ...",
        user_response="Much better. Keep being direct like that.",
    )

    policy = await imprint.get_policy(user_id=user_id)
    memories = await imprint.list_memories(user_id)

    print(f"Stored {len(memories)} memories in Turso.")
    for m in memories:
        print(f"  [{m.type.value}] {m.content}")
    print(f"\nCompiled policy: {policy.text}")

    # Clean up: deactivate the memories we wrote so re-runs start fresh.
    for m in memories:
        await imprint.deactivate_memory(user_id, m.id)
    print(f"\nCleaned up {len(memories)} test memories from remote store.")

    await imprint.close()


if __name__ == "__main__":
    asyncio.run(main())
