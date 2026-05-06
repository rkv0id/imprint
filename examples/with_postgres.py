"""
with_postgres.py -- PostgresMemoryStore as a drop-in replacement for SQLite.

The entire observe/get_policy API is identical to the SQLite examples.
Changing one constructor argument moves storage to a Postgres instance.
This is the right backend for imprint-server and any multi-instance
deployment where multiple service replicas share the same memory store.

Demonstrates:
  - PostgresMemoryStore setup (local Postgres via Docker)
  - PostgresVectorStore setup (optional, requires pgvector extension)
  - That the API surface is identical to SQLiteMemoryStore
  - A complete observe -> get_policy cycle against a real Postgres store

Requirements:
  pip install imprint-mem[postgres]
  export ANTHROPIC_API_KEY=sk-ant-...
  export IMPRINT_POSTGRES_URL=postgres://imprint:imprint@localhost/imprint_test

Setup -- start a local Postgres with pgvector via Docker:
  docker run --rm --name imprint-pg \\
    -e POSTGRES_DB=imprint_test \\
    -e POSTGRES_USER=imprint \\
    -e POSTGRES_PASSWORD=imprint \\
    -p 5432:5432 \\
    pgvector/pgvector:pg16

Or with just:
  just postgres-dev   (in one terminal)
  python examples/with_postgres.py

Usage:
  python examples/with_postgres.py
"""

import asyncio
import os

from imprint import Imprint
from imprint.stores.postgres import PostgresMemoryStore


def _require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise SystemExit(
            f"Missing required environment variable: {name}\n"
            "See the module docstring for setup instructions."
        )
    return val


async def main() -> None:
    db_url = _require_env("IMPRINT_POSTGRES_URL")

    # PostgresMemoryStore uses asyncpg with a connection pool.
    # URL format: postgres://user:pass@host[:port]/dbname
    # or:         postgresql://user:pass@host[:port]/dbname
    store = PostgresMemoryStore(db_url)

    # PostgresVectorStore is optional. When configured alongside an embedder,
    # it enables hybrid retrieval (BM25 tsvector + dense + RRF). It requires
    # the pgvector extension in your Postgres instance -- the pgvector/pgvector
    # Docker image includes it by default.
    #
    # Uncomment to enable:
    # from imprint.stores.postgres import PostgresVectorStore
    # from imprint.providers.voyage import VoyageEmbedder
    # embedder = VoyageEmbedder(os.environ["VOYAGE_API_KEY"], model="voyage-3")
    # vector_store = PostgresVectorStore(store.pool, dim=embedder.dim)
    # await vector_store.init_schema()

    imprint = Imprint(
        agent_id="postgres_assistant",
        store=store,
        processing_mode="frugal",
        # embedder=embedder,
        # vector_store=vector_store,
    )
    await imprint.connect()

    print("=== PostgresMemoryStore -- Shared Storage ===\n")
    print(f"Connected to: {db_url}\n")

    user_id = "pg_user"

    # Everything from here is identical to the SQLite examples.
    # Swap PostgresMemoryStore for SQLiteMemoryStore and nothing else changes.

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

    print(f"Stored {len(memories)} memories in Postgres.")
    for m in memories:
        print(f"  [{m.type.value}] {m.content}")
    print(f"\nCompiled policy: {policy.text}")

    # Clean up: deactivate the memories we wrote so re-runs start fresh.
    for m in memories:
        await imprint.deactivate_memory(user_id, m.id)
    print(f"\nCleaned up {len(memories)} test memories from Postgres store.")

    await imprint.close()


if __name__ == "__main__":
    asyncio.run(main())
