"""
with_production_server.py -- full production stack via imprint-server.

Shows the complete feature set working together through the HTTP client:
  - Voyage AI embedder for dense retrieval
  - Hybrid BM25 + vector search (semantic ordering)
  - Gradient decay learning from session outcomes
  - Redis policy cache (second request served from cache)
  - Postgres persistent store shared across sessions
  - Pagination over growing memory sets

This example assumes a running imprint-server with the production stack
configured. The fastest way to start it:

  just server-compose-live-test   # starts and tears down automatically
                                   # (use for CI / automated runs)

Or start it manually and run the example against it:

  docker compose -f imprint-server/docker-compose.live.yml up --build --wait

Requirements:
  pip install imprint-mem[client]
  export ANTHROPIC_API_KEY=sk-ant-...
  export VOYAGE_API_KEY=pa-...

  # Start the server:
  docker compose -f imprint-server/docker-compose.live.yml up --build --wait

Usage:
  python examples/with_production_server.py
"""

import asyncio
import time

from imprint.client import ImprintClient

SERVER_URL = "http://localhost:18001"
AGENT = "prod-demo-agent"
USER = "prod-demo-user"


async def main() -> None:
    async with ImprintClient(SERVER_URL) as client:
        agent = client.agent(AGENT)

        print("=== imprint-server production stack demo ===")
        print(f"Server: {SERVER_URL}")

        # Verify the stack is fully up.
        health = await client._get("/health/ready")
        body = health.json()  # type: ignore[attr-defined]
        print(f"Health: store={body['store']} redis={body['redis']} db_ok={body['db_ok']}")
        assert body["redis"] == "ok", "Redis not connected -- is the full stack running?"

        # -- Seed memories on two clearly different topics --------------------

        print("\n[1] Seeding memories on two domains...")
        await agent.observe_directions(
            USER,
            directions=[
                "Morgan always wants Python code examples, never JavaScript.",
                "Morgan prefers snake_case variable names in all Python code.",
                "For Python, always include type annotations on function signatures.",
            ],
            scope="python",
        )
        await agent.observe_directions(
            USER,
            directions=[
                "When writing in French, use formal 'vous' not informal 'tu'.",
                "Morgan's French writing is for academic and professional contexts.",
            ],
            scope="french",
        )
        print("  Stored 5 preferences across two scopes.")

        # -- Semantic search shows vector retrieval ordering ------------------

        print("\n[2] Semantic search -- Python naming conventions...")
        results = await agent.search_memories(USER, "variable naming in Python")
        print(f"  {len(results)} results (vector-ranked):")
        for r in results[:3]:
            print(f"    [{r.scope or 'global':10}] {r.content[:65]}")
        if results:
            assert any("snake_case" in r.content or "Python" in r.content for r in results[:2]), (
                "Expected Python-related memory in top 2 results"
            )

        # -- Policy compilation with LLM (balanced mode) ----------------------

        print("\n[3] Compiling policy via LLM (balanced mode)...")
        t0 = time.perf_counter()
        policy = await agent.get_policy(USER, context="Python code review")
        first_ms = (time.perf_counter() - t0) * 1000
        print(f"  First request: {first_ms:.0f}ms")
        print(f"  Policy text: {policy.text[:120]}...")

        # Second identical request -- should hit Redis cache.
        t1 = time.perf_counter()
        policy2 = await agent.get_policy(USER, context="Python code review")
        second_ms = (time.perf_counter() - t1) * 1000
        print(f"  Cache hit:    {second_ms:.0f}ms (speedup: {first_ms / second_ms:.1f}x)")
        assert policy.text == policy2.text, "Cache returned different policy text"

        # -- Session lifecycle with gradient decay ----------------------------

        print("\n[4] Running two sessions -- gradient decay learning...")
        for i, outcome_label in enumerate(["positive", "negative"]):
            session_id = await client.open_session(AGENT, USER, context="code-review")
            _ = await agent.get_policy(USER, context="code-review")

            if outcome_label == "positive":
                applied = await client.reinforce(AGENT, USER, session_id=session_id)
                print(f"  Session {i + 1}: reinforced={applied}")
            else:
                mem_id = await client.correct(
                    AGENT,
                    USER,
                    "Do not use camelCase -- stick to snake_case.",
                    session_id=session_id,
                )
                print(f"  Session {i + 1}: correction stored as {mem_id}")

        # -- Pagination over accumulated memories -----------------------------

        print("\n[5] Paginating memories (limit=2)...")
        page = await agent.paginate_memories(USER, limit=2)
        total = len(page.items)
        pages = 1
        while page.has_more:
            page = await agent.paginate_memories(USER, limit=2, cursor=page.next_cursor)
            total += len(page.items)
            pages += 1
        print(f"  {total} memories across {pages} pages")

        # -- Health summary ---------------------------------------------------

        mem_health = await agent.memory_health(USER)
        print(
            f"\n[6] Memory health: "
            f"total={mem_health.total} "
            f"active={mem_health.active} "
            f"pinned={mem_health.pinned}"
        )

        print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
