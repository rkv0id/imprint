"""
with_server_client.py -- using imprint-server via the typed HTTP client.

A customer support agent learns per-user communication preferences over
several turns, demonstrating the full imprint-server client surface:

  - ImprintClient and AgentClient
  - get_policy / observe / correct / reinforce
  - paginate_memories (cursor-based pagination)
  - search_memories (semantic search)
  - Session lifecycle with outcome signaling

This example runs against a local imprint-server instance and uses frugal
mode so no LLM API key is required for the server. The client itself is
pure HTTP -- no local Imprint object is used.

Requirements:
  pip install imprint-mem[client]
  pip install imprint-server

Start the server in another terminal before running:
  IMPRINT_DEFAULT_MODE=frugal imprint-server serve

Or with just:
  just server-dev

Usage:
  python examples/with_server_client.py
"""

import asyncio

from imprint.client import ImprintClient, MemoryRecord, PageResult

SERVER_URL = "http://localhost:8000"
AGENT = "support-agent"


async def main() -> None:
    async with ImprintClient(SERVER_URL) as client:
        # Use AgentClient to avoid repeating agent_id on every call.
        agent = client.agent(AGENT)
        user = "alice"

        print("=== imprint-server client demo ===\n")

        # -- Register explicit directions for a user --------------------------
        # In production these come from real observe() turns. Here we seed
        # them directly so the example runs without an LLM on the client.

        await agent.observe_directions(
            user_id=user,
            directions=[
                "Alice prefers short replies -- three sentences maximum.",
                "Alice is a developer; use technical language freely.",
                "Alice dislikes emojis in support responses.",
                "Always link to the relevant docs page when mentioning a feature.",
                "Alice is in the APAC timezone; mention business hours in UTC+8.",
            ],
        )

        # -- Fetch the compiled policy ----------------------------------------

        policy = await agent.get_policy(user)
        print(f"[policy] {policy.memory_count} memories loaded")
        if policy.text:
            print(f"  {policy.text[:120]}...")
        else:
            print("  (no policy text yet -- try balanced mode for LLM compilation)")

        # -- Observe a turn ---------------------------------------------------

        await agent.observe(
            user_id=user,
            agent_output="We support three authentication methods: password, OAuth, and SSO. "
            "Let me know which one you'd like help with! 😊",
            user_response="Stop with the emojis, and just tell me where the OAuth docs are.",
        )

        print("\n[observe] turn recorded")

        # -- Signal a correction (negative learning signal) -------------------

        # The agent used an emoji despite the preference. Correct it.
        mem_id = await agent.correct(
            user_id=user,
            content="Do not use emojis. Alice explicitly dislikes them.",
        )
        print(f"[correct] correction stored as memory {mem_id}")

        # -- Session lifecycle with positive reinforcement --------------------

        print("\n[session] running session with positive outcome...")
        async with client.session(AGENT, user, context="oauth-setup") as sess:
            policy = await sess.get_policy()
            print(f"  session policy: {policy.memory_count} memories")

            await sess.observe(
                agent_output="OAuth 2.0 setup guide: https://docs.example.com/oauth.",
                user_response="That was exactly what I needed, thanks.",
            )
            # Signal this was a good session.
            sess.set_outcome(0.9)

        print("  session closed with positive outcome")

        # -- Search memories semantically -------------------------------------

        print("\n[search] searching for tone/style memories...")
        results = await agent.search_memories(user, "communication tone style")
        for m in results[:3]:
            print(f"  - {m.content[:80]}")

        # -- Paginate all memories --------------------------------------------

        print("\n[paginate] fetching all memories in pages of 2...")
        page: PageResult[MemoryRecord] = await agent.paginate_memories(user, limit=2)
        all_memories: list[MemoryRecord] = list(page.items)
        pages = 1

        while page.has_more:
            page = await agent.paginate_memories(user, limit=2, cursor=page.next_cursor)
            all_memories.extend(page.items)
            pages += 1

        print(f"  fetched {len(all_memories)} memories across {pages} page(s)")
        for m in all_memories:
            print(f"  [{m.type}] {m.content[:70]}")

        # -- Health check -----------------------------------------------------

        health = await agent.memory_health(user)
        print(f"\n[health] active={health.active} pinned={health.pinned}")

        print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
