"""
writing_assistant.py -- multi-user, scopes, directions, and observability.

A writing assistant serves two authors from the same Imprint instance.
Alice writes literary fiction; Bob writes technical documentation. Their
memories stay completely separate despite sharing one agent.

Demonstrates:
  - observe_directions() for explicit preference setup
  - Multi-user isolation via user_id
  - Scopes to separate context-specific memories
  - Consolidation when a preference changes (contradict action)
  - memory_health() and list_events() from the observability API

Requirements:
  pip install imprint-mem
  export ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python examples/writing_assistant.py
"""

import asyncio
from pathlib import Path

from imprint import Imprint

DB_PATH = "writing_assistant.db"


async def main() -> None:
    # Always start from a clean slate regardless of previous runs.
    Path(DB_PATH).unlink(missing_ok=True)

    imprint = Imprint(
        agent_id="writing_assistant",
        store=DB_PATH,
        processing_mode="balanced",
        scopes=["genre:fiction", "genre:technical"],
    )
    await imprint.connect()

    print("=== Writing Assistant (two authors, one agent) ===\n")

    # ------------------------------------------------------------------
    # Alice -- literary fiction
    # ------------------------------------------------------------------

    # observe_directions() stores explicit preferences without running detection.
    # Good for onboarding flows where the user directly configures the agent.
    await imprint.observe_directions(
        user_id="alice",
        scope="genre:fiction",
        directions=[
            "Use vivid sensory detail and immersive scene-setting.",
            "Write in long, flowing paragraphs -- not short choppy sentences.",
            "Prefer active voice throughout.",
        ],
    )
    print("[alice] 3 baseline directions stored")

    # Alice refines her preference for paragraph length -- the new direction
    # contradicts the old one. Imprint runs consolidation and the old memory
    # is deactivated (superseded).
    await imprint.observe(
        user_id="alice",
        agent_output="The morning light fell in long golden shafts"
        " across the worn floorboards, each particle of dust"
        " suspended in amber silence...",
        user_response="Too much. I want lyrical but restrained."
        " Short sentences that earn their weight. Every word must do work.",
        scope="genre:fiction",
    )

    # One more positive signal.
    await imprint.observe(
        user_id="alice",
        agent_output="Light fell across the floor. Dust moved in it. She did not.",
        user_response="Yes. Exactly that. Precise and earned.",
        scope="genre:fiction",
    )

    alice_policy = await imprint.get_policy(user_id="alice", scopes=["genre:fiction"])
    print(f"[alice] {len(alice_policy.memories)} memories after learning")
    print(f"[alice] policy: {alice_policy.text}\n")

    # ------------------------------------------------------------------
    # Bob -- technical documentation
    # ------------------------------------------------------------------

    await imprint.observe_directions(
        user_id="bob",
        scope="genre:technical",
        directions=[
            "Use plain language. Avoid jargon unless the term is established in the field.",
            "One idea per sentence. Short sentences.",
            "Follow every concept with a concrete code example.",
        ],
    )
    print("[bob] 3 baseline directions stored")

    await imprint.observe(
        user_id="bob",
        agent_output="The authentication module validates user credentials"
        " against the identity provider.",
        user_response="Good start, but you forgot the code example."
        " Always show code after the concept.",
        scope="genre:technical",
    )

    await imprint.observe(
        user_id="bob",
        agent_output="The function returns a JWT on success.\n\n"
        "```python\ntoken = auth.login(user, password)\n```",
        user_response="Perfect. Keep the code blocks in that style.",
        scope="genre:technical",
    )

    bob_policy = await imprint.get_policy(user_id="bob", scopes=["genre:technical"])
    print(f"[bob] {len(bob_policy.memories)} memories after learning")
    print(f"[bob] policy: {bob_policy.text}\n")

    # ------------------------------------------------------------------
    # Scope inference -- get_policy without explicit scopes.
    #
    # Passing context= without scopes= lets imprint infer which scopes
    # are relevant. In balanced mode it uses embedding similarity between
    # the context string and scope names, falling back to an LLM call when
    # the signal is ambiguous. In eager mode it always uses the LLM.
    #
    # This is what production usage looks like: the agent passes its current
    # context and imprint selects the right memories automatically.
    # ------------------------------------------------------------------

    p_alice_inferred = await imprint.get_policy(
        user_id="alice",
        context="writing a literary short story with sparse, precise prose",
        # no scopes= -- imprint infers genre:fiction from context
    )
    p_bob_inferred = await imprint.get_policy(
        user_id="bob",
        context="writing API reference documentation for a Python library",
        # no scopes= -- imprint infers genre:technical from context
    )
    print("--- Scope inference (no explicit scopes= passed) ---")
    print(f"alice (fiction context): {len(p_alice_inferred.memories)} memories")
    print(f"bob   (technical context): {len(p_bob_inferred.memories)} memories")

    # ------------------------------------------------------------------
    # Isolation check
    # ------------------------------------------------------------------

    alice_memories = await imprint.list_memories("alice")
    bob_memories = await imprint.list_memories("bob")
    print(f"\nalice: {len(alice_memories)} memories | bob: {len(bob_memories)} memories")
    print(f"alice scopes: {sorted({m.scope for m in alice_memories})}")
    print(f"bob   scopes: {sorted({m.scope for m in bob_memories})}")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    alice_health = await imprint.memory_health("alice")
    print("\n--- memory_health (alice) ---")
    print(f"  total={alice_health.total}  active={alice_health.active}")
    print(f"  by_type={alice_health.by_type}")
    print(f"  avg_recall_count={alice_health.avg_recall_count:.2f}")

    events = await imprint.list_events("alice", limit=6)
    print(f"\n--- list_events (alice, last {len(events)}) ---")
    for e in events:
        print(f"  {e.event_type:12s}  {e.memory_id[:20]}...")

    await imprint.close()


if __name__ == "__main__":
    asyncio.run(main())
    Path(DB_PATH).unlink(missing_ok=True)
