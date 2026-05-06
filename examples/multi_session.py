"""
multi_session.py -- memory that persists and evolves across sessions.

Shows the full MemoryLoop lifecycle (open -> get_policy -> observe ->
set_outcome -> finalize) across three separate connect/close cycles.
Each cycle represents a distinct user session as an agent would handle
it in production.

Demonstrates:
  - Memory persistence across Imprint.connect() / Imprint.close() cycles
  - Policy evolving as more memories accumulate across sessions
  - Stability increasing from positive outcome signals (finalize_loop)
  - Stability decreasing from negative outcome signals (correction session)
  - recall_count accumulating across independent sessions
  - observe() capturing corrections that inform outcome attribution

Requirements:
  pip install imprint-mem
  export ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python examples/multi_session.py
"""

import asyncio
from pathlib import Path

from imprint import Imprint

DB_PATH = "multi_session.db"
AGENT_ID = "coding_assistant"
USER_ID = "rami"


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_memories(memories: list) -> None:
    for m in memories:
        print(f"  [{m.scope:12s}] s={m.stability:.3f} r={m.recall_count:2d}  {m.content[:55]}")


async def session_one() -> None:
    """First contact. No prior memories. Agent learns two preferences
    through observe(), then runs a MemoryLoop with a strong positive outcome."""

    _header("Session 1 -- first contact")

    im = Imprint(
        agent_id=AGENT_ID,
        store=DB_PATH,
        processing_mode="balanced",
        scopes=["python", "general"],
    )
    await im.connect()

    memories = await im.list_memories(USER_ID)
    print(f"\nMemories on connect: {len(memories)}")
    policy = await im.get_policy(user_id=USER_ID)
    print(f"Policy: {policy.text or '(none yet)'}")

    # Two preference signals picked up via observe().
    print("\n-- Learning phase --")

    await im.observe(
        user_id=USER_ID,
        agent_output="Here is the function:\n\ndef add(a, b):\n    return a + b",
        user_response=("Always add type hints. Python functions without them are incomplete."),
        scope="python",
    )

    await im.observe(
        user_id=USER_ID,
        agent_output="Sure, let me explain that concept.",
        user_response=("Skip the preamble. Lead with the answer, then explain if needed."),
        scope="general",
    )

    memories = await im.list_memories(USER_ID)
    print(f"Memories after observe: {len(memories)}")
    _print_memories(memories)

    # First MemoryLoop: positive outcome (agent was helpful).
    print("\n-- MemoryLoop (positive outcome) --")
    loop = await im.open_loop(user_id=USER_ID)
    policy = await im.get_policy(user_id=USER_ID, loop=loop)
    print(f"Policy injected into agent:\n  {policy.text}")

    # Simulated agent turn inside the loop.
    await im.observe(
        user_id=USER_ID,
        agent_output=("def greet(name: str) -> str:\n    return f'Hello, {name}'"),
        user_response="Good. That's exactly what I want.",
    )

    loop.set_outcome(0.90)
    await im.finalize_loop(loop)

    memories = await im.list_memories(USER_ID)
    print("\nMemories after finalize (stability boosted by positive outcome):")
    _print_memories(memories)

    await im.close()
    print("\nSession 1 closed.")


async def session_two() -> None:
    """Second session. Prior memories are loaded from disk. Two more loops
    run -- one strong positive, one moderate -- showing stability compounding."""

    _header("Session 2 -- returning user")

    im = Imprint(
        agent_id=AGENT_ID,
        store=DB_PATH,
        processing_mode="balanced",
        scopes=["python", "general"],
    )
    await im.connect()

    memories = await im.list_memories(USER_ID)
    print(f"\nMemories loaded from disk: {len(memories)}")
    _print_memories(memories)

    # A new preference picked up this session.
    await im.observe(
        user_id=USER_ID,
        agent_output="You can use either a list or a generator here.",
        user_response=(
            "In Python, prefer generators over lists when you don't need"
            " random access. Saves memory."
        ),
        scope="python",
    )

    # Strong positive loop.
    print("\n-- Loop A (outcome=0.95) --")
    loop_a = await im.open_loop(user_id=USER_ID)
    policy = await im.get_policy(user_id=USER_ID, loop=loop_a)
    print(f"Policy: {policy.text[:120]}...")
    loop_a.set_outcome(0.95)
    await im.finalize_loop(loop_a)

    # Moderate positive loop.
    print("\n-- Loop B (outcome=0.65) --")
    loop_b = await im.open_loop(user_id=USER_ID)
    await im.get_policy(user_id=USER_ID, loop=loop_b)
    loop_b.set_outcome(0.65)
    await im.finalize_loop(loop_b)

    memories = await im.list_memories(USER_ID)
    print("\nMemories after two loops (stability reflects outcome history):")
    _print_memories(memories)

    await im.close()
    print("\nSession 2 closed.")


async def session_three() -> None:
    """Third session. One memory is corrected via a negative outcome loop,
    showing stability decay. Final policy reflects the full history."""

    _header("Session 3 -- correction")

    im = Imprint(
        agent_id=AGENT_ID,
        store=DB_PATH,
        processing_mode="balanced",
        scopes=["python", "general"],
    )
    await im.connect()

    memories = await im.list_memories(USER_ID)
    print(f"\nMemories loaded from disk: {len(memories)}")
    _print_memories(memories)

    # Negative outcome: agent retrieved memories but the response still
    # missed the mark. Stability of retrieved memories decays.
    print("\n-- Loop C (outcome=-0.30, agent missed the mark) --")
    loop_c = await im.open_loop(user_id=USER_ID)
    policy = await im.get_policy(
        user_id=USER_ID,
        loop=loop_c,
        context="writing a Python utility function",
    )
    print(f"Policy going in: {policy.text[:120]}...")

    # The agent made a mistake; user corrects.
    await im.observe(
        user_id=USER_ID,
        agent_output=("def items(data: list) -> list:\n    return [x for x in data if x]"),
        user_response=(
            "Use a generator expression here, not a list comprehension. I told you this last time."
        ),
    )

    loop_c.set_outcome(-0.30)
    await im.finalize_loop(loop_c)

    memories = await im.list_memories(USER_ID)

    print("\nFinal memory state across all three sessions:")
    print(f"  {'scope':12s}  {'stability':>9}  {'recalls':>7}  content")
    print(f"  {'-' * 12}  {'-' * 9}  {'-' * 7}  {'-' * 40}")
    for m in sorted(memories, key=lambda x: x.stability, reverse=True):
        print(f"  [{m.scope:12s}] {m.stability:9.3f}  {m.recall_count:7d}  {m.content[:50]}")

    print("\n-- Final compiled policy --")
    policy = await im.get_policy(
        user_id=USER_ID,
        context="writing Python utility functions",
    )
    print(policy.text)

    await im.close()
    print("\nSession 3 closed.")


async def main() -> None:
    Path(DB_PATH).unlink(missing_ok=True)

    print("=== Multi-Session Persistence ===")
    print(f"Store: {DB_PATH}")

    await session_one()
    await session_two()
    await session_three()

    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
    Path(DB_PATH).unlink(missing_ok=True)
