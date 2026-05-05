"""
minimal.py -- the core observe/get_policy loop, no extras required.

A code review assistant learns a developer's style preferences over
three conversation turns. Shows how imprint detects signals, stores
memories, and compiles them into a behavioral policy.

Requirements:
  pip install imprint-mem
  export ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python examples/minimal.py
"""

import asyncio

from imprint import Imprint


async def main() -> None:
    # SQLiteMemoryStore(":memory:") is explicit in-memory storage.
    # Without a store= argument, Imprint defaults to ~/.imprint/imprint.db
    # which persists across runs -- not what we want for a demo.
    imprint = Imprint(
        agent_id="code_reviewer",
        store=":memory:",
        processing_mode="frugal",
    )
    await imprint.connect()

    user_id = "dev"

    # --- Before any learning ---
    policy = await imprint.get_policy(user_id=user_id)
    print("=== Code Review Assistant ===\n")
    print(f"[start] {len(policy.memories)} memories, policy: '{policy.text or '(none yet)'}'")

    # --- Turn 1: agent uses inline comments, developer wants full sentences ---
    # These are pre-scripted to keep the example deterministic.
    # In a real deployment, agent_output and user_response come from your pipeline.
    await imprint.observe(
        user_id=user_id,
        agent_output=(
            "// line 42: consider extracting this into a helper.\n// line 58: missing null check."
        ),
        user_response="Please write your feedback in full sentences, not inline comments. "
        "Inline comments are hard to read during review.",
    )

    policy = await imprint.get_policy(user_id=user_id)
    print(f"\n[turn 1] {len(policy.memories)} memories")
    print(f"  policy: {policy.text}")

    # --- Turn 2: agent uses bullet points, developer corrects again ---
    await imprint.observe(
        user_id=user_id,
        agent_output=(
            "Issues:\n- Missing null check on line 42\n- Inconsistent naming\n- No error handling"
        ),
        user_response="I said full sentences. Do not use bullet points either.",
    )

    policy = await imprint.get_policy(user_id=user_id)
    print(f"\n[turn 2] {len(policy.memories)} memories")
    print(f"  policy: {policy.text}")

    # --- Turn 3: agent writes in full sentences, developer reinforces ---
    await imprint.observe(
        user_id=user_id,
        agent_output="Line 42 is missing a null check that could cause a runtime error. "
        "The variable naming across the module is inconsistent, which makes "
        "the code harder to follow. Error handling should be added for the "
        "external API call on line 58.",
        user_response="Much better. Please keep writing exactly like that.",
    )

    policy = await imprint.get_policy(user_id=user_id)
    print(f"\n[turn 3] {len(policy.memories)} memories")
    print(f"  policy: {policy.text}")

    # List what was stored.
    memories = await imprint.list_memories(user_id)
    print("\n--- Stored memories ---")
    for m in memories:
        print(f"  [{m.type.value}] {m.content}")

    await imprint.close()


if __name__ == "__main__":
    asyncio.run(main())
