"""
dynamic_scopes.py -- scope vocabulary that grows from interactions.

A coding assistant starts with no declared scopes. As the developer works
in different languages, imprint proposes and registers new scope names
based on what it observes. The scope list grows automatically -- no upfront
vocabulary needed.

User responses explicitly name the language so the derivation LLM has a
clear signal to create separate scopes (e.g. python, typescript) rather
than collapsing everything into a generic theme.

Demonstrates:
  - dynamic_scopes=True with no initial scopes declared
  - imprint.scopes growing as new scopes are created
  - Scope inference (get_policy with context=) routing to the right scope
  - consolidate_scopes() reorganizing the vocabulary when triggered

Note: dynamic scope creation requires balanced or eager mode. frugal mode
uses heuristic derivation that always falls back to "global" for scope.

Requirements:
  pip install imprint-mem
  export ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python examples/dynamic_scopes.py
"""

import asyncio
from pathlib import Path

from imprint import Imprint

DB_PATH = "dynamic_scopes.db"


async def main() -> None:
    Path(DB_PATH).unlink(missing_ok=True)

    imprint = Imprint(
        agent_id="coding_assistant",
        store=DB_PATH,
        processing_mode="balanced",
        dynamic_scopes=True,
        scope_consolidation_threshold=5,
    )
    await imprint.connect()

    print("=== Coding Assistant -- Dynamic Scope Creation ===\n")
    print(f"Starting scopes: {imprint.scopes}\n")

    user_id = "dev"

    # ------------------------------------------------------------------
    # Python session.
    # User responses explicitly mention Python so the derivation LLM
    # has a clear language signal when choosing a scope name.
    # ------------------------------------------------------------------

    print("--- Python session ---")

    await imprint.observe(
        user_id=user_id,
        agent_output=(
            "def process(items, threshold):\n"
            "    result = []\n"
            "    for x in items:\n"
            "        if x > threshold:\n"
            "            result.append(x)\n"
            "    return result"
        ),
        user_response=(
            "In Python, always add type hints to every function parameter "
            "and the return type. Never leave them implicit."
        ),
    )
    print(f"After turn 1: {imprint.scopes}")

    await imprint.observe(
        user_id=user_id,
        agent_output="config = {'host': 'localhost', 'port': 8080, 'debug': True}",
        user_response=(
            "For Python code, use dataclasses instead of plain dicts "
            "whenever the data has a fixed structure."
        ),
    )
    print(f"After turn 2: {imprint.scopes}")

    await imprint.observe(
        user_id=user_id,
        agent_output="import os, sys, json, re, math",
        user_response=(
            "Python imports should be grouped: stdlib first, third-party second, "
            "local last. One import per line."
        ),
    )
    print(f"After turn 3: {imprint.scopes}")

    # ------------------------------------------------------------------
    # TypeScript session.
    # User responses explicitly mention TypeScript so the LLM creates
    # a distinct scope rather than merging with the Python one.
    # ------------------------------------------------------------------

    print("\n--- TypeScript session ---")

    await imprint.observe(
        user_id=user_id,
        agent_output="type UserConfig = { host: string; port: number; debug: boolean }",
        user_response=(
            "In TypeScript, always use interface instead of type alias "
            "for object shapes. Rename this to interface UserConfig."
        ),
    )
    print(f"After turn 4: {imprint.scopes}")

    await imprint.observe(
        user_id=user_id,
        agent_output="function getUser(id) { return users[id] }",
        user_response=(
            "TypeScript functions must always have explicit return types declared. "
            "Never rely on implicit any."
        ),
    )
    print(f"After turn 5: {imprint.scopes}")

    await imprint.observe(
        user_id=user_id,
        agent_output="const items = data.map(x => x.value)",
        user_response=(
            "For TypeScript arrow functions, always annotate the parameter "
            "types and the return type explicitly."
        ),
    )
    print(f"After turn 6: {imprint.scopes}")

    # ------------------------------------------------------------------
    # Show what was stored and in which scopes.
    # ------------------------------------------------------------------

    all_memories = await imprint.list_memories(user_id)
    print(f"\n--- Stored memories ({len(all_memories)} total) ---")
    for m in all_memories:
        print(f"  [{m.scope:20s}] {m.content[:60]}")

    # ------------------------------------------------------------------
    # Scope inference: get_policy with context= but no scopes=.
    # imprint picks the relevant scope from the vocabulary it built.
    # ------------------------------------------------------------------

    print("\n--- Scope inference (no explicit scopes= passed) ---")

    p_py = await imprint.get_policy(
        user_id=user_id,
        context="writing a Python function to parse JSON config files",
    )
    print(f"\nPython context -> {len(p_py.memories)} memories:")
    for m in p_py.memories:
        print(f"  [{m.scope:20s}] {m.content[:60]}")

    p_ts = await imprint.get_policy(
        user_id=user_id,
        context="defining TypeScript interfaces for a REST API client",
    )
    print(f"\nTypeScript context -> {len(p_ts.memories)} memories:")
    for m in p_ts.memories:
        print(f"  [{m.scope:20s}] {m.content[:60]}")

    print(f"\nFinal scope vocabulary: {imprint.scopes}")

    # ------------------------------------------------------------------
    # Force scope consolidation.
    # If multiple scopes exist that overlap or could be renamed more
    # precisely, the LLM reorganizes them here.
    # consolidate_scopes() also fires automatically in the background
    # every scope_consolidation_threshold memories (set to 5 above).
    # ------------------------------------------------------------------

    print("\n--- Triggering scope consolidation ---")
    print(f"Before: {imprint.scopes}")
    await imprint.consolidate_scopes(user_id=user_id)
    print(f"After:  {imprint.scopes}")

    all_memories_after = await imprint.list_memories(user_id)
    print(f"\n--- Memories after consolidation ({len(all_memories_after)} total) ---")
    for m in all_memories_after:
        print(f"  [{m.scope:20s}] {m.content[:60]}")

    await imprint.close()


if __name__ == "__main__":
    asyncio.run(main())
    Path(DB_PATH).unlink(missing_ok=True)
