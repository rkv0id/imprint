"""
dynamic_scopes.py -- scope vocabulary that grows from interactions.

A coding assistant starts with no declared scopes. As the developer works
in different languages, imprint proposes and registers new scope names
(lang:python, lang:typescript) based on what it observes. The scope list
grows automatically -- no upfront vocabulary needed.

This is useful for:
  - Agents serving many contexts where domains are not known in advance
  - imprint-server deployments where each agent accumulates its own scopes
  - Any long-running agent where the scope taxonomy should emerge from use

Demonstrates:
  - dynamic_scopes=True with no initial scopes declared
  - imprint.scopes growing as new scopes are created
  - Scope inference (get_policy with context=) working against the
    dynamically created vocabulary

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

    # No scopes declared. dynamic_scopes=True allows the derivation LLM to
    # propose new scope names when memories clearly belong to a distinct context.
    imprint = Imprint(
        agent_id="coding_assistant",
        store=DB_PATH,
        processing_mode="balanced",
        dynamic_scopes=True,
        # scopes= not passed -- starts empty
    )
    await imprint.connect()

    print("=== Coding Assistant -- Dynamic Scope Creation ===\n")
    print(f"Starting scopes: {imprint.scopes}\n")

    user_id = "dev"

    # ------------------------------------------------------------------
    # Python session -- user corrects Python-specific behavior.
    # The derivation LLM sees an empty scope list and is expected to
    # propose a new scope (e.g. lang:python) for these memories.
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
        user_response="Always add type hints to every function parameter and return type.",
    )
    print(f"After turn 1: {imprint.scopes}")

    await imprint.observe(
        user_id=user_id,
        agent_output="config = {'host': 'localhost', 'port': 8080, 'debug': True}",
        user_response="Use dataclasses instead of plain dicts for structured data.",
    )
    print(f"After turn 2: {imprint.scopes}")

    await imprint.observe(
        user_id=user_id,
        agent_output="import os, sys, json, re, math",
        user_response=(
            "Group imports: stdlib first, then third-party, then local. One import per line."
        ),
    )
    print(f"After turn 3: {imprint.scopes}")

    # ------------------------------------------------------------------
    # TypeScript session -- user corrects TypeScript-specific behavior.
    # The derivation LLM now sees whatever scopes were created above and
    # should propose a distinct scope for TypeScript memories.
    # ------------------------------------------------------------------

    print("\n--- TypeScript session ---")

    await imprint.observe(
        user_id=user_id,
        agent_output="type UserConfig = { host: string; port: number; debug: boolean }",
        user_response=(
            "Prefer interfaces over type aliases for object shapes. Use interface UserConfig."
        ),
    )
    print(f"After turn 4: {imprint.scopes}")

    await imprint.observe(
        user_id=user_id,
        agent_output="function getUser(id) { return users[id] }",
        user_response="Always use explicit return types on functions. Never use implicit any.",
    )
    print(f"After turn 5: {imprint.scopes}")

    # ------------------------------------------------------------------
    # Show what was stored and in which scopes.
    # ------------------------------------------------------------------

    all_memories = await imprint.list_memories(user_id)
    print(f"\n--- Stored memories ({len(all_memories)} total) ---")
    for m in all_memories:
        print(f"  [{m.scope:20s}] {m.content[:60]}")

    # ------------------------------------------------------------------
    # Scope inference: get_policy with context= but no scopes=.
    # imprint now has a real vocabulary to infer from.
    # ------------------------------------------------------------------

    print("\n--- Scope inference against the dynamic vocabulary ---")

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

    await imprint.close()


if __name__ == "__main__":
    asyncio.run(main())
    Path(DB_PATH).unlink(missing_ok=True)
