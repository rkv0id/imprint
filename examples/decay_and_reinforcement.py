"""
decay_and_reinforcement.py -- stability, token budget, and pinning.

A personal assistant for "maya" has accumulated 19 memories. A tight token
budget means not all memories can fit in every policy compilation. The ones
that get recalled most often have higher recall_count, which influences their
stability ranking. Pinned memories always survive regardless of budget.

Demonstrates:
  - Token budget enforcement: get_policy() truncates when memories exceed budget
  - Memory stability and recall_count tracking
  - pin_memory() to guarantee a memory always appears in the policy
  - list_memories() showing stability values before and after repeated recall
  - FSRSStaticDecay (the default): stability is also time-based in production
    but this example focuses on recall_count as the observable signal

Note: in production, FSRSStaticDecay also decays memories over elapsed time
since last recall. Here we focus on the recall_count axis since we cannot
fake time in a short example.

Requirements:
  pip install imprint-mem
  export ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python examples/decay_and_reinforcement.py
"""

import asyncio
from pathlib import Path

from imprint import Imprint

DB_PATH = "decay_reinforcement.db"

# Tight budget -- forces imprint to drop roughly half the 19 memories.
# Adjust upward to see fewer truncations, downward for more.
TIGHT_BUDGET = 300  # tokens


def _print_memories(memories: list, label: str) -> None:
    print(f"\n--- {label} ({len(memories)} total) ---")
    print(f"  {'content':50s}  {'stability':9s}  {'recalls':7s}  {'pinned':6s}")
    print(f"  {'-' * 50}  {'-' * 9}  {'-' * 7}  {'-' * 6}")
    for m in sorted(memories, key=lambda x: x.stability, reverse=True):
        pinned = "YES" if m.pinned else ""
        print(f"  {m.content[:50]:50s}  {m.stability:9.3f}  {m.recall_count:7d}  {pinned:6s}")


async def main() -> None:
    # Always start from a clean slate regardless of previous runs.
    Path(DB_PATH).unlink(missing_ok=True)

    imprint = Imprint(
        agent_id="personal_assistant",
        store=DB_PATH,
        processing_mode="frugal",
    )
    await imprint.connect()

    print("=== Personal Assistant -- Stability and Budget ===\n")
    print(f"Token budget: {TIGHT_BUDGET} tokens (will force truncation with 19 memories)\n")

    user_id = "maya"

    # ------------------------------------------------------------------
    # Seed 19 memories across different topics.
    # Some will be recalled often (communication style).
    # Some will rarely come up (niche preferences).
    # ------------------------------------------------------------------

    communication = [
        "Always respond in a warm, conversational tone.",
        "Keep responses under 200 words unless explicitly asked for more detail.",
        "Start with the most important information first, then supporting detail.",
    ]
    format_prefs = [
        "Use numbered lists only for sequential steps, not for general points.",
        "Prefer plain paragraphs over formatted tables for comparisons.",
        "When writing code examples, always include the import statements.",
    ]
    scheduling = [
        "Maya works in CET timezone (UTC+1 or UTC+2 in summer).",
        "Meetings should never be scheduled before 9 AM or after 6 PM CET.",
        "Friday afternoons are reserved -- do not schedule anything after 2 PM.",
    ]
    technical = [
        "Maya's primary language is Python 3.12. Avoid suggesting older syntax.",
        "She uses uv for dependency management, not pip or poetry.",
        "Prefer async/await patterns over threading for concurrent code.",
    ]
    niche = [
        "Maya has a standing weekly sync with the Berlin team every Tuesday at 11 AM.",
        "Her preferred IDE is Zed, not VS Code or PyCharm.",
        "She reads academic papers in the evening -- save long reads for then.",
        "Budget approvals over 5000 EUR require sign-off from her director.",
        "Her team uses Linear for issue tracking, not Jira.",
        "Expense reports are due on the last working day of each month.",
        "She has an ergonomic keyboard and prefers keyboard shortcuts over menus.",
    ]

    # Niche memories go in first (oldest). Communication goes in last (newest).
    # list_memories() returns newest-first, so communication memories surface
    # first and fit within the tight budget. Niche memories get cut.
    all_directions = niche + technical + scheduling + format_prefs + communication
    await imprint.observe_directions(user_id=user_id, directions=all_directions)

    memories = await imprint.list_memories(user_id)
    print(f"Seeded {len(memories)} memories.\n")

    # Pin one critical memory so it always appears in the policy.
    budget_memory = next(
        (m for m in memories if "Budget" in m.content or "5000" in m.content),
        memories[0],
    )
    await imprint.pin_memory(budget_memory.id)
    print(f"Pinned: '{budget_memory.content[:60]}'")

    # ------------------------------------------------------------------
    # First policy compilation -- all 19 memories compete for the budget.
    # Roughly half will be dropped. Print which survived.
    # ------------------------------------------------------------------

    p1 = await imprint.get_policy(
        user_id=user_id,
        max_input_tokens=TIGHT_BUDGET,
    )
    memories_now = await imprint.list_memories(user_id)
    _print_memories(memories_now, "After first policy (all start equal)")
    print(f"\nPolicy used {len(p1.memories)} of {len(memories_now)} memories.")
    print(
        f"Pinned memory in policy: "
        f"{'YES' if any(m.id == budget_memory.id for m in p1.memories) else 'NO'}"
    )

    # ------------------------------------------------------------------
    # Repeatedly recall communication and format memories.
    # Each get_policy() call increments recall_count for retrieved memories.
    # After enough recalls, these memories have higher recall_count
    # and slightly higher effective stability.
    # ------------------------------------------------------------------

    print("\n--- Simulating 6 recalls for communication and format topics ---")
    for _ in range(6):
        await imprint.get_policy(
            user_id=user_id,
            context="drafting a reply to a colleague",
            max_input_tokens=TIGHT_BUDGET,
        )

    # ------------------------------------------------------------------
    # Compare stability and recall_count after repeated retrieval.
    # ------------------------------------------------------------------

    memories_after = await imprint.list_memories(user_id)
    _print_memories(memories_after, "After 6 context-specific recalls")

    p2 = await imprint.get_policy(
        user_id=user_id,
        context="drafting a reply to a colleague",
        max_input_tokens=TIGHT_BUDGET,
    )
    print(f"\nPolicy (with context) used {len(p2.memories)} of {len(memories_after)} memories.")
    print(
        f"Pinned memory in policy: "
        f"{'YES' if any(m.id == budget_memory.id for m in p2.memories) else 'NO'}"
    )
    print("\nTop memories by recall_count:")
    for m in sorted(memories_after, key=lambda x: x.recall_count, reverse=True)[:5]:
        pinned = " [PINNED]" if m.pinned else ""
        print(f"  recall_count={m.recall_count}  '{m.content[:55]}'{pinned}")

    await imprint.close()


if __name__ == "__main__":
    asyncio.run(main())
    Path(DB_PATH).unlink(missing_ok=True)
