"""
online_learning.py -- FSRSGradientDecay learning per-agent decay parameters.

The default FSRSStaticDecay uses fixed FSRS constants to predict how quickly
a memory's relevance decays over time. FSRSGradientDecay starts from the same
place but learns the decay parameters from actual feedback -- it adapts to
this specific agent's interaction patterns.

Two imprint instances run the same simulated interactions side by side:
one with FSRSStaticDecay (fixed), one with FSRSGradientDecay (adaptive).
After 20 interactions with explicit outcomes, the gradient model's
effective_stability predictions differ from the static model's, reflecting
what it learned from the feedback.

Demonstrates:
  - FSRSGradientDecay setup and comparison with FSRSStaticDecay
  - MemoryLoop with set_outcome() feeding the learning signal
  - decay.learn() being called by finalize_loop() in the background
  - effective_stability() diverging between static and gradient models
  - drain() to flush background learning tasks before reading state

Requirements:
  pip install imprint-mem[online]
  export ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python examples/online_learning.py
"""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from imprint import FSRSGradientDecay, FSRSStaticDecay, Imprint, SQLiteMemoryStore
from imprint.types import Memory

DB_STATIC = "online_static.db"
DB_GRADIENT = "online_gradient.db"


async def seed_and_run(
    imprint: Imprint,
    *,
    label: str,
    outcomes: list[float],
) -> list[Memory]:
    """Seed memories, run loops with outcomes, return memories after learning."""
    user_id = "user"

    await imprint.observe_directions(
        user_id=user_id,
        directions=[
            "Always respond in English.",
            "Keep explanations concise -- one idea per paragraph.",
            "Prefer code examples over abstract descriptions.",
            "When uncertain, ask a clarifying question rather than guessing.",
            "Summarize long answers with a one-sentence takeaway at the end.",
        ],
    )

    memories_before = await imprint.list_memories(user_id)
    print(f"[{label}] seeded {len(memories_before)} memories")

    # Run loops with provided outcomes to feed the learning signal.
    contexts = [
        "answering a technical question about async Python",
        "explaining a billing discrepancy to a customer",
        "helping debug a production incident",
        "drafting a project summary for the team",
        "reviewing a pull request for code style",
    ]

    for i, outcome in enumerate(outcomes):
        ctx = contexts[i % len(contexts)]
        loop = await imprint.open_loop(user_id=user_id)
        await imprint.get_policy(user_id=user_id, context=ctx, loop=loop)
        loop.set_outcome(outcome)
        await imprint.finalize_loop(loop)

    return await imprint.list_memories(user_id)


async def main() -> None:
    # Always start from a clean slate regardless of previous runs.
    for p in [DB_STATIC, DB_GRADIENT]:
        Path(p).unlink(missing_ok=True)

    # Same sequence of outcomes for both models so comparison is fair.
    # Strongly positive first (model should learn these memories are worth keeping),
    # then weaker (model learns some contexts don't reinforce well).
    outcomes = [
        0.95,
        0.90,
        0.88,
        0.92,
        0.85,  # strong positives
        0.75,
        0.70,
        0.65,
        0.60,
        0.72,  # moderate
        0.40,
        0.35,
        0.45,
        0.30,
        0.50,  # weak
        0.20,
        0.25,
        0.15,
        0.30,
        0.22,  # negative
    ]

    print("=== Online Learning -- Static vs Gradient Decay ===\n")
    print(f"Running {len(outcomes)} loops. Same outcomes for both models.\n")

    # ------------------------------------------------------------------
    # Static decay instance
    # ------------------------------------------------------------------
    static_decay = FSRSStaticDecay()
    store_static = SQLiteMemoryStore(DB_STATIC)
    imprint_static = Imprint(
        agent_id="agent_static",
        store=store_static,
        decay_model=static_decay,
        processing_mode="frugal",
    )
    await imprint_static.connect()
    memories_static = await seed_and_run(imprint_static, label="static", outcomes=outcomes)
    await imprint_static.close()
    await store_static.close()

    # ------------------------------------------------------------------
    # Gradient decay instance (learns from feedback)
    # ------------------------------------------------------------------
    gradient_decay = FSRSGradientDecay(learning_rate=0.05)
    store_gradient = SQLiteMemoryStore(DB_GRADIENT)
    imprint_gradient = Imprint(
        agent_id="agent_gradient",
        store=store_gradient,
        decay_model=gradient_decay,
        processing_mode="frugal",
    )
    await imprint_gradient.connect()
    memories_gradient = await seed_and_run(imprint_gradient, label="gradient", outcomes=outcomes)
    await imprint_gradient.close()
    await store_gradient.close()

    # ------------------------------------------------------------------
    # Compare effective_stability predictions at three time horizons.
    # The gradient model learns from the mixed outcomes above.
    # With more positive outcomes early, it may predict higher stability
    # (memories are worth keeping longer). With negative outcomes later,
    # it adjusts downward. The static model stays fixed.
    # ------------------------------------------------------------------

    now = datetime.now(UTC)
    horizons = [
        ("now", now),
        ("7 days", now + timedelta(days=7)),
        ("30 days", now + timedelta(days=30)),
    ]

    print("\n--- effective_stability comparison (per memory, per time horizon) ---")
    print(f"  {'memory content':45s}  ", end="")
    for label, _ in horizons:
        print(f"  {'static@' + label:14s}  {'gradient@' + label:16s}", end="")
    print()
    print(f"  {'-' * 45}  " + "  " + ("  " + "-" * 14 + "  " + "-" * 16) * len(horizons))

    for ms, mg in zip(
        sorted(memories_static, key=lambda m: m.id),
        sorted(memories_gradient, key=lambda m: m.id),
        strict=False,
    ):
        print(f"  {ms.content[:45]:45s}  ", end="")
        for _, t in horizons:
            s_val = static_decay.effective_stability(ms, t)
            g_val = gradient_decay.effective_stability(mg, t)
            diff = "^" if g_val > s_val + 0.05 else ("v" if g_val < s_val - 0.05 else "=")
            print(f"  {s_val:6.3f}        {g_val:6.3f} {diff}        ", end="")
        print()

    print("\nKey: ^ gradient predicts higher stability, v lower, = roughly equal")
    print("With more training loops the divergence grows. This example shows the")
    print("mechanism -- in production, hundreds of interactions shape the model.")


if __name__ == "__main__":
    asyncio.run(main())
    for p in [DB_STATIC, DB_GRADIENT]:
        Path(p).unlink(missing_ok=True)
