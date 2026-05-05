"""
retrieval_tuning.py -- MemoryLoop and the BanditAlphaTuner learning in action.

A support agent serves "jordan" across 16 simulated interaction loops.
The first batch has consistently positive outcomes (good retrieval).
The second batch has mixed outcomes. The BanditAlphaTuner adjusts the
sparse/dense retrieval balance (alpha) based on which retrieved memories
led to good outcomes.

alpha controls how much weight goes to sparse (BM25) vs dense (vector)
retrieval. alpha=0.1 means 10% sparse, 90% dense. alpha=0.9 means
90% sparse, 10% dense. The bandit samples from a Beta distribution
per arm and picks the highest sample (Thompson sampling).

Demonstrates:
  - MemoryLoop: open, get_policy, set_outcome, finalize_loop
  - loop.alpha_used showing which retrieval balance was sampled
  - BanditAlphaTuner state (successes/failures per arm) evolving
  - drain() to ensure background learning tasks complete

Requirements:
  pip install imprint-mem[vector,openai]
  export ANTHROPIC_API_KEY=sk-ant-...
  export OPENAI_API_KEY=sk-...

Usage:
  python examples/retrieval_tuning.py
"""

import asyncio
from pathlib import Path

from imprint import BanditAlphaTuner, Imprint, SQLiteMemoryStore, SQLiteVecStore
from imprint.openai import OpenAIEmbedder

DB_PATH = "retrieval_tuning.db"
DIM = 512


def _arm_summary(tuner: BanditAlphaTuner) -> str:
    """Print successes/failures per arm to show bandit state."""
    arms = [0.1, 0.3, 0.5, 0.7, 0.9]
    state = tuner.get_state()
    parts = []
    for i, arm in enumerate(arms):
        s = state["s"][i]
        f = state["f"][i]
        parts.append(f"alpha={arm:.1f}: s={s:.1f} f={f:.1f}")
    return " | ".join(parts)


async def main() -> None:
    # Always start from a clean slate regardless of previous runs.
    Path(DB_PATH).unlink(missing_ok=True)

    store = SQLiteMemoryStore(DB_PATH)
    await store.connect()

    embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=DIM)
    tuner = BanditAlphaTuner()  # keep a reference so we can inspect state

    imprint = Imprint(
        agent_id="support_agent",
        store=store,
        vector_store=SQLiteVecStore(store.conn, dim=DIM),
        embedder=embedder,
        alpha_tuner=tuner,
        processing_mode="balanced",
        scopes=["topic:product", "topic:billing", "topic:support"],
    )
    await imprint.connect()

    print("=== Support Agent -- Retrieval Tuning ===\n")

    # ------------------------------------------------------------------
    # Seed memories across three support topics.
    # ------------------------------------------------------------------

    await imprint.observe_directions(
        user_id="jordan",
        scope="topic:product",
        directions=[
            "Always check if the reported issue is a known limitation before escalating.",
            "Link to the changelog when mentioning a recent feature change.",
            "Confirm which product version the customer is running before diagnosing.",
        ],
    )
    await imprint.observe_directions(
        user_id="jordan",
        scope="topic:billing",
        directions=[
            "Never quote a price without confirming the customer's current plan first.",
            "Refund requests above $500 require manager approval before proceeding.",
            "Always send a confirmation email after any billing change.",
        ],
    )
    await imprint.observe_directions(
        user_id="jordan",
        scope="topic:support",
        directions=[
            "Start every response by acknowledging the customer's frustration.",
            "Avoid technical jargon unless the customer uses it first.",
            "Always end with a clear next step or expected resolution time.",
        ],
    )

    memories = await imprint.list_memories("jordan")
    print(f"Seeded {len(memories)} memories across 3 topics.\n")
    print("Initial bandit state:")
    print(f"  {_arm_summary(tuner)}\n")

    # ------------------------------------------------------------------
    # Batch A: 8 loops with strong positive outcomes (0.85-1.0).
    # Good retrieval -> bandit should reward the arms that were sampled.
    # ------------------------------------------------------------------

    print("--- Batch A: 8 loops, positive outcomes (good retrieval) ---")
    outcomes_a = [0.95, 0.88, 0.92, 0.85, 0.97, 0.90, 0.88, 0.93]
    contexts_a = [
        "customer cannot log in after recent update",
        "billing charge appeared twice this month",
        "feature stopped working after upgrade to v4.2",
        "customer wants to downgrade their subscription",
        "error 503 on the dashboard since yesterday",
        "invoice shows wrong currency for EU customers",
        "response time degraded over the last week",
        "refund request for duplicate charge",
    ]

    for i, (ctx, outcome) in enumerate(zip(contexts_a, outcomes_a, strict=True)):
        loop = await imprint.open_loop(user_id="jordan")
        await imprint.get_policy(user_id="jordan", context=ctx, loop=loop)
        loop.set_outcome(outcome)
        await imprint.finalize_loop(loop)
        await imprint.drain()
        print(
            f"  loop {i + 1:2d}  alpha_used={loop.alpha_used:.1f}  outcome={outcome:.2f}  "
            f"memories_retrieved={len(loop.retrieved_memories)}"
        )

    print("\nAfter batch A:")
    print(f"  {_arm_summary(tuner)}\n")

    # ------------------------------------------------------------------
    # Batch B: 8 loops with weaker outcomes (0.2-0.5).
    # Mixed or poor retrieval -> bandit should penalize the sampled arms.
    # ------------------------------------------------------------------

    print("--- Batch B: 8 loops, weak outcomes (poor retrieval match) ---")
    outcomes_b = [0.3, 0.45, 0.25, 0.4, 0.35, 0.5, 0.2, 0.4]
    contexts_b = [
        "general product question about roadmap",
        "customer wants to export all their data",
        "API rate limit documentation unclear",
        "onboarding assistance for new enterprise customer",
        "request for custom integration support",
        "question about SLA terms in enterprise contract",
        "need help migrating from legacy version",
        "bulk user import failing with 400 error",
    ]

    for i, (ctx, outcome) in enumerate(zip(contexts_b, outcomes_b, strict=True)):
        loop = await imprint.open_loop(user_id="jordan")
        await imprint.get_policy(user_id="jordan", context=ctx, loop=loop)
        loop.set_outcome(outcome)
        await imprint.finalize_loop(loop)
        await imprint.drain()
        print(
            f"  loop {i + 1:2d}  alpha_used={loop.alpha_used:.1f}  outcome={outcome:.2f}  "
            f"memories_retrieved={len(loop.retrieved_memories)}"
        )

    print("\nAfter batch B:")
    print(f"  {_arm_summary(tuner)}")

    # ------------------------------------------------------------------
    # The arm with the best success/failure ratio will be sampled most
    # often in future loops. Thompson sampling explores other arms too,
    # so the winning arm is not chosen every time -- it is chosen more often.
    # With more loops (hundreds of real interactions) the convergence becomes
    # much clearer. This example shows the mechanism, not the final state.
    # ------------------------------------------------------------------

    st = tuner.get_state()
    arms = [0.1, 0.3, 0.5, 0.7, 0.9]
    winning_arm = max(
        range(5),
        key=lambda i: st["s"][i] / (st["s"][i] + st["f"][i]),
    )
    print(
        f"\nCurrent best arm: alpha={arms[winning_arm]:.1f} "
        f"(s={st['s'][winning_arm]:.1f} / f={st['f'][winning_arm]:.1f})"
    )
    print("Run more loops to see clearer convergence in production.")

    await imprint.close()
    await store.close()  # store is owned by us, not imprint


if __name__ == "__main__":
    asyncio.run(main())
    Path(DB_PATH).unlink(missing_ok=True)
