"""seed_demo.py -- populate a running imprint-server with demo data.

Seeds four agents (frugal, balanced, eager, frugal) with behavioral
memories, sessions with varied outcome signals to exercise online learning,
consolidation to show decay pruning, memory stability readouts, and API
keys for all roles.

Usage:
    just server-dev          # start server in one terminal
    just demo-seed           # seed data in another terminal

    # Or one-shot:
    just demo

    # Or directly:
    uv run python imprint-server/examples/seed_demo.py [--url http://localhost:8000]

Modes demonstrated:
  frugal:   peripheral-assistant, onboarding-guide
            -- observe_directions() + get_policy() without LLM
  balanced: code-review-bot
            -- policy compilation may call LLM if ANTHROPIC_API_KEY is set
  eager:    research-assistant
            -- policy always calls LLM; skipped gracefully if no key

Online learning signal path:
  Each session is opened, policy retrieved (creates recall events, updates
  retrieval stability), then closed with an outcome (updates BanditAlphaTuner
  and FSRS decay parameters). Multiple sessions for alice with different
  outcomes demonstrate the bandit adapting its alpha estimate.

Memory decay:
  After the learning sessions, alice's memories are listed showing current
  stability values. Then consolidate() is called with a low prune_threshold
  to demonstrate decay-based pruning.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

BASE = "http://localhost:8000"

AGENTS = [
    {
        "agent_id": "peripheral-assistant",
        "processing_mode": "frugal",
        "agent_description": "BI assistant for Peripheral platform users",
        "scopes": ["formatting", "reports", "data"],
        "dynamic_scopes": False,
    },
    {
        "agent_id": "code-review-bot",
        "processing_mode": "balanced",
        "agent_description": "Code review and engineering assistant",
        "scopes": ["style", "security", "performance"],
        "dynamic_scopes": True,
    },
    {
        "agent_id": "onboarding-guide",
        "processing_mode": "frugal",
        "agent_description": "New user onboarding and documentation agent",
        "scopes": ["global"],
        "dynamic_scopes": False,
    },
    {
        "agent_id": "research-assistant",
        "processing_mode": "eager",
        "agent_description": "Deep research and synthesis agent (LLM policy)",
        "scopes": ["research", "synthesis", "citations"],
        "dynamic_scopes": True,
    },
]

DIRECTIONS: dict[str, dict[str, list[str]]] = {
    "peripheral-assistant": {
        "alice": [
            "Always use Markdown tables when presenting financial data.",
            "User prefers quarterly breakdowns rather than monthly.",
            "Never use bullet points in executive summaries -- prose only.",
            "When uncertain about a metric definition, ask before assuming.",
            "Alice prefers responses under 200 words unless the topic requires depth.",
        ],
        "bob": [
            "Bob prefers concise answers with a link to full details.",
            "Always include a confidence level when making projections.",
            "Bob works across timezones -- always specify timezone when referencing times.",
        ],
    },
    "code-review-bot": {
        "carol": [
            "Carol works in Python. Use Python examples unless she specifies otherwise.",
            "Always suggest type annotations for new functions.",
            "Prefer pathlib.Path over os.path in any file handling code.",
            "Carol's team uses conventional commits -- mention this when relevant.",
            "Flag any use of mutable default arguments immediately.",
        ],
        "dave": [
            "Dave is a senior engineer. Skip basic explanations.",
            "Focus on security implications first, then style.",
            "Dave appreciates references to relevant PEPs or RFCs.",
        ],
    },
    "onboarding-guide": {
        "eve": [
            "Eve is a first-time user. Use friendly, encouraging language.",
            "Always offer a concrete next step at the end of each response.",
            "Eve prefers visual explanations -- suggest diagrams or tables.",
        ],
        "frank": [
            "Frank is migrating from a competitor product. Highlight differences proactively.",
            "Frank prefers CLI examples over GUI walkthroughs.",
        ],
    },
    "research-assistant": {
        "grace": [
            "Grace is a PhD researcher. Assume familiarity with academic literature.",
            "Always cite sources when making factual claims.",
            "Prefer primary sources over secondary summaries.",
            "Grace uses APA citation format in all her work.",
            "Flag when evidence is preliminary or contested.",
            "Grace works in computational biology -- prioritize that domain when ambiguous.",
        ],
    },
}

# Sessions: (agent_id, user_id, context, outcome).
# Multiple sessions for alice with different outcomes show the bandit adapting.
SESSIONS = [
    # alice: three sessions, outcomes vary -- demonstrates alpha tuner adaptation
    ("peripheral-assistant", "alice", "Q3 board report review", 0.9),
    ("peripheral-assistant", "alice", "revenue forecast", 0.6),
    ("peripheral-assistant", "alice", "churn analysis", 0.85),
    # other users
    ("peripheral-assistant", "bob", "pipeline metrics", 0.8),
    ("code-review-bot", "carol", "auth module review", 1.0),
    ("code-review-bot", "carol", "database layer review", 0.7),
    ("code-review-bot", "dave", "security audit", 0.9),
    ("onboarding-guide", "eve", "first login walkthrough", 0.75),
    ("onboarding-guide", "frank", "CLI quickstart", 0.85),
    # research-assistant/grace handled separately (eager, LLM required)
]

KEYS = [
    # Master
    {"label": "ci-master-key", "agent_id": None, "user_id": None},
    # Agent-scoped
    {"label": "peripheral-prod", "agent_id": "peripheral-assistant", "user_id": None},
    {"label": "code-review-prod", "agent_id": "code-review-bot", "user_id": None},
    {"label": "onboarding-prod", "agent_id": "onboarding-guide", "user_id": None},
    {"label": "research-prod", "agent_id": "research-assistant", "user_id": None},
    # User-bound (MCP multi-user pattern)
    {"label": "alice-personal", "agent_id": "peripheral-assistant", "user_id": "alice"},
    {"label": "bob-personal", "agent_id": "peripheral-assistant", "user_id": "bob"},
    {"label": "carol-personal", "agent_id": "code-review-bot", "user_id": "carol"},
    {"label": "dave-personal", "agent_id": "code-review-bot", "user_id": "dave"},
    {"label": "grace-personal", "agent_id": "research-assistant", "user_id": "grace"},
]


# -- Helpers ------------------------------------------------------------------


async def _run_session(
    client: httpx.AsyncClient,
    agent_id: str,
    user_id: str,
    context: str,
    outcome: float,
) -> int:
    """Open, get policy, close. Returns memory_count from policy."""
    open_r = await client.post(
        f"/v1/agents/{agent_id}/sessions",
        json={"user_id": user_id, "context": context},
    )
    if open_r.status_code != 200:
        print(f"  {agent_id}/{user_id}: session open ERROR {open_r.status_code}")
        return 0
    sid = open_r.json()["session_id"]

    pol_r = await client.post(
        f"/v1/agents/{agent_id}/sessions/{sid}/policy",
        json={"context": context},
    )
    mem_count = pol_r.json().get("memory_count", 0) if pol_r.status_code == 200 else 0

    close_r = await client.post(
        f"/v1/agents/{agent_id}/sessions/{sid}/close",
        json={"outcome": outcome},
    )
    ok = close_r.status_code == 200
    print(
        f"  {agent_id}/{user_id:<8} context={context!r:<32}"
        f" mems={mem_count} outcome={outcome} -> {'ok' if ok else 'ERROR'}"
    )
    return mem_count


def _stability_bar(val: float, width: int = 20) -> str:
    filled = round(val * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {val:.3f}"


# -- Main ---------------------------------------------------------------------


async def run(base: str) -> None:
    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        print(f"\nConnecting to {base} ...")
        health = await client.get("/health")
        if health.status_code != 200:
            print(f"  ERROR: server returned {health.status_code}. Is it running?")
            sys.exit(1)
        h = health.json()
        print(
            f"  store: {h.get('store', '?')}  "
            f"redis: {h.get('redis', '?')}  "
            f"status: {h.get('status', '?')}"
        )

        # -- Agents -----------------------------------------------------------
        print("\nCreating agents ...")
        for spec in AGENTS:
            r = await client.post("/v1/agents", json=spec)
            created = r.status_code == 200 and r.json().get("created")
            mode = spec["processing_mode"]
            print(
                f"  [{mode:8}] {spec['agent_id']:30} {'created' if created else 'already exists'}"
            )

        # -- Directions -------------------------------------------------------
        print("\nStoring behavioral directions ...")
        for agent_id, users in DIRECTIONS.items():
            for user_id, directions in users.items():
                r = await client.post(
                    f"/v1/agents/{agent_id}/memories/{user_id}/directions",
                    json={"directions": directions},
                )
                n = r.json().get("stored", 0) if r.status_code == 200 else f"ERROR {r.status_code}"
                print(f"  {agent_id}/{user_id}: {n} direction(s)")

        # -- Online learning sessions -----------------------------------------
        print(
            "\nRunning sessions with outcome signals ..."
            "\n  (each session retrieves memories, creates recall events,"
            "\n   and closes with an outcome that updates the BanditAlphaTuner"
            "\n   and FSRS decay parameters)"
        )
        for agent_id, user_id, context, outcome in SESSIONS:
            await _run_session(client, agent_id, user_id, context, outcome)

        # -- Standalone policy calls (policy event log) -----------------------
        print("\nCompiling standalone policies (populates policy event log) ...")
        for agent_id, users in DIRECTIONS.items():
            for user_id in users:
                r = await client.post(
                    f"/v1/agents/{agent_id}/policy",
                    json={"user_id": user_id, "context": "demo standalone"},
                )
                if r.status_code == 200:
                    n = r.json().get("memory_count", 0)
                    print(f"  {agent_id}/{user_id}: {n} memories in policy")

        # -- Eager agent session (LLM required) -------------------------------
        print("\nEager agent: research-assistant/grace ...")
        open_r = await client.post(
            "/v1/agents/research-assistant/sessions",
            json={"user_id": "grace", "context": "protein folding literature review"},
        )
        if open_r.status_code == 200:
            sid = open_r.json()["session_id"]
            pol_r = await client.post(
                f"/v1/agents/research-assistant/sessions/{sid}/policy",
                json={"context": "protein folding literature review"},
            )
            if pol_r.status_code == 200:
                n = pol_r.json().get("memory_count", 0)
                close_r = await client.post(
                    f"/v1/agents/research-assistant/sessions/{sid}/close",
                    json={"outcome": 0.95},
                )
                ok = close_r.status_code == 200
                print(f"  {n} memories, outcome=0.95 -> {'ok' if ok else 'ERROR'}")
            else:
                await client.post(f"/v1/agents/research-assistant/sessions/{sid}/close", json={})
                print("  eager policy skipped (set ANTHROPIC_API_KEY to enable LLM compilation)")

        # -- Memory stability readout -----------------------------------------
        # Show alice's current stability values to make the learning effect visible.
        # Memories that were recalled in sessions with high outcomes have higher
        # stability; those recalled after low-outcome sessions decay faster.
        print("\nMemory stability after online learning (peripheral-assistant/alice) ...")
        mems_r = await client.get("/v1/agents/peripheral-assistant/memories/alice")
        if mems_r.status_code == 200:
            mems = sorted(mems_r.json(), key=lambda m: m.get("stability", 0), reverse=True)
            for m in mems:
                stab = m.get("stability", 0.0)
                rc = m.get("recall_count", 0)
                content = m.get("content", "")[:50]
                bar = _stability_bar(stab)
                active = "active" if m.get("active") else "inactive"
                print(f"  {bar}  rc={rc}  [{active}]  {content!r}")

        health_r = await client.get("/v1/agents/peripheral-assistant/health/alice")
        if health_r.status_code == 200:
            hh = health_r.json()
            print(
                f"\n  summary: total={hh['total']} active={hh['active']}"
                f" pinned={hh['pinned']} avg_recalls={hh['avg_recall_count']:.2f}"
            )

        # -- Decay and pruning demo -------------------------------------------
        print("\nRunning consolidation for alice (decay-based pruning) ...")
        # prune_threshold=0.95 is aggressive -- prunes anything with stability < 0.95.
        # In a fresh demo all memories start at 1.0 and decay slightly after sessions.
        # Use 0.85 to show what consolidation does without wiping everything.
        cons_r = await client.post(
            "/v1/agents/peripheral-assistant/memories/alice/consolidate",
            params={"prune_threshold": 0.01},  # very low -- shows the mechanism
        )
        if cons_r.status_code == 200:
            pruned = cons_r.json().get("pruned", 0)
            print(
                f"  consolidate(prune_threshold=0.01): {pruned} memories pruned"
                f"  (threshold is intentionally low to demonstrate the API)"
            )

        # -- Correction -------------------------------------------------------
        print("\nApplying a correction for bob ...")
        corr_r = await client.post(
            "/v1/agents/peripheral-assistant/correct/bob",
            json={"content": "Do not include raw SQL in executive summaries."},
        )
        if corr_r.status_code == 200:
            mid = corr_r.json().get("memory_id", "") or ""
            print(f"  correction stored: {mid[:20]}...")

        # -- Deactivate one memory (diff demo) --------------------------------
        print("\nDeactivating one memory (for the diff endpoint demo) ...")
        mems_r2 = await client.get("/v1/agents/peripheral-assistant/memories/bob")
        if mems_r2.status_code == 200:
            bobs = mems_r2.json()
            if bobs:
                mid = bobs[-1]["id"]
                dr = await client.delete(f"/v1/agents/peripheral-assistant/memories/bob/{mid}")
                if dr.status_code == 200:
                    print(f"  deactivated {mid[:20]}...")

        # -- API keys via REST ------------------------------------------------
        print(f"\nCreating {len(KEYS)} API keys ...")
        for spec in KEYS:
            payload: dict[str, object] = {}
            if spec["label"]:
                payload["label"] = spec["label"]
            if spec["agent_id"]:
                payload["agent_id"] = spec["agent_id"]
            if spec["user_id"]:
                payload["user_id"] = spec["user_id"]
            r = await client.post("/v1/keys", json=payload)
            if r.status_code == 200:
                body = r.json()
                scope = spec["agent_id"] or "master"
                user = spec["user_id"] or "--"
                print(
                    f"  {spec['label']:22} scope={scope:26} user={user:8}"
                    f" hash={body.get('key_hash', '')[:12]}..."
                )
            else:
                print(f"  {spec['label']:22} ERROR {r.status_code}")

        # -- Summary ----------------------------------------------------------
        print("\nFetching summary ...")
        agents = (await client.get("/v1/agents")).json()
        keys_all = (await client.get("/v1/keys")).json()
        active_keys = [k for k in keys_all if k.get("active")]

        print(
            f"""
========================================
  Demo data seeded successfully.

  Agents:     {len(agents)} ({", ".join(a["agent_id"] for a in agents)})
  API keys:   {len(active_keys)} active ({len(active_keys)} total)
  Sessions:   {len(SESSIONS)} + 1 eager

  Open the admin dashboard:
    {base}/admin

  Memory Browser (try these):
    peripheral-assistant / alice   -- 3 sessions, stability evolution
    code-review-bot      / carol   -- 2 sessions
    research-assistant   / grace   -- eager mode, 6 directions

  Events panel (try these):
    peripheral-assistant / alice   -- recall events from 3 sessions
    code-review-bot      / carol   -- recall events from 2 sessions

  Memory diff (try this via API):
    GET /v1/agents/peripheral-assistant/memories/alice/diff
        ?since=<timestamp from 5 min ago>

  Metrics (includes extended gauges if IMPRINT_METRICS_EXTENDED=true):
    GET /metrics
========================================
"""
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed imprint-server with demo data.")
    parser.add_argument("--url", default=BASE, help="Server base URL")
    args = parser.parse_args()
    asyncio.run(run(args.url))


if __name__ == "__main__":
    main()
