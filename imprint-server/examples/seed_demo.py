"""seed_demo.py -- populate a running imprint-server with demo data.

Seeds three agents with different configurations, behavioral memories per
agent, sessions with outcome signals to demonstrate online learning, API
keys created via REST, and a deactivated memory so the diff endpoint has
something to show.

Usage:
    # Start the server first (in a separate terminal):
    just server-dev

    # Then seed the demo data:
    just demo-seed

    # Or run directly:
    uv run python imprint-server/examples/seed_demo.py [--url http://localhost:8000]

Note: sessions with outcome signals require observe() to have stored enough
memories for the policy to retrieve them. In frugal mode this works without
any LLM API keys.
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
            "Eve prefers visual explanations -- suggest diagrams or tables where appropriate.",
        ],
        "frank": [
            "Frank is migrating from a competitor product. Highlight differences proactively.",
            "Frank prefers CLI examples over GUI walkthroughs.",
        ],
    },
}

KEYS = [
    {"label": "ci-master-key", "agent_id": None, "user_id": None},
    {"label": "peripheral-prod", "agent_id": "peripheral-assistant", "user_id": None},
    {"label": "alice-personal", "agent_id": "peripheral-assistant", "user_id": "alice"},
    {"label": "carol-personal", "agent_id": "code-review-bot", "user_id": "carol"},
]

# Sessions to open, observe, get policy, then close with an outcome.
# This exercises the online learning signal path and creates memory events.
SESSIONS = [
    {
        "agent_id": "peripheral-assistant",
        "user_id": "alice",
        "context": "Q3 board report review",
        "outcome": 0.9,
    },
    {
        "agent_id": "peripheral-assistant",
        "user_id": "alice",
        "context": "revenue forecast",
        "outcome": 0.6,
    },
    {
        "agent_id": "peripheral-assistant",
        "user_id": "bob",
        "context": "pipeline metrics",
        "outcome": 0.8,
    },
    {
        "agent_id": "code-review-bot",
        "user_id": "carol",
        "context": "auth module review",
        "outcome": 1.0,
    },
    {
        "agent_id": "onboarding-guide",
        "user_id": "eve",
        "context": "first login walkthrough",
        "outcome": 0.75,
    },
]


async def run(base: str) -> None:
    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        print(f"\nConnecting to {base} ...")
        health = await client.get("/health")
        if health.status_code != 200:
            print(f"  ERROR: server returned {health.status_code}. Is it running?")
            sys.exit(1)
        h = health.json()
        store = h.get("store", "?")
        redis = h.get("redis", "?")
        status = h.get("status", "?")
        print(f"  store: {store}  redis: {redis}  status: {status}")

        # -- Agents -----------------------------------------------------------
        print("\nCreating agents ...")
        for spec in AGENTS:
            r = await client.post("/v1/agents", json=spec)
            created = r.status_code == 200 and r.json().get("created")
            verb = "created" if created else "already exists"
            print(f"  {spec['agent_id']:30} {verb}")

        # -- Directions -------------------------------------------------------
        print("\nStoring behavioral directions ...")
        for agent_id, users in DIRECTIONS.items():
            for user_id, directions in users.items():
                r = await client.post(
                    f"/v1/agents/{agent_id}/memories/{user_id}/directions",
                    json={"directions": directions},
                )
                if r.status_code == 200:
                    stored = r.json().get("stored", 0)
                    print(f"  {agent_id}/{user_id}: {stored} direction(s) stored")
                else:
                    print(f"  {agent_id}/{user_id}: ERROR {r.status_code}")

        # -- Sessions with outcomes (online learning) -------------------------
        print("\nRunning sessions with outcome signals (online learning) ...")
        for spec in SESSIONS:
            agent_id = spec["agent_id"]
            user_id = spec["user_id"]
            context = spec["context"]
            outcome = spec["outcome"]

            # Open session
            open_r = await client.post(
                f"/v1/agents/{agent_id}/sessions",
                json={"user_id": user_id, "context": context},
            )
            if open_r.status_code != 200:
                print(f"  {agent_id}/{user_id}: session open ERROR {open_r.status_code}")
                continue
            sid = open_r.json()["session_id"]

            # Get policy (retrieves memories, creates recall events)
            pol_r = await client.post(
                f"/v1/agents/{agent_id}/sessions/{sid}/policy",
                json={"context": context},
            )
            mem_count = pol_r.json().get("memory_count", 0) if pol_r.status_code == 200 else 0

            # Close session with outcome (applies learning signal)
            close_r = await client.post(
                f"/v1/agents/{agent_id}/sessions/{sid}/close",
                json={"outcome": outcome},
            )
            ok = close_r.status_code == 200
            status_str = "ok" if ok else f"ERROR {close_r.status_code}"
            print(
                f"  {agent_id}/{user_id}: {mem_count} memories, outcome={outcome} -> {status_str}"
            )

        # -- Policy calls (outside sessions, populates policy event log) ------
        print("\nCompiling standalone policies ...")
        for agent_id, users in DIRECTIONS.items():
            for user_id in users:
                r = await client.post(
                    f"/v1/agents/{agent_id}/policy",
                    json={"user_id": user_id, "context": "demo standalone policy"},
                )
                if r.status_code == 200:
                    body = r.json()
                    print(f"  {agent_id}/{user_id}: {body['memory_count']} memories in policy")

        # -- Deactivate one memory to demo diff endpoint ----------------------
        print("\nDeactivating one memory to demonstrate diff endpoint ...")
        r = await client.get("/v1/agents/peripheral-assistant/memories/alice")
        if r.status_code == 200:
            mems = r.json()
            if mems:
                mid = mems[-1]["id"]
                dr = await client.delete(f"/v1/agents/peripheral-assistant/memories/alice/{mid}")
                if dr.status_code == 200:
                    print(f"  Deactivated memory {mid[:20]}...")

        # -- Apply a correction (negative signal + memory) --------------------
        print("\nApplying a correction for bob ...")
        corr_r = await client.post(
            "/v1/agents/peripheral-assistant/correct/bob",
            json={"content": "Do not include raw SQL in executive summaries."},
        )
        if corr_r.status_code == 200:
            mid = corr_r.json().get("memory_id", "")
            print(f"  Correction stored, memory_id={mid[:20] if mid else 'n/a'}...")

        # -- API keys via REST ------------------------------------------------
        print("\nCreating API keys ...")
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
                print(f"  {spec['label']:25} hash: {body.get('key_hash', '')[:16]}...")
                # Raw key shown once -- operators should copy it
                raw = body.get("raw_key", "")
                if raw:
                    print(f"    raw (copy now): {raw}")
            else:
                print(f"  {spec['label']:25} ERROR {r.status_code}: {r.text[:80]}")

        # -- Summary ----------------------------------------------------------
        print("\nFetching summary ...")
        agents_r = await client.get("/v1/agents")
        keys_r = await client.get("/v1/keys")
        agents = agents_r.json() if agents_r.status_code == 200 else []
        keys = keys_r.json() if keys_r.status_code == 200 else []

        print(
            f"""
========================================
  Demo data seeded successfully.

  Agents loaded:  {len(agents)}
  API keys:       {len([k for k in keys if k.get("active")])} active
  Sessions run:   {len(SESSIONS)}

  Open the admin dashboard:
    {base}/admin

  Try the Memory Browser:
    Agent: peripheral-assistant, User: alice
    Agent: code-review-bot, User: carol

  Try the Events panel to see recall events from sessions:
    Agent: peripheral-assistant, User: alice
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
