"""seed_demo.py -- populate a running imprint-server with demo data.

Seeds three agents with different configurations, a handful of behavioral
memories per agent, several API keys, and some policy calls so the admin
dashboard at http://localhost:8000/admin has something interesting to show.

Usage:
    # Start the server first (in a separate terminal):
    just server-dev

    # Then seed the demo data:
    just demo-seed

    # Or run directly:
    uv run python imprint-server/examples/seed_demo.py [--url http://localhost:8000]
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

DIRECTIONS = {
    "peripheral-assistant": {
        "alice": [
            "Always use Markdown tables when presenting financial data.",
            "User prefers quarterly breakdowns rather than monthly.",
            "Never use bullet points in executive summaries -- prose only.",
            "When uncertain about a metric definition, ask before assuming.",
        ],
        "bob": [
            "Bob prefers concise answers with a link to full details.",
            "Always include a confidence level when making projections.",
        ],
    },
    "code-review-bot": {
        "carol": [
            "Carol works in Python. Use Python examples unless she specifies otherwise.",
            "Always suggest type annotations for new functions.",
            "Prefer pathlib.Path over os.path in any file handling code.",
            "Carol's team uses conventional commits -- mention this when relevant.",
        ],
        "dave": [
            "Dave is a senior engineer. Skip basic explanations.",
            "Focus on security implications first, then style.",
        ],
    },
    "onboarding-guide": {
        "eve": [
            "Eve is a first-time user. Use friendly, encouraging language.",
            "Always offer a concrete next step at the end of each response.",
        ],
    },
}


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

        # -- Policy calls (to populate policy events) -------------------------
        print("\nCompiling policies (populates policy event log) ...")
        for agent_id, users in DIRECTIONS.items():
            for user_id in users:
                r = await client.post(
                    f"/v1/agents/{agent_id}/policy",
                    json={"user_id": user_id, "context": "demo seed run"},
                )
                if r.status_code == 200:
                    body = r.json()
                    print(f"  {agent_id}/{user_id}: {body['memory_count']} memories in policy")
                else:
                    print(f"  {agent_id}/{user_id}: policy ERROR {r.status_code}")

        # -- Deactivate one memory to show diff/deactivation ------------------
        print("\nDeactivating one memory to demonstrate diff endpoint ...")
        r = await client.get("/v1/agents/peripheral-assistant/memories/alice")
        if r.status_code == 200:
            mems = r.json()
            if mems:
                mem_id = mems[0]["id"]
                dr = await client.delete(f"/v1/agents/peripheral-assistant/memories/alice/{mem_id}")
                if dr.status_code == 200:
                    print(f"  Deactivated memory {mem_id[:20]}...")
                else:
                    print(f"  Deactivation returned {dr.status_code}")

        # -- Summary ----------------------------------------------------------
        print("\nFetching summary ...")
        agents_r = await client.get("/v1/agents")
        keys_r = await client.get("/v1/keys")
        agents = agents_r.json() if agents_r.status_code == 200 else []
        keys = keys_r.json() if keys_r.status_code == 200 else []
        # -- Summary ----------------------------------------------------------
        agents_r = await client.get("/v1/agents")
        keys_r = await client.get("/v1/keys")
        agents = agents_r.json() if agents_r.status_code == 200 else []
        keys = keys_r.json() if keys_r.status_code == 200 else []

        print(f"""
========================================
  Demo data seeded successfully.

  Agents loaded:  {len(agents)}
  API keys:       {len([k for k in keys if k.get("active")])} active

  Open the admin dashboard:
    {base}/admin
========================================
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed imprint-server with demo data.")
    parser.add_argument("--url", default=BASE, help="Server base URL")
    args = parser.parse_args()
    asyncio.run(run(args.url))


if __name__ == "__main__":
    main()
