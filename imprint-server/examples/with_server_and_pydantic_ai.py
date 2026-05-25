"""
with_server_and_pydantic_ai.py -- PydanticAI agent with imprint-server as
the memory backend (multi-service architecture).

Contrast with with_pydantic_ai.py which runs the memory store in the same
process. Here the agent and memory are separate services:

  imprint-server  <-- HTTP -->  PydanticAI agent (this script)

The agent gets memory tools backed by ImprintClient rather than a local
Imprint instance. This is the production pattern: imprint-server runs
continuously, agents connect and disconnect, all share a persistent store.

The tools are defined as plain PydanticAI Tool callables that wrap the
ImprintClient methods. There is no make_pydantic_ai_tools equivalent for
the HTTP client -- the wrapping is explicit and easy to customise.

Shows:
  - Manual tool definitions wrapping ImprintClient
  - Session lifecycle: open -> policy -> observe -> reinforce
  - Pagination to inspect memory state after the conversation
  - Correct/reinforce via the HTTP client

Requirements:
  pip install imprint-mem[client]
  pip install pydantic-ai-slim
  pip install imprint-server

  export ANTHROPIC_API_KEY=sk-ant-...

Start the server in another terminal before running:
  IMPRINT_DEFAULT_MODE=balanced imprint-server serve
  # or: just server-dev (uses frugal mode, no policy text but still functional)

Usage:
  python examples/with_server_and_pydantic_ai.py
"""

import asyncio

from imprint.client import ImprintClient, MemoryRecord, PageResult
from pydantic_ai import Agent, Tool

SERVER_URL = "http://localhost:8000"
AGENT_ID = "pa-agent"
USER = "morgan"
MODEL = "anthropic:claude-haiku-4-5-20251001"


def make_server_tools(client: ImprintClient, *, agent_id: str, user_id: str) -> list[Tool]:
    """Build PydanticAI tools backed by ImprintClient.

    Unlike make_pydantic_ai_tools (which takes a local Imprint instance),
    these tools talk to imprint-server over HTTP. Session management is
    handled externally -- open/close sessions via client.open_session() /
    client.reinforce() / client.correct() around the agent run.
    """

    async def recall(context: str | None = None) -> str:
        """Retrieve the compiled behavioral policy for this user."""
        policy = await client.get_policy(agent_id, user_id, context=context)
        return policy.text or "(no policy yet)"

    async def remember(content: str, scope: str | None = None) -> str:
        """Store something worth remembering about this user."""
        await client.observe_directions(agent_id, user_id, directions=[content], scope=scope)
        return "stored"

    async def search(query: str) -> str:
        """Search memories by semantic similarity."""
        results = await client.search_memories(agent_id, user_id, query, limit=5)
        if not results:
            return "(no results)"
        return "\n".join(f"- [{m.type}] {m.content}" for m in results)

    return [Tool(recall), Tool(remember), Tool(search)]


SYSTEM_PROMPT = """\
You are a helpful personal assistant backed by a persistent memory server.
At the start of each conversation call recall() to load your instructions.
Call remember() when the user shares something worth keeping long-term.
Call search() to look up specific preferences before answering.
Be concise -- two sentences maximum per response.\
"""

TURNS = [
    "Hi, I'm Morgan. I always want metric units, never imperial. "
    "And I prefer code examples in Python.",
    "What's 5 miles in metric?",
    "Show me how to open a file in your preferred language.",
]


async def run_turn(
    agent: Agent,
    client: ImprintClient,
    turn: int,
    message: str,
    session_id: str,
) -> None:
    print(f"\n--- Turn {turn + 1} ---")
    print(f"Morgan: {message}")
    result = await agent.run(message)
    print(f"Agent:  {result.output}")

    # Record the exchange so imprint learns from this session.
    await client.observe(
        AGENT_ID,
        USER,
        agent_output=result.output,
        user_response=message,
    )


async def main() -> None:
    async with ImprintClient(SERVER_URL) as client:
        tools = make_server_tools(client, agent_id=AGENT_ID, user_id=USER)
        agent = Agent(MODEL, system_prompt=SYSTEM_PROMPT, tools=tools)

        print("=== PydanticAI + imprint-server (multi-service) ===")
        print(f"Server: {SERVER_URL}  |  Agent: {AGENT_ID}  |  User: {USER}\n")

        # Open a session so observe() calls are tracked together for learning.
        session_id = await client.open_session(AGENT_ID, USER, context="demo")
        print(f"Session: {session_id}")

        for turn, message in enumerate(TURNS):
            await run_turn(agent, client, turn, message, session_id)

        # Close the session with a positive signal.
        applied = await client.reinforce(AGENT_ID, USER, session_id=session_id)
        print(f"\nSession closed (reinforced={applied})")

        # Inspect memory state via pagination.
        print("\n=== Memory state after conversation ===")
        page: PageResult[MemoryRecord] = await client.paginate_memories(AGENT_ID, USER, limit=10)
        print(f"Active memories ({len(page.items)} shown, has_more={page.has_more}):")
        for m in page.items:
            print(f"  [{m.type:12}] {m.content[:70]}")

        health = await client.memory_health(AGENT_ID, USER)
        print(f"\nHealth: total={health.total} active={health.active} pinned={health.pinned}")


if __name__ == "__main__":
    asyncio.run(main())
