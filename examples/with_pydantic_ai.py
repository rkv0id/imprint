"""
with_pydantic_ai.py -- imprint memory tools wired into a PydanticAI agent.

A personal assistant agent that accumulates behavioral preferences across
three conversations. After each conversation it can call:

  recall()         -- compile and retrieve the current policy
  remember(...)    -- explicitly store something worth keeping
  search(...)      -- semantic search over existing memories
  correct(...)     -- store a correction + apply negative learning signal
  reinforce()      -- apply a positive learning signal
  signal_outcome() -- set a precise outcome score
  forget(...)      -- deactivate a specific memory by ID

The agent starts fresh (no memories) and progressively learns how the user
wants it to behave. After three turns the policy reflects all accumulated
preferences. The MemoryLoop is opened before each turn and closed after so
the learning signal is associated with the right retrieved memories.

This pattern is the single-process equivalent of the imprint-server MCP
tools -- use it when the agent and the memory store run in the same process.
For multi-service deployments, see with_server_and_pydantic_ai.py.

Requirements:
  pip install imprint-mem
  pip install pydantic-ai-slim

  export ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python examples/with_pydantic_ai.py
"""

import asyncio

from pydantic_ai import Agent

from imprint import Imprint
from imprint.integrations.tools import make_pydantic_ai_tools
from imprint.stores.sqlite import SQLiteMemoryStore

USER = "alex"
AGENT_ID = "assistant"
MODEL = "anthropic:claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are a helpful personal assistant. At the start of each conversation
call recall() to load your behavioral instructions for this user. During
the conversation call remember() when the user tells you something worth
keeping. If the user corrects you, call correct() with what they said.
Call reinforce() or signal_outcome() at the end to record how the
conversation went. Be concise.\
"""

CONVERSATIONS = [
    # Turn 1: user establishes two preferences, agent should pick them up.
    "Hi. My name is Alex. I prefer short answers -- never more than two sentences. "
    "And please always skip the pleasantries, just get to the point.",
    # Turn 2: agent gets format wrong, user corrects.
    "What are the main cloud providers?",
    # Turn 3: check that the agent applies what it learned.
    "Summarize what you know about my preferences.",
]

EXPECTED_CORRECTIONS = {
    1: None,  # no correction expected on turn 0
    2: "I said no bullet points. Use a plain sentence.",  # inject a correction
    3: None,
}


async def run_conversation(imprint: Imprint, turn: int, user_message: str) -> None:
    print(f"\n--- Turn {turn + 1} ---")
    print(f"Alex: {user_message}")

    # Open a MemoryLoop so recall() records which memories were retrieved
    # and correct/reinforce can close the loop with the right signal.
    async with imprint.loop(user_id=USER) as loop:
        tools = make_pydantic_ai_tools(imprint, user_id=USER, loop=loop)
        agent = Agent(MODEL, system_prompt=SYSTEM_PROMPT, tools=tools)

        result = await agent.run(user_message)
        print(f"Agent: {result.output}")

        # Simulate a user correction on turn 2.
        correction = EXPECTED_CORRECTIONS.get(turn)
        if correction:
            print(f"Alex: {correction}")
            # Store the correction as a memory and close the loop with a
            # negative signal. In production the agent calls correct() itself.
            await imprint.observe_directions(
                user_id=USER,
                directions=[correction],
            )
            loop.set_outcome(-1.0, correction=correction)
        else:
            # Positive outcome -- the agent handled it well.
            loop.set_outcome(0.8)


async def main() -> None:
    store = SQLiteMemoryStore(":memory:")
    imprint = Imprint(
        agent_id=AGENT_ID,
        store=store,
        processing_mode="balanced",
    )
    await imprint.connect()

    print("=== PydanticAI + imprint memory tools ===")
    print(f"Agent: {AGENT_ID}  |  User: {USER}  |  Model: {MODEL}\n")

    for turn, message in enumerate(CONVERSATIONS):
        await run_conversation(imprint, turn, message)

    print("\n\n=== Final memory state ===")
    memories = await imprint.list_memories(USER)
    print(f"Total active memories: {len(memories)}")
    for m in memories:
        print(f"  [{m.type.value:10}] {m.content[:75]}")

    print("\n=== Compiled policy after three turns ===")
    policy = await imprint.get_policy(user_id=USER)
    print(policy.text or "(no policy text -- add more turns to build up memories)")

    await imprint.drain()


if __name__ == "__main__":
    asyncio.run(main())
