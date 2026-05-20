"""
with_pydantic_ai.py -- imprint memory tools wired into a PydanticAI agent.

A personal assistant that uses seven imprint tools:

  recall()         -- compile and retrieve the current behavioral policy
  remember(...)    -- store something worth keeping about this user
  search(...)      -- semantic search over existing memories
  correct(...)     -- store a correction + negative learning signal
  reinforce()      -- positive learning signal
  signal_outcome() -- precise outcome score
  forget(...)      -- deactivate a specific memory by ID

Three scripted turns show the full lifecycle:

  Turn 1: user states preferences; the agent calls remember() and stores them.
  Turn 2: agent breaks a stated preference; user corrects it.
  Turn 3: agent recalls all accumulated preferences and applies them.

For the multi-service pattern (agent + imprint-server over HTTP), see
with_server_and_pydantic_ai.py.

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
You are a helpful personal assistant with a persistent memory layer.

At the start of each conversation call recall() to retrieve behavioral
instructions for this user. If the user states a preference, call remember()
to store it. When the user corrects you, call correct() with what they said.
When the conversation ends well, call reinforce().

Keep responses to two sentences maximum.\
"""

CONVERSATIONS = [
    "Hi. My name is Alex. I prefer short answers -- never more than two sentences. "
    "And please always skip the pleasantries, just get to the point.",
    "What are the main cloud providers?",
    "Summarize what you know about my preferences.",
]

# Simulated correction injected after turn 2 (agent tends to use bullet points).
CORRECTION_AFTER_TURN = 1
CORRECTION_TEXT = "I said no bullet points. Use a plain sentence."


async def run_conversation(imprint: Imprint, turn: int, user_message: str) -> None:
    print(f"\n--- Turn {turn + 1} ---")
    print(f"Alex: {user_message}")

    async with imprint.loop(user_id=USER) as loop:
        tools = make_pydantic_ai_tools(imprint, user_id=USER, loop=loop)
        agent = Agent(MODEL, system_prompt=SYSTEM_PROMPT, tools=tools)

        result = await agent.run(user_message)
        print(f"Agent: {result.output}")

        if turn == CORRECTION_AFTER_TURN:
            # Inject a correction: store it and close the loop with -1.0.
            print(f"Alex: {CORRECTION_TEXT}")
            await imprint.observe_directions(user_id=USER, directions=[CORRECTION_TEXT])
            loop.set_outcome(-1.0, correction=CORRECTION_TEXT)
        else:
            loop.set_outcome(0.9)


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
    print(policy.text or "(no policy text yet)")

    # Give background signal detection tasks up to 30 s to finish before
    # the event loop closes. Without this, asyncio.run() cancels them
    # mid-flight and the HTTP connection teardown can hang on Python 3.14+.
    import contextlib

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(imprint.drain(), timeout=30.0)


if __name__ == "__main__":
    asyncio.run(main())
