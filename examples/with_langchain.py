"""
with_langchain.py -- ImprintCallbackHandler wired into a LangChain chain.

ImprintCallbackHandler hooks into LangChain's callback system. When a chain
or agent runs, the handler:
  - captures user input from on_chain_start
  - captures the last LLM generation from on_llm_end
  - fires imprint.observe() when on_agent_finish triggers

This example simulates the LangChain callbacks directly so it does not
require langchain_anthropic or a full agent loop. In a real deployment,
attach the handler to any chain or agent via callbacks=[handler].

For full LangChain integration without the simulation, see the comment
block at the bottom of this file.

Demonstrates:
  - ImprintCallbackHandler construction and callback interface
  - flush() to await pending observe() tasks
  - How the handler feeds real agent turns into imprint automatically
  - AnthropicAPITokenCounter as an optional swap-in for the token counter

Requirements:
  pip install imprint-mem[langchain]
  export ANTHROPIC_API_KEY=sk-ant-...

  # For AnthropicAPITokenCounter (optional, more precise budget enforcement):
  # pip install imprint-mem[anthropic]  (already included in [langchain] if using [all])

Usage:
  python examples/with_langchain.py
"""

import asyncio
from unittest.mock import MagicMock

from imprint import Imprint
from imprint.integrations.langchain import ImprintCallbackHandler


def _make_llm_response(text: str) -> MagicMock:
    """Build a minimal LLMResult mock as LangChain would produce."""
    gen = MagicMock()
    gen.text = text
    response = MagicMock()
    response.generations = [[gen]]
    return response


def _make_finish(output: str) -> MagicMock:
    """Build a minimal AgentFinish mock."""
    finish = MagicMock()
    finish.return_values = {"output": output}
    return finish


async def simulate_turn(
    handler: ImprintCallbackHandler,
    *,
    user_input: str,
    agent_response: str,
) -> None:
    """Simulate one LangChain agent turn by firing callbacks manually."""
    # LangChain fires these in order during a real chain run.
    handler.on_chain_start({}, {"input": user_input})
    handler.on_llm_end(_make_llm_response(agent_response))
    handler.on_agent_finish(_make_finish(agent_response))

    # flush() awaits the observe() tasks that on_agent_finish scheduled.
    await handler.flush()


async def main() -> None:
    imprint = Imprint(
        agent_id="langchain_agent",
        store=":memory:",
        processing_mode="frugal",
    )
    await imprint.connect()

    user_id = "langchain_user"

    handler = ImprintCallbackHandler(
        imprint=imprint,
        user_id=user_id,
        context="customer support",
    )

    print("=== LangChain Integration -- ImprintCallbackHandler ===\n")

    # ------------------------------------------------------------------
    # Simulate three agent turns. In a real deployment these would be
    # actual LangChain agent invocations -- no simulation needed.
    # ------------------------------------------------------------------

    turns = [
        (
            "Can you help me?",
            "Of course! What seems to be the problem? Would you like me to list the options?",
        ),
        (
            "Don't ask if I want a list. Just help me directly.",
            "Understood. Let me address your issue directly without asking for preferences first.",
        ),
        (
            "Yes, that's exactly how I want you to respond.",
            "Got it. I will always respond directly and concisely without preamble.",
        ),
    ]

    for i, (user_input, agent_response) in enumerate(turns, 1):
        print(f"[turn {i}]")
        print(f"  user: {user_input}")
        print(f"  agent: {agent_response[:60]}...")
        await simulate_turn(handler, user_input=user_input, agent_response=agent_response)

    # ------------------------------------------------------------------
    # Check what imprint learned from the simulated turns.
    # ------------------------------------------------------------------

    memories = await imprint.list_memories(user_id)
    policy = await imprint.get_policy(user_id=user_id)

    print("\n--- Results ---")
    print(f"Stored {len(memories)} memories via callback handler.")
    for m in memories:
        print(f"  [{m.type.value}] {m.content}")
    print(f"\nCompiled policy: {policy.text}")

    await imprint.close()


# ------------------------------------------------------------------
# Real LangChain integration (no simulation):
#
# from langchain_anthropic import ChatAnthropic
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
#
# chain = (
#     ChatPromptTemplate.from_template("{input}")
#     | ChatAnthropic(model="claude-haiku-4-5-20251001")
#     | StrOutputParser()
# )
#
# handler = ImprintCallbackHandler(imprint=imprint, user_id="user")
# chain_with_memory = chain.with_config(callbacks=[handler])
#
# result = await chain_with_memory.ainvoke({"input": user_message})
# await handler.flush()
# ------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(main())
