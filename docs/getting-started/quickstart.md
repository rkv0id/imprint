# Quick start

## Five-minute example

```python
import asyncio
from imprint import Imprint

async def main():
    imprint = Imprint(
        agent_id="assistant",
        model="anthropic:claude-haiku-4-5-20251001",  # reads ANTHROPIC_API_KEY
        processing_mode="balanced",                    # frugal | balanced | eager
    )
    await imprint.connect()

    # After each agent turn, record what was said.
    # Most interactions carry no learnable signal -- nothing is stored.
    await imprint.observe(
        user_id="alice",
        agent_output="Here is a summary with bullet points.",
        user_response="Please write in prose, not bullets.",
    )

    # Before each agent turn, compile a policy for this user.
    # Returns text ready to inject into the system prompt.
    policy = await imprint.get_policy(user_id="alice")
    print(policy.text)
    # -> "Write responses in prose rather than bullet points."

    await imprint.close()

asyncio.run(main())
```

That's the core loop. `observe()` stores nothing most of the time -- it detects
a signal only when the user's response carries a learnable behavioral preference.
`get_policy()` compiles all stored memories into a concise instruction block.

## Using context managers

```python
async with Imprint(agent_id="assistant", model="...") as imprint:
    await imprint.observe(user_id="alice", ...)
    policy = await imprint.get_policy(user_id="alice")
```

## Environment-based configuration

```python
# Reads IMPRINT_AGENT_ID, IMPRINT_STORE, IMPRINT_MODEL from env
async with Imprint.from_env() as imprint:
    policy = await imprint.get_policy(user_id="alice")
```

## Injecting the policy

The policy text is designed to be injected directly into the system prompt:

```python
system_prompt = (
    "You are a helpful assistant.\n\n"
    + policy.text
)
```

In PydanticAI:

```python
from pydantic_ai import Agent

agent = Agent(
    model="anthropic:claude-haiku-4-5-20251001",
    system_prompt=lambda: system_prompt,
)
```

## Explicit directions

Skip signal detection and store behavioral preferences directly -- useful for
settings screens and onboarding flows:

```python
await imprint.observe_directions(
    user_id="alice",
    directions=[
        "Always respond in English.",
        "Never use bullet points.",
        "Keep responses under 200 words.",
    ],
)
```

## Frugal mode (no LLM cost for observation)

```python
imprint = Imprint(
    agent_id="assistant",
    processing_mode="frugal",  # heuristics only
)
```

In frugal mode, `observe()` uses pattern heuristics and never calls the LLM.
`get_policy()` still calls the LLM to compile the policy unless all memories
fit within `existing_instructions` directly.

!!! tip "Frugal + directions"
    For cost-sensitive deployments, use `frugal` mode with `observe_directions()`
    for explicit preferences and rely on `observe()` only for implicit signal.
    This gives you zero LLM cost for observation while still capturing explicitly
    stated preferences.

## Next steps

- [Concepts](../library/concepts.md) -- understand detection, derivation, consolidation, and decay
- [API reference](../library/api.md) -- every public method documented
- [Examples](../library/examples.md) -- runnable examples for every feature
- [imprint-server](../server/overview.md) -- run imprint as a networked service
