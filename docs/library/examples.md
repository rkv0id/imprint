# Examples

Eleven runnable examples in the `examples/` directory. Each is self-contained.

| Example | Extras | API Keys | Teaches |
|---|---|---|---|
| `minimal.py` | none | ANTHROPIC | core loop: observe, get_policy |
| `writing_assistant.py` | none | ANTHROPIC | multi-user, scopes, directions, consolidation |
| `with_retrieval.py` | vector, openai | ANTHROPIC + OPENAI | scope filtering, hybrid BM25 + dense retrieval |
| `retrieval_tuning.py` | vector, openai | ANTHROPIC + OPENAI | MemoryLoop, outcome signals, BanditAlphaTuner |
| `decay_and_reinforcement.py` | none | ANTHROPIC | stability, token budget, pinning, recall tracking |
| `online_learning.py` | online | ANTHROPIC | FSRSGradientDecay vs FSRSStaticDecay |
| `with_postgres.py` | postgres | ANTHROPIC | PostgresMemoryStore, pgvector |
| `with_langchain.py` | langchain | ANTHROPIC | ImprintCallbackHandler |
| `multi_session.py` | none | ANTHROPIC | MemoryLoop lifecycle, persistence across sessions |
| `dynamic_scopes.py` | none | ANTHROPIC | scope inference, dynamic creation, consolidation |
| `with_pydantic_ai.py` | none | ANTHROPIC | make_pydantic_ai_tools, single-process pattern |

See `examples/README.md` for setup instructions and per-example extras.

For examples using imprint-server (HTTP client, multi-service pattern), see
[imprint-server/examples/](../server/overview.md).

## Running examples

```sh
# Run all library examples (skips Postgres and server-requiring examples):
just run-examples

# Run a single example:
uv run python examples/minimal.py
```

## Minimal example

```python
import asyncio
from imprint import Imprint

async def main():
    async with Imprint(
        agent_id="assistant",
        model="anthropic:claude-haiku-4-5-20251001",
    ) as imprint:
        await imprint.observe(
            user_id="alice",
            agent_output="Here is a bullet list.",
            user_response="Please use prose.",
        )
        policy = await imprint.get_policy(user_id="alice")
        print(policy.text)

asyncio.run(main())
```

## Writing assistant (scopes, directions, consolidation)

```python
import asyncio
from imprint import Imprint

async def main():
    async with Imprint(
        agent_id="writing_assistant",
        model="anthropic:claude-haiku-4-5-20251001",
        scopes=["style", "tone", "format"],
    ) as imprint:
        # Explicit directions (no LLM cost for detection)
        await imprint.observe_directions(
            user_id="alice",
            directions=["Write in first person.", "Avoid passive voice."],
            scope="style",
        )

        # Observed signal from dialogue
        await imprint.observe(
            user_id="alice",
            agent_output="The document was written by the team.",
            user_response="Please write actively: 'The team wrote the document.'",
        )

        policy = await imprint.get_policy(
            user_id="alice",
            context="editing a business memo",
            scopes=["style"],
        )
        print(policy.text)

asyncio.run(main())
```

## Session with outcome signal (online learning)

```python
import asyncio
from imprint import Imprint

async def main():
    async with Imprint(
        agent_id="assistant",
        model="anthropic:claude-haiku-4-5-20251001",
    ) as imprint:
        # Record behavioral preferences
        await imprint.observe_directions(
            user_id="alice",
            directions=["Always cite your sources."],
        )

        # Open a loop -- tracks which memories were retrieved
        loop = await imprint.open_loop(user_id="alice", context="research")
        policy = await imprint.get_policy(user_id="alice", loop=loop)

        # ... agent responds ...

        # Signal outcome: 1.0 = ideal, 0.0 = correction needed
        loop.set_outcome(0.9)
        await imprint.finalize_loop(loop)

        # Stability of recalled memories increases after a positive outcome

asyncio.run(main())
```
