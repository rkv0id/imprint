<h1>
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/media/mark-dark.svg"><img src="docs/media/mark-light.svg" width="48" alt="" valign="middle" /></picture>&nbsp;imprint
</h1>

Detect, distill, compile. Memory for AI agents.

Not a database of past conversations. A system that detects what matters in
interactions, distills it into typed memories (facts, rules, decisions, context),
consolidates redundant or contradicted memories as new ones arrive, and compiles
a behavioral policy the agent injects into its prompt. The library is
self-contained: SQLite for storage, pydantic-ai for LLM calls.

## Install

```sh
pip install imprint-mem
```

Optional extras:

```sh
pip install imprint-mem[vector]           # SQLiteVecStore for dense retrieval
pip install imprint-mem[voyage]           # VoyageEmbedder and VoyageTokenCounter
pip install imprint-mem[anthropic] # exact token counting via the Anthropic API
pip install imprint-mem[online]           # FSRSGradientDecay via River
pip install imprint-mem[all]              # everything above
```

## Quick example

```python
from imprint import Imprint

imprint = Imprint(
    agent_id="reviewer",
    agent_description="A code reviewer that suggests improvements to pull requests.",
    model="anthropic:claude-haiku-4-5-20251001",   # reads ANTHROPIC_API_KEY from env
    store="sqlite:///~/.imprint/reviewer.db",
    processing_mode="balanced",                    # frugal | balanced | eager
    scopes=["project:alpha", "role:reviewer"],     # optional: declared scope set
)
await imprint.connect()

# After each user turn, hand imprint the agent's last output and the user's reply.
# Most replies carry no signal and nothing is stored.
await imprint.observe(
    user_id="rami",
    agent_output="I suggest using bullet points here.",
    user_response="No, write in paragraphs.",
)

# Before each agent turn, compile a behavioral policy for this user.
# The output is a ready-to-inject text block, filtered to memories that
# match the requested scopes and deduplicated against existing instructions.
policy = await imprint.get_policy(
    user_id="rami",
    existing_instructions="You are a helpful code reviewer.",
    scopes=["project:alpha"],
    max_output_tokens=400,
)

print(policy.text)
# -> "Write feedback in paragraphs rather than bullet points."
```

Models use [pydantic-ai](https://ai.pydantic.dev) under the hood. Any provider
string pydantic-ai supports works: `"openai:gpt-4o"`, `"google:gemini-2.5-pro"`,
`"ollama:llama3"`, etc. For more control pass a `pydantic_ai.models.Model`
instance directly.

## How it works

`observe()` runs four stages in order:

1. **Detection** decides whether the user's response carries a signal worth
   capturing. Pattern heuristics fire first; an LLM call runs as fallback in
   balanced mode or always in eager mode. Most observations stop here at zero
   LLM cost.
2. **Derivation** converts the signal into a canonical typed memory: what type
   (FACT, RULE, DECISION, CONTEXT), what content, what scope.
3. **Persistence** writes the memory and its supporting signal to SQLite and
   keeps the full-text search index in sync.
4. **Consolidation** compares the new memory against existing ones and decides
   for each: merge (redundant), contradict (now wrong), or distinct (keep both).
   Memories marked merged or contradicted are deactivated.

`get_policy()` lists the active memories that match the requested scopes, hashes
the inputs into a cache key, and returns a cached compile if one is available.
Otherwise it asks the LLM to compile a behavioral policy and caches the result.
The cache invalidates whenever a new memory is written for that user.

When a vector store and embedder are configured, `get_policy()` switches to
hybrid retrieval: BM25 sparse search over the full-text index fused with dense
vector search via Reciprocal Rank Fusion. A `BanditAlphaTuner` learns the
optimal sparse/dense balance per agent from implicit feedback.

## Processing modes

- **frugal** - pattern heuristics only; zero LLM cost for observation. Misses
  subtle signals. Good for high-volume or cost-sensitive deployments.
- **balanced** *(default)* - heuristics first, LLM fallback when silent. One
  LLM call per ambiguous observation.
- **eager** - always uses the LLM for detection, derivation, and validation.
  Best signal recall, highest cost. Adds a pre-validation pass for
  `observe_directions()` and LLM-based correction attribution in the feedback loop.

## Scopes

Scopes let one Imprint instance hold context-specific memories without cross-
contamination. Declare the candidate set on construction:

```python
imprint = Imprint(
    agent_id="reviewer",
    scopes=["project:alpha", "project:beta", "role:reviewer"],
)
```

A memory is tagged with one scope at write time. The LLM picks from the declared
set during derivation, or the caller passes `scope=` to `observe()` explicitly.
Unknown scopes fall back to `"global"`. The `"global"` scope is always available
and matches every retrieval call.

`get_policy(scopes=...)` filters which memories are visible. A memory matches
when its scope is `"global"` or appears in the requested list. Omitting `scopes`
returns all memories for that user.

## Injecting directives directly

`observe_directions()` persists explicit instructions without the detect stage.
Useful for onboarding flows, settings screens, or any surface where the user
explicitly configures how the agent should behave.

```python
memories = await imprint.observe_directions(
    user_id="rami",
    directions=[
        "Always respond in English.",
        "Never use bullet points.",
        "Keep responses under 200 words.",
    ],
    source=MemorySource.USER_EDIT,
)
```

In eager mode a batched LLM validation pre-pass filters out hedges and
non-directives before any memory is written.

## Feedback loop

Imprint tracks an open feedback loop per user session. The loop opens when
`get_policy()` is called and closes on the next `observe()` or an explicit
`observe_feedback()` call.

```python
# Explicit quality signal from the application layer.
# outcome: -1.0 = clear failure, 0.0 = neutral, 1.0 = clear success.
await imprint.observe_feedback(user_id="rami", outcome=0.9)
```

Implicit signals are also extracted automatically: a CORRECTION closes the loop
with a negative reward, a REINFORCEMENT with a positive one. When an embedder is
configured, corrections trigger an embedding-based attribution pass that
identifies which ranked memory was most relevant to the correction and adjusts
the retrieval alpha accordingly. In eager mode an LLM call makes this attribution
more precise.

Loops expire lazily after `feedback_timeout` seconds (default: 1 hour).

## Extras

### Vector retrieval (`imprint-mem[vector]`)

```python
from imprint import Imprint, SQLiteVecStore
from imprint.voyage import VoyageEmbedder

imprint = Imprint(
    agent_id="assistant",
    vector_store=SQLiteVecStore("assistant.db"),
    embedder=VoyageEmbedder(),                 # reads VOYAGE_API_KEY from env
)
```

When a vector store and embedder are provided, `observe()` embeds each new
memory alongside it. `get_policy()` switches to hybrid BM25 + dense retrieval
when a `context` string is provided.

### Online decay (`imprint-mem[online]`)

```python
from imprint import Imprint, FSRSGradientDecay

imprint = Imprint(
    agent_id="assistant",
    decay_model=FSRSGradientDecay(),
)
```

`FSRSGradientDecay` replaces the default static decay formula with a River
online regression model that learns per-agent decay parameters from feedback
events. State is persisted to the agent database and survives restarts.

## Layout

```
src/imprint/
  _core.py               # Imprint facade, Policy, open loop tracking
  store.py               # SQLiteMemoryStore, event logging, FTS5 index
  types.py               # Memory, Signal, MemoryType, SignalType, enums
  protocols.py           # adapter protocols (10 interfaces)
  retrieval.py           # StaticAlphaTuner, BanditAlphaTuner, RRF fusion
  decay.py               # FSRSStaticDecay
  online.py              # FSRSGradientDecay (requires imprint-mem[online])
  detect.py              # heuristic signal detection
  budget.py              # HeuristicTokenCounter
  tokens.py              # AnthropicAPITokenCounter
  vector.py              # SQLiteVecStore (requires imprint-mem[vector])
  voyage.py              # VoyageEmbedder, VoyageTokenCounter (requires imprint-mem[voyage])
  prompts/               # one module per LLM-call prompt
tests/
  test_imprint.py        # unit tests + live-marked integration tests
  test_store.py          # SQLite store tests
  test_detect.py         # heuristic detection tests
  test_types.py          # type model tests
  test_ascii_audit.py    # encoding guard
```

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```sh
just sync         # install all extras into .venv
just check        # lint, format-check, typecheck, test
just fmt          # auto-format
just test-live    # run live tests (require ANTHROPIC_API_KEY)
just clean        # remove caches and local SQLite databases
```

Live tests are excluded from the default run. They hit the real Anthropic API
and need a key. Copy `.env.example` to `.env` and fill in the values.

## API stability

The public API is shaped but not stable. Breaking changes between 0.x versions
should be expected. The core observe/get_policy contract is the most stable
part; adapter protocols and optional extra APIs may shift.

## License

[Apache 2.0](LICENSE).
