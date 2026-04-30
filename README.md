<h1>
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/media/mark-dark.svg"><img src="docs/media/mark-light.svg" width="48" alt="" valign="middle" /></picture>&nbsp;imprint
</h1>

Detect, distill, compile. Memory for AI agents.

Not a database of past conversations. A system that detects what matters in
interactions, distills it into typed memories (facts, rules, decisions, context),
consolidates redundant or contradicted memories as new ones arrive, and compiles
a behavioral policy the agent injects into its prompt. Storage is SQLite or Turso.
LLM calls go through pydantic-ai so any provider works.

## Install

```sh
pip install imprint-mem
```

Optional extras:

```sh
pip install imprint-mem[vector]     # SQLiteVecStore for dense retrieval
pip install imprint-mem[voyage]     # VoyageEmbedder and VoyageTokenCounter
pip install imprint-mem[anthropic]  # exact token counting + Anthropic tool definitions
pip install imprint-mem[online]     # FSRSGradientDecay via River
pip install imprint-mem[turso]      # TursoMemoryStore for remote/cloud storage
pip install imprint-mem[all]        # everything above
```

## Quick start

```python
from imprint import Imprint

imprint = Imprint(
    agent_id="reviewer",
    agent_description="A code reviewer that suggests improvements to pull requests.",
    model="anthropic:claude-haiku-4-5-20251001",  # reads ANTHROPIC_API_KEY from env
    processing_mode="balanced",                    # frugal | balanced | eager
    scopes=["project:alpha", "role:reviewer"],
)
await imprint.connect()

# After each user turn, hand imprint the agent's last output and the user's reply.
# Most responses carry no signal. Nothing is stored if detection finds nothing.
await imprint.observe(
    user_id="rami",
    agent_output="I suggest using bullet points here.",
    user_response="No, write in paragraphs.",
)

# Before each agent turn, compile a behavioral policy for this user.
# Returns a ready-to-inject text block. Cached until memories change.
policy = await imprint.get_policy(
    user_id="rami",
    existing_instructions="You are a helpful code reviewer.",
    context="Rami is reviewing a Python PR.",
    scopes=["project:alpha"],
)

print(policy.text)
# -> "Write feedback in paragraphs rather than bullet points."
```

Any provider string pydantic-ai supports works as `model`: `"openai:gpt-4o"`,
`"google:gemini-2.5-pro"`, `"ollama:llama3"`, etc. Pass a `pydantic_ai.models.Model`
instance directly for more control.

## How it works

`observe()` runs four stages:

**Detection** decides whether the user's response carries a learnable signal.
Pattern heuristics fire first (zero LLM cost). In balanced mode the LLM runs as
fallback when heuristics are silent. In eager mode the LLM always runs. Most
observations stop at detection with nothing stored.

**Derivation** converts the signal into a typed memory: what type (FACT, RULE,
DECISION, CONTEXT), what content (canonical third-person phrasing), what scope.

**Persistence** writes the memory and supporting signal to the store, keeps the
FTS5 index in sync, and embeds the memory if a vector store is configured.

**Consolidation** compares the new memory against existing ones -- merge if
redundant, contradict if the old one is now wrong, distinct if unrelated.
Merged and contradicted memories are deactivated. Learning updates (bandit,
gradient decay) run as non-blocking background tasks so `observe()` returns
as soon as persistence is done.

`get_policy()` lists active memories matching the requested scopes, hashes
inputs into a cache key, and returns a cached compile if available. Otherwise
the LLM compiles a behavioral policy and the result is cached. The cache
invalidates whenever a new memory is written for that user. When a vector store
and embedder are configured, `get_policy()` switches to hybrid retrieval: BM25
sparse search fused with dense vector search via Reciprocal Rank Fusion.

## Processing modes

**frugal** uses pattern heuristics only. Zero LLM cost for observation. Misses
subtle signals. Right for high-volume or cost-sensitive deployments.

**balanced** *(default)* uses heuristics first with LLM fallback when silent.
One LLM call per ambiguous observation. Good default for most agents.

**eager** always uses the LLM for detection, derivation, and validation. Highest
signal recall. Adds a validation pre-pass for `observe_directions()` and LLM
attribution for corrections in the feedback loop.

## Scopes

Scopes let one Imprint instance hold context-specific memories without
cross-contamination. Declare the candidate set on construction:

```python
imprint = Imprint(
    agent_id="reviewer",
    scopes=["project:alpha", "project:beta", "role:reviewer"],
)
```

A memory is tagged with one scope at write time. The LLM picks from the declared
set during derivation, or the caller passes `scope=` explicitly. Unknown scopes
fall back to `"global"`. The `"global"` scope is always available.

`get_policy(scopes=...)` filters which memories are visible. Pass `context=`
without `scopes=` to let imprint infer which scopes are relevant automatically:
in balanced mode it uses embedding similarity with LLM fallback, in eager mode
it uses the LLM directly.

## Injecting directives

`observe_directions()` persists explicit instructions without the detect stage.
Useful for onboarding flows, settings screens, or any surface where the user
explicitly configures how the agent should behave:

```python
memories = await imprint.observe_directions(
    user_id="rami",
    directions=[
        "Always respond in English.",
        "Never use bullet points.",
        "Keep responses under 200 words.",
    ],
)
```

In eager mode a batched LLM validation pass filters out hedges and
non-directives before any memory is written.

## Feedback loop

Imprint tracks an open feedback loop per user session. The loop opens when
`get_policy()` is called and closes on the next `observe()` or an explicit signal:

```python
# Explicit quality signal from the application layer.
# outcome: -1.0 = clear failure, 0.0 = neutral, 1.0 = clear success.
await imprint.observe_feedback(user_id="rami", outcome=0.9, session_id="s1")

# Or close the loop directly:
await imprint.close_loop(user_id="rami", outcome=0.9, session_id="s1")
```

Implicit signals are also extracted automatically: a CORRECTION closes the loop
with a negative reward, a REINFORCEMENT with a positive one. When an embedder is
configured, corrections trigger an embedding-based attribution pass that
identifies which retrieved memory was most responsible and adjusts the
sparse/dense retrieval balance accordingly.

Loops expire after `feedback_timeout` seconds (default: 3600).

## Tools interface

Expose imprint as callable tools so the LLM can manage its own memory:

```python
from imprint import make_pydantic_ai_tools
from pydantic_ai import Agent

tools = make_pydantic_ai_tools(imprint, user_id="rami", session_id="s1")
agent = Agent(model="anthropic:claude-haiku-4-5-20251001", tools=tools)
```

For Anthropic's messages API directly (requires `imprint-mem[anthropic]`):

```python
from imprint import make_anthropic_tools
import anthropic

tool_defs, dispatch = make_anthropic_tools(imprint, user_id="rami")
client = anthropic.Anthropic()
response = client.messages.create(tools=tool_defs, ...)
for block in response.content:
    if block.type == "tool_use":
        result = await dispatch(block.name, block.input)
```

Six tools are exposed: `remember`, `recall`, `search`, `forget`, `correct`,
`reinforce`.

## Memory management

```python
# List active memories for a user
memories = await imprint.list_memories("rami", scopes=["project:alpha"])

# Search by semantic query (falls back to list order without embedder)
results = await imprint.search_memories("rami", "coding style preferences")

# Deactivate a specific memory (returns True if found)
found = await imprint.deactivate_memory("rami", memory_id)

# Pin a memory so it is never dropped by token budget truncation
await imprint.pin_memory(memory_id)

# Wait for all pending background learning tasks (useful in tests)
await imprint.drain()
```

## Turso storage

Use Turso or a local sqld instance instead of SQLite:

```python
imprint = Imprint(
    agent_id="assistant",
    store="libsql://your-db.turso.io?auth_token=your-token",
)
```

`TursoMemoryStore` is selected automatically for `libsql://`, `turso://`,
`ws://`, `wss://`, `http://`, and `https://` URLs. Requires `imprint-mem[turso]`.

For local development, run sqld via Docker:

```sh
just turso-dev                                         # starts sqld on :8080
TURSO_DATABASE_URL=http://127.0.0.1:8080 just test-live
```

## Extras

### Vector retrieval (`imprint-mem[vector]` + `imprint-mem[voyage]`)

```python
from imprint import Imprint, SQLiteVecStore, SQLiteMemoryStore
from imprint.voyage import VoyageEmbedder

store = SQLiteMemoryStore("assistant.db")
await store.connect()

imprint = Imprint(
    agent_id="assistant",
    store=store,
    vector_store=SQLiteVecStore(store.conn, dim=1024),
    embedder=VoyageEmbedder(),   # reads VOYAGE_API_KEY from env
)
```

With a vector store configured, `observe()` embeds each new memory and
`get_policy()` switches to hybrid BM25 + dense retrieval when `context` is
provided. A `BanditAlphaTuner` learns the optimal sparse/dense balance from
implicit feedback.

### Online decay (`imprint-mem[online]`)

```python
from imprint import Imprint, FSRSGradientDecay

imprint = Imprint(
    agent_id="assistant",
    decay_model=FSRSGradientDecay(),
)
```

Replaces the default static FSRS decay formula with a River online regression
model that learns per-agent decay parameters from feedback. State persists across
restarts.

### Exact token counting (`imprint-mem[anthropic]`)

```python
from imprint import Imprint, AnthropicAPITokenCounter

imprint = Imprint(
    agent_id="assistant",
    token_counter=AnthropicAPITokenCounter(),
)
```

Uses the Anthropic count_tokens endpoint for precise budget enforcement. The
default `HeuristicTokenCounter` (chars / 4) is sufficient for most cases.

## Layout

```
src/imprint/
  _core.py       Imprint facade, Policy, open loop tracking
  store.py       SQLiteMemoryStore, event logging, FTS5 index
  turso.py       TursoMemoryStore (requires imprint-mem[turso])
  tools.py       make_pydantic_ai_tools, make_anthropic_tools
  types.py       Memory, Signal, MemoryType, SignalType
  protocols.py   adapter protocols (10 interfaces)
  retrieval.py   StaticAlphaTuner, BanditAlphaTuner, RRF fusion
  decay.py       FSRSStaticDecay
  online.py      FSRSGradientDecay (requires imprint-mem[online])
  detect.py      heuristic signal detection
  budget.py      HeuristicTokenCounter
  tokens.py      AnthropicAPITokenCounter (requires imprint-mem[anthropic])
  vector.py      SQLiteVecStore (requires imprint-mem[vector])
  voyage.py      VoyageEmbedder, VoyageTokenCounter (requires imprint-mem[voyage])
  prompts/       one module per LLM-call prompt
```

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```sh
just sync         # install all extras into .venv
just check        # lint, format-check, typecheck, test
just fmt          # auto-format
just test-live    # run live tests (require ANTHROPIC_API_KEY)
just turso-dev    # start local sqld on :8080 via Docker
just clean        # remove caches and local SQLite databases
```

Copy `.env.example` to `.env` and fill in the values before running live tests.

## API stability

The public API is shaped but not stable. Breaking changes between 0.x versions
should be expected. The `observe` / `get_policy` contract is the most stable
part. Adapter protocols and optional extra APIs may shift.

## License

[Apache 2.0](LICENSE).
