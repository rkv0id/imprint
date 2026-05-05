<h1>
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/media/mark-dark.svg"><img src="docs/media/mark-light.svg" width="48" alt="" valign="middle" /></picture>&nbsp;imprint
</h1>

Detect, distill, compile. Memory for AI agents.

Most memory systems store what was said. Imprint learns what to do differently.
It watches interactions, extracts typed memories (FACT, RULE, DECISION, CONTEXT),
consolidates as new ones arrive, and compiles a behavioral policy the agent
injects into its system prompt. The policy is the output -- not a database
the agent queries.

```
observe() -> detect -> derive -> persist -> consolidate
get_policy() -> filter -> rank -> compile -> cache
```

Storage is SQLite (embedded, no setup) or Turso (remote, scales across
instances). LLM calls go through pydantic-ai so any provider works.

## Install

```sh
pip install imprint-mem
```

Optional extras:

```sh
pip install imprint-mem[vector]      # SQLiteVecStore for dense retrieval
pip install imprint-mem[voyage]      # VoyageEmbedder, VoyageTokenCounter
pip install imprint-mem[anthropic]   # AnthropicAPITokenCounter
pip install imprint-mem[openai]      # OpenAIEmbedder, OpenAITokenCounter
pip install imprint-mem[online]      # FSRSGradientDecay via River
pip install imprint-mem[turso]       # TursoMemoryStore (httpx, hrana-over-HTTP)
pip install imprint-mem[langchain]   # ImprintCallbackHandler for LangChain
pip install imprint-mem[llamaindex]  # ImprintEventHandler for LlamaIndex
pip install imprint-mem[all]         # everything above
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

# After each user turn, pass the agent's last output and the user's reply.
# Most responses carry no signal. Nothing is stored when detection finds nothing.
await imprint.observe(
    user_id="rami",
    agent_output="I suggest using bullet points here.",
    user_response="No, please write in paragraphs.",
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

await imprint.close()
```

Imprint can also be used as an async context manager and configured from
environment variables:

```python
async with Imprint.from_env() as imprint:
    # IMPRINT_AGENT_ID, IMPRINT_DATABASE_URL, IMPRINT_MODEL from env
    policy = await imprint.get_policy(user_id="rami")
```

Any provider string pydantic-ai supports works as `model`: `"openai:gpt-4o"`,
`"google:gemini-2.5-pro"`, `"ollama:llama3"`, etc.

## How it works

`observe()` runs four stages in sequence.

**Detection** decides whether the user's response carries a learnable signal.
Pattern heuristics fire first (zero LLM cost). In balanced mode the LLM runs
as fallback when heuristics are silent. In eager mode the LLM always runs.
Most observations stop at detection with nothing stored.

**Derivation** converts the signal into a typed memory: what type (FACT, RULE,
DECISION, CONTEXT), what content (canonical third-person phrasing), what scope.

**Persistence** writes the memory and signal to the store, keeps the FTS5 index
in sync, and embeds the memory if a vector store is configured.

**Consolidation** compares the new memory against existing ones and picks one
of four actions: merge if redundant, contradict if the old one is now wrong,
scope_override if the conflict is scope-specific, or distinct if unrelated.
Deactivated memories stay in the store for lineage tracking. Learning updates
(bandit alpha, gradient decay) run as non-blocking background tasks.

`get_policy()` lists active memories matching the requested scopes, hashes
inputs into a cache key, and returns a cached compile if available. Otherwise
the LLM compiles a behavioral policy and caches the result. The cache
invalidates whenever a new memory is written for that user. With a vector store
and embedder configured, retrieval switches to hybrid BM25 + dense search fused
via Reciprocal Rank Fusion.

## Processing modes

**frugal** uses pattern heuristics only. Zero LLM cost for observation. Misses
subtle signals -- complex preferences, implicit corrections, and nuanced
directives frequently go undetected. Use this for high-volume or cost-sensitive
deployments where recall matters less than cost.

**balanced** *(default)* uses heuristics first with LLM fallback when silent.
One LLM call per ambiguous observation. Good default for most agents.

**eager** always runs the LLM for detection, derivation, and validation. Highest
signal recall. Adds a validation pre-pass for `observe_directions()` and LLM
attribution for corrections.

## Explicit memory loops

The `MemoryLoop` model tracks a single agent turn end-to-end, carrying the
retrieved memories, the retrieval parameters, and any outcome signal:

```python
# Open a loop before the agent responds.
loop = await imprint.open_loop(user_id="rami", context="code review")

# Get the policy using the loop -- memories retrieved here are tracked.
policy = await imprint.get_policy(user_id="rami", loop=loop)

# Feed the loop into tools so the agent can signal its own outcome.
tools = make_pydantic_ai_tools(imprint, user_id="rami", loop=loop)

# After the turn, close the loop with an explicit outcome.
loop.set_outcome(0.8)
await imprint.finalize_loop(loop)
```

Or use the context manager form:

```python
async with imprint.loop(user_id="rami") as loop:
    policy = await imprint.get_policy(user_id="rami", loop=loop)
    # outcome is set inside the loop; finalize_loop runs on exit
```

Loops that are never closed expire after `feedback_timeout` seconds (default
3600) and are swept on the next `observe()` call.

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
fall back to `"global"`. The `"global"` scope is always included.

`get_policy(scopes=...)` filters which memories are visible. Pass `context=`
without `scopes=` to let imprint infer scope automatically.

When a consolidated memory conflicts with an existing one at a different scope,
the more specific scope wins at compile time. Both memories stay active.

## Injecting directives

`observe_directions()` persists explicit instructions without the detect stage.
Useful for onboarding flows, settings screens, or any surface where the user
configures agent behavior directly:

```python
await imprint.observe_directions(
    user_id="rami",
    directions=[
        "Always respond in English.",
        "Never use bullet points.",
        "Keep responses under 200 words.",
    ],
)
```

In eager mode a batched LLM validation pass filters out hedges and non-directives
before any memory is written.

## Tools interface

Expose imprint as callable tools so the agent can manage its own memory:

```python
from imprint import make_pydantic_ai_tools
from pydantic_ai import Agent

tools = make_pydantic_ai_tools(imprint, user_id="rami", loop=loop)
agent = Agent(model="anthropic:claude-haiku-4-5-20251001", tools=tools)
```

For Anthropic's messages API directly (requires `imprint-mem[anthropic]`):

```python
from imprint import make_anthropic_tools

tool_defs, dispatch = make_anthropic_tools(imprint, user_id="rami", loop=loop)
```

Seven tools are exposed: `remember`, `recall`, `search`, `forget`, `correct`,
`reinforce`, `signal_outcome`. The `signal_outcome` tool lets the agent close
the loop with an explicit quality score from within the conversation.

## Observability

Imprint logs every memory lifecycle event (derive, merge, contradict, recall)
and exposes three observability methods:

```python
# Recent events for a user (newest first).
events = await imprint.list_events("rami", limit=50)

# Full history of one memory: origin signal, supersession chain, events.
lineage = await imprint.memory_lineage(memory_id)

# Aggregate health statistics for a user's memory store.
health = await imprint.memory_health("rami")
print(health.total, health.active, health.by_scope, health.avg_recall_count)
```

## Memory management

```python
# List active memories.
memories = await imprint.list_memories("rami", scopes=["project:alpha"])

# Semantic search (falls back to list order without an embedder).
results = await imprint.search_memories("rami", "coding style preferences")

# Deactivate a specific memory (returns True if found and active).
found = await imprint.deactivate_memory("rami", memory_id)

# Pin a memory so it is never dropped by token budget truncation.
await imprint.pin_memory(memory_id)

# Await all pending background learning tasks (useful in tests).
await imprint.drain()
```

## Framework integrations

### LangChain (`imprint-mem[langchain]`)

```python
from imprint.integrations.langchain import ImprintCallbackHandler

handler = ImprintCallbackHandler(
    imprint=imprint,
    user_id="rami",
    loop=loop,       # optional MemoryLoop
    context="code",  # optional scope context
)

# Attach to any chain or agent.
chain = your_chain.with_config(callbacks=[handler])
await chain.ainvoke({"input": user_message})

# Flush pending observe() tasks after the turn.
await handler.flush()
```

`on_chain_start` captures the user input, `on_llm_end` captures the last LLM
generation, and `on_agent_finish` fires `observe()`. For exact turn-level
control, call `imprint.observe()` directly.

### LlamaIndex (`imprint-mem[llamaindex]`)

```python
from llama_index.core.instrumentation import get_dispatcher
from imprint.integrations.llamaindex import ImprintEventHandler

handler = ImprintEventHandler(imprint=imprint, user_id="rami")
get_dispatcher().add_event_handler(handler)

# Now any query engine call feeds into imprint automatically.
response = await query_engine.aquery("What changed in this PR?")
await handler.flush()
```

Event matching uses class name lookup rather than isinstance so the integration
stays stable across LlamaIndex version changes.

## Extras

### Vector retrieval (`imprint-mem[vector]` + embedder extra)

```python
from imprint import Imprint, SQLiteMemoryStore, SQLiteVecStore
from imprint.voyage import VoyageEmbedder

store = SQLiteMemoryStore("assistant.db")
await store.connect()

imprint = Imprint(
    agent_id="assistant",
    store=store,
    vector_store=SQLiteVecStore(store.conn, dim=1024),
    embedder=VoyageEmbedder(),      # reads VOYAGE_API_KEY from env
)
```

`OpenAIEmbedder` is also available from `imprint-mem[openai]`:

```python
from imprint.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=512)
```

With a vector store configured, `observe()` embeds each new memory and
`get_policy()` switches to hybrid BM25 + dense retrieval when `context` is
provided. A `BanditAlphaTuner` learns the optimal sparse/dense balance from
implicit feedback.

### Token counting

The default `HeuristicTokenCounter` uses tiktoken when installed (opportunistic),
falling back to ceil(chars / 4). For exact counts:

```python
# Exact counting via Anthropic count_tokens endpoint (imprint-mem[anthropic]).
from imprint.anthropic import AnthropicAPITokenCounter
imprint = Imprint(..., token_counter=AnthropicAPITokenCounter())

# Local tiktoken counting for OpenAI models (imprint-mem[openai], no API call).
from imprint.openai import OpenAITokenCounter
imprint = Imprint(..., token_counter=OpenAITokenCounter(model="gpt-4o"))
```

### Online decay (`imprint-mem[online]`)

```python
from imprint import FSRSGradientDecay

imprint = Imprint(agent_id="assistant", decay_model=FSRSGradientDecay())
```

Replaces the default static FSRS formula with a River online regression model
that learns per-agent decay parameters from feedback. State persists across
restarts.

## Turso storage

Use Turso or a local sqld instance instead of SQLite:

```python
from imprint import Imprint, TursoMemoryStore

store = TursoMemoryStore(
    "libsql://your-db.turso.io",
    auth_token="your-token",     # omit for local sqld without auth
)
imprint = Imprint(agent_id="assistant", store=store)
await imprint.connect()
```

`TursoMemoryStore` calls sqld's hrana-over-HTTP API using httpx. No Rust
extension, no cmake. Works on any Python version. URL schemes accepted:
`http://`, `https://`, `libsql://` (converted to https), `ws://`, `wss://`.

Requires `imprint-mem[turso]`. For local development:

```sh
just turso-dev                                         # starts sqld on :8080
TURSO_DATABASE_URL=http://127.0.0.1:8080 just test-live
```

## Environment variables

`Imprint.from_env()` reads configuration from the environment:

```
IMPRINT_AGENT_ID         required  agent identifier
IMPRINT_DATABASE_URL     optional  SQLite path or Turso URL (default: :memory:)
IMPRINT_MODEL            optional  model string (default: anthropic:claude-haiku-4-5-20251001)
IMPRINT_MODE             optional  frugal | balanced | eager (default: balanced)
ANTHROPIC_API_KEY        required  for the default Anthropic LLM pipeline
OPENAI_API_KEY           optional  for OpenAIEmbedder / OpenAITokenCounter
VOYAGE_API_KEY           optional  for VoyageEmbedder / VoyageTokenCounter
TURSO_DATABASE_URL       optional  for TursoMemoryStore live tests
TURSO_AUTH_TOKEN         optional  for Turso cloud authentication
```

## Examples

The `examples/` directory has eight runnable examples covering the full
feature range, from the bare minimum to online learning and framework
integrations. Each example is self-contained and includes setup instructions
in its module docstring. See `examples/README.md` for an overview table,
required extras, and API keys per example.

## Layout

```
src/imprint/
  _core.py              Imprint facade, MemoryLoop, pipeline logic
  store.py              SQLiteMemoryStore, event logging, FTS5
  turso.py              TursoMemoryStore (httpx, hrana-over-HTTP)
  tools.py              make_pydantic_ai_tools, make_anthropic_tools
  types.py              Memory, Signal, MemoryEvent, MemoryLineage, MemoryHealth
  protocols.py          adapter protocols (MemoryStore, Embedder, ...)
  budget.py             HeuristicTokenCounter, truncate_to_budget
  anthropic.py          AnthropicAPITokenCounter
  openai.py             OpenAIEmbedder, OpenAITokenCounter
  voyage.py             VoyageEmbedder, VoyageTokenCounter
  retrieval.py          StaticAlphaTuner, BanditAlphaTuner, RRF fusion
  decay.py              FSRSStaticDecay
  online.py             FSRSGradientDecay (imprint-mem[online])
  detect.py             heuristic signal detection
  vector.py             SQLiteVecStore (imprint-mem[vector])
  integrations/
    langchain.py        ImprintCallbackHandler (imprint-mem[langchain])
    llamaindex.py       ImprintEventHandler (imprint-mem[llamaindex])
  prompts/              one module per LLM-call prompt
```

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```sh
just sync         # install all extras into .venv
just check        # lint, format-check, typecheck, test
just fmt          # auto-format
just test-live    # run live tests (require API keys in env)
just turso-dev    # start local sqld on :8080 via Docker
just clean        # remove caches and local SQLite databases
```

Copy `.env.example` to `.env` and fill in the relevant keys before running
live tests.

## API stability

The public API is shaped but not stable. Breaking changes between 0.x versions
should be expected. The `observe` / `get_policy` contract is the most stable
part. Adapter protocols and optional extra APIs may shift.

## License

[Apache 2.0](LICENSE).
