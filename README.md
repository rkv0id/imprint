<h1>
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/media/mark-dark.svg"><img src="docs/media/mark-light.svg" width="48" alt="" valign="middle" /></picture>&nbsp;imprint
</h1>

[![PyPI](https://img.shields.io/pypi/v/imprint-mem?color=0d9488&label=imprint-mem)](https://pypi.org/project/imprint-mem/)
[![License](https://img.shields.io/badge/license-Apache%202.0-0d9488)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/imprint-mem?color=0d9488)](https://pypi.org/project/imprint-mem/)

Detect, distill, compile. Memory for AI agents.

Most memory systems store what was said. Imprint learns what to do differently.
It watches interactions, extracts typed memories (FACT, RULE, DECISION, CONTEXT),
consolidates as new ones arrive, and compiles a behavioral policy the agent
injects into its system prompt. The policy is the output -- not a database
the agent queries.

```
observe() -> detect -> derive -> persist -> consolidate
get_policy() -> filter -> rank -> compile -> cache
session.outcome -> FSRS decay update -> BanditAlphaTuner feedback
```

Storage is SQLite (embedded, no setup) or PostgreSQL (server deployments,
pgvector). LLM calls go through pydantic-ai so any provider works.

## Install

```sh
pip install imprint-mem
```

Optional extras:

```sh
pip install imprint-mem[client]      # ImprintClient for imprint-server HTTP API
pip install imprint-mem[vector]      # SQLiteVecStore for dense retrieval
pip install imprint-mem[voyage]      # VoyageEmbedder, VoyageTokenCounter
pip install imprint-mem[anthropic]   # AnthropicAPITokenCounter
pip install imprint-mem[openai]      # OpenAIEmbedder, OpenAITokenCounter
pip install imprint-mem[online]      # FSRSGradientDecay via River
pip install imprint-mem[postgres]    # PostgresMemoryStore, PostgresVectorStore (asyncpg)
pip install imprint-mem[langchain]   # ImprintCallbackHandler for LangChain
pip install imprint-mem[llamaindex]  # ImprintEventHandler for LlamaIndex
pip install imprint-mem[all]         # everything above
```

## Quick start

```python
from imprint import Imprint

imprint = Imprint(
    agent_id="reviewer",
    model="anthropic:claude-haiku-4-5-20251001",  # reads ANTHROPIC_API_KEY from env
    processing_mode="balanced",                    # frugal | balanced | eager
    scopes=["project:alpha", "role:reviewer"],
)
await imprint.connect()

# After each user turn, pass the agent's last output and the user's reply.
# Most responses carry no signal. Nothing is stored when detection finds nothing.
await imprint.observe(
    user_id="alice",
    agent_output="I suggest using bullet points here.",
    user_response="No, please write in paragraphs.",
)

# Before each agent turn, compile a behavioral policy for this user.
# Returns a ready-to-inject text block. Cached until memories change.
policy = await imprint.get_policy(
    user_id="alice",
    existing_instructions="You are a helpful code reviewer.",
    context="Alice is reviewing a Python PR.",
    scopes=["project:alpha"],
)

print(policy.text)
# -> "Write feedback in paragraphs rather than bullet points."

await imprint.close()
```

## Online learning

Imprint improves retrieval quality over time from session outcome signals.

```python
# Open a loop before the agent responds.
loop = await imprint.open_loop(user_id="alice", context="code review")
policy = await imprint.get_policy(user_id="alice", loop=loop)

# ... agent generates response ...

# Signal the outcome: 0 = correction, 0.5 = neutral, 1 = ideal.
loop.set_outcome(0.9)
await imprint.finalize_loop(loop)
```

Two learning mechanisms run on every finalized loop:

**FSRS-inspired memory decay** -- memories recalled in high-outcome sessions
get a stability boost. Memories never recalled gradually decay. Consolidation
prunes low-stability memories automatically.

**BanditAlphaTuner** -- learns the optimal blend between BM25 keyword search
and dense vector retrieval from outcome signals. Adapts per agent without
manual configuration.

With `imprint-mem[online]`, `FSRSGradientDecay` replaces the static decay
formula with a River online regression model that learns per-agent parameters:

```python
from imprint import FSRSGradientDecay

imprint = Imprint(agent_id="assistant", decay_model=FSRSGradientDecay())
```

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
Deactivated memories stay in the store for lineage tracking.

`get_policy()` lists active memories matching the requested scopes, hashes
inputs into a cache key, and returns a cached compile if available. Otherwise
the LLM compiles a behavioral policy and caches the result. The cache
invalidates whenever a new memory is written for that user. With a vector store
and embedder configured, retrieval switches to hybrid BM25 + dense search fused
via Reciprocal Rank Fusion.

## Processing modes

**frugal** uses pattern heuristics only. Zero LLM cost for observation. Use
for high-volume or cost-sensitive deployments.

**balanced** *(default)* uses heuristics first with LLM fallback when silent.
One LLM call per ambiguous observation. Good default for most agents.

**eager** always runs the LLM for detection, derivation, and validation.
Highest signal recall.

## Memory loops

The `MemoryLoop` model tracks a single agent turn end-to-end:

```python
# Context manager form.
async with imprint.loop(user_id="alice") as loop:
    policy = await imprint.get_policy(user_id="alice", loop=loop)
    # set_outcome inside the block; finalize_loop runs on exit

# Or explicitly:
loop = await imprint.open_loop(user_id="alice", context="code review")
policy = await imprint.get_policy(user_id="alice", loop=loop)
loop.set_outcome(0.8)
await imprint.finalize_loop(loop)
```

## Scopes

Scopes let one Imprint instance hold context-specific memories without
cross-contamination. Declare the candidate set on construction:

```python
imprint = Imprint(
    agent_id="reviewer",
    scopes=["project:alpha", "project:beta", "role:reviewer"],
)
```

`get_policy(scopes=...)` filters which memories are visible. Pass `context=`
without `scopes=` to let imprint infer scope automatically.

With `dynamic_scopes=True`, imprint can create new scope names on the fly
during derivation and auto-consolidate the vocabulary at configurable intervals.

## Explicit directions

`observe_directions()` persists explicit instructions without the detect stage.
Useful for onboarding flows and settings screens:

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

## Memory management

```python
# List active memories.
memories = await imprint.list_memories("alice", scopes=["project:alpha"])

# Semantic search (falls back to list order without an embedder).
results = await imprint.search_memories("alice", "coding style preferences")

# Temporal diff -- what changed in a user's memory between two timestamps.
diff = await imprint.diff_memories("alice", since=t1, until=t2)
print(diff.summary)  # {"added": 3, "deactivated": 1, "superseded": 2}

# Deactivate a specific memory (soft delete, stays in store for lineage).
found = await imprint.deactivate_memory("alice", memory_id)

# Pin a memory so it is never dropped by token budget truncation.
await imprint.pin_memory(memory_id)

# Hard delete all memories and events for a user. Irreversible.
await imprint.forget("alice")

# Prune decayed memories and run scope consolidation.
pruned = await imprint.consolidate("alice", prune_threshold=0.5)
```

## Observability

```python
# Recent events for a user (newest first).
events = await imprint.list_events("alice", limit=50)

# Full history of one memory: origin signal, supersession chain, events.
lineage = await imprint.memory_lineage(memory_id)

# Aggregate health statistics for a user's memory store.
health = await imprint.memory_health("alice")
print(health.total, health.active, health.by_scope, health.avg_recall_count)
```

## Tools interface

```python
from imprint import make_pydantic_ai_tools
from pydantic_ai import Agent

tools = make_pydantic_ai_tools(imprint, user_id="alice", loop=loop)
agent = Agent(model="anthropic:claude-haiku-4-5-20251001", tools=tools)
```

Seven tools: `remember`, `recall`, `search`, `forget`, `correct`,
`reinforce`, `signal_outcome`.

## Framework integrations

### LangChain (`imprint-mem[langchain]`)

```python
from imprint.integrations.langchain import ImprintCallbackHandler

handler = ImprintCallbackHandler(imprint=imprint, user_id="alice", loop=loop)
chain = your_chain.with_config(callbacks=[handler])
await chain.ainvoke({"input": user_message})
await handler.flush()
```

### LlamaIndex (`imprint-mem[llamaindex]`)

```python
from llama_index.core.instrumentation import get_dispatcher
from imprint.integrations.llamaindex import ImprintEventHandler

handler = ImprintEventHandler(imprint=imprint, user_id="alice")
get_dispatcher().add_event_handler(handler)
```

## Extras

### Vector retrieval

```python
from imprint import Imprint, SQLiteMemoryStore, SQLiteVecStore, VoyageEmbedder

store = SQLiteMemoryStore("assistant.db")
await store.connect()

imprint = Imprint(
    agent_id="assistant",
    store=store,
    vector_store=SQLiteVecStore(store.conn, dim=1024),
    embedder=VoyageEmbedder(),      # reads VOYAGE_API_KEY from env
)
```

### PostgreSQL storage

```python
from imprint import Imprint, PostgresMemoryStore, PostgresVectorStore, VoyageEmbedder

url = "postgres://user:pass@host/dbname"
store = PostgresMemoryStore(url)
await store.connect()

imprint = Imprint(
    agent_id="assistant",
    store=store,
    vector_store=PostgresVectorStore(store.pool, dim=1024),
    embedder=VoyageEmbedder(),
)
```

## Environment variables

`Imprint.from_env()` reads configuration from the environment:

```
IMPRINT_AGENT_ID         required  agent identifier
IMPRINT_STORE            optional  SQLite path or Postgres URL (default: ~/.imprint/imprint.db)
IMPRINT_MODEL            optional  model string (default: anthropic:claude-haiku-4-5-20251001)
IMPRINT_MODE             optional  frugal | balanced | eager (default: balanced)
IMPRINT_DYNAMIC_SCOPES   optional  true | 1 | yes to enable dynamic scope creation
ANTHROPIC_API_KEY        required  for the default Anthropic LLM pipeline
OPENAI_API_KEY           optional  for OpenAIEmbedder / OpenAITokenCounter
VOYAGE_API_KEY           optional  for VoyageEmbedder / VoyageTokenCounter
IMPRINT_POSTGRES_URL     optional  for PostgresMemoryStore (postgres://user:pass@host/db)
```

## Examples

The `examples/` directory has thirteen runnable examples covering the full
feature range. See `examples/README.md` for an overview table, required extras,
and API keys per example.

```sh
just run-examples   # run all examples (skips Postgres and server-requiring ones)
```

## imprint-server

Run imprint-mem as a networked service with a REST API and MCP SSE endpoint.

```sh
pip install imprint-server
imprint-server serve
```

See [imprint-server/README.md](imprint-server/README.md) for the full API
reference, auth setup, CLI commands, admin dashboard, and Docker deployment guide.

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```sh
just sync-all       # install all packages and extras into .venv
just check          # lint, format-check, typecheck, test (library)
just server-check   # lint, typecheck, test (imprint-server)
just test-all       # full suite: library + server + Postgres + Redis (requires Docker)
just live-all       # live tests against real APIs (requires API keys in .env)
just docs-serve     # preview the documentation site locally
just demo           # start server + seed demo data + open admin dashboard
```

Copy `.env.example` to `.env` and fill in the relevant keys before running
live tests.

## API stability

The public API is shaped but not stable. Breaking changes between 0.x versions
should be expected. The `observe` / `get_policy` contract is the most stable
part.

## License

[Apache 2.0](LICENSE).
