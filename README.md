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

Storage is SQLite (embedded, no setup)
or PostgreSQL (server deployments, pgvector). LLM calls go through pydantic-ai
so any provider works.

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
    # IMPRINT_AGENT_ID, IMPRINT_STORE, IMPRINT_MODEL from env
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

## Scope inference

When `get_policy()` is called with `context=` but without `scopes=`, imprint
infers which declared scopes are relevant automatically. In balanced mode it
uses embedding similarity between the context and scope names, falling back to
an LLM call when the signal is ambiguous or no embedder is configured. In eager
mode the LLM always decides. Frugal mode skips inference and fetches all memories.

```python
# Explicit: tell imprint exactly which scope to use.
policy = await imprint.get_policy(user_id="rami", scopes=["project:alpha"])

# Inferred: imprint picks from the declared scope list based on context.
policy = await imprint.get_policy(
    user_id="rami",
    context="reviewing the pull request for the checkout flow",
)
```

## Dynamic scope creation

With `dynamic_scopes=True`, imprint can create new scope names on the fly
during derivation. The LLM proposes a scope, imprint deduplicates it against
existing scopes (near-duplicates within edit distance 2 are collapsed), then
registers it in a dedicated `scopes` table that persists across reconnects.

```python
imprint = Imprint(
    agent_id="coding_assistant",
    dynamic_scopes=True,
    processing_mode="balanced",
    scope_consolidation_threshold=5,  # auto-consolidate every N memories
)
await imprint.connect()
# No scopes declared -- none needed.

await imprint.observe(
    user_id="dev",
    agent_output="def process(items): ...",
    user_response="In Python, always add type hints to function parameters.",
)
# imprint creates scope "python" and registers it.

await imprint.observe(
    user_id="dev",
    agent_output="function getUser(id) { return users[id] }",
    user_response="TypeScript functions must always have explicit return types.",
)
# imprint creates scope "typescript" and registers it.

print(imprint.scopes)  # -> ['python', 'typescript']
```

Scope names must be short (a couple of words), lowercase, no spaces. The LLM
proposes whatever fits the context naturally. Near-duplicates are collapsed to
the existing name rather than creating a new one.

## Scope consolidation

`consolidate_scopes()` reorganizes the scope vocabulary by asking the LLM
whether any scopes should be merged, renamed, or split. It runs automatically
in the background when memory count crosses `scope_consolidation_threshold`,
and can be force-triggered at any time:

```python
# Force consolidation -- useful after a seeding phase.
await imprint.consolidate_scopes(user_id="rami")
print(imprint.scopes)  # scopes may have been renamed, merged, or split
```

The LLM sees each scope name, its memory count, and sample memory contents.
It can merge overlapping scopes, rename vague ones, or split a scope that
contains clearly distinct topics by reassigning memories individually.
No-op in frugal mode. No-op when fewer than two scopes exist.

Enable dynamic scopes via environment variable:

```sh
IMPRINT_DYNAMIC_SCOPES=true python your_agent.py
```

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

# Hard delete all memories and events for a user. Irreversible.
await imprint.forget("rami")

# Prune decayed memories and run scope consolidation.
pruned = await imprint.consolidate("rami", prune_threshold=0.5)

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

`OpenAIEmbedder` is also available from `imprint-mem[openai]`:

```python
from imprint import OpenAIEmbedder

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
from imprint import AnthropicAPITokenCounter
imprint = Imprint(..., token_counter=AnthropicAPITokenCounter())

# Local tiktoken counting for OpenAI models (imprint-mem[openai], no API call).
from imprint import OpenAITokenCounter
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

## PostgreSQL storage

Use Postgres for multi-instance server deployments. Requires `imprint-mem[postgres]`
and a Postgres instance with the pgvector extension (`pgvector/pgvector:pg16`
Docker image ships with it pre-installed).

```python
from imprint import Imprint, PostgresMemoryStore

imprint = Imprint(
    agent_id="assistant",
    store=PostgresMemoryStore("postgres://user:pass@host/dbname"),
)
await imprint.connect()
```

`PostgresMemoryStore` uses asyncpg with a connection pool. FTS is backed by a
`TSVECTOR` generated column with a partial GIN index over active memories.

For dense retrieval, pair it with `PostgresVectorStore` (same connection pool,
separate `memory_vectors` table with an HNSW index):

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

For local development:

```sh
just postgres-dev    # starts pgvector/pgvector:pg16 on :5432
IMPRINT_POSTGRES_URL=postgres://imprint:imprint@localhost/imprint_test python examples/with_postgres.py
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
feature range, from the bare minimum to online learning and framework
integrations. Each example is self-contained and includes setup instructions
in its module docstring. See `examples/README.md` for an overview table,
required extras, and API keys per example.

## Layout

```
src/imprint/
  __init__.py           public API surface
  types.py              Memory, Signal, MemoryEvent, MemoryLineage, MemoryHealth
  protocols.py          adapter protocols (MemoryStore, Embedder, Compiler, ...)

  _core.py              Imprint, LLMCompiler, MemoryLoop, Policy
  _detect.py            heuristic signal detection
  _feedback.py          loop finalization, attribution, bandit updates
  _observe.py           observe path: detect -> derive -> persist -> consolidate
  _policy.py            get_policy: scope inference, hybrid retrieval, compile, cache
  _scope.py             scope management: accept, register, consolidate, infer
  _utils.py             pure utilities: URL parsing, cosine, cache key, IDs

  budget.py             HeuristicTokenCounter, truncate_to_budget
  decay.py              FSRSStaticDecay
  online.py             FSRSGradientDecay (imprint-mem[online])
  client.py             ImprintClient, AgentClient, Session (imprint-mem[client])
  retrieval.py          StaticAlphaTuner, BanditAlphaTuner, RRF fusion

  stores/
    sqlite.py           SQLiteMemoryStore, SQLiteEventLogger
    postgres.py         PostgresMemoryStore, PostgresVectorStore (asyncpg)
    vector.py           SQLiteVecStore (imprint-mem[vector])

  providers/
    anthropic.py        AnthropicAPITokenCounter (imprint-mem[anthropic])
    openai.py           OpenAIEmbedder, OpenAITokenCounter (imprint-mem[openai])
    voyage.py           VoyageEmbedder, VoyageTokenCounter (imprint-mem[voyage])

  integrations/
    langchain.py        ImprintCallbackHandler (imprint-mem[langchain])
    llamaindex.py       ImprintEventHandler (imprint-mem[llamaindex])
    tools.py            make_pydantic_ai_tools, make_anthropic_tools

  prompts/              one module per LLM-call prompt (system prompt + output model)
```

## imprint-server

Run imprint-mem as a networked service with a REST API and MCP SSE endpoint.

```sh
pip install imprint-server
```

```sh
# SQLite (local, zero infrastructure)
imprint-server serve

# Postgres (production)
IMPRINT_STORE=postgres://user:pass@host/db imprint-server serve

# Docker Compose (Postgres + server, one command)
docker compose -f imprint-server/docker-compose.yml up
```

Connect Claude Code or Cursor via MCP:

```sh
IMPRINT_MCP_AGENT_ID=my-agent IMPRINT_MCP_USER_ID=me imprint-server serve
# then add http://localhost:8000/mcp/sse as an MCP server
```

Use the typed Python client from any agent:

```python
from imprint.client import ImprintClient

async with ImprintClient("http://localhost:8000") as client:
    policy = await client.get_policy("my-agent", "user-1")
    await client.observe("my-agent", "user-1",
        agent_output="Here is a bullet list.",
        user_response="No bullet points please.")

    # Search memories by semantic similarity (falls back to list order without embedder):
    memories = await client.search_memories("my-agent", "user-1", "formatting style")

    # Paginate when the list is large:
    page = await client.paginate_memories("my-agent", "user-1", limit=50)
    while page.has_more:
        page = await client.paginate_memories("my-agent", "user-1",
            limit=50, cursor=page.next_cursor)

    # Signal a correction and store it as a memory:
    await client.correct("my-agent", "user-1", "No bullet points please.")

# Session-scoped usage (enables learning signal):
async with client.session("my-agent", "user-1", context="coding") as sess:
    policy = await sess.get_policy()
    await sess.observe("output", "response")
    sess.set_outcome(0.9)
```

See [imprint-server/README.md](imprint-server/README.md) for the full API reference,
auth setup, CLI commands, and Docker deployment guide.

For PydanticAI integration: `examples/with_pydantic_ai.py` shows `make_pydantic_ai_tools`
(single-process); `examples/with_server_and_pydantic_ai.py` shows manual tool wrapping
of `ImprintClient` for the multi-service pattern.

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```sh
just sync-all       # install all packages and extras into .venv
just check          # lint, format-check, typecheck, test (library)
just server-check   # lint, typecheck, test (imprint-server)
just test-all       # full suite: library + server + Postgres + Redis (requires Docker)
just live-all       # live tests against real APIs (requires API keys in .env)
just fmt            # auto-format
just test-live      # run library live tests (require API keys in env)
just run-examples   # run all examples and write output to examples/output/
just postgres-dev   # start local pgvector on :5432 via Docker
just clean          # remove caches and local SQLite databases
```

Copy `.env.example` to `.env` and fill in the relevant keys before running
live tests.

## API stability

The public API is shaped but not stable. Breaking changes between 0.x versions
should be expected. The `observe` / `get_policy` contract is the most stable
part. Adapter protocols and optional extra APIs may shift.

## License

[Apache 2.0](LICENSE).
