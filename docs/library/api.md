# API reference

## Imprint

The main entry point. One instance per agent.

```python
from imprint import Imprint

imprint = Imprint(
    agent_id="assistant",           # required -- unique identifier for this agent
    agent_description="...",        # optional -- used by the LLM for context
    model="anthropic:...",          # pydantic-ai model string
    processing_mode="balanced",     # frugal | balanced | eager
    scopes=[],                      # candidate scope names
    dynamic_scopes=False,           # allow runtime scope creation
    store=None,                     # MemoryStore instance (default: SQLiteMemoryStore)
    vector_store=None,              # VectorStore instance (optional)
    embedder=None,                  # Embedder instance (optional)
    token_counter=None,             # TokenCounter instance (default: HeuristicTokenCounter)
    decay_model=None,               # DecayModel instance (default: FSRSStaticDecay)
    max_input_tokens=8000,          # token budget for memory retrieval
    max_output_tokens=3000,         # token budget for policy compilation
    feedback_timeout=3600,          # seconds before an unclosed loop expires
    scope_consolidation_threshold=20, # auto-consolidate after N memories
)
```

### Connection

```python
await imprint.connect()    # open the store connection and initialize schema
await imprint.close()      # flush pending tasks and close

# Context manager form (recommended)
async with Imprint(...) as imprint:
    ...

# Environment-based construction
imprint = Imprint.from_env()  # reads IMPRINT_AGENT_ID, IMPRINT_STORE, IMPRINT_MODEL
```

### Core loop

```python
# Record an agent-user exchange. Stores a memory only when a signal is detected.
await imprint.observe(
    user_id="alice",
    agent_output="Here is a list.",   # the agent's last response
    user_response="Use prose please.", # the user's reply
    context=None,                     # optional scope context hint
    scope=None,                       # optional explicit scope
)

# Compile a behavioral policy for the user.
policy = await imprint.get_policy(
    user_id="alice",
    existing_instructions=None,       # prepend to the compiled policy
    context=None,                     # for scope inference and retrieval ranking
    scopes=None,                      # explicit scope filter
    max_input_tokens=8000,
    max_output_tokens=3000,
    loop=None,                        # MemoryLoop for tracking
)
print(policy.text)                    # inject into system prompt
print(policy.memories)                # list[Memory] -- memories used
```

### Explicit directions

```python
# Store behavioral preferences without the detect/derive stages.
await imprint.observe_directions(
    user_id="alice",
    directions=["Always write in prose.", "Keep responses brief."],
    context=None,
    scope=None,
)
```

### Memory management

```python
# List active memories.
memories = await imprint.list_memories(user_id, scopes=None)

# Semantic search (list order when no embedder).
results = await imprint.search_memories(user_id, query, limit=20)

# Pin a memory (never dropped by token budget truncation).
await imprint.pin_memory(memory_id)

# Soft-deactivate a memory (stays in store for lineage).
found = await imprint.deactivate_memory(user_id, memory_id)  # -> bool

# Hard delete all memories for a user.
await imprint.forget(user_id)

# Prune decayed memories + run scope consolidation.
pruned = await imprint.consolidate(user_id, prune_threshold=0.5)

# Correct a mistake -- stores a negative direction and applies a -1 learning signal.
await imprint.correct(user_id, content, loop=None)

# Temporal diff -- what changed in a user's memory between two timestamps.
diff = await imprint.diff_memories(user_id, since, until)
```

### Observability

```python
# Recent events for a user (newest first).
events = await imprint.list_events(user_id, memory_id=None, limit=50)

# Full creation and mutation history of one memory.
lineage = await imprint.memory_lineage(memory_id)

# Aggregate health statistics.
health = await imprint.memory_health(user_id)
print(health.total, health.active, health.by_scope, health.avg_recall_count)
```

### Memory loops

```python
# Open a loop to track retrieval and enable outcome signals.
loop = await imprint.open_loop(user_id, context=None)
policy = await imprint.get_policy(user_id, loop=loop)
loop.set_outcome(0.9)               # 0=correction, 0.5=neutral, 1=ideal
await imprint.finalize_loop(loop)

# Context manager form.
async with imprint.loop(user_id) as loop:
    policy = await imprint.get_policy(user_id, loop=loop)
    # set_outcome inside the block; finalize_loop runs on exit
```

---

## Memory

```python
from imprint import Memory

memory.id           # str -- "mem_<hash>"
memory.agent_id     # str
memory.user_id      # str
memory.type         # MemoryType -- RULE | PREFERENCE | FACT | DECISION | CONTEXT
memory.scope        # str -- e.g. "global", "project:alpha"
memory.content      # str -- canonical third-person phrasing
memory.source       # MemorySource -- DETECTED | DIRECTION | CORRECTION
memory.stability    # float 0-1
memory.recall_count # int
memory.pinned       # bool
memory.active       # bool
memory.valid_from   # datetime
memory.valid_until  # datetime | None
memory.created_at   # datetime
memory.updated_at   # datetime
```

---

## MemoryDiff

```python
from imprint import MemoryDiff

diff = await imprint.diff_memories(user_id, since, until)
diff.since          # datetime
diff.until          # datetime
diff.added          # list[Memory] -- created in window and currently active
diff.deactivated    # list[Memory] -- deactivated in window with no replacement
diff.superseded     # list[SupersededPair] -- (old, new) replacement pairs
diff.summary        # dict[str, int] -- {"added": N, "deactivated": N, "superseded": N}
```

---

## MemoryHealth

```python
health = await imprint.memory_health(user_id)
health.total            # int
health.active           # int
health.pinned           # int
health.by_scope         # dict[str, int]
health.by_type          # dict[str, int]
health.avg_recall_count # float
health.oldest_active    # datetime | None
health.newest_active    # datetime | None
```

---

## ImprintClient (imprint-mem[client])

HTTP client for imprint-server. Mirrors the library API.

```python
from imprint.client import ImprintClient

async with ImprintClient("http://localhost:8000", api_key="sk-imp-...") as client:
    policy = await client.get_policy("agent-id", "user-id")
    await client.observe("agent-id", "user-id",
        agent_output="...", user_response="...")
    await client.observe_directions("agent-id", "user-id",
        directions=["Always be concise."])
    memories = await client.list_memories("agent-id", "user-id")
    results = await client.search_memories("agent-id", "user-id", "query")
    page = await client.paginate_memories("agent-id", "user-id", limit=50)
    await client.correct("agent-id", "user-id", content="...")
    await client.pin_memory("agent-id", memory_id)
    await client.deactivate_memory("agent-id", "user-id", memory_id)
    await client.forget("agent-id", "user-id")

# Agent-scoped shorthand
agent = client.agent("agent-id")
policy = await agent.get_policy("user-id")
```

### Session-scoped client

```python
async with client.session("agent-id", "user-id", context="...") as sess:
    policy = await sess.get_policy()
    await sess.observe(agent_output="...", user_response="...")
    sess.set_outcome(0.9)
# finalize_loop runs on exit
```

---

## Tools

```python
from imprint import make_pydantic_ai_tools

tools = make_pydantic_ai_tools(imprint, user_id="alice", loop=loop)
# Returns list of pydantic-ai Tool objects:
# remember, recall, search, forget, correct, reinforce, signal_outcome
```

```python
from imprint import make_anthropic_tools

tool_defs, dispatch = make_anthropic_tools(imprint, user_id="alice", loop=loop)
```

---

## Storage backends

```python
from imprint import SQLiteMemoryStore, PostgresMemoryStore

# SQLite (default)
store = SQLiteMemoryStore("path/to/db.db")

# Postgres
store = PostgresMemoryStore(
    "postgres://user:pass@host/dbname",
    min_size=1,
    max_size=10,
)
```

## Vector stores

```python
from imprint import SQLiteVecStore, PostgresVectorStore

# SQLite-vec
vec = SQLiteVecStore(store.conn, dim=1024)

# pgvector
vec = PostgresVectorStore(store.pool, dim=1024)
```

## Embedders

```python
from imprint import VoyageEmbedder, OpenAIEmbedder

voyage = VoyageEmbedder(model="voyage-3", dim=1024)     # VOYAGE_API_KEY
openai = OpenAIEmbedder(model="text-embedding-3-small", dimensions=512)  # OPENAI_API_KEY
```

## Decay models

```python
from imprint import FSRSStaticDecay         # default
from imprint import FSRSGradientDecay       # imprint-mem[online] -- learns from feedback
```
