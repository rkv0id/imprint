# Examples

Ten runnable examples covering imprint's features progressively.
Start with `minimal.py` if you are new to the library.

## Overview

| Example | Extras | Default API Keys | Teaches |
|---|---|---|---|
| minimal.py | none | ANTHROPIC | core loop: observe, get_policy |
| writing_assistant.py | none | ANTHROPIC | multi-user, scopes, directions, consolidation, scope inference, observability |
| with_retrieval.py | vector, openai | ANTHROPIC + OPENAI | scope filtering, hybrid BM25 + dense retrieval |
| retrieval_tuning.py | vector, openai | ANTHROPIC + OPENAI | MemoryLoop, outcome signals, BanditAlphaTuner adaptation |
| decay_and_reinforcement.py | none | ANTHROPIC | stability, token budget, pinning, recall tracking |
| online_learning.py | online | ANTHROPIC | FSRSGradientDecay vs FSRSStaticDecay, learned decay parameters |
| with_postgres.py | postgres | ANTHROPIC | PostgresMemoryStore, pgvector, shared storage for multi-instance deployments |
| with_langchain.py | langchain | ANTHROPIC | ImprintCallbackHandler, LangChain integration |
| multi_session.py | none | ANTHROPIC | MemoryLoop lifecycle, persistence across sessions, stability from outcomes |
| dynamic_scopes.py | none | ANTHROPIC | scope inference, dynamic scope creation, vocabulary consolidation |
| with_server_client.py | client | none (server in frugal mode) | ImprintClient, paginate_memories, search_memories, correct, reinforce, sessions |
| with_pydantic_ai.py | none | ANTHROPIC | make_pydantic_ai_tools, MemoryLoop, Tool definitions, single-process pattern |
| with_server_and_pydantic_ai.py | client | ANTHROPIC | manual Tool wrapping of ImprintClient, multi-service pattern, session lifecycle |
| with_production_server.py | client | none (server has API keys) | full stack: Voyage embedder, semantic search, gradient decay, Redis cache, pagination |

## Common setup

Clone the repo and install the base package:

```sh
git clone https://github.com/rkv0id/imprint
cd imprint
pip install -e .
```

Or install from PyPI:

```sh
pip install imprint-mem
```

All examples default to `anthropic:claude-haiku-4-5-20251001` as the LLM,
which requires an Anthropic API key. This is not a hard requirement -- imprint
uses pydantic-ai under the hood, so any supported provider works. Swap the
model string and the key requirement follows:

```sh
# Anthropic (default in examples)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-...
# then pass model="openai:gpt-4o-mini" to Imprint(...)

# Google
export GEMINI_API_KEY=...
# then pass model="google-gla:gemini-2.0-flash"

# Ollama (local, no key needed)
# pass model="ollama:llama3.2"
```

Run any example from the repo root:

```sh
python examples/minimal.py
```

---

## minimal.py

No extras required. Shows the core observe/get_policy loop using a code review
assistant that learns a developer's style preferences over three turns.

```sh
pip install imprint-mem
export ANTHROPIC_API_KEY=sk-ant-...
python examples/minimal.py
```

---

## writing_assistant.py

No extras required. A writing assistant serves two authors (alice and bob) from
the same Imprint instance. Shows multi-user isolation, scopes, observe_directions
for baseline setup, consolidation when a preference changes, and the observability
API (memory_health, list_events).

```sh
pip install imprint-mem
export ANTHROPIC_API_KEY=sk-ant-...
python examples/writing_assistant.py
```

---

## with_retrieval.py

Requires `imprint-mem[vector,openai]`. A research assistant has memories across
three scientific domains. Without context, get_policy returns a mix. With context,
hybrid BM25 + dense retrieval surfaces the relevant domain.

Voyage alternative: swap `OpenAIEmbedder` for `VoyageEmbedder` (see comment in
the file). Use `pip install imprint-mem[vector,voyage]` and `VOYAGE_API_KEY` instead.

```sh
pip install imprint-mem[vector,openai]
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
python examples/with_retrieval.py
```

---

## retrieval_tuning.py

Requires `imprint-mem[vector,openai]`. Runs 16 MemoryLoops against a support agent.
The first batch has strong positive outcomes; the second has weak outcomes. Prints
BanditAlphaTuner arm state after each batch to show how the sparse/dense retrieval
balance adapts.

```sh
pip install imprint-mem[vector,openai]
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
python examples/retrieval_tuning.py
```

---

## decay_and_reinforcement.py

No extras required. Seeds 16 memories into a personal assistant, pins one critical
memory, and forces a tight token budget. Shows which memories survive repeated
policy compilations and how recall_count accumulates for frequently-retrieved ones.

```sh
pip install imprint-mem
export ANTHROPIC_API_KEY=sk-ant-...
python examples/decay_and_reinforcement.py
```

---

## online_learning.py

Requires `imprint-mem[online]`. Runs 20 MemoryLoops with the same outcomes through
two Imprint instances -- one with FSRSStaticDecay (fixed constants) and one with
FSRSGradientDecay (learns from feedback). Prints effective_stability predictions at
three time horizons to show the divergence between the static and learned models.

```sh
pip install imprint-mem[online]
export ANTHROPIC_API_KEY=sk-ant-...
python examples/online_learning.py
```

---

## with_postgres.py

Requires `imprint-mem[postgres]` and a running Postgres instance. Shows that the
observe/get_policy API is identical to the SQLite examples -- only the store
constructor changes. Uses `pgvector/pgvector:pg16` which ships with the pgvector
extension pre-installed for dense retrieval.

**Start a local Postgres with pgvector via Docker:**

```sh
docker run --rm --name imprint-pg \
  -e POSTGRES_DB=imprint_test \
  -e POSTGRES_USER=imprint \
  -e POSTGRES_PASSWORD=imprint \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Or with just: `just postgres-dev`

Then in another terminal:

```sh
pip install imprint-mem[postgres]
export ANTHROPIC_API_KEY=sk-ant-...
export IMPRINT_POSTGRES_URL=postgres://imprint:imprint@localhost/imprint_test
python examples/with_postgres.py
```

The example includes a commented-out block showing how to add `PostgresVectorStore`
with `VoyageEmbedder` for hybrid retrieval.

---

## with_langchain.py

Requires `imprint-mem[langchain]`. Shows ImprintCallbackHandler attached to a
simulated LangChain chain. The simulation fires on_chain_start, on_llm_end, and
on_agent_finish manually so no additional LangChain packages (langchain_anthropic,
etc.) are required. A comment block at the bottom shows the real integration.

```sh
pip install imprint-mem[langchain]
export ANTHROPIC_API_KEY=sk-ant-...
python examples/with_langchain.py
```

**Real LangChain integration:**

```sh
pip install imprint-mem[langchain] langchain-anthropic
```

Then follow the comment block at the bottom of `with_langchain.py`.

LlamaIndex note: `imprint-mem[llamaindex]` provides `ImprintEventHandler` for
the LlamaIndex Instrumentation dispatcher. The pattern is similar to LangChain --
see `src/imprint/integrations/llamaindex.py` and the README for usage.

---

## dynamic_scopes.py

No extras required. A coding assistant that starts with zero declared scopes.
As the developer works in Python then TypeScript, imprint creates scope names
(python, typescript) from scratch. Shows scope inference routing each context
query to the right scope automatically, and scope consolidation reorganizing
the vocabulary when triggered.

```sh
pip install imprint-mem
export ANTHROPIC_API_KEY=sk-ant-...
python examples/dynamic_scopes.py
```

Note: requires balanced or eager mode. frugal mode always returns "global"
for scope because it uses heuristic derivation without an LLM call.

---

## multi_session.py

No extras required. A coding assistant across three separate connect/close
cycles simulating real production sessions. Session 1 learns preferences and
runs a positive MemoryLoop. Session 2 reconnects, finds memories on disk, adds
more and runs two more loops. Session 3 reconnects again, runs a negative
outcome loop showing stability decay for retrieved memories.

Teaches the full MemoryLoop lifecycle: open_loop -> get_policy -> set_outcome
-> finalize_loop, and how stability compounds from repeated positive outcomes
and decays from negative ones across independent sessions.

```sh
pip install imprint-mem
export ANTHROPIC_API_KEY=sk-ant-...
python examples/multi_session.py
```

---

## with_pydantic_ai.py

No extras required beyond PydanticAI. A personal assistant learns user preferences
across three conversations using `make_pydantic_ai_tools` -- the single-process
pattern where the agent and memory run in the same process. Shows the full tool
set: `recall`, `remember`, `search`, `correct`, `reinforce`, `signal_outcome`,
`forget`. The MemoryLoop is opened before each turn so the learning signal
is associated with the right retrieved memories.

```sh
pip install imprint-mem pydantic-ai-slim
export ANTHROPIC_API_KEY=sk-ant-...
python examples/with_pydantic_ai.py
```

---

## with_server_and_pydantic_ai.py

Requires `imprint-mem[client]` and a running imprint-server. The multi-service
pattern: the PydanticAI agent runs in one process and calls imprint-server over
HTTP for all memory operations. Tools are defined manually wrapping `ImprintClient`
rather than using `make_pydantic_ai_tools` (which takes a local `Imprint` instance).
This is the production architecture for separate agent and memory deployments.

**Start the server first:**

```sh
pip install imprint-server
IMPRINT_DEFAULT_MODE=balanced imprint-server serve
```

**Run the example:**

```sh
pip install imprint-mem[client] pydantic-ai-slim
export ANTHROPIC_API_KEY=sk-ant-...
python examples/with_server_and_pydantic_ai.py
```

---

## with_production_server.py

Requires `imprint-mem[client]` and the full production stack running via Docker.
Shows all v0.3.x features working together through the HTTP client: Voyage
semantic search returning vector-ranked results, LLM policy compilation in
balanced mode, Redis policy cache hit on the second identical request, gradient
decay learning from session outcomes, and cursor-based pagination.

**Start the full stack:**

```sh
docker compose -f imprint-server/docker-compose.live.yml up --build --wait
# or: just server-compose-live-test (auto teardown after tests)
```

**Run the example:**

```sh
pip install imprint-mem[client]
# API keys are passed to the server via docker-compose.live.yml, not the client.
python examples/with_production_server.py
```

---

## with_server_client.py

Requires `imprint-mem[client]` and a running imprint-server instance. A customer
support agent learns per-user communication preferences over several turns, using
the typed HTTP client rather than the library directly. Demonstrates
`ImprintClient`, `AgentClient`, `paginate_memories` (cursor-based pagination),
`search_memories`, `correct`, `reinforce`, and the session lifecycle. The server
runs in frugal mode so no LLM API key is needed.

**Start the server in one terminal:**

```sh
pip install imprint-server
IMPRINT_DEFAULT_MODE=frugal imprint-server serve
# or: just server-dev
```

**Run the example in another terminal:**

```sh
pip install imprint-mem[client]
python examples/with_server_client.py
```

---

## Extras reference

```sh
pip install imprint-mem[vector]      # SQLiteVecStore for dense retrieval
pip install imprint-mem[voyage]      # VoyageEmbedder, VoyageTokenCounter
pip install imprint-mem[anthropic]   # AnthropicAPITokenCounter
pip install imprint-mem[openai]      # OpenAIEmbedder, OpenAITokenCounter
pip install imprint-mem[online]      # FSRSGradientDecay via River
pip install imprint-mem[postgres]    # PostgresMemoryStore, PostgresVectorStore (asyncpg, pgvector)
pip install imprint-mem[langchain]   # ImprintCallbackHandler
pip install imprint-mem[llamaindex]  # ImprintEventHandler
pip install imprint-mem[all]         # everything above
```
