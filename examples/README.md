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
| with_turso.py | turso | ANTHROPIC | TursoMemoryStore, remote storage, multi-instance pattern |
| with_langchain.py | langchain | ANTHROPIC | ImprintCallbackHandler, LangChain integration |
| multi_session.py | none | ANTHROPIC | MemoryLoop lifecycle, persistence across sessions, stability from outcomes |

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

## with_turso.py

Requires `imprint-mem[turso]` and a running sqld instance. Shows that the
observe/get_policy API is identical to the SQLite examples -- only the store
constructor changes.

**Start a local sqld server via Docker:**

```sh
docker run --rm -p 8080:8080 ghcr.io/tursodatabase/libsql-server:latest
```

Then in another terminal:

```sh
pip install imprint-mem[turso]
export ANTHROPIC_API_KEY=sk-ant-...
export TURSO_DATABASE_URL=http://127.0.0.1:8080
python examples/with_turso.py
```

**Turso cloud instead of local sqld:**

```sh
export TURSO_DATABASE_URL=libsql://your-db.turso.io
export TURSO_AUTH_TOKEN=your-token
python examples/with_turso.py
```

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

## Extras reference

```sh
pip install imprint-mem[vector]      # SQLiteVecStore for dense retrieval
pip install imprint-mem[voyage]      # VoyageEmbedder, VoyageTokenCounter
pip install imprint-mem[anthropic]   # AnthropicAPITokenCounter
pip install imprint-mem[openai]      # OpenAIEmbedder, OpenAITokenCounter
pip install imprint-mem[online]      # FSRSGradientDecay via River
pip install imprint-mem[turso]       # TursoMemoryStore (httpx, hrana-over-HTTP)
pip install imprint-mem[langchain]   # ImprintCallbackHandler
pip install imprint-mem[llamaindex]  # ImprintEventHandler
pip install imprint-mem[all]         # everything above
```
