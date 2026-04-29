# <picture><source media="(prefers-color-scheme: dark)" srcset="docs/media/mark-dark.svg"><img src="docs/media/mark-light.svg" alt="" height="40" align="left"></picture>&nbsp;imprint

A Python library that gives AI agents memory.

Not a database of past conversations. A system that detects what matters in
interactions, distills it into typed memories (facts, rules, decisions, context),
consolidates redundant or contradicted memories as new ones arrive, and compiles
a behavioral policy the agent injects into its prompt. The library is
self-contained: it talks directly to SQLite for storage and directly to the
configured LLM provider for the internal LLM calls it needs.

Early-stage. Built incrementally. The public API is shaped but not stable; see
the API stability note at the bottom.

## Quick example

```python
from imprint import Imprint

imprint = Imprint(
    agent_id="reviewer",
    agent_description="A code reviewer that suggests improvements to pull requests.",
    model="anthropic:claude-haiku-4-5-20251001",   # reads ANTHROPIC_API_KEY from env
    store="sqlite:///~/.imprint/imprint.db",
    detection_mode="balanced",                      # frugal | balanced | eager
    scopes=["project:imprint", "role:reviewer"],   # optional: declared scope set
)
await imprint.connect()

# After each user turn, hand Imprint the agent's last output and the user's reply.
# Most replies don't carry a signal and nothing is stored.
await imprint.observe(
    user_id="rami",
    agent_output="I suggest using bullet points.",
    user_response="No, write in paragraphs.",
)

# Before each agent turn, ask Imprint to compile a behavioral policy for this user.
# The output is a ready-to-inject text block, deduplicated against the existing
# system prompt and filtered to memories that match the requested scopes.
policy = await imprint.get_policy(
    user_id="rami",
    existing_instructions="You are a helpful code reviewer.",
    scopes=["project:imprint"],
    max_tokens=400,
)

print(policy.text)
# -> e.g. "Write feedback in paragraphs rather than bullet points."
```

Models use [pydantic-ai](https://ai.pydantic.dev) under the hood. Any provider
string pydantic-ai supports works (`"openai:gpt-5"`, `"google:gemini-2.5-pro"`,
`"ollama:llama3"`, etc.). For more control, pass a `pydantic_ai.models.Model`
instance directly.

## How it works

`observe()` runs four internal stages in order:

1. **Detection** decides whether the user's response carries a signal worth
   capturing. Heuristics first; LLM fallback in balanced mode; LLM-only in
   eager mode. Most observations stop here.
2. **Derivation** asks the LLM to convert the signal into a canonical memory:
   what type (FACT, RULE, DECISION, CONTEXT), what content, what scope.
3. **Persistence** writes the memory and its supporting signal to SQLite.
4. **Consolidation** asks the LLM to compare the new memory against existing
   ones and decide for each: merge (redundant), contradict (now wrong), or
   distinct (keep both). Memories the LLM marks merged or contradicted get
   deactivated.

`get_policy()` lists the active memories that match the requested scopes,
hashes the inputs into a cache key, and returns a cached compile if one is
available. Otherwise it asks the LLM to compile a behavioral policy and caches
the result. Cache invalidates whenever `observe()` writes a new memory.

## Detection modes

- **frugal** - pattern heuristics only; zero LLM cost. Misses subtle signals.
- **balanced** *(default)* - heuristics first, LLM fallback when silent. One
  LLM call per ambiguous observation.
- **eager** - always uses the LLM. Best recall, highest cost.

## Scopes

Scopes let one Imprint instance hold context-specific memories. Declare the
candidate set on construction:

```python
imprint = Imprint(
    agent_id="reviewer",
    scopes=["project:alpha", "project:beta", "role:reviewer"],
)
```

A memory is tagged with a scope at write time. The LLM picks one from the
declared set during derivation, or a caller can pass `scope=` to `observe()`
explicitly. Unknown scopes silently fall back to `"global"`. The `"global"`
scope is reserved and always available.

`get_policy(scopes=...)` filters retrieval. A memory matches when its scope is
`"global"` or appears in the requested set. Passing `scopes=[]` means "globals
only"; passing no `scopes` argument returns everything.

## Layout

```
src/imprint/
  _core.py               # Imprint facade, Policy dataclass
  store.py               # SQLite store
  types.py               # Memory, Signal, ContextStat, enums
  detect.py              # heuristic signal detection
  prompts/               # one module per LLM-call prompt
tests/                   # unit tests + live-marked integration tests
docs/media/              # logo, social preview, diagrams
```

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```sh
just sync         # install dependencies into .venv
just check        # lint, format-check, typecheck, test
just fmt          # auto-format
just test-live    # run live tests (require ANTHROPIC_API_KEY)
just clean        # remove caches and local SQLite databases
```

Live tests are excluded from the default test run and from CI. They hit the
real Anthropic API and need an API key. Copy `.env.example` to `.env` and fill
in the key; `just` loads it automatically.

## API stability

The public API is shaped but not stable. Breaking changes between 0.x versions
should be expected.

The most recent breaking change was the removal of `SignalType.IMPLICIT`, which
the LLM-based detector never produced reliably. If you have stored memories
that reference it, drop them before upgrading.

## License

[Apache 2.0](LICENSE).
