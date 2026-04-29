# Imprint

A Python library that gives AI agents memory.

Not a database of past conversations. A system that detects what matters in
interactions, distills it into typed memories (facts, rules, decisions, context),
and compiles a behavioral policy the agent injects into its prompt. The library
is self-contained: it talks directly to SQLite for storage, and directly to the
configured LLM provider (Anthropic today; OpenAI, Ollama, others to follow) for
the internal LLM calls it needs for signal detection, memory derivation, and
policy compilation.

Early-stage. Built incrementally. The public API is shaped but not stable.

## Quick example

```python
from imprint import Imprint

imprint = Imprint(
    agent_id="reviewer",
    model="anthropic:claude-haiku-4-5-20251001",   # reads ANTHROPIC_API_KEY from env
    store="sqlite:///~/.imprint/imprint.db",
    detection_mode="balanced",                      # frugal | balanced | eager
)
await imprint.connect()

# After each user turn, hand Imprint the agent's last output and the user's reply.
# Most replies don't carry a signal and nothing is stored. The ones that do are
# classified and persisted.
await imprint.observe(
    user_id="rami",
    agent_output="I suggest using bullet points.",
    user_response="No, write in paragraphs.",
)

# Before each agent turn, ask Imprint to compile a behavioral policy for this user.
# The output is a ready-to-inject text block, deduplicated against your existing
# system prompt.
policy = await imprint.get_policy(
    user_id="rami",
    existing_instructions="You are a helpful code reviewer.",
    max_tokens=400,
)

print(policy.text)
# -> e.g. "Write feedback in paragraphs rather than bullet points."
```

Models use [pydantic-ai](https://ai.pydantic.dev) under the hood. Any provider
string pydantic-ai supports works (`"openai:gpt-5"`, `"google:gemini-2.5-pro"`,
`"ollama:llama3"`, etc.). For more control, pass a `pydantic_ai.models.Model`
instance directly.

## Detection modes

`observe()` runs a detector before storing anything. Three modes:

- **frugal** - pattern heuristics only; zero LLM cost. Misses subtle signals.
- **balanced** *(default)* - heuristics first, LLM fallback when silent. One LLM
  call per ambiguous observation.
- **eager** - always uses the LLM. Best recall, highest cost.

## Layout

```
src/imprint/             # the library
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
```

Live tests are excluded from the default test run and from CI. They hit the
real Anthropic API and need an API key. Copy `.env.example` to `.env` and fill
in the key - `just` loads it automatically.

## License

[Apache 2.0](LICENSE).
