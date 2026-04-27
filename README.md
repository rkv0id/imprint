# Imprint

A Python library that gives AI agents memory.

Not a database of past conversations. A system that detects what matters in
interactions, distills it into typed memories (facts, rules, decisions, context),
and compiles a behavioral policy the agent injects into its prompt. The library is
self-contained: it talks directly to SQLite (or Turso for distributed deployments)
for storage, and directly to the configured LLM provider (Anthropic, OpenAI,
Ollama, ...) for the internal LLM calls it needs for signal detection, memory
derivation, and policy compilation.

Pre-implementation. Built incrementally.

## Layout

```
src/imprint/   # the library
tests/         # tests
docs/media/    # logo, social preview, diagrams
```

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```sh
just sync     # install dependencies into .venv
just check    # lint, format-check, typecheck, test
```

## License

Apache 2.0.
