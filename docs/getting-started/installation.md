# Installation

## Requirements

- Python 3.12 or later
- An LLM API key (Anthropic recommended; any pydantic-ai provider works)

## imprint-mem (library)

```sh
pip install imprint-mem
```

### Optional extras

Install only what you need:

```sh
# HTTP client for imprint-server
pip install imprint-mem[client]

# Dense vector retrieval (SQLite-vec)
pip install imprint-mem[vector]

# Embedder providers
pip install imprint-mem[voyage]     # VoyageEmbedder (recommended)
pip install imprint-mem[openai]     # OpenAIEmbedder

# Exact token counting
pip install imprint-mem[anthropic]  # AnthropicAPITokenCounter
pip install imprint-mem[openai]     # OpenAITokenCounter

# Online decay learning via River
pip install imprint-mem[online]     # FSRSGradientDecay

# Postgres storage + pgvector
pip install imprint-mem[postgres]

# Framework integrations
pip install imprint-mem[langchain]
pip install imprint-mem[llamaindex]

# Everything
pip install imprint-mem[all]
```

## imprint-server (networked service)

```sh
pip install imprint-server
```

imprint-server includes imprint-mem automatically. The server binary exposes
`imprint-server serve`, `imprint-server migrate`, and `imprint-server keys`.

### Docker

The official image ships with all extras pre-installed:

```sh
docker pull ghcr.io/rkv0id/imprint-server:latest
```

Or use the included Compose stacks:

```sh
# Postgres + imprint-server
docker compose -f imprint-server/docker-compose.yml up

# Full production stack: Postgres + Redis + imprint-server
docker compose -f imprint-server/docker-compose.live.yml up --build --wait
```

## API keys

=== "Anthropic (default)"

    ```sh
    export ANTHROPIC_API_KEY=sk-ant-...
    ```

    The default model is `anthropic:claude-haiku-4-5-20251001`. Any
    pydantic-ai model string works as a drop-in replacement.

=== "OpenAI"

    ```sh
    export OPENAI_API_KEY=sk-...
    ```

    ```python
    imprint = Imprint(agent_id="assistant", model="openai:gpt-4o-mini")
    ```

=== "Other providers"

    Any provider that pydantic-ai supports works:

    ```python
    imprint = Imprint(agent_id="assistant", model="google:gemini-2.5-pro")
    imprint = Imprint(agent_id="assistant", model="ollama:llama3")
    ```

## Development setup

The full development environment requires [uv](https://docs.astral.sh/uv/)
and [just](https://github.com/casey/just).

```sh
git clone https://github.com/rkv0id/imprint
cd imprint
just sync-all     # install all packages and extras into .venv
just check        # lint + typecheck + test (library)
just server-check # lint + typecheck + test (imprint-server)
```

Copy `.env.example` to `.env` and fill in API keys before running live tests.
