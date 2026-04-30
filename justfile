set dotenv-load := true

default:
    @just --list

# Install all deps into .venv
sync:
    uv sync

# Install all extras into .venv (vector, voyage, online, anthropic-tokens)
sync-all:
    uv sync --all-extras

# Run tests
test *ARGS:
    uv run pytest {{ARGS}}

# Run live integration tests (require API keys in env)
test-live:
    uv run pytest -m live

# Lint
lint:
    uv run ruff check .

# Format
fmt:
    uv run ruff format .

# Format check (CI)
fmt-check:
    uv run ruff format --check .

# Type check
typecheck:
    uv run pyright

# Run all checks (mirrors CI)
check: lint fmt-check typecheck test

# Wipe the venv and re-sync
fresh:
    rm -rf .venv uv.lock
    uv sync

# Remove caches and local SQLite databases
clean:
    rm -rf .pytest_cache .ruff_cache .pyright_cache
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name '*.db' -not -path './.venv/*' -delete
    find . -type f -name '*.db-journal' -not -path './.venv/*' -delete
    find . -type f -name '*.db-wal' -not -path './.venv/*' -delete
    find . -type f -name '*.db-shm' -not -path './.venv/*' -delete
