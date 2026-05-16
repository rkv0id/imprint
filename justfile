set dotenv-load := true

default:
    @just --list

# Install all deps into .venv
sync:
    uv sync

# Install deps + all extras
sync-all:
    uv sync --all-extras

# Run tests
test *ARGS:
    uv run pytest {{ARGS}}

# Run live integration tests (require API keys in env, not running servers)
# For server store tests use: just postgres-test
test-live:
    uv run pytest -m live --ignore=tests/test_postgres.py

# Start a local Postgres with pgvector via Docker (for Postgres live tests)
postgres-dev port="5432":
    docker run --rm \
        --name imprint-pg \
        -e POSTGRES_DB=imprint_test \
        -e POSTGRES_USER=imprint \
        -e POSTGRES_PASSWORD=imprint \
        -p {{port}}:5432 \
        pgvector/pgvector:pg16

# Run Postgres live tests against a local instance (start postgres-dev first)
postgres-test port="5432":
    IMPRINT_POSTGRES_URL=postgres://imprint:imprint@localhost:{{port}}/imprint_test \
        uv run --extra postgres pytest tests/test_postgres.py -m live -v

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

# Build source and wheel distributions
build:
    rm -rf dist/
    uv build

# Publish to PyPI (run `just build` first, reads UV_PUBLISH_TOKEN from .env)
publish:
    uv publish

# Tag, build, publish, and create a GitHub release in one shot.
# Usage: just release 0.4.2
#        just release 0.4.2 "Bug fixes and new PostgreSQL support."
release version notes="": build
    git tag v{{version}}
    git push origin v{{version}}
    uv publish
    gh release create v{{version}} dist/* \
        --title "v{{version}}" \
        --notes "{{notes}}"
