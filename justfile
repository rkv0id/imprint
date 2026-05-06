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
# For server store tests use: just turso-test or just postgres-test
test-live:
    uv run pytest -m live --ignore=tests/test_turso.py --ignore=tests/test_postgres.py

# Start a local Turso/sqld server via Docker (for Turso live tests)
turso-dev port="8080":
    docker run --rm -p {{port}}:8080 ghcr.io/tursodatabase/libsql-server:latest

# Run Turso live tests against a local sqld instance (start turso-dev first)
turso-test port="8080":
    TURSO_DATABASE_URL=http://127.0.0.1:{{port}} \
        uv run --extra turso pytest tests/test_turso.py -m live -v

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
