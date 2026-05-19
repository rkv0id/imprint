set dotenv-load := true

default:
    @just --list

# Install all deps into .venv
sync:
    uv sync

# Install deps + all extras for all workspace packages
sync-all:
    uv sync --all-packages --all-extras

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
postgres-test port="5432" db="imprint_test":
    IMPRINT_POSTGRES_URL=postgres://imprint:imprint@localhost:{{port}}/{{db}} \
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

# Run the full test suite across all packages.
# Suites run independently -- a failure in one does not skip the rest.
# Exits non-zero if any suite failed. Pipe to tee for a log file:
#   just test-all 2>&1 | tee ~/imprint-test-$(date +%Y%m%d-%H%M%S).log
#
# Prerequisites:
#   - Docker running (for server-integration-test)
#   - .env and imprint-server/.env with any required API keys
#
# To also run library Postgres tests against a local imprint_test database:
#   just postgres-test
test-all:
    #!/usr/bin/env bash

    # Load env files so API keys and IMPRINT_* vars reach all subprocesses.
    set -a
    [ -f .env ] && source .env
    [ -f imprint-server/.env ] && source imprint-server/.env
    set +a

    # Dependency sync must succeed before anything else runs.
    echo "========================================"
    echo "=== sync-all"
    echo "========================================"
    just sync-all || { echo "[FATAL] sync-all failed -- aborting"; exit 1; }

    # Run each suite and record its outcome.
    declare -a results
    declare -a failed

    _run() {
        local name="$1"; shift
        echo ""
        echo "========================================"
        echo "=== $name"
        echo "========================================"
        if "$@"; then
            results+=("[PASS] $name")
        else
            results+=("[FAIL] $name")
            failed+=("$name")
        fi
    }

    _run "library: lint + typecheck + tests"   just check
    _run "server:  lint + typecheck + tests"   just server-check
    _run "server:  postgres integration tests" just server-integration-test

    # Summary
    echo ""
    echo "========================================"
    echo "=== Summary"
    echo "========================================"
    for r in "${results[@]}"; do
        echo "  $r"
    done
    echo "========================================"

    if [ ${#failed[@]} -gt 0 ]; then
        echo "  FAILED suites: ${failed[*]}"
        exit 1
    else
        echo "  All suites passed."
        exit 0
    fi

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

# ---- imprint-server ---------------------------------------------------------

# Run all imprint-server checks (mirrors CI)
server-check: server-lint server-typecheck server-test

# Lint imprint-server
server-lint:
    cd imprint-server && uv run ruff check src tests
    cd imprint-server && uv run ruff format --check src tests

# Type-check imprint-server
server-typecheck:
    cd imprint-server && uv run pyright

# Run imprint-server unit tests
server-test *ARGS:
    cd imprint-server && uv run pytest {{ARGS}}

# Run imprint-server against a local SQLite store (auth disabled)
server-dev:
    cd imprint-server && \
        IMPRINT_STORE=sqlite:///~/.imprint/imprint-dev.db \
        IMPRINT_AUTH_DISABLED=true \
        uv run imprint-server serve

# Run imprint-server against a local Postgres instance (start postgres-dev first)
server-postgres-dev:
    cd imprint-server && \
        IMPRINT_STORE=postgres://imprint:imprint@localhost:5432/imprint_test \
        IMPRINT_AUTH_DISABLED=true \
        uv run imprint-server serve

# Run imprint-server Postgres integration tests (start postgres-dev first)
server-postgres-test port="5432":
    IMPRINT_STORE=postgres://imprint:imprint@localhost:{{port}}/imprint_test \
        cd imprint-server && uv run --extra postgres pytest -m postgres -v

# Build imprint-server wheel
server-build:
    cd imprint-server && rm -rf dist/ && uv build

# Publish imprint-server to PyPI (run server-build first)
server-publish:
    cd imprint-server && uv publish

# Tag, build, publish, and create a GitHub release for imprint-server
server-release version notes="": server-build
    git tag imprint-server-v{{version}} -m "{{notes}}"
    git push origin imprint-server-v{{version}}
    cd imprint-server && uv publish
    gh release create imprint-server-v{{version}} imprint-server/dist/* \
        --title "imprint-server v{{version}}" \
        --notes "{{notes}}"

# Build the imprint-server Docker image (context = repo root)
server-docker-build tag="imprint-server:latest":
    docker build -t {{tag}} -f imprint-server/Dockerfile .

# Start imprint-server + Postgres via docker-compose
server-docker-up:
    docker compose -f imprint-server/docker-compose.yml up

# Start in detached mode
server-docker-up-d:
    docker compose -f imprint-server/docker-compose.yml up -d

# Stop and remove containers (data volume preserved)
server-docker-down:
    docker compose -f imprint-server/docker-compose.yml down

# Stop and remove containers AND the data volume
server-docker-reset:
    docker compose -f imprint-server/docker-compose.yml down -v

# Start Postgres via Docker Compose, run Postgres integration tests, then tear down.
# Tests run even if the server container is not yet built (only Postgres needed).
# Cleanup runs via trap so it fires even on test failure.
server-integration-test:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    docker compose -f "$REPO_ROOT/imprint-server/docker-compose.yml" up -d --wait postgres
    trap 'docker compose -f "$REPO_ROOT/imprint-server/docker-compose.yml" down' EXIT
    cd imprint-server && \
        IMPRINT_STORE=postgres://imprint:imprint@localhost:5432/imprint \
        uv run --extra postgres pytest -m postgres -v
