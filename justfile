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
#   - Docker running (for server-integration-test and server-redis-test)
#   - .env and imprint-server/.env with any required API keys
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
    _run "server:  redis integration tests"   just server-redis-test

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

# Run imprint-server live retrieval tests (requires VOYAGE_API_KEY; OPENAI_API_KEY optional)
# --override-ini clears the default -m filter so -m live is the only active marker expression.
server-live-test:
    cd imprint-server && uv run pytest tests/test_live_retrieval.py -m live -v --override-ini="addopts=-ra"

# Run all library examples and write output to examples/output/.
# Requires ANTHROPIC_API_KEY (and OPENAI_API_KEY or VOYAGE_API_KEY for retrieval examples).
# with_server_client.py is excluded -- it requires a running imprint-server.
# Set SKIP_SLOW=1 to skip retrieval_tuning.py (makes many API calls).
#
# Output per example is written to examples/output/<name>.log.
# A summary of pass/fail is printed at the end.
#
# Usage:
#   just run-examples
#   just run-examples 2>&1 | tee examples/output/session.log
run-examples:
    #!/usr/bin/env bash
    set -a
    [ -f .env ] && source .env
    set +a

    mkdir -p examples/output

    declare -a results
    declare -a failed

    _run_example() {
        local name="$1"
        local file="examples/${name}"
        local log="examples/output/${name%.py}.log"

        echo ""
        echo "========================================"
        echo "=== $name"
        echo "========================================"

        if uv run python "$file" > "$log" 2>&1; then
            results+=("[PASS] $name")
            tail -5 "$log"
        else
            results+=("[FAIL] $name")
            failed+=("$name")
            echo "[ERROR] last 20 lines of output:"
            tail -20 "$log"
        fi
    }

    _run_example minimal.py
    _run_example writing_assistant.py
    _run_example decay_and_reinforcement.py
    _run_example multi_session.py
    _run_example dynamic_scopes.py
    _run_example online_learning.py
    _run_example with_langchain.py

    if [ "${SKIP_SLOW:-0}" != "1" ]; then
        _run_example with_retrieval.py
        _run_example retrieval_tuning.py
    else
        echo ""
        echo "========================================"
        echo "=== skipping retrieval examples (SKIP_SLOW=1)"
        echo "========================================"
    fi

    # with_postgres.py skipped unless Postgres is actually reachable.
    if [ -n "${IMPRINT_POSTGRES_URL:-}" ] && \
       python3 -c "import socket; s=socket.create_connection(('localhost',5432),timeout=1); s.close()" 2>/dev/null; then
        _run_example with_postgres.py
    else
        echo ""
        echo "========================================"
        echo "=== skipping with_postgres.py (Postgres not reachable at localhost:5432)"
        echo "=== start with: just postgres-dev"
        echo "========================================"
    fi

    # with_server_client.py skipped unless a server is reachable.
    if curl -sf http://localhost:8000/health/live > /dev/null 2>&1; then
        _run_example with_server_client.py
    else
        echo ""
        echo "========================================"
        echo "=== skipping with_server_client.py (no server at localhost:8000)"
        echo "=== start with: just server-dev"
        echo "========================================"
    fi

    echo ""
    echo "========================================"
    echo "=== Summary"
    echo "========================================"
    for r in "${results[@]}"; do
        echo "  $r"
    done
    echo ""
    echo "  Output logs: examples/output/"
    echo "========================================"

    if [ ${#failed[@]} -gt 0 ]; then
        echo "  FAILED: ${failed[*]}"
        exit 1
    else
        echo "  All examples passed."
        exit 0
    fi
    docker run --rm \
        --name imprint-redis \
        -p 6379:6379 \
        redis:7-alpine redis-server --save "" --appendonly no

# Run server tests that require a live Redis instance.
# Starts Redis via Docker, runs the tests, tears down on exit.
server-redis-test:
    #!/usr/bin/env bash
    set -e
    docker run --rm -d \
        --name imprint-redis-test \
        -p 6379:6379 \
        redis:7-alpine redis-server --save "" --appendonly no
    trap 'docker stop imprint-redis-test 2>/dev/null || true' EXIT
    # Wait for Redis to be ready.
    for i in $(seq 1 10); do
        docker exec imprint-redis-test redis-cli ping > /dev/null 2>&1 && break
        sleep 0.5
    done
    cd imprint-server && uv run pytest tests/test_middleware.py -m redis -v --override-ini="addopts=-ra"

# Run ALL live tests across both packages (library + server).
# Requires API keys set in .env or the environment:
#   ANTHROPIC_API_KEY  -- library LLM calls (balanced/eager mode observe + policy)
#   VOYAGE_API_KEY     -- server retrieval pipeline (embedder + vector store)
#   OPENAI_API_KEY     -- optional, for OpenAI embedder smoke test
#
# Redis and Postgres integration tests are in test-all (infrastructure-dependent,
# not API-key-dependent). This recipe covers external API calls only.
live-all:
    #!/usr/bin/env bash
    set -a
    [ -f .env ] && source .env
    [ -f imprint-server/.env ] && source imprint-server/.env
    set +a

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

    _run "library: live tests"       uv run pytest -m live --ignore=tests/test_postgres.py -v
    _run "server:  live tests"       just server-live-test
    _run "server:  registry live"    just server-test tests/test_registry.py -m live -v --override-ini="addopts=-ra"
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
        echo "  All live suites passed."
        exit 0
    fi

# Run imprint-server against a local SQLite store (auth disabled)
server-dev:
    cd imprint-server && \
        IMPRINT_STORE=sqlite:///~/.imprint/imprint-dev.db \
        IMPRINT_AUTH_DISABLED=true \
        uv run imprint-server serve

# Run imprint-server with MCP enabled against a local SQLite store.
# Set IMPRINT_MCP_AGENT_ID and IMPRINT_MCP_USER_ID in .env or pass inline:
#   just server-mcp-dev agent=my-agent user=me
server-mcp-dev agent="default" user="me":
    cd imprint-server && \
        IMPRINT_STORE=sqlite:///~/.imprint/imprint-dev.db \
        IMPRINT_AUTH_DISABLED=true \
        IMPRINT_MCP_AGENT_ID={{agent}} \
        IMPRINT_MCP_USER_ID={{user}} \
        uv run imprint-server serve

# Run imprint-server against a local Postgres instance (start postgres-dev first)
server-postgres-dev:
    cd imprint-server && \
        IMPRINT_STORE=postgres://imprint:imprint@localhost:5432/imprint \
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
