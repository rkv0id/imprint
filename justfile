default:
    @just --list

# Install all deps into .venv
sync:
    uv sync

# Run tests
test *ARGS:
    uv run pytest {{ARGS}}

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
