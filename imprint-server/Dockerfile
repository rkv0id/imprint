# imprint-server Dockerfile
# Multi-stage build: builder installs deps, runtime runs the server.
#
# Build:
#   docker build -t imprint-server -f imprint-server/Dockerfile .
#
# Run (SQLite, local dev):
#   docker run -p 8000:8000 \
#     -v ~/.imprint:/data \
#     -e IMPRINT_STORE=sqlite:////data/imprint.db \
#     imprint-server
#
# Run (Postgres, production):
#   docker run -p 8000:8000 \
#     -e IMPRINT_STORE=postgres://user:pass@host/db \
#     -e IMPRINT_AUTH_DISABLED=false \
#     imprint-server
#
# The build context must be the repository root (not imprint-server/).
# Both packages (imprint-mem and imprint-server) are installed from source.

# -- Builder ------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system

# Copy workspace config and lockfile first so Docker layer cache is effective.
COPY pyproject.toml uv.lock README.md ./
COPY imprint-server/pyproject.toml ./imprint-server/
COPY imprint-server/README.md ./imprint-server/

# Install all production dependencies (both packages, with postgres extra).
# --no-dev skips test/lint tools. --no-editable produces a portable venv.
RUN uv sync \
    --all-packages \
    --extra postgres \
    --extra vector \
    --extra online \
    --extra redis \
    --no-dev \
    --no-editable \
    --locked

# Copy source after deps so changes to source don't bust the dep cache layer.
COPY src/ ./src/
COPY imprint-server/src/ ./imprint-server/src/

# Re-sync with source to install the packages themselves (deps already cached).
RUN uv sync \
    --all-packages \
    --extra postgres \
    --extra vector \
    --extra online \
    --extra redis \
    --no-dev \
    --no-editable \
    --locked

# -- Runtime ------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Create a non-root user for the server process.
RUN addgroup --system imprint && \
    adduser --system --ingroup imprint --no-create-home imprint

# Copy the venv from the builder. The venv contains both imprint-mem and
# imprint-server with all their dependencies.
COPY --from=builder /build/.venv /app/.venv

# Activate the venv for all subsequent commands.
ENV PATH="/app/.venv/bin:$PATH"

# Default configuration. Override with -e or an env file.
ENV IMPRINT_HOST=0.0.0.0 \
    IMPRINT_PORT=8000 \
    IMPRINT_WORKERS=1 \
    IMPRINT_AUTH_DISABLED=true \
    IMPRINT_LOG_FORMAT=json \
    IMPRINT_STORE=sqlite:////data/imprint.db

# Data directory for SQLite mode. Mount a volume here for persistence.
RUN mkdir -p /data && chown imprint:imprint /data
VOLUME ["/data"]

EXPOSE 8000

USER imprint

# Health check using the /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${IMPRINT_PORT}/health')" || exit 1

ENTRYPOINT ["imprint-server"]
CMD ["serve"]
