"""Shared pytest fixtures and configuration for imprint-server tests.

Environment isolation
---------------------
Tests must never depend on or be affected by local .env values. This is
particularly important for IMPRINT_REDIS_URL: if the user has a Redis URL
in their .env for live testing, every ServerConfig() call in every fixture
would try to connect to Redis, causing connection errors and test hangs.

The `isolate_imprint_env` fixture below clears all IMPRINT_* env vars that
could cause unexpected external connections (Redis, Postgres URL overrides)
before each test. It is autouse so no test file needs to opt in.

Tests that require Redis (marked with @pytest.mark.redis) must explicitly
set the redis_url in their fixture rather than relying on the env var.
"""

from __future__ import annotations

import pytest

_VARS_TO_CLEAR = (
    "IMPRINT_REDIS_URL",
    "IMPRINT_RATE_LIMIT_ENABLED",
)


@pytest.fixture(autouse=True)
def isolate_imprint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear env vars that would cause unexpected external connections in tests.

    Cleared vars are restored after each test (monkeypatch is function-scoped).
    Redis-specific tests set redis_url explicitly in their fixture, so they
    are unaffected by this isolation.
    """
    for var in _VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
