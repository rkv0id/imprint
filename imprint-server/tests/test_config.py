"""Tests for ServerConfig: defaults, store detection, env overrides, validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from imprint_server.config import ServerConfig


@pytest.fixture(autouse=True)
def clean_imprint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all IMPRINT_* env vars before each test.

    Prevents interference from .env files or shell env in the developer's
    environment. Each test starts from a known baseline.
    """
    import os

    for key in list(os.environ.keys()):
        if key.startswith("IMPRINT_"):
            monkeypatch.delenv(key, raising=False)


# -- Defaults -----------------------------------------------------------------


def test_defaults_store() -> None:
    cfg = ServerConfig()
    assert cfg.store == "sqlite:///~/.imprint/imprint.db"


def test_defaults_server() -> None:
    cfg = ServerConfig()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.workers == 1


def test_defaults_model_and_mode() -> None:
    cfg = ServerConfig()
    assert cfg.default_model == "anthropic:claude-haiku-4-5-20251001"
    assert cfg.default_mode == "balanced"


def test_defaults_auth_disabled() -> None:
    cfg = ServerConfig()
    assert cfg.auth_disabled is True


def test_defaults_logging_and_cors() -> None:
    cfg = ServerConfig()
    assert cfg.log_format == "text"
    assert cfg.cors_origins == "*"


def test_defaults_pool() -> None:
    cfg = ServerConfig()
    assert cfg.pool_min == 2
    assert cfg.pool_max == 10


def test_defaults_scheduler() -> None:
    cfg = ServerConfig()
    assert cfg.consolidate_interval == 86400
    assert cfg.session_ttl == 3600


def test_defaults_confusion() -> None:
    cfg = ServerConfig()
    assert cfg.confusion_window == 10
    assert cfg.confusion_threshold == pytest.approx(0.3)


def test_defaults_alpha() -> None:
    cfg = ServerConfig()
    assert cfg.alpha_min_samples == 20


def test_mcp_field_defaults() -> None:
    """MCP fields have empty string as their declared default.

    Uses model_fields inspection rather than instantiation so that local
    .env files with IMPRINT_MCP_AGENT_ID/IMPRINT_MCP_USER_ID set do not
    cause false failures.
    """
    assert ServerConfig.model_fields["mcp_agent_id"].default == ""
    assert ServerConfig.model_fields["mcp_user_id"].default == ""


# -- Store detection ----------------------------------------------------------


def test_is_sqlite_default() -> None:
    cfg = ServerConfig()
    assert cfg.is_sqlite is True
    assert cfg.is_postgres is False


def test_is_sqlite_explicit_path() -> None:
    cfg = ServerConfig(store="sqlite:///tmp/test.db")
    assert cfg.is_sqlite is True
    assert cfg.is_postgres is False


def test_is_sqlite_memory() -> None:
    cfg = ServerConfig(store=":memory:")
    assert cfg.is_sqlite is True
    assert cfg.is_postgres is False


def test_is_postgres_url() -> None:
    cfg = ServerConfig(store="postgres://user:pass@localhost/db")
    assert cfg.is_postgres is True
    assert cfg.is_sqlite is False


def test_is_postgres_postgresql_scheme() -> None:
    cfg = ServerConfig(store="postgresql://user:pass@localhost/db")
    assert cfg.is_postgres is True
    assert cfg.is_sqlite is False


# -- CORS origins list --------------------------------------------------------


def test_cors_origins_list_wildcard() -> None:
    cfg = ServerConfig()
    assert cfg.cors_origins_list == ["*"]


def test_cors_origins_list_single() -> None:
    cfg = ServerConfig(cors_origins="https://app.example.com")
    assert cfg.cors_origins_list == ["https://app.example.com"]


def test_cors_origins_list_multiple() -> None:
    cfg = ServerConfig(cors_origins="https://app.example.com,https://admin.example.com")
    assert cfg.cors_origins_list == [
        "https://app.example.com",
        "https://admin.example.com",
    ]


def test_cors_origins_list_strips_whitespace() -> None:
    cfg = ServerConfig(cors_origins=" https://a.com , https://b.com ")
    assert cfg.cors_origins_list == ["https://a.com", "https://b.com"]


def test_cors_origins_list_ignores_empty_segments() -> None:
    cfg = ServerConfig(cors_origins="https://a.com,,https://b.com")
    assert cfg.cors_origins_list == ["https://a.com", "https://b.com"]


# -- Env var overrides --------------------------------------------------------


def test_env_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_PORT", "9000")
    cfg = ServerConfig()
    assert cfg.port == 9000


def test_env_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_STORE", "postgres://user:pw@host/db")
    cfg = ServerConfig()
    assert cfg.is_postgres is True


def test_env_auth_disabled_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_AUTH_DISABLED", "false")
    cfg = ServerConfig()
    assert cfg.auth_disabled is False


def test_env_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_DEFAULT_MODE", "eager")
    cfg = ServerConfig()
    assert cfg.default_mode == "eager"


def test_env_log_format_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_LOG_FORMAT", "json")
    cfg = ServerConfig()
    assert cfg.log_format == "json"


def test_env_confusion_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_CONFUSION_THRESHOLD", "0.5")
    cfg = ServerConfig()
    assert cfg.confusion_threshold == pytest.approx(0.5)


def test_env_mcp_agent_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_MCP_AGENT_ID", "my-agent")
    cfg = ServerConfig()
    assert cfg.mcp_agent_id == "my-agent"


def test_env_mcp_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_MCP_USER_ID", "rami")
    cfg = ServerConfig()
    assert cfg.mcp_user_id == "rami"


# -- Validation errors --------------------------------------------------------


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValidationError, match="default_mode"):
        ServerConfig(default_mode="turbo")


def test_invalid_log_format_raises() -> None:
    with pytest.raises(ValidationError, match="log_format"):
        ServerConfig(log_format="xml")


def test_workers_zero_raises() -> None:
    with pytest.raises(ValidationError, match="workers"):
        ServerConfig(workers=0)


def test_workers_negative_raises() -> None:
    with pytest.raises(ValidationError, match="workers"):
        ServerConfig(workers=-1)


def test_port_out_of_range_raises() -> None:
    with pytest.raises(ValidationError, match="port"):
        ServerConfig(port=99999)


def test_pool_min_zero_raises() -> None:
    with pytest.raises(ValidationError, match="pool_min"):
        ServerConfig(pool_min=0)


def test_pool_min_gt_pool_max_raises() -> None:
    with pytest.raises(ValidationError, match="pool_min"):
        ServerConfig(pool_min=10, pool_max=5)


def test_confusion_threshold_above_one_raises() -> None:
    with pytest.raises(ValidationError, match="confusion_threshold"):
        ServerConfig(confusion_threshold=1.1)


def test_confusion_threshold_below_zero_raises() -> None:
    with pytest.raises(ValidationError, match="confusion_threshold"):
        ServerConfig(confusion_threshold=-0.1)


def test_sqlite_multi_worker_raises() -> None:
    with pytest.raises(ValidationError, match="SQLite"):
        ServerConfig(store="sqlite:///test.db", workers=2)


def test_postgres_multi_worker_allowed() -> None:
    cfg = ServerConfig(store="postgres://user:pw@host/db", workers=4)
    assert cfg.workers == 4


def test_alpha_min_samples_zero_raises() -> None:
    with pytest.raises(ValidationError, match="alpha_min_samples"):
        ServerConfig(alpha_min_samples=0)


# -- Extra env vars ignored ---------------------------------------------------


def test_unknown_imprint_env_vars_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_SOME_FUTURE_SETTING", "whatever")
    cfg = ServerConfig()
    assert cfg.port == 8000


# -- Embedder defaults --------------------------------------------------------


def test_defaults_embedder() -> None:
    cfg = ServerConfig()
    assert cfg.embedder == "none"
    assert cfg.embedder_model == "voyage-3"
    assert cfg.embedder_dim == 1024


def test_defaults_vector_store() -> None:
    cfg = ServerConfig()
    assert cfg.vector_store == "none"


def test_defaults_decay_model() -> None:
    cfg = ServerConfig()
    assert cfg.decay_model == "static"


# -- Embedder validation ------------------------------------------------------


def test_embedder_voyage_accepted() -> None:
    cfg = ServerConfig(embedder="voyage")
    assert cfg.embedder == "voyage"


def test_embedder_openai_accepted() -> None:
    cfg = ServerConfig(embedder="openai")
    assert cfg.embedder == "openai"


def test_embedder_unknown_raises() -> None:
    with pytest.raises(ValidationError, match="embedder"):
        ServerConfig(embedder="cohere")


def test_embedder_dim_zero_raises() -> None:
    with pytest.raises(ValidationError, match="embedder_dim"):
        ServerConfig(embedder_dim=0)


def test_embedder_dim_negative_raises() -> None:
    with pytest.raises(ValidationError, match="embedder_dim"):
        ServerConfig(embedder_dim=-1)


def test_embedder_dim_custom() -> None:
    cfg = ServerConfig(embedder_dim=768)
    assert cfg.embedder_dim == 768


# -- Vector store validation --------------------------------------------------


def test_vector_store_unknown_raises() -> None:
    with pytest.raises(ValidationError, match="vector_store"):
        ServerConfig(vector_store="pinecone")


def test_vector_store_sqlite_vec_with_sqlite_store() -> None:
    cfg = ServerConfig(
        store="sqlite:///tmp/test.db",
        embedder="voyage",
        vector_store="sqlite-vec",
    )
    assert cfg.vector_store == "sqlite-vec"


def test_vector_store_sqlite_vec_with_postgres_raises() -> None:
    with pytest.raises(ValidationError, match="sqlite-vec"):
        ServerConfig(
            store="postgres://user:pw@host/db",
            embedder="voyage",
            vector_store="sqlite-vec",
        )


def test_vector_store_postgres_with_postgres_store() -> None:
    cfg = ServerConfig(
        store="postgres://user:pw@host/db",
        embedder="voyage",
        vector_store="postgres",
    )
    assert cfg.vector_store == "postgres"


def test_vector_store_postgres_with_sqlite_raises() -> None:
    with pytest.raises(ValidationError, match="vector_store"):
        ServerConfig(
            store="sqlite:///tmp/test.db",
            embedder="voyage",
            vector_store="postgres",
        )


def test_vector_store_requires_embedder() -> None:
    with pytest.raises(ValidationError, match="embedder"):
        ServerConfig(
            store="sqlite:///tmp/test.db",
            embedder="none",
            vector_store="sqlite-vec",
        )


# -- Decay model validation ---------------------------------------------------


def test_decay_model_static_accepted() -> None:
    cfg = ServerConfig(decay_model="static")
    assert cfg.decay_model == "static"


def test_decay_model_unknown_raises() -> None:
    with pytest.raises(ValidationError, match="decay_model"):
        ServerConfig(decay_model="exponential")


def test_decay_model_gradient_accepted_when_online_installed() -> None:
    """gradient is valid when imprint-mem[online] (River) is installed."""
    import importlib

    try:
        importlib.import_module("imprint.online")
        online_available = True
    except ImportError:
        online_available = False

    if online_available:
        cfg = ServerConfig(decay_model="gradient")
        assert cfg.decay_model == "gradient"
    else:
        with pytest.raises(ValidationError, match="gradient"):
            ServerConfig(decay_model="gradient")


# -- Env var overrides --------------------------------------------------------


def test_env_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_EMBEDDER", "voyage")
    cfg = ServerConfig()
    assert cfg.embedder == "voyage"


def test_env_embedder_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_EMBEDDER_MODEL", "voyage-3-lite")
    cfg = ServerConfig()
    assert cfg.embedder_model == "voyage-3-lite"


def test_env_embedder_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_EMBEDDER_DIM", "512")
    cfg = ServerConfig()
    assert cfg.embedder_dim == 512


def test_env_decay_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPRINT_DECAY_MODEL", "static")
    cfg = ServerConfig()
    assert cfg.decay_model == "static"
