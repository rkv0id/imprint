"""Server configuration loaded from environment variables.

All settings are read from env vars prefixed with IMPRINT_.
Unknown IMPRINT_* vars are silently ignored (extra="ignore").

SQLite mode: IMPRINT_STORE=sqlite:///path/to/imprint.db (default)
  - Single process only.
  - Scheduler runs without job coordination.
  - Auth disabled by default.
  - Right for local use and Claude Code MCP integration.

Postgres mode: IMPRINT_STORE=postgres://user:pass@host/db
  - Multi-worker safe via jobs table (SELECT FOR UPDATE SKIP LOCKED).
  - Requires imprint-server[postgres] (asyncpg).
  - Required for production deployments.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_MODES = frozenset({"frugal", "balanced", "eager"})
_VALID_LOG_FORMATS = frozenset({"text", "json"})
_POSTGRES_PREFIXES = ("postgres://", "postgresql://")


class ServerConfig(BaseSettings):
    """All imprint-server configuration, sourced from IMPRINT_* env vars."""

    model_config = SettingsConfigDict(
        env_prefix="IMPRINT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Store ----------------------------------------------------------------

    store: str = "sqlite:///~/.imprint/imprint.db"
    """Store URL. sqlite:///path or postgres://user:pass@host/db."""

    # -- Server ---------------------------------------------------------------

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    """Number of uvicorn workers. SQLite mode must remain at 1."""

    # -- Library defaults (applied to agents with no explicit config) ---------

    default_model: str = "anthropic:claude-haiku-4-5-20251001"
    default_mode: str = "balanced"

    # -- Auth -----------------------------------------------------------------

    auth_disabled: bool = True
    """Disable API key authentication. Default true for local use.
    Set false (IMPRINT_AUTH_DISABLED=false) to enable auth in production.
    On first auth-enabled startup with no keys in DB, a master key is
    auto-generated and printed to stdout."""

    # -- Logging + CORS -------------------------------------------------------

    log_format: str = "text"
    """Log format: text (human-readable) or json (structured, for production)."""

    cors_origins: str = "*"
    """Comma-separated allowed CORS origins. Default * for local dev.
    Example: https://app.example.com,https://admin.example.com"""

    # -- Connection pool (Postgres only) --------------------------------------

    pool_min: int = Field(default=2, ge=1)
    pool_max: int = Field(default=10, ge=1)

    # -- Scheduler intervals (seconds) ----------------------------------------

    consolidate_interval: int = Field(default=86400, ge=60)
    """How often to run scheduled consolidation. Default 24h."""

    session_ttl: int = Field(default=3600, ge=1)
    """HTTP session lifetime in seconds before expiry sweep closes it. Default 1h."""

    # -- Confusion-based consolidation ----------------------------------------

    confusion_window: int = Field(default=10, ge=1)
    """Number of recent observations to check for contradiction rate."""

    confusion_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    """Correction/contradiction rate above which early consolidation is triggered."""

    # -- Per-context alpha tuning ---------------------------------------------

    alpha_min_samples: int = Field(default=20, ge=1)
    """Minimum policy_events per context hash before per-context alpha is used."""

    # -- MCP ------------------------------------------------------------------

    mcp_agent_id: str = ""
    """Agent ID used by the MCP endpoint. Required when using MCP tools."""

    mcp_user_id: str = ""
    """User namespace used by the MCP endpoint. Required when using MCP tools."""

    # -- Computed properties --------------------------------------------------

    @property
    def is_postgres(self) -> bool:
        """True when the store URL points at a Postgres instance."""
        return self.store.startswith(_POSTGRES_PREFIXES)

    @property
    def is_sqlite(self) -> bool:
        """True when the store URL is a SQLite path (local or :memory:)."""
        return not self.is_postgres

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS origins as a parsed list, ready for FastAPI CORSMiddleware."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # -- Validators -----------------------------------------------------------

    @field_validator("default_mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in _VALID_MODES:
            raise ValueError(f"default_mode must be one of {sorted(_VALID_MODES)!r}; got {v!r}")
        return v

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        if v not in _VALID_LOG_FORMATS:
            raise ValueError(f"log_format must be one of {sorted(_VALID_LOG_FORMATS)!r}; got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_pool_bounds(self) -> ServerConfig:
        if self.pool_min > self.pool_max:
            raise ValueError(f"pool_min ({self.pool_min}) must be <= pool_max ({self.pool_max})")
        return self

    @model_validator(mode="after")
    def validate_sqlite_workers(self) -> ServerConfig:
        if self.is_sqlite and self.workers > 1:
            raise ValueError(
                f"SQLite store supports only 1 worker; got workers={self.workers}. "
                "Use a Postgres store for multi-worker deployments."
            )
        return self
