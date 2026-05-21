-- Initial server schema for imprint-server.
-- Creates the five tables managed by imprint-server on top of the
-- imprint-mem library schema. All statements are CREATE TABLE IF NOT EXISTS
-- so this migration is safe to run against a pre-existing database.

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    context       TEXT,
    retrieved_ids TEXT,
    alpha_used    REAL DEFAULT 0.3,
    outcome       REAL,
    correction    TEXT,
    opened_at     TIMESTAMPTZ NOT NULL,
    expires_at    TIMESTAMPTZ NOT NULL,
    closed_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sessions_agent_user
    ON sessions(agent_id, user_id, closed_at);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    user_id       TEXT,
    job_type      TEXT NOT NULL,
    payload       JSONB,
    status        TEXT DEFAULT 'pending',
    priority      INT DEFAULT 5,
    created_at    TIMESTAMPTZ NOT NULL,
    locked_at     TIMESTAMPTZ,
    locked_by     TEXT,
    completed_at  TIMESTAMPTZ,
    error         TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_claim
    ON jobs(status, priority DESC, created_at)
    WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS api_keys (
    key_hash      TEXT PRIMARY KEY,
    agent_id      TEXT,
    label         TEXT,
    created_at    TIMESTAMPTZ NOT NULL,
    expires_at    TIMESTAMPTZ,
    active        BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS policy_events (
    id            TEXT PRIMARY KEY,
    session_id    TEXT,
    agent_id      TEXT NOT NULL,
    user_id       TEXT,
    retrieved_ids TEXT NOT NULL,
    filtered_ids  TEXT NOT NULL,
    alpha_used    REAL NOT NULL,
    context_hash  TEXT,
    occurred_at   TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policy_events_agent
    ON policy_events(agent_id, occurred_at DESC);

CREATE TABLE IF NOT EXISTS agent_ext_config (
    agent_id       TEXT PRIMARY KEY,
    dynamic_scopes BOOLEAN NOT NULL DEFAULT FALSE
);
