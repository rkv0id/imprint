-- Add user_id column to api_keys for multi-user MCP support.
-- Keys with a user_id set are scoped to that user namespace.
-- Keys without user_id remain master keys.
--
-- Note: SQLite does not support ADD COLUMN IF NOT EXISTS. The migration
-- runner catches OperationalError for "duplicate column" and skips silently.

ALTER TABLE api_keys ADD COLUMN user_id TEXT;
