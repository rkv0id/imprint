-- Add user_id column to api_keys for multi-user MCP support.
-- Keys with a user_id set are scoped to that user namespace.
-- Keys without user_id remain master keys.

ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS user_id TEXT;
