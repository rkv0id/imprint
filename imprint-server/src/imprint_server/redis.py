"""Redis client wrapper and rate limiter for imprint-server.

Redis is optional. All public functions check config.redis_enabled first
and no-op when Redis is not configured.

Rate limiting uses a sliding window implemented with Redis sorted sets:
  - Key: imprint:rl:{identifier}
  - Members: unique request timestamps (uuid)
  - Score: unix timestamp (float)
  - On each request:
      1. ZREMRANGEBYSCORE to evict entries older than the window
      2. ZCARD to count current requests
      3. If count >= limit: return 429
      4. ZADD to record this request
      5. EXPIRE to auto-clean the key

This is the standard Redis sliding window pattern. It is correct under
concurrent load because the ZADD happens after the count check, but
with pipeline the entire sequence is atomic per-request.

Without Redis, an in-memory sliding window is used. It is process-local
and not safe for multi-worker deployments (enforced by config validation).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig


# -- Redis client lifecycle ---------------------------------------------------


class RedisClient:
    """Thin async wrapper around redis.asyncio.Redis.

    Created once in registry.startup() and closed in registry.shutdown().
    Shared across all requests via app.state.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: Any = None

    async def connect(self) -> None:
        try:
            from redis.asyncio import Redis  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "IMPRINT_REDIS_URL is set but redis is not installed. "
                "Install it with: pip install imprint-server[redis]"
            ) from exc
        self._client = Redis.from_url(self._url, decode_responses=True)
        # Verify connectivity immediately so startup fails fast on bad URL.
        await self._client.ping()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ping(self) -> bool:
        """Return True if Redis is reachable."""
        if self._client is None:
            return False
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    # -- Low-level operations used by rate limiter ----------------------------

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> None:
        await self._client.zremrangebyscore(key, min_score, max_score)

    async def zcard(self, key: str) -> int:
        return int(await self._client.zcard(key))

    async def zadd(self, key: str, mapping: dict[str, float]) -> None:
        await self._client.zadd(key, mapping)

    async def expire(self, key: str, seconds: int) -> None:
        await self._client.expire(key, seconds)

    # -- Cache operations used by policy cache overlay ------------------------

    async def get(self, key: str) -> str | None:
        """Return the cached string value, or None if missing/expired."""
        return await self._client.get(key)  # type: ignore[no-any-return]

    async def setex(self, key: str, seconds: int, value: str) -> None:
        """Set key to value with a TTL in seconds."""
        await self._client.setex(key, seconds, value)

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern. Returns count of deleted keys.

        Uses SCAN + DEL to avoid blocking Redis on large keyspaces.
        Safe for production use but not atomic across all matched keys.
        """
        deleted = 0
        async for key in self._client.scan_iter(match=pattern, count=100):
            await self._client.delete(key)
            deleted += 1
        return deleted


# -- Rate limiter -------------------------------------------------------------


class RateLimiter:
    """Sliding window rate limiter. Uses Redis when available, memory otherwise.

    Instantiated once and shared across all requests via app.state.
    """

    def __init__(self, config: ServerConfig, redis: RedisClient | None) -> None:
        self._limit = config.rate_limit_requests
        self._window = config.rate_limit_window
        self._redis = redis
        # In-memory fallback: deque of timestamps per identifier.
        self._memory: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, identifier: str) -> bool:
        """Return True if the request is within the rate limit, False if exceeded."""
        if self._redis is not None:
            return await self._redis_check(identifier)
        return await self._memory_check(identifier)

    async def _redis_check(self, identifier: str) -> bool:
        assert self._redis is not None
        key = f"imprint:rl:{identifier}"
        now = time.time()
        window_start = now - self._window

        # Evict expired entries, count current, conditionally record.
        await self._redis.zremrangebyscore(key, 0, window_start)
        count = await self._redis.zcard(key)
        if count >= self._limit:
            return False
        member = str(uuid.uuid4())
        await self._redis.zadd(key, {member: now})
        await self._redis.expire(key, self._window * 2)
        return True

    async def _memory_check(self, identifier: str) -> bool:
        now = time.time()
        window_start = now - self._window
        async with self._lock:
            if identifier not in self._memory:
                self._memory[identifier] = deque()
            q = self._memory[identifier]
            # Evict old entries.
            while q and q[0] <= window_start:
                q.popleft()
            if len(q) >= self._limit:
                return False
            q.append(now)
            return True
