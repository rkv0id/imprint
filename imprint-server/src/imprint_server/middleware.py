"""Request middleware for imprint-server.

RequestIDMiddleware  -- inject / echo X-Request-ID on every request.
AccessLogMiddleware  -- structured per-request access log line.
RateLimitMiddleware  -- sliding window rate limiting per API key or IP.

Middleware is added in app.py in this order (innermost to outermost):
  CORS -> Auth -> RateLimit -> AccessLog -> RequestID

So the execution order per request is:
  RequestID -> AccessLog (start) -> RateLimit -> Auth -> CORS -> handler
  handler -> CORS -> Auth -> RateLimit -> AccessLog (end) -> RequestID
"""

from __future__ import annotations

import logging
import re as _re
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig

_access_log = logging.getLogger("imprint.server.access")

# Paths exempt from rate limiting.
_RL_EXEMPT = frozenset({"/health", "/health/live", "/health/ready", "/metrics"})

_AGENT_RE = _re.compile(r"^/v1/agents/([^/]+)")
_USER_RE = _re.compile(r"^/v1/agents/[^/]+/(?:memories|events|health|correct|reinforce)/([^/]+)")


def _extract_ids(path: str) -> tuple[str | None, str | None]:
    agent_match = _AGENT_RE.match(path)
    agent_id = agent_match.group(1) if agent_match else None
    user_match = _USER_RE.match(path)
    user_id = user_match.group(1) if user_match else None
    return agent_id, user_id


def _problem_json(status: int, title: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": "about:blank", "title": title, "status": status, "detail": detail},
        headers={"Content-Type": "application/problem+json"},
    )


# -- RequestIDMiddleware ------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request and response.

    Uses X-Request-ID if the client supplies one; generates a UUID4 otherwise.
    The ID is stored on request.state.request_id for downstream log middleware.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# -- AccessLogMiddleware ------------------------------------------------------


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Log one structured line per completed request.

    Fields logged: method, path, status, duration_ms, request_id,
    agent_id (when present in path), user_id (when present in path).

    Format controlled by IMPRINT_LOG_FORMAT:
      text  -> key=value pairs on one line
      json  -> JSON object (use with a log aggregator)
    """

    def __init__(self, app: object, config: ServerConfig) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._json = config.log_format == "json"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        request_id = getattr(request.state, "request_id", "-")
        agent_id, user_id = _extract_ids(request.url.path)

        if self._json:
            import json

            msg = json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                    "request_id": request_id,
                    "agent_id": agent_id,
                    "user_id": user_id,
                },
                separators=(",", ":"),
            )
        else:
            parts = [
                f"method={request.method}",
                f"path={request.url.path}",
                f"status={response.status_code}",
                f"duration_ms={duration_ms}",
                f"request_id={request_id}",
            ]
            if agent_id:
                parts.append(f"agent_id={agent_id}")
            if user_id:
                parts.append(f"user_id={user_id}")
            msg = " ".join(parts)

        level = logging.WARNING if response.status_code >= 500 else logging.INFO
        _access_log.log(level, msg)
        return response


# -- RateLimitMiddleware ------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiting per API key (auth enabled) or client IP.

    Returns 429 Too Many Requests with a Retry-After header when the limit
    is exceeded. Exempt paths (/health, /metrics) are never rate-limited.

    The rate limiter is fetched from the registry lazily on each request so
    the middleware is safe to add before registry.startup() is called.
    """

    def __init__(
        self,
        app: object,
        config: ServerConfig,
        registry: object,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._config = config
        self._registry = registry

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _RL_EXEMPT:
            return await call_next(request)

        from imprint_server.redis import RateLimiter

        limiter: RateLimiter | None = getattr(self._registry, "rate_limiter", None)
        if limiter is None:
            return await call_next(request)

        identifier = self._get_identifier(request)
        allowed = await limiter.is_allowed(identifier)
        if not allowed:
            rejection = _problem_json(
                429,
                "Too Many Requests",
                f"Rate limit exceeded: {self._config.rate_limit_requests} requests "
                f"per {self._config.rate_limit_window}s. Retry after "
                f"{self._config.rate_limit_window}s.",
            )
            rejection.headers["Retry-After"] = str(self._config.rate_limit_window)
            return rejection
        response = await call_next(request)
        response.headers["Retry-After"] = str(self._config.rate_limit_window)
        return response

    def _get_identifier(self, request: Request) -> str:
        """Return the rate limit key: API key hash when auth enabled, else client IP."""
        if not self._config.auth_disabled:
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                raw = auth[len("Bearer ") :]
                from imprint_server.stores.api_keys import hash_key

                return f"key:{hash_key(raw)}"
        # Fall back to client IP.
        client = request.client
        ip = client.host if client else "unknown"
        return f"ip:{ip}"
