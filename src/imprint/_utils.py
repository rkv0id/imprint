"""Pure utility functions and constants shared across imprint internals.

No imprint imports -- this module sits at the bottom of the dependency graph.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imprint.types import Memory

# -- Constants ----------------------------------------------------------------

_MAX_SCOPE_LEN = 50

_TURSO_SCHEMES = ("libsql://", "ws://", "wss://", "https://", "http://", "turso://")
_POSTGRES_SCHEMES = ("postgres://", "postgresql://")

# -- ID generation ------------------------------------------------------------


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# -- Scope helpers ------------------------------------------------------------


def _resolve_scope(requested: str | None) -> str:
    """Return the requested scope, or 'global' if absent or blank."""
    if not requested or not requested.strip():
        return "global"
    return requested.strip()


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# -- URL helpers --------------------------------------------------------------


def _is_turso_url(url: str) -> bool:
    return any(url.startswith(s) for s in _TURSO_SCHEMES)


def _is_postgres_url(url: str) -> bool:
    return any(url.startswith(s) for s in _POSTGRES_SCHEMES)


def _parse_turso_url(url: str) -> tuple[str, str | None]:
    """Parse a Turso store URL, extracting auth_token from query string if present.

    Returns (url_without_token, auth_token_or_None).
    Accepts turso:// as an alias for libsql://.
    """
    if url.startswith("turso://"):
        url = "libsql://" + url[len("turso://") :]
    if "?auth_token=" in url:
        base, token = url.split("?auth_token=", 1)
        return base, token
    return url, None


def _parse_store_url(url: str) -> str:
    """Parse a store URL into a SQLite path. Accepts:

    - sqlite:///abs/path -> /abs/path
    - sqlite:///:memory: -> :memory:
    - :memory: -> :memory:
    - bare absolute or relative path -> path (with ~ expansion)

    Rejects empty strings, Turso/libSQL URLs, Postgres URLs, and unknown schemes.
    Postgres and Turso URLs are handled before this function is called.
    """
    if not url:
        raise ValueError("store URL must be non-empty")
    if _is_turso_url(url):
        raise ValueError(
            f"Turso/libSQL URLs are handled automatically; pass the URL directly "
            f"as the store parameter: Imprint(store={url!r})"
        )
    if _is_postgres_url(url):
        raise ValueError(
            f"Postgres URLs are handled automatically; pass the URL directly "
            f"as the store parameter: Imprint(store={url!r})"
        )
    if "://" in url and not url.startswith("sqlite://"):
        scheme = url.split("://", 1)[0]
        raise ValueError(f"unsupported store URL scheme: {scheme!r} (expected 'sqlite')")
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///") :]
        if path == ":memory:":
            return ":memory:"
        return os.path.expanduser(path)
    return os.path.expanduser(url)


# -- Vector math --------------------------------------------------------------


try:
    import numpy as _np  # type: ignore[import-untyped,import-not-found]

    def _cosine(a: list[float], b: list[float]) -> float:
        va = _np.array(a, dtype=_np.float32)  # type: ignore[reportUnknownMemberType]
        vb = _np.array(b, dtype=_np.float32)  # type: ignore[reportUnknownMemberType]
        denom = float(_np.linalg.norm(va) * _np.linalg.norm(vb))  # type: ignore[reportUnknownMemberType]
        return float(_np.dot(va, vb) / denom) if denom > 0.0 else 0.0  # type: ignore[reportUnknownMemberType]

except ImportError:

    def _cosine(a: list[float], b: list[float]) -> float:  # type: ignore[misc]
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)


# -- Policy cache key ---------------------------------------------------------


def _policy_cache_key(
    *,
    agent_id: str,
    user_id: str,
    memories: list[Memory],
    context: str | None,
    existing_instructions: str | None,
    max_output_tokens: int,
    scopes: list[str] | None,
) -> str:
    h = hashlib.sha256()
    h.update(b"agent\x00")
    h.update(agent_id.encode("utf-8"))
    h.update(b"\x00user\x00")
    h.update(user_id.encode("utf-8"))
    h.update(b"\x00mem\x00")
    for m in sorted(memories, key=lambda x: x.id):
        h.update(m.id.encode("utf-8"))
        h.update(b"|")
        h.update(m.updated_at.isoformat().encode("utf-8"))
        h.update(b"\x00")
    h.update(b"\x00ctx\x00")
    h.update((context or "").encode("utf-8"))
    h.update(b"\x00inst\x00")
    h.update((existing_instructions or "").encode("utf-8"))
    h.update(b"\x00max\x00")
    h.update(str(max_output_tokens).encode("utf-8"))
    h.update(b"\x00scopes\x00")
    if scopes is None:
        h.update(b"<none>")
    else:
        for s in sorted(scopes):
            h.update(s.encode("utf-8"))
            h.update(b"|")
    return h.hexdigest()


# -- Datetime helper ----------------------------------------------------------
# Re-exported here so mixin modules don't need to import from store.py.
# store.py is the canonical definition; this avoids a cross-dependency from
# the mixin files back into the store layer.


def _parse_dt(value: str) -> datetime:
    """Parse an ISO datetime string as a timezone-aware UTC datetime."""
    from datetime import UTC

    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
