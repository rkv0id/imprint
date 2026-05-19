"""Async client for imprint-server.

Provides a typed Python interface to the imprint-server REST API. Agents
using a remote server use this instead of calling the library directly.

Quick start:
  from imprint.client import ImprintClient

  async with ImprintClient("http://localhost:8000", api_key="sk-imp-...") as client:
      policy = await client.get_policy("my-agent", "user-1")
      print(policy.text)

  # Session-scoped usage (tracks retrieval for learning signal):
  async with client.session("my-agent", "user-1", context="coding") as sess:
      policy = await sess.get_policy()
      await sess.observe(agent_output="Here is a list...", user_response="No bullet points.")
      await sess.close(outcome=0.8)

  # Agent-scoped shortcut (avoids repeating agent_id):
  agent = client.agent("my-agent")
  policy = await agent.get_policy("user-1")

Install:
  pip install imprint-mem[client]   # adds httpx dependency
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

__all__ = [
    "AgentClient",
    "ImprintClient",
    "ImprintClientError",
    "ImprintConnectionError",
    "ImprintError",
    "MemoryHealth",
    "MemoryRecord",
    "PolicyResult",
    "ServerHealth",
    "Session",
]

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "imprint-server client requires httpx. Install it with: pip install imprint-mem[client]"
    ) from exc


# -- Exceptions ---------------------------------------------------------------


class ImprintClientError(Exception):
    """Base class for client-side errors."""


class ImprintError(ImprintClientError):
    """Raised when the server returns a 4xx or 5xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class ImprintConnectionError(ImprintClientError):
    """Raised when the client cannot reach the server."""


# -- Response types -----------------------------------------------------------


@dataclass
class PolicyResult:
    """Compiled behavioral policy returned by get_policy()."""

    text: str
    memory_count: int
    dropped_count: int
    compiled_at: str

    @property
    def has_memories(self) -> bool:
        return self.memory_count > 0


@dataclass
class MemoryRecord:
    """A single memory record returned by list_memories()."""

    id: str
    agent_id: str
    user_id: str
    type: str
    scope: str | None
    content: str
    source: str
    stability: float
    recall_count: int
    pinned: bool
    active: bool
    valid_from: str
    valid_until: str | None
    created_at: str
    updated_at: str


@dataclass
class MemoryHealth:
    """Aggregate memory health statistics."""

    total: int
    active: int
    pinned: int
    by_scope: dict[str, int]
    by_type: dict[str, int]
    avg_recall_count: float
    oldest_active: str | None
    newest_active: str | None


@dataclass
class ServerHealth:
    """Server health status from GET /health."""

    status: str
    store: str
    agents_loaded: int
    db_ok: bool

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.db_ok


# -- Core client --------------------------------------------------------------


class ImprintClient:
    """Async HTTP client for imprint-server.

    Manages an httpx.AsyncClient, injects auth headers, and retries on
    transient server errors with exponential backoff.

    Use as an async context manager to ensure the underlying connection
    pool is properly closed:

        async with ImprintClient("http://localhost:8000") as client:
            policy = await client.get_policy("my-agent", "user-1")

    Or manage the lifecycle manually:

        client = ImprintClient("http://localhost:8000")
        await client.aclose()
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """
        Args:
            base_url:      imprint-server base URL (e.g. "http://localhost:8000").
            api_key:       API key for auth-enabled servers. Omit if auth is disabled.
            timeout:       Request timeout in seconds (default 30).
            max_retries:   Max retry attempts on 5xx or connection errors (default 3).
            retry_backoff: Base sleep in seconds between retries (doubles each attempt).
            transport:     Custom httpx transport (useful for testing).
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> ImprintClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._http.aclose()

    # -- Shortcuts ------------------------------------------------------------

    def agent(self, agent_id: str) -> AgentClient:
        """Return a client pre-scoped to one agent ID."""
        return AgentClient(self, agent_id)

    def session(
        self,
        agent_id: str,
        user_id: str,
        *,
        context: str | None = None,
    ) -> Session:
        """Return a session context manager for a MemoryLoop over HTTP.

        Usage:
            async with client.session("my-agent", "user-1", context="coding") as sess:
                policy = await sess.get_policy()
                await sess.observe("output", "response")
                await sess.close(outcome=0.9)
        """
        return Session(self, agent_id=agent_id, user_id=user_id, context=context)

    # -- Observe --------------------------------------------------------------

    async def observe(
        self,
        agent_id: str,
        user_id: str,
        *,
        agent_output: str,
        user_response: str,
        context: str | None = None,
        scope: str | None = None,
    ) -> None:
        """Record a turn-by-turn exchange."""
        await self._post(
            f"/v1/agents/{agent_id}/observe",
            json={
                "user_id": user_id,
                "agent_output": agent_output,
                "user_response": user_response,
                "context": context,
                "scope": scope,
            },
        )

    async def observe_directions(
        self,
        agent_id: str,
        user_id: str,
        directions: list[str],
        *,
        context: str | None = None,
        scope: str | None = None,
    ) -> int:
        """Store explicit behavioral directions, bypassing signal detection.

        Returns the number of direction memories stored.
        """
        resp = await self._post(
            f"/v1/agents/{agent_id}/memories/{user_id}/directions",
            json={"directions": directions, "context": context, "scope": scope},
        )
        return int(resp.json()["stored"])

    # -- Policy ---------------------------------------------------------------

    async def get_policy(
        self,
        agent_id: str,
        user_id: str,
        *,
        context: str | None = None,
        scopes: list[str] | None = None,
        existing_instructions: str | None = None,
        max_input_tokens: int = 8000,
        max_output_tokens: int = 3000,
    ) -> PolicyResult:
        """Compile and return a behavioral policy."""
        resp = await self._post(
            f"/v1/agents/{agent_id}/policy",
            json={
                "user_id": user_id,
                "context": context,
                "scopes": scopes,
                "existing_instructions": existing_instructions,
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
            },
        )
        data = resp.json()
        return PolicyResult(
            text=data["policy_text"],
            memory_count=data["memory_count"],
            dropped_count=data["dropped_count"],
            compiled_at=data["compiled_at"],
        )

    # -- Memories -------------------------------------------------------------

    async def list_memories(
        self,
        agent_id: str,
        user_id: str,
        *,
        scopes: list[str] | None = None,
    ) -> list[MemoryRecord]:
        """Return active memories for a user namespace."""
        params: dict[str, Any] = {}
        if scopes:
            params["scopes"] = ",".join(scopes)
        resp = await self._get(f"/v1/agents/{agent_id}/memories/{user_id}", params=params)
        return [_memory_from_dict(m) for m in resp.json()]

    async def memory_health(self, agent_id: str, user_id: str) -> MemoryHealth:
        """Return aggregate memory health statistics."""
        resp = await self._get(f"/v1/agents/{agent_id}/health/{user_id}")
        data = resp.json()
        return MemoryHealth(
            total=data["total"],
            active=data["active"],
            pinned=data["pinned"],
            by_scope=data["by_scope"],
            by_type=data["by_type"],
            avg_recall_count=data["avg_recall_count"],
            oldest_active=data.get("oldest_active"),
            newest_active=data.get("newest_active"),
        )

    async def forget(self, agent_id: str, user_id: str) -> None:
        """Hard delete all memories for a user namespace. Irreversible."""
        await self._request("DELETE", f"/v1/agents/{agent_id}/memories/{user_id}")

    async def consolidate(
        self,
        agent_id: str,
        user_id: str,
        *,
        prune_threshold: float = 0.5,
    ) -> int:
        """Prune decayed memories. Returns number pruned."""
        resp = await self._post(
            f"/v1/agents/{agent_id}/memories/{user_id}/consolidate",
            params={"prune_threshold": prune_threshold},
        )
        return int(resp.json()["pruned"])

    # -- Sessions -------------------------------------------------------------

    async def open_session(
        self,
        agent_id: str,
        user_id: str,
        *,
        context: str | None = None,
    ) -> str:
        """Open a new MemoryLoop session. Returns the session_id."""
        resp = await self._post(
            f"/v1/agents/{agent_id}/sessions",
            json={"user_id": user_id, "context": context},
        )
        return str(resp.json()["session_id"])

    async def session_policy(
        self,
        agent_id: str,
        session_id: str,
        *,
        context: str | None = None,
        scopes: list[str] | None = None,
        existing_instructions: str | None = None,
        max_input_tokens: int = 8000,
        max_output_tokens: int = 3000,
    ) -> PolicyResult:
        """Compile a policy within an open session."""
        resp = await self._post(
            f"/v1/agents/{agent_id}/sessions/{session_id}/policy",
            json={
                "context": context,
                "scopes": scopes,
                "existing_instructions": existing_instructions,
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
            },
        )
        data = resp.json()
        return PolicyResult(
            text=data["policy_text"],
            memory_count=data["memory_count"],
            dropped_count=data["dropped_count"],
            compiled_at=data["compiled_at"],
        )

    async def session_observe(
        self,
        agent_id: str,
        session_id: str,
        *,
        agent_output: str,
        user_response: str,
        scope: str | None = None,
    ) -> None:
        """Record a turn within an open session."""
        await self._post(
            f"/v1/agents/{agent_id}/sessions/{session_id}/observe",
            json={
                "agent_output": agent_output,
                "user_response": user_response,
                "scope": scope,
            },
        )

    async def close_session(
        self,
        agent_id: str,
        session_id: str,
        *,
        outcome: float | None = None,
        correction: str | None = None,
    ) -> None:
        """Close a session and apply the learning signal."""
        await self._post(
            f"/v1/agents/{agent_id}/sessions/{session_id}/close",
            json={"outcome": outcome, "correction": correction},
        )

    # -- Server health --------------------------------------------------------

    async def health(self) -> ServerHealth:
        """Return server health status."""
        resp = await self._get("/health")
        data = resp.json()
        return ServerHealth(
            status=data["status"],
            store=data["store"],
            agents_loaded=data["agents_loaded"],
            db_ok=data["db_ok"],
        )

    # -- HTTP internals -------------------------------------------------------

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        return await self._request("GET", path, params=params)

    async def _post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        return await self._request("POST", path, json=json, params=params)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                await asyncio.sleep(self._retry_backoff * (2 ** (attempt - 1)))
            try:
                response = await self._http.request(method, path, json=json, params=params)
            except httpx.TransportError as exc:
                last_exc = exc
                continue

            if response.status_code >= 500 and attempt < self._max_retries:
                last_exc = ImprintError(response.status_code, response.text)
                continue

            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                raise ImprintError(response.status_code, detail)

            return response

        if isinstance(last_exc, ImprintError):
            raise last_exc
        raise ImprintConnectionError(
            f"Failed to reach imprint-server at {self._base_url} "
            f"after {self._max_retries + 1} attempts: {last_exc}"
        ) from last_exc


# -- AgentClient --------------------------------------------------------------


class AgentClient:
    """ImprintClient pre-scoped to one agent ID.

    Avoids repeating agent_id on every call:

        agent = client.agent("my-agent")
        policy = await agent.get_policy("user-1")
        await agent.observe("user-1", agent_output="...", user_response="...")
    """

    def __init__(self, client: ImprintClient, agent_id: str) -> None:
        self._client = client
        self._agent_id = agent_id

    def session(self, user_id: str, *, context: str | None = None) -> Session:
        """Return a session context manager for this agent."""
        return self._client.session(self._agent_id, user_id, context=context)

    async def observe(
        self,
        user_id: str,
        *,
        agent_output: str,
        user_response: str,
        context: str | None = None,
        scope: str | None = None,
    ) -> None:
        await self._client.observe(
            self._agent_id,
            user_id,
            agent_output=agent_output,
            user_response=user_response,
            context=context,
            scope=scope,
        )

    async def observe_directions(
        self,
        user_id: str,
        directions: list[str],
        *,
        context: str | None = None,
        scope: str | None = None,
    ) -> int:
        return await self._client.observe_directions(
            self._agent_id, user_id, directions, context=context, scope=scope
        )

    async def get_policy(
        self,
        user_id: str,
        *,
        context: str | None = None,
        scopes: list[str] | None = None,
        existing_instructions: str | None = None,
    ) -> PolicyResult:
        return await self._client.get_policy(
            self._agent_id,
            user_id,
            context=context,
            scopes=scopes,
            existing_instructions=existing_instructions,
        )

    async def list_memories(
        self, user_id: str, *, scopes: list[str] | None = None
    ) -> list[MemoryRecord]:
        return await self._client.list_memories(self._agent_id, user_id, scopes=scopes)

    async def memory_health(self, user_id: str) -> MemoryHealth:
        return await self._client.memory_health(self._agent_id, user_id)

    async def forget(self, user_id: str) -> None:
        await self._client.forget(self._agent_id, user_id)

    async def consolidate(self, user_id: str, *, prune_threshold: float = 0.5) -> int:
        return await self._client.consolidate(
            self._agent_id, user_id, prune_threshold=prune_threshold
        )


# -- Session ------------------------------------------------------------------


class Session:
    """Async context manager for a MemoryLoop session over HTTP.

    Tracks which memories were retrieved and at what alpha weight, then
    applies a learning signal when the session closes. Equivalent to the
    library's MemoryLoop context manager but over HTTP.

    Usage:
        async with client.session("my-agent", "user-1", context="coding") as sess:
            policy = await sess.get_policy()
            await sess.observe("output", "response")
            # Set outcome before exiting:
            sess.set_outcome(0.9)
        # close() is called automatically on exit.

    Or close explicitly:
        policy = await sess.get_policy()
        await sess.observe(...)
        await sess.close(outcome=0.9, correction="Was too verbose.")
    """

    def __init__(
        self,
        client: ImprintClient,
        *,
        agent_id: str,
        user_id: str,
        context: str | None,
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._user_id = user_id
        self._context = context
        self._session_id: str | None = None
        self._outcome: float | None = None
        self._correction: str | None = None
        self._closed = False

    @property
    def session_id(self) -> str:
        """The server-assigned session ID. Available after __aenter__."""
        if self._session_id is None:
            raise RuntimeError("Session is not open. Use 'async with client.session(...)'.")
        return self._session_id

    def set_outcome(
        self,
        outcome: float,
        *,
        correction: str | None = None,
    ) -> None:
        """Set the outcome score (0.0-1.0) to send when the session closes.

        Call this before the context manager exits, or pass outcome to
        close() explicitly.
        """
        self._outcome = outcome
        self._correction = correction

    async def __aenter__(self) -> Session:
        self._session_id = await self._client.open_session(
            self._agent_id, self._user_id, context=self._context
        )
        return self

    async def __aexit__(self, exc_type: object, *_: object) -> None:
        if not self._closed and self._session_id is not None:
            # On exception, close without outcome so we don't spuriously
            # penalize the learning signal.
            outcome = None if exc_type is not None else self._outcome
            correction = None if exc_type is not None else self._correction
            await self.close(outcome=outcome, correction=correction)

    async def get_policy(
        self,
        *,
        context: str | None = None,
        scopes: list[str] | None = None,
        existing_instructions: str | None = None,
    ) -> PolicyResult:
        """Compile a policy within this session."""
        return await self._client.session_policy(
            self._agent_id,
            self.session_id,
            context=context or self._context,
            scopes=scopes,
            existing_instructions=existing_instructions,
        )

    async def observe(
        self,
        agent_output: str,
        user_response: str,
        *,
        scope: str | None = None,
    ) -> None:
        """Record a turn within this session."""
        await self._client.session_observe(
            self._agent_id,
            self.session_id,
            agent_output=agent_output,
            user_response=user_response,
            scope=scope,
        )

    async def direct(
        self,
        instruction: str,
        *,
        scope: str | None = None,
    ) -> int:
        """Store an explicit behavioral direction for this session's user.

        Note: directions are written to the user namespace (not scoped to the
        session). They persist beyond this session and appear in all future
        policy compilations for this user.
        """
        return await self._client.observe_directions(
            self._agent_id,
            self._user_id,
            [instruction],
            scope=scope,
        )

    async def close(
        self,
        *,
        outcome: float | None = None,
        correction: str | None = None,
    ) -> None:
        """Close the session and apply the learning signal."""
        if self._closed:
            return
        self._closed = True
        await self._client.close_session(
            self._agent_id,
            self.session_id,
            outcome=outcome,
            correction=correction,
        )


# -- Internal helpers ---------------------------------------------------------


def _memory_from_dict(data: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        id=data["id"],
        agent_id=data["agent_id"],
        user_id=data["user_id"],
        type=data["type"],
        scope=data.get("scope"),
        content=data["content"],
        source=data["source"],
        stability=float(data["stability"]),
        recall_count=int(data["recall_count"]),
        pinned=bool(data["pinned"]),
        active=bool(data["active"]),
        valid_from=data["valid_from"],
        valid_until=data.get("valid_until"),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )
