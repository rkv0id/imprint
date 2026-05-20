"""Tests for the imprint-server HTTP client.

Uses httpx.MockTransport to intercept requests without a real server.
Tests cover all public methods, retry logic, error handling, and the
Session context manager.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from imprint.client import (
    AgentClient,
    ImprintClient,
    ImprintConnectionError,
    ImprintError,
    MemoryHealth,
    MemoryRecord,
    PolicyResult,
    ServerHealth,
    Session,
)

AGENT = "test-agent"
USER = "test-user"
BASE_URL = "http://test.imprint"

# -- Mock transport helpers ---------------------------------------------------


def _response(body: Any, *, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        headers={"Content-Type": "application/json"},
        content=json.dumps(body).encode(),
    )


def _ok() -> httpx.Response:
    return _response({"ok": True})


class _QueuedTransport(httpx.AsyncBaseTransport):
    """Replay a fixed sequence of responses, then raise if exhausted."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self._index = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._index >= len(self._responses):
            raise AssertionError(
                f"Unexpected request {self._index + 1}: {request.method} {request.url}"
            )
        resp = self._responses[self._index]
        self._index += 1
        return resp


def _client(*responses: httpx.Response) -> ImprintClient:
    return ImprintClient(
        BASE_URL,
        transport=_QueuedTransport(*responses),
        max_retries=0,  # disable retry in most tests for speed
    )


# -- observe ------------------------------------------------------------------


async def test_observe_sends_correct_payload() -> None:
    received: list[dict[str, Any]] = []

    class _Capturing(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.content))
            return _ok()

    client = ImprintClient(BASE_URL, transport=_Capturing(), max_retries=0)
    await client.observe(AGENT, USER, agent_output="out", user_response="resp", scope="coding")

    assert received[0]["agent_output"] == "out"
    assert received[0]["user_response"] == "resp"
    assert received[0]["scope"] == "coding"
    assert received[0]["user_id"] == USER


async def test_observe_directions_returns_stored_count() -> None:
    client = _client(_response({"stored": 2}))
    count = await client.observe_directions(AGENT, USER, ["dir 1", "dir 2"])
    assert count == 2


# -- get_policy ---------------------------------------------------------------


async def test_get_policy_returns_policy_result() -> None:
    client = _client(
        _response(
            {
                "policy_text": "always use prose",
                "memory_count": 3,
                "dropped_count": 1,
                "compiled_at": "2025-01-01T00:00:00+00:00",
            }
        )
    )
    pol = await client.get_policy(AGENT, USER)
    assert isinstance(pol, PolicyResult)
    assert pol.text == "always use prose"
    assert pol.memory_count == 3
    assert pol.dropped_count == 1
    assert pol.has_memories is True


async def test_get_policy_empty_has_no_memories() -> None:
    client = _client(
        _response(
            {
                "policy_text": "",
                "memory_count": 0,
                "dropped_count": 0,
                "compiled_at": "2025-01-01T00:00:00+00:00",
            }
        )
    )
    pol = await client.get_policy(AGENT, USER)
    assert pol.has_memories is False
    assert pol.text == ""


# -- list_memories / health ---------------------------------------------------


async def test_list_memories_returns_records() -> None:
    memory_dict = {
        "id": "m_001",
        "agent_id": AGENT,
        "user_id": USER,
        "type": "preference",
        "scope": "coding",
        "content": "no bullet points",
        "source": "signal",
        "stability": 0.8,
        "recall_count": 2,
        "pinned": False,
        "active": True,
        "valid_from": "2025-01-01T00:00:00+00:00",
        "valid_until": None,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }
    client = _client(_response([memory_dict]))
    records = await client.list_memories(AGENT, USER)
    assert len(records) == 1
    assert isinstance(records[0], MemoryRecord)
    assert records[0].content == "no bullet points"
    assert records[0].scope == "coding"


async def test_memory_health_returns_health_object() -> None:
    client = _client(
        _response(
            {
                "total": 5,
                "active": 4,
                "pinned": 1,
                "by_scope": {"coding": 3, "writing": 1},
                "by_type": {"preference": 4},
                "avg_recall_count": 2.5,
                "oldest_active": "2025-01-01T00:00:00+00:00",
                "newest_active": "2025-01-02T00:00:00+00:00",
            }
        )
    )
    h = await client.memory_health(AGENT, USER)
    assert isinstance(h, MemoryHealth)
    assert h.total == 5
    assert h.active == 4
    assert h.by_scope["coding"] == 3


async def test_forget_sends_delete_request() -> None:
    methods: list[str] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            methods.append(request.method)
            return _ok()

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.forget(AGENT, USER)
    assert methods == ["DELETE"]


async def test_consolidate_returns_pruned_count() -> None:
    client = _client(_response({"pruned": 3}))
    pruned = await client.consolidate(AGENT, USER)
    assert pruned == 3


# -- server health ------------------------------------------------------------


async def test_health_returns_server_health() -> None:
    client = _client(
        _response(
            {
                "status": "ok",
                "store": "sqlite",
                "agents_loaded": 2,
                "db_ok": True,
            }
        )
    )
    h = await client.health()
    assert isinstance(h, ServerHealth)
    assert h.ok is True
    assert h.store == "sqlite"
    assert h.agents_loaded == 2


async def test_health_degraded_not_ok() -> None:
    client = _client(
        _response(
            {
                "status": "degraded",
                "store": "postgres",
                "agents_loaded": 0,
                "db_ok": False,
            }
        )
    )
    h = await client.health()
    assert h.ok is False


# -- session endpoints --------------------------------------------------------


async def test_open_session_returns_session_id() -> None:
    client = _client(_response({"session_id": "sess_abc123"}))
    sid = await client.open_session(AGENT, USER, context="coding")
    assert sid == "sess_abc123"


async def test_session_policy_returns_policy_result() -> None:
    client = _client(
        _response(
            {
                "policy_text": "be concise",
                "memory_count": 1,
                "dropped_count": 0,
                "compiled_at": "2025-01-01T00:00:00+00:00",
            }
        )
    )
    pol = await client.session_policy(AGENT, "sess_001")
    assert pol.text == "be concise"


async def test_session_observe_succeeds() -> None:
    client = _client(_ok())
    await client.session_observe(AGENT, "sess_001", agent_output="x", user_response="y")


async def test_close_session_succeeds() -> None:
    client = _client(_ok())
    await client.close_session(AGENT, "sess_001", outcome=0.9)


# -- error handling -----------------------------------------------------------


async def test_4xx_raises_imprint_error() -> None:
    client = _client(_response({"detail": "session not found"}, status=404))
    with pytest.raises(ImprintError) as exc_info:
        await client.get_policy(AGENT, USER)
    assert exc_info.value.status_code == 404
    assert "session not found" in exc_info.value.detail


async def test_401_raises_imprint_error() -> None:
    client = _client(_response({"detail": "unauthorized"}, status=401))
    with pytest.raises(ImprintError) as exc_info:
        await client.get_policy(AGENT, USER)
    assert exc_info.value.status_code == 401


async def test_transport_error_raises_connection_error() -> None:
    class _Failing(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

    client = ImprintClient(BASE_URL, transport=_Failing(), max_retries=0)
    with pytest.raises(ImprintConnectionError):
        await client.get_policy(AGENT, USER)


# -- retry logic --------------------------------------------------------------


async def test_retries_on_5xx_and_succeeds() -> None:
    client = ImprintClient(
        BASE_URL,
        transport=_QueuedTransport(
            _response({"error": "overloaded"}, status=503),
            _response(
                {
                    "policy_text": "retried ok",
                    "memory_count": 0,
                    "dropped_count": 0,
                    "compiled_at": "2025-01-01T00:00:00+00:00",
                }
            ),
        ),
        max_retries=2,
        retry_backoff=0.0,
    )
    pol = await client.get_policy(AGENT, USER)
    assert pol.text == "retried ok"


async def test_exhausted_retries_raises_imprint_error() -> None:
    client = ImprintClient(
        BASE_URL,
        transport=_QueuedTransport(
            _response({"error": "overloaded"}, status=503),
            _response({"error": "overloaded"}, status=503),
        ),
        max_retries=1,
        retry_backoff=0.0,
    )
    with pytest.raises(ImprintError) as exc_info:
        await client.get_policy(AGENT, USER)
    assert exc_info.value.status_code == 503


async def test_no_retry_on_4xx() -> None:
    """4xx errors must not be retried -- they are client errors."""
    call_count = 0

    class _Counting(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _response({"detail": "not found"}, status=404)

    client = ImprintClient(BASE_URL, transport=_Counting(), max_retries=3)
    with pytest.raises(ImprintError):
        await client.get_policy(AGENT, USER)
    assert call_count == 1


async def test_transport_error_retried() -> None:
    call_count = 0

    class _FailThenOk(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("refused")
            return _response(
                {
                    "policy_text": "",
                    "memory_count": 0,
                    "dropped_count": 0,
                    "compiled_at": "2025-01-01T00:00:00+00:00",
                }
            )

    client = ImprintClient(BASE_URL, transport=_FailThenOk(), max_retries=2, retry_backoff=0.0)
    pol = await client.get_policy(AGENT, USER)
    assert call_count == 2
    assert pol.text == ""


# -- AgentClient --------------------------------------------------------------


async def test_agent_client_scopes_agent_id() -> None:
    received_paths: list[str] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received_paths.append(str(request.url.path))
            return _response(
                {
                    "policy_text": "",
                    "memory_count": 0,
                    "dropped_count": 0,
                    "compiled_at": "2025-01-01T00:00:00+00:00",
                }
            )

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    agent = client.agent("scoped-agent")
    assert isinstance(agent, AgentClient)
    await agent.get_policy(USER)
    assert "/scoped-agent/" in received_paths[0]


async def test_agent_client_session_returns_session() -> None:
    client = ImprintClient(
        BASE_URL,
        transport=_QueuedTransport(
            _response({"session_id": "sess_scoped"}),
            _ok(),
        ),
        max_retries=0,
    )
    agent = client.agent(AGENT)
    sess = agent.session(USER, context="test")
    assert isinstance(sess, Session)


# -- Session context manager --------------------------------------------------


async def test_session_context_manager_lifecycle() -> None:
    """open -> get_policy -> observe -> close with outcome."""
    client = ImprintClient(
        BASE_URL,
        transport=_QueuedTransport(
            _response({"session_id": "sess_xyz"}),  # open
            _response(
                {  # get_policy
                    "policy_text": "be concise",
                    "memory_count": 1,
                    "dropped_count": 0,
                    "compiled_at": "2025-01-01T00:00:00+00:00",
                }
            ),
            _ok(),  # observe
            _ok(),  # close
        ),
        max_retries=0,
    )
    async with client.session(AGENT, USER, context="coding") as sess:
        assert sess.session_id == "sess_xyz"
        pol = await sess.get_policy()
        assert pol.text == "be concise"
        await sess.observe("output", "response")
        sess.set_outcome(0.9)


async def test_session_auto_closes_on_exit() -> None:
    close_called = False
    close_payload: dict[str, Any] = {}

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal close_called, close_payload
            if "close" in str(request.url):
                close_called = True
                close_payload = json.loads(request.content)
            path = str(request.url.path)
            is_open = (
                "sessions" in path
                and "close" not in path
                and "policy" not in path
                and "observe" not in path
            )
            return _response({"session_id": "sess_auto"}) if is_open else _ok()

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    async with client.session(AGENT, USER) as sess:
        sess.set_outcome(0.8)

    assert close_called
    assert abs(close_payload.get("outcome", 0) - 0.8) < 0.001


async def test_session_closes_without_outcome_on_exception() -> None:
    """On exception, close must be called with outcome=None."""
    close_payloads: list[dict[str, Any]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if "close" in str(request.url):
                close_payloads.append(json.loads(request.content))
            if "sessions" in str(request.url.path) and "close" not in str(request.url.path):
                return _response({"session_id": "sess_exc"})
            return _ok()

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    with pytest.raises(ValueError):
        async with client.session(AGENT, USER) as sess:
            sess.set_outcome(1.0)
            raise ValueError("something went wrong")

    assert len(close_payloads) == 1
    assert close_payloads[0]["outcome"] is None


async def test_session_id_raises_before_open() -> None:
    client = ImprintClient(BASE_URL, transport=_QueuedTransport(), max_retries=0)
    sess = client.session(AGENT, USER)
    with pytest.raises(RuntimeError, match="not open"):
        _ = sess.session_id


async def test_session_close_is_idempotent() -> None:
    """Calling close() twice must not send two requests."""
    close_count = 0

    class _Counting(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal close_count
            if "close" in str(request.url):
                close_count += 1
            if "close" in str(request.url):
                return _ok()
            return _response({"session_id": "sess_idem"})

    client = ImprintClient(BASE_URL, transport=_Counting(), max_retries=0)
    async with client.session(AGENT, USER) as sess:
        await sess.close()  # explicit close
    # __aexit__ should not send another close

    assert close_count == 1


# -- auth header --------------------------------------------------------------


async def test_api_key_sent_as_bearer_header() -> None:
    received_headers: list[dict[str, str]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received_headers.append(dict(request.headers))
            return _ok()

    client = ImprintClient(BASE_URL, api_key="sk-imp-abc", transport=_Spy(), max_retries=0)
    await client.forget(AGENT, USER)
    assert received_headers[0]["authorization"] == "Bearer sk-imp-abc"


async def test_no_auth_header_without_key() -> None:
    received_headers: list[dict[str, str]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received_headers.append(dict(request.headers))
            return _ok()

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.forget(AGENT, USER)
    assert "authorization" not in received_headers[0]


# -- context manager ----------------------------------------------------------


async def test_client_context_manager_closes_http_client() -> None:
    async with ImprintClient(
        BASE_URL,
        transport=_QueuedTransport(),
        max_retries=0,
    ) as client:
        assert not client._http.is_closed
    assert client._http.is_closed


# -- AgentClient delegation ---------------------------------------------------


async def test_agent_client_observe_delegates_with_agent_id() -> None:
    """AgentClient.observe must inject the pre-scoped agent_id."""
    received: list[dict[str, Any]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received.append(
                {
                    "path": str(request.url.path),
                    "body": json.loads(request.content),
                }
            )
            return _ok()

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    agent = client.agent("delegate-agent")
    await agent.observe(USER, agent_output="x", user_response="y", scope="test")

    assert "/delegate-agent/" in received[0]["path"]
    assert received[0]["body"]["scope"] == "test"
    assert received[0]["body"]["user_id"] == USER


async def test_agent_client_get_policy_delegates() -> None:
    client = ImprintClient(
        BASE_URL,
        transport=_QueuedTransport(
            _response(
                {
                    "policy_text": "delegated",
                    "memory_count": 0,
                    "dropped_count": 0,
                    "compiled_at": "2025-01-01T00:00:00+00:00",
                }
            )
        ),
        max_retries=0,
    )
    pol = await client.agent("agent-x").get_policy(USER)
    assert pol.text == "delegated"


# -- search_memories ----------------------------------------------------------


async def test_search_memories_returns_records() -> None:
    memory_dict = {
        "id": "m_search_01",
        "agent_id": AGENT,
        "user_id": USER,
        "type": "preference",
        "scope": None,
        "content": "write in prose",
        "source": "signal",
        "stability": 0.9,
        "recall_count": 1,
        "pinned": False,
        "active": True,
        "valid_from": "2025-01-01T00:00:00+00:00",
        "valid_until": None,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }
    client = _client(_response([memory_dict]))
    records = await client.search_memories(AGENT, USER, "prose style")
    assert len(records) == 1
    assert isinstance(records[0], MemoryRecord)
    assert records[0].content == "write in prose"


async def test_search_memories_sends_q_and_limit_params() -> None:
    received: list[dict[str, str]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received.append(dict(request.url.params))
            return _response([])

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.search_memories(AGENT, USER, "bullet points", limit=5)

    assert len(received) == 1
    assert received[0]["q"] == "bullet points"
    assert received[0]["limit"] == "5"


async def test_search_memories_default_limit_is_20() -> None:
    received: list[dict[str, str]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received.append(dict(request.url.params))
            return _response([])

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.search_memories(AGENT, USER, "some query")
    assert received[0]["limit"] == "20"


async def test_search_memories_hits_correct_path() -> None:
    paths: list[str] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            paths.append(str(request.url.path))
            return _response([])

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.search_memories(AGENT, USER, "query")
    assert paths[0] == f"/v1/agents/{AGENT}/memories/{USER}/search"


async def test_agent_client_search_memories_delegates() -> None:
    received: list[dict[str, Any]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received.append({"path": str(request.url.path), "params": dict(request.url.params)})
            return _response([])

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    agent = client.agent("search-agent")
    await agent.search_memories(USER, "my query", limit=10)

    assert "/search-agent/" in received[0]["path"]
    assert "/search" in received[0]["path"]
    assert received[0]["params"]["q"] == "my query"
    assert received[0]["params"]["limit"] == "10"


# -- pin_memory / deactivate_memory -------------------------------------------


async def test_pin_memory_hits_correct_path() -> None:
    paths: list[str] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            paths.append(str(request.url.path))
            return _ok()

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.pin_memory(AGENT, "mem_abc")
    assert paths[0] == f"/v1/agents/{AGENT}/memories/mem_abc/pin"


async def test_deactivate_memory_returns_true_on_200() -> None:
    client = _client(_ok())
    result = await client.deactivate_memory(AGENT, USER, "mem_abc")
    assert result is True


async def test_deactivate_memory_returns_false_on_404() -> None:
    client = _client(_response({"detail": "not found"}, status=404))
    result = await client.deactivate_memory(AGENT, USER, "mem_gone")
    assert result is False


async def test_deactivate_memory_raises_on_500() -> None:
    client = ImprintClient(
        BASE_URL,
        transport=_QueuedTransport(_response({"detail": "server error"}, status=500)),
        max_retries=0,
    )
    with pytest.raises(ImprintError) as exc_info:
        await client.deactivate_memory(AGENT, USER, "mem_x")
    assert exc_info.value.status_code == 500


async def test_deactivate_memory_sends_delete_method() -> None:
    methods: list[str] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            methods.append(request.method)
            return _ok()

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.deactivate_memory(AGENT, USER, "mem_abc")
    assert methods == ["DELETE"]


# -- correct / reinforce ------------------------------------------------------


async def test_correct_returns_memory_id() -> None:
    client = _client(_response({"ok": True, "memory_id": "mem_new_01"}))
    result = await client.correct(AGENT, USER, "No bullet points.")
    assert result == "mem_new_01"


async def test_correct_returns_none_when_memory_id_absent() -> None:
    client = _client(_response({"ok": True, "memory_id": None}))
    result = await client.correct(AGENT, USER, "content")
    assert result is None


async def test_correct_sends_content_and_session_id() -> None:
    received: list[dict[str, Any]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.content))
            return _response({"ok": True, "memory_id": "m1"})

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.correct(AGENT, USER, "Too verbose.", session_id="sess_abc")

    assert received[0]["content"] == "Too verbose."
    assert received[0]["session_id"] == "sess_abc"


async def test_correct_hits_correct_path() -> None:
    paths: list[str] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            paths.append(str(request.url.path))
            return _response({"ok": True, "memory_id": "m1"})

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.correct(AGENT, USER, "content")
    assert paths[0] == f"/v1/agents/{AGENT}/correct/{USER}"


async def test_reinforce_returns_true_when_applied() -> None:
    client = _client(_response({"ok": True, "applied": True}))
    result = await client.reinforce(AGENT, USER, session_id="sess_xyz")
    assert result is True


async def test_reinforce_returns_false_when_not_applied() -> None:
    client = _client(_response({"ok": True, "applied": False}))
    result = await client.reinforce(AGENT, USER)
    assert result is False


async def test_reinforce_sends_session_id() -> None:
    received: list[dict[str, Any]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.content))
            return _response({"ok": True, "applied": True})

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.reinforce(AGENT, USER, session_id="sess_reinforce")
    assert received[0]["session_id"] == "sess_reinforce"


async def test_reinforce_hits_correct_path() -> None:
    paths: list[str] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            paths.append(str(request.url.path))
            return _response({"ok": True, "applied": False})

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.reinforce(AGENT, USER)
    assert paths[0] == f"/v1/agents/{AGENT}/reinforce/{USER}"


# -- AgentClient delegation for new methods -----------------------------------


async def test_agent_client_pin_memory_delegates() -> None:
    paths: list[str] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            paths.append(str(request.url.path))
            return _ok()

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.agent("pin-agent").pin_memory("mem_xyz")
    assert "/pin-agent/" in paths[0]
    assert "/mem_xyz/pin" in paths[0]


async def test_agent_client_deactivate_memory_delegates() -> None:
    client = _client(_ok())
    result = await client.agent("del-agent").deactivate_memory(USER, "mem_del")
    assert result is True


async def test_agent_client_correct_delegates() -> None:
    received: list[dict[str, Any]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received.append({"path": str(request.url.path), "body": json.loads(request.content)})
            return _response({"ok": True, "memory_id": "m1"})

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    await client.agent("corr-agent").correct(USER, "Be concise.", session_id="s1")
    assert "/corr-agent/correct/" in received[0]["path"]
    assert received[0]["body"]["content"] == "Be concise."
    assert received[0]["body"]["session_id"] == "s1"


async def test_agent_client_reinforce_delegates() -> None:
    received: list[dict[str, Any]] = []

    class _Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received.append({"path": str(request.url.path), "body": json.loads(request.content)})
            return _response({"ok": True, "applied": True})

    client = ImprintClient(BASE_URL, transport=_Spy(), max_retries=0)
    result = await client.agent("reinf-agent").reinforce(USER, session_id="s2")
    assert result is True
    assert "/reinf-agent/reinforce/" in received[0]["path"]
    assert received[0]["body"]["session_id"] == "s2"
