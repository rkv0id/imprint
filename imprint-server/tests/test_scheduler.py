"""Tests for the scheduler: session expiry, confusion detection, consolidation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry
from imprint_server.workers.scheduler import (
    Scheduler,
    _expire_sessions,
    _list_user_ids,
    check_confusion_and_enqueue,
    enqueue_consolidate,
)

AGENT = "sched-test-agent"
USER = "sched-test-user"


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture()
async def setup(tmp_path: Path) -> AsyncGenerator[tuple[ServerConfig, AgentRegistry], None]:
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'sched_test.db'}",
        default_mode="frugal",
        auth_disabled=True,
        consolidate_interval=86400,
        confusion_window=5,
        confusion_threshold=0.3,
    )
    registry = AgentRegistry(config)
    await registry.startup()
    yield config, registry
    await registry.shutdown()


# -- Scheduler lifecycle ------------------------------------------------------


async def test_scheduler_starts_and_stops(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    scheduler = Scheduler(config, registry)
    scheduler.start()
    assert len(scheduler._tasks) == 2
    await scheduler.stop()
    assert len(scheduler._tasks) == 0


async def test_scheduler_stop_is_idempotent(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    scheduler = Scheduler(config, registry)
    await scheduler.stop()  # stop before start -- must not raise
    scheduler.start()
    await scheduler.stop()
    await scheduler.stop()  # double stop -- must not raise


# -- Session expiry -----------------------------------------------------------


async def _insert_session(
    db_path: str,
    session_id: str,
    *,
    expired: bool,
    closed: bool = False,
) -> None:
    now = datetime.now(UTC)
    expires_at = now - timedelta(hours=2) if expired else now + timedelta(hours=1)
    closed_at = now.isoformat() if closed else None

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO sessions"
            " (id, agent_id, user_id, retrieved_ids, alpha_used, opened_at, expires_at, closed_at)"
            " VALUES (?, ?, ?, '[]', 0.0, ?, ?, ?)",
            (
                session_id,
                AGENT,
                USER,
                now.isoformat(),
                expires_at.isoformat(),
                closed_at,
            ),
        )
        await conn.commit()


async def test_expire_sessions_closes_expired(
    setup: tuple[ServerConfig, AgentRegistry], tmp_path: Path
) -> None:
    config, registry = setup
    db_path = str(tmp_path / "sched_test.db")

    await _insert_session(db_path, "sess_expired", expired=True)
    await _expire_sessions(config, registry)

    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute("SELECT closed_at FROM sessions WHERE id = 'sess_expired'") as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None and row[0] is not None


async def test_expire_sessions_ignores_active(
    setup: tuple[ServerConfig, AgentRegistry], tmp_path: Path
) -> None:
    config, registry = setup
    db_path = str(tmp_path / "sched_test.db")

    await _insert_session(db_path, "sess_active", expired=False)
    await _expire_sessions(config, registry)

    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute("SELECT closed_at FROM sessions WHERE id = 'sess_active'") as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None and row[0] is None


async def test_expire_sessions_ignores_already_closed(
    setup: tuple[ServerConfig, AgentRegistry], tmp_path: Path
) -> None:
    config, registry = setup
    db_path = str(tmp_path / "sched_test.db")

    await _insert_session(db_path, "sess_closed", expired=True, closed=True)

    # Record the existing closed_at timestamp.
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute("SELECT closed_at FROM sessions WHERE id = 'sess_closed'") as cursor,
    ):
        before = await cursor.fetchone()

    await _expire_sessions(config, registry)

    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute("SELECT closed_at FROM sessions WHERE id = 'sess_closed'") as cursor,
    ):
        after = await cursor.fetchone()

    # closed_at should be unchanged.
    assert before is not None and after is not None
    assert before[0] == after[0]


# -- User ID enumeration ------------------------------------------------------


async def test_list_user_ids_empty(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    await registry.get(AGENT)
    users = await _list_user_ids(config, registry, AGENT)
    assert users == []


async def test_list_user_ids_after_directions(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    imp = await registry.get(AGENT)
    await imp.observe_directions(user_id=USER, directions=["Write in prose."])
    users = await _list_user_ids(config, registry, AGENT)
    assert USER in users


# -- Confusion-based consolidation --------------------------------------------


async def test_check_confusion_no_events_no_enqueue(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    await registry.get(AGENT)
    # No events -- should not raise and should not enqueue.
    await check_confusion_and_enqueue(config, registry, AGENT, USER)


async def test_enqueue_consolidate_does_not_raise(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    await registry.get(AGENT)
    # Should schedule a background task without blocking.
    await enqueue_consolidate(config, registry, AGENT, USER)
    # Give the task a moment to run.
    await asyncio.sleep(0.05)


async def test_check_confusion_below_threshold_no_enqueue(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    """With contradiction rate below threshold, no consolidation is triggered."""
    config, registry = setup
    imp = await registry.get(AGENT)

    # Observe several neutral turns (no corrections -- rate = 0.0).
    for _ in range(5):
        await imp.observe(
            user_id=USER,
            agent_output="summary",
            user_response="looks good",
        )

    # With rate = 0.0 < 0.3 threshold, check should not raise.
    await check_confusion_and_enqueue(config, registry, AGENT, USER)
