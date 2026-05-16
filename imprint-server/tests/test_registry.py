"""Tests for AgentRegistry.

Unit tests use a mocked MemoryStore so no live database is required.
Integration tests (marked live) use a real SQLite file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

# -- Fixtures -----------------------------------------------------------------


@pytest.fixture()
def sqlite_config(tmp_path: Path) -> ServerConfig:
    db_path = tmp_path / "registry_test.db"
    return ServerConfig(store=f"sqlite:///{db_path}")


@pytest.fixture()
def mock_store() -> MagicMock:
    """A mock MemoryStore with async methods."""
    store = MagicMock()
    store.connect = AsyncMock()
    store.close = AsyncMock()
    store.init_schema = AsyncMock()
    store.get_agent_config = AsyncMock(return_value=None)
    store.put_agent_config = AsyncMock()
    store.list_scopes = AsyncMock(return_value=[])
    store.insert_scope = AsyncMock()
    store.clear_scopes = AsyncMock()
    store.put_alpha_tuner_state = AsyncMock()
    store.put_gradient_state = AsyncMock()
    return store


@pytest.fixture()
def registry(sqlite_config: ServerConfig) -> AgentRegistry:
    return AgentRegistry(sqlite_config)


# -- store property -----------------------------------------------------------


def test_store_property_raises_before_startup(registry: AgentRegistry) -> None:
    with pytest.raises(RuntimeError, match="startup"):
        _ = registry.store


# -- startup / shutdown -------------------------------------------------------


@pytest.mark.live
async def test_startup_creates_store(sqlite_config: ServerConfig) -> None:
    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    try:
        assert reg.store is not None
    finally:
        await reg.shutdown()


@pytest.mark.live
async def test_shutdown_clears_store(sqlite_config: ServerConfig) -> None:
    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    await reg.shutdown()
    with pytest.raises(RuntimeError, match="startup"):
        _ = reg.store


@pytest.mark.live
async def test_startup_is_idempotent_schema(sqlite_config: ServerConfig) -> None:
    """Calling startup twice must not raise (all DDL uses IF NOT EXISTS)."""
    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    try:
        # Manually re-run schema steps -- should not error.
        from imprint_server.db import init_server_schema

        await reg.store.init_schema()
        await init_server_schema(sqlite_config, reg.store)
    finally:
        await reg.shutdown()


# -- get() / lazy init --------------------------------------------------------


@pytest.mark.live
async def test_get_returns_imprint_instance(sqlite_config: ServerConfig) -> None:
    from imprint import Imprint

    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    try:
        imp = await reg.get("agent-a")
        assert isinstance(imp, Imprint)
    finally:
        await reg.shutdown()


@pytest.mark.live
async def test_get_same_agent_returns_same_instance(sqlite_config: ServerConfig) -> None:
    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    try:
        imp1 = await reg.get("agent-a")
        imp2 = await reg.get("agent-a")
        assert imp1 is imp2
    finally:
        await reg.shutdown()


@pytest.mark.live
async def test_get_different_agents_different_instances(
    sqlite_config: ServerConfig,
) -> None:
    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    try:
        imp_a = await reg.get("agent-a")
        imp_b = await reg.get("agent-b")
        assert imp_a is not imp_b
        assert imp_a.agent_id == "agent-a"
        assert imp_b.agent_id == "agent-b"
    finally:
        await reg.shutdown()


@pytest.mark.live
async def test_concurrent_get_same_agent_initializes_once(
    sqlite_config: ServerConfig,
) -> None:
    """Two concurrent get() calls for the same agent must produce one instance."""
    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    try:
        results = await asyncio.gather(
            reg.get("agent-x"),
            reg.get("agent-x"),
            reg.get("agent-x"),
        )
        # All three must return the exact same object.
        assert results[0] is results[1]
        assert results[1] is results[2]
        assert reg.agent_count == 1
    finally:
        await reg.shutdown()


@pytest.mark.live
async def test_instance_count_grows_per_agent(sqlite_config: ServerConfig) -> None:
    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    try:
        await reg.get("agent-a")
        assert reg.agent_count == 1
        await reg.get("agent-b")
        assert reg.agent_count == 2
        await reg.get("agent-a")  # cached -- no new instance
        assert reg.agent_count == 2
    finally:
        await reg.shutdown()


@pytest.mark.live
async def test_agent_ids_sorted(sqlite_config: ServerConfig) -> None:
    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    try:
        await reg.get("zebra")
        await reg.get("apple")
        await reg.get("mango")
        assert reg.agent_ids() == ["apple", "mango", "zebra"]
    finally:
        await reg.shutdown()


# -- default_mode seeding -----------------------------------------------------


@pytest.mark.live
async def test_default_mode_seeded_for_new_agent(sqlite_config: ServerConfig) -> None:
    """A new agent must be initialized with config.default_mode, not 'balanced'."""
    config = ServerConfig(
        store=sqlite_config.store,
        default_mode="frugal",
    )
    reg = AgentRegistry(config)
    await reg.startup()
    try:
        imp = await reg.get("new-agent")
        assert imp.processing_mode == "frugal"
    finally:
        await reg.shutdown()


@pytest.mark.live
async def test_existing_agent_config_preserved(sqlite_config: ServerConfig) -> None:
    """An agent already in the DB keeps its DB config, not config.default_mode."""
    reg1 = AgentRegistry(ServerConfig(store=sqlite_config.store, default_mode="frugal"))
    await reg1.startup()
    try:
        imp1 = await reg1.get("my-agent")
        assert imp1.processing_mode == "frugal"
        # Manually update the DB config to eager.
        await reg1.store.put_agent_config(
            agent_id="my-agent",
            processing_mode="eager",
            agent_description=None,
            scopes=[],
        )
    finally:
        await reg1.shutdown()

    # Second registry startup -- default_mode is still "frugal" in config,
    # but the DB says "eager". DB must win.
    reg2 = AgentRegistry(ServerConfig(store=sqlite_config.store, default_mode="frugal"))
    await reg2.startup()
    try:
        imp2 = await reg2.get("my-agent")
        assert imp2.processing_mode == "eager"
    finally:
        await reg2.shutdown()


# -- get_op_lock() ------------------------------------------------------------


async def test_get_op_lock_returns_asyncio_lock(registry: AgentRegistry) -> None:
    lock = await registry.get_op_lock("agent-a")
    assert isinstance(lock, asyncio.Lock)


async def test_get_op_lock_same_agent_same_lock(registry: AgentRegistry) -> None:
    lock1 = await registry.get_op_lock("agent-a")
    lock2 = await registry.get_op_lock("agent-a")
    assert lock1 is lock2


async def test_get_op_lock_different_agents_different_locks(
    registry: AgentRegistry,
) -> None:
    lock_a = await registry.get_op_lock("agent-a")
    lock_b = await registry.get_op_lock("agent-b")
    assert lock_a is not lock_b


# -- reload_config() ----------------------------------------------------------


async def test_reload_config_noop_for_unknown_agent(registry: AgentRegistry) -> None:
    """reload_config for an agent that has not been initialized must not raise."""
    await registry.reload_config("does-not-exist")


@pytest.mark.live
async def test_reload_config_applies_db_update(sqlite_config: ServerConfig) -> None:
    """reload_config must re-read the DB and apply changes to the live instance."""
    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    try:
        imp = await reg.get("reload-agent")
        assert imp.processing_mode == "balanced"

        # Update the DB directly.
        await reg.store.put_agent_config(
            agent_id="reload-agent",
            processing_mode="frugal",
            agent_description=None,
            scopes=[],
        )

        # Reload -- instance must pick up the change.
        await reg.reload_config("reload-agent")
        assert imp.processing_mode == "frugal"
    finally:
        await reg.shutdown()


# -- drain_all() --------------------------------------------------------------


async def test_drain_all_empty_registry_does_not_raise(registry: AgentRegistry) -> None:
    await registry.drain_all()


@pytest.mark.live
async def test_drain_all_with_instances(sqlite_config: ServerConfig) -> None:
    reg = AgentRegistry(sqlite_config)
    await reg.startup()
    try:
        await reg.get("agent-a")
        await reg.get("agent-b")
        await reg.drain_all()  # must not raise
    finally:
        await reg.shutdown()
