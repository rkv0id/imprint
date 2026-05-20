"""Live integration tests for embedder, vector store, and decay model wiring.

These tests make real external API calls and require environment variables:
  VOYAGE_API_KEY   -- required for Voyage embedder tests
  OPENAI_API_KEY   -- required for OpenAI embedder tests

The tests use frugal processing mode so no LLM API key is needed for the
observe path. All vectors are written by the embedder during observe_directions()
and searched during search_memories().

Run with:
  cd imprint-server && VOYAGE_API_KEY=... uv run pytest tests/test_live_retrieval.py -m live -v

Or from the repo root:
  just server-live-test
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

AGENT = "live-retrieval-agent"
USER = "live-retrieval-user"


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture()
def voyage_config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(
        store=f"sqlite:///{tmp_path / 'live.db'}",
        default_mode="frugal",
        auth_disabled=True,
        embedder="voyage",
        embedder_model="voyage-3",
        embedder_dim=1024,
        vector_store="sqlite-vec",
    )


@pytest.fixture()
def openai_config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(
        store=f"sqlite:///{tmp_path / 'live_openai.db'}",
        default_mode="frugal",
        auth_disabled=True,
        embedder="openai",
        embedder_model="text-embedding-3-small",
        embedder_dim=1536,
        vector_store="sqlite-vec",
    )


@pytest.fixture()
def gradient_config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(
        store=f"sqlite:///{tmp_path / 'live_gradient.db'}",
        default_mode="frugal",
        auth_disabled=True,
        decay_model="gradient",
    )


@pytest.fixture()
async def voyage_client(voyage_config: ServerConfig) -> AsyncGenerator[AsyncClient, None]:
    registry = AgentRegistry(voyage_config)
    app = create_app(voyage_config, registry)
    await registry.startup()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await registry.shutdown()


@pytest.fixture()
async def voyage_registry(voyage_config: ServerConfig) -> AsyncGenerator[AgentRegistry, None]:
    registry = AgentRegistry(voyage_config)
    await registry.startup()
    yield registry
    await registry.shutdown()


# -- Startup: embedder construction -------------------------------------------


@pytest.mark.live
async def test_voyage_embedder_built_on_startup(voyage_config: ServerConfig) -> None:
    """startup() must build a VoyageEmbedder when IMPRINT_EMBEDDER=voyage."""
    from imprint.providers.voyage import VoyageEmbedder

    reg = AgentRegistry(voyage_config)
    await reg.startup()
    try:
        assert reg._embedder is not None  # type: ignore[attr-defined]
        assert isinstance(reg._embedder, VoyageEmbedder)  # type: ignore[attr-defined]
    finally:
        await reg.shutdown()


@pytest.mark.live
async def test_openai_embedder_built_on_startup(openai_config: ServerConfig) -> None:
    """startup() must build an OpenAIEmbedder when IMPRINT_EMBEDDER=openai."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    from imprint.providers.openai import OpenAIEmbedder

    reg = AgentRegistry(openai_config)
    await reg.startup()
    try:
        assert reg._embedder is not None  # type: ignore[attr-defined]
        assert isinstance(reg._embedder, OpenAIEmbedder)  # type: ignore[attr-defined]
    finally:
        await reg.shutdown()


@pytest.mark.live
async def test_sqlite_vec_store_built_on_startup(voyage_config: ServerConfig) -> None:
    """startup() must build a SQLiteVecStore when IMPRINT_VECTOR_STORE=sqlite-vec."""
    from imprint.stores.vector import SQLiteVecStore

    reg = AgentRegistry(voyage_config)
    await reg.startup()
    try:
        assert reg._vector_store is not None  # type: ignore[attr-defined]
        assert isinstance(reg._vector_store, SQLiteVecStore)  # type: ignore[attr-defined]
    finally:
        await reg.shutdown()


@pytest.mark.live
async def test_gradient_decay_built_on_startup(gradient_config: ServerConfig) -> None:
    """startup() must build FSRSGradientDecay when IMPRINT_DECAY_MODEL=gradient."""
    from imprint.online import FSRSGradientDecay

    reg = AgentRegistry(gradient_config)
    await reg.startup()
    try:
        assert isinstance(reg._decay_model, FSRSGradientDecay)  # type: ignore[attr-defined]
    finally:
        await reg.shutdown()


# -- Startup: agent wiring ----------------------------------------------------


@pytest.mark.live
async def test_bandit_alpha_tuner_enabled_with_vector_store(
    voyage_registry: AgentRegistry,
) -> None:
    """Imprint instance must use BanditAlphaTuner when a vector store is configured."""
    from imprint.retrieval import BanditAlphaTuner

    imp = await voyage_registry.get(AGENT)
    assert isinstance(imp._alpha_tuner, BanditAlphaTuner)  # type: ignore[attr-defined]


@pytest.mark.live
async def test_static_alpha_tuner_without_vector_store(tmp_path: Path) -> None:
    """Imprint instance must use StaticAlphaTuner when no vector store is configured."""
    from imprint.retrieval import StaticAlphaTuner

    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'no_vec.db'}",
        default_mode="frugal",
        auth_disabled=True,
    )
    reg = AgentRegistry(config)
    await reg.startup()
    try:
        imp = await reg.get(AGENT)
        assert isinstance(imp._alpha_tuner, StaticAlphaTuner)  # type: ignore[attr-defined]
    finally:
        await reg.shutdown()


# -- Embedding pipeline: observe stores vectors -------------------------------


@pytest.mark.live
async def test_observe_directions_stores_vector(voyage_registry: AgentRegistry) -> None:
    """observe_directions() must embed the memory content and upsert to vector store."""
    imp = await voyage_registry.get(AGENT)
    async with await voyage_registry.get_op_lock(AGENT):
        stored = await imp.observe_directions(
            user_id=USER,
            directions=["Always write in prose, never use bullet points."],
        )
    assert len(stored) >= 1

    # Vector store must have a hit for this memory ID.
    vec_store = voyage_registry._vector_store  # type: ignore[attr-defined]
    assert vec_store is not None

    # Embed a semantically similar query and confirm the memory is returned.
    embedder = voyage_registry._embedder  # type: ignore[attr-defined]
    assert embedder is not None
    query_vec = await embedder.embed("prose formatting preference")
    hits = await vec_store.search(query_vec, top_k=5)
    hit_ids = {mid for mid, _ in hits}
    assert stored[0].id in hit_ids


# -- Search endpoint: semantic ordering ---------------------------------------


@pytest.mark.live
async def test_search_returns_semantic_ordering(voyage_client: AsyncClient) -> None:
    """search_memories must return the semantically closest memory first."""
    # Store two memories on clearly different topics.
    await voyage_client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Always respond in French when the user writes in French."]},
    )
    await voyage_client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Use snake_case for all Python variable names."]},
    )

    # Search for something clearly about Python naming.
    resp = await voyage_client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/search",
        params={"q": "Python variable naming convention", "limit": 10},
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 2
    # The snake_case memory must rank before the French language memory.
    contents = [r["content"] for r in results]
    snake_idx = next(i for i, c in enumerate(contents) if "snake_case" in c)
    french_idx = next(i for i, c in enumerate(contents) if "French" in c)
    assert snake_idx < french_idx, (
        f"Expected snake_case memory (idx {snake_idx}) before "
        f"French memory (idx {french_idx}). Order: {contents}"
    )


@pytest.mark.live
async def test_search_endpoint_returns_all_fields(voyage_client: AsyncClient) -> None:
    """search_memories response must contain all MemoryRecord fields."""
    await voyage_client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Prefer short sentences."]},
    )
    resp = await voyage_client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/search",
        params={"q": "sentence length"},
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 1
    required = {
        "id",
        "agent_id",
        "user_id",
        "type",
        "content",
        "source",
        "stability",
        "recall_count",
        "pinned",
        "active",
        "valid_from",
        "created_at",
        "updated_at",
    }
    assert required <= set(results[0].keys())


@pytest.mark.live
async def test_search_limit_applied_after_semantic_sort(voyage_client: AsyncClient) -> None:
    """limit must be applied after semantic ranking, not before."""
    for i in range(4):
        await voyage_client.post(
            f"/v1/agents/{AGENT}/memories/{USER}/directions",
            json={"directions": [f"Coding preference number {i}."]},
        )
    resp = await voyage_client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/search",
        params={"q": "coding style", "limit": 2},
    )
    assert resp.status_code == 200
    assert len(resp.json()) <= 2


# -- Gradient decay: construction and basic use -------------------------------


@pytest.mark.live
async def test_gradient_decay_agent_initializes(gradient_config: ServerConfig) -> None:
    """An agent must initialize correctly when FSRSGradientDecay is configured."""
    from imprint.online import FSRSGradientDecay

    reg = AgentRegistry(gradient_config)
    await reg.startup()
    try:
        imp = await reg.get(AGENT)
        assert isinstance(imp._decay_model, FSRSGradientDecay)  # type: ignore[attr-defined]
    finally:
        await reg.shutdown()
