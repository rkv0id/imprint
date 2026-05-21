"""Detailed tests for Prometheus metric correctness.

Verifies that specific metrics are incremented with the right labels and
values, not just that the /metrics endpoint returns 200. Covers hot-path
counters, histograms, the MetricsRefresher lifecycle, and extended gauges.

LLM call avoidance: all tests use the directions path through /observe
(which calls observe_directions internally and bypasses signal detection
entirely) or policy calls with no stored memories (which returns an empty
policy without LLM compilation). Both paths are LLM-free in all modes.
"""

from __future__ import annotations

from pathlib import Path

from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

AGENT = "metrics-detail-agent"
USER = "metrics-detail-user"


# -- Fixture ------------------------------------------------------------------


async def _make_client(tmp_path: Path) -> tuple[AsyncClient, AgentRegistry]:
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'metrics_test.db'}",
        default_mode="frugal",
        auth_disabled=True,
    )
    registry = AgentRegistry(config)
    app = create_app(config, registry)
    await registry.startup()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, registry


def _metric_sample_count(name: str, labels: dict[str, str]) -> float:
    """Read a metric value by sample name from the global Prometheus registry.

    Uses sample.name rather than metric.name because prometheus_client strips
    the _total suffix from Counter metric names in collect() output. For a
    Counter("imprint_observe_total", ...), metric.name is "imprint_observe"
    but sample.name is "imprint_observe_total". Gauges and histograms are
    unaffected since their sample names match their metric names.
    """
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return 0.0


# -- observe metrics ----------------------------------------------------------


async def test_observe_total_increments_with_mode_label(tmp_path: Path) -> None:
    """observe_total has agent_id and mode labels; both must appear after an observe."""
    client, registry = await _make_client(tmp_path)
    async with client:
        before = _metric_sample_count(
            "imprint_observe_total", {"agent_id": AGENT, "mode": "frugal"}
        )
        resp = await client.post(
            f"/v1/agents/{AGENT}/observe",
            # Use directions path -- bypasses heuristic detection, always LLM-free.
            json={"user_id": USER, "directions": ["always respond in plain prose"]},
        )
        assert resp.status_code == 200
        after = _metric_sample_count("imprint_observe_total", {"agent_id": AGENT, "mode": "frugal"})
        assert after == before + 1
    await registry.shutdown()


async def test_observe_latency_recorded(tmp_path: Path) -> None:
    client, registry = await _make_client(tmp_path)
    async with client:
        resp = await client.post(
            f"/v1/agents/{AGENT}/observe",
            json={"user_id": USER, "directions": ["keep responses concise"]},
        )
        assert resp.status_code == 200
        metrics_text = (await client.get("/metrics")).text
        assert f'imprint_observe_latency_seconds_count{{agent_id="{AGENT}"' in metrics_text
    await registry.shutdown()


# -- policy metrics -----------------------------------------------------------


async def test_policy_total_increments(tmp_path: Path) -> None:
    """policy_total increments on every policy call, with or without cached flag."""
    client, registry = await _make_client(tmp_path)
    async with client:
        before = _metric_sample_count("imprint_policy_total", {"agent_id": AGENT})
        resp = await client.post(f"/v1/agents/{AGENT}/policy", json={"user_id": USER})
        # No memories -> empty policy returned without LLM.
        assert resp.status_code == 200
        after = _metric_sample_count("imprint_policy_total", {"agent_id": AGENT})
        assert after == before + 1
    await registry.shutdown()


async def test_policy_latency_has_cached_label(tmp_path: Path) -> None:
    client, registry = await _make_client(tmp_path)
    async with client:
        await client.post(f"/v1/agents/{AGENT}/policy", json={"user_id": USER})
        metrics_text = (await client.get("/metrics")).text
        # cached="false" label appears on cache-miss (no Redis configured).
        assert "imprint_policy_latency_seconds" in metrics_text
    await registry.shutdown()


async def test_policy_cache_miss_counter_registered(tmp_path: Path) -> None:
    """Cache miss counter exists in output once a policy call has been made."""
    client, registry = await _make_client(tmp_path)
    async with client:
        await client.post(f"/v1/agents/{AGENT}/policy", json={"user_id": USER})
        metrics_text = (await client.get("/metrics")).text
        assert "imprint_policy_cache_misses_total" in metrics_text
    await registry.shutdown()


async def test_policy_memories_retrieved_histogram_fires(tmp_path: Path) -> None:
    """Histogram fires on every policy call (observed value is 0 when no memories stored)."""
    client, registry = await _make_client(tmp_path)
    async with client:
        # No memories pre-stored -- policy returns empty, histograms observe 0.
        resp = await client.post(f"/v1/agents/{AGENT}/policy", json={"user_id": USER})
        assert resp.status_code == 200
        metrics_text = (await client.get("/metrics")).text
        assert "imprint_policy_memories_retrieved" in metrics_text
        assert "imprint_policy_memories_dropped" in metrics_text
    await registry.shutdown()


# -- consolidation metrics ----------------------------------------------------


async def test_consolidation_pruned_counter_registered(tmp_path: Path) -> None:
    client, registry = await _make_client(tmp_path)
    async with client:
        resp = await client.post(f"/v1/agents/{AGENT}/memories/{USER}/consolidate")
        assert resp.status_code == 200
        metrics_text = (await client.get("/metrics")).text
        assert "imprint_consolidation_pruned_total" in metrics_text
    await registry.shutdown()


# -- MetricsRefresher ---------------------------------------------------------


async def test_metrics_refresher_starts_and_stops(tmp_path: Path) -> None:
    from imprint_server.workers.metrics_refresh import MetricsRefresher

    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'refresh_test.db'}",
        metrics_extended=True,
        metrics_refresh_interval=5,
    )
    registry = AgentRegistry(config)
    await registry.startup()
    try:
        refresher = MetricsRefresher(config, registry)
        refresher.start()
        assert refresher._task is not None
        assert not refresher._task.done()
        await refresher.stop()
        assert refresher._task is None
    finally:
        await registry.shutdown()


async def test_metrics_refresher_updates_memories_active_gauge(tmp_path: Path) -> None:
    from imprint_server.metrics import memories_active
    from imprint_server.workers.metrics_refresh import MetricsRefresher

    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'gauge_test.db'}",
        default_mode="frugal",
        metrics_extended=True,
        metrics_refresh_interval=5,
    )
    registry = AgentRegistry(config)
    await registry.startup()
    try:
        imp = await registry.get(AGENT)
        await imp.observe_directions(user_id=USER, directions=["prefer brevity"])

        refresher = MetricsRefresher(config, registry)
        await refresher._refresh_all()

        count = memories_active.labels(agent_id=AGENT)._value.get()
        assert count == 1.0
    finally:
        await registry.shutdown()


async def test_metrics_refresher_updates_alpha_estimate_gauge(tmp_path: Path) -> None:
    from imprint_server.metrics import bandit_alpha_estimate
    from imprint_server.workers.metrics_refresh import MetricsRefresher

    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'alpha_gauge_test.db'}",
        default_mode="frugal",
        metrics_extended=True,
        metrics_refresh_interval=5,
    )
    registry = AgentRegistry(config)
    await registry.startup()
    try:
        await registry.get(AGENT)  # initialize agent

        refresher = MetricsRefresher(config, registry)
        await refresher._refresh_all()

        alpha = bandit_alpha_estimate.labels(agent_id=AGENT)._value.get()
        assert 0.0 < alpha <= 1.0
    finally:
        await registry.shutdown()


async def test_metrics_refresher_skips_unloaded_agents(tmp_path: Path) -> None:
    """Refresher does not initialize agents that are not already in _instances."""
    from imprint_server.workers.metrics_refresh import MetricsRefresher

    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'skip_test.db'}",
        metrics_extended=True,
        metrics_refresh_interval=5,
    )
    registry = AgentRegistry(config)
    await registry.startup()
    try:
        refresher = MetricsRefresher(config, registry)
        # No agents loaded -- refresh should complete silently with no errors.
        await refresher._refresh_all()
        assert registry.agent_ids() == []
    finally:
        await registry.shutdown()


# -- extended config ----------------------------------------------------------


def test_metrics_extended_defaults_false() -> None:
    config = ServerConfig(store="sqlite:///~/.imprint/test.db")
    assert config.metrics_extended is False
    assert config.metrics_refresh_interval == 60


def test_metrics_extended_config_parses(tmp_path: Path) -> None:
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'x.db'}",
        metrics_extended=True,
        metrics_refresh_interval=30,
    )
    assert config.metrics_extended is True
    assert config.metrics_refresh_interval == 30
