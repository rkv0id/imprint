"""Prometheus metrics for imprint-server.

Hot-path metrics (always on, zero DB overhead):
  imprint_observe_total{agent_id, mode}
  imprint_observe_latency_seconds{agent_id, mode}
  imprint_observe_errors_total{agent_id, mode}
  imprint_policy_total{agent_id}
  imprint_policy_latency_seconds{agent_id, cached}
  imprint_policy_errors_total{agent_id}
  imprint_policy_cache_hits_total{agent_id}
  imprint_policy_cache_misses_total{agent_id}
  imprint_policy_memories_retrieved{agent_id}
  imprint_policy_memories_dropped{agent_id}
  imprint_redis_invalidations_total{agent_id}
  imprint_consolidation_pruned_total{agent_id}
  imprint_session_total{agent_id}
  imprint_scheduler_job_total{job_type, status}

Extended gauges (IMPRINT_METRICS_EXTENDED=true, refreshed in background):
  imprint_memories_active{agent_id}
  imprint_bandit_alpha_estimate{agent_id}

/metrics is exposed by api/health.py via prometheus_client.generate_latest().
Import metric objects directly and call .labels(...).inc() / .observe().
"""

from prometheus_client import Counter, Gauge, Histogram

# -- observe() ----------------------------------------------------------------

observe_total: Counter = Counter(
    "imprint_observe_total",
    "Total observe() calls received by the server.",
    ["agent_id", "mode"],
)

observe_latency: Histogram = Histogram(
    "imprint_observe_latency_seconds",
    "Wall-clock time for observe() calls, including LLM latency in balanced/eager mode.",
    ["agent_id", "mode"],
)

observe_errors: Counter = Counter(
    "imprint_observe_errors_total",
    "Failed observe() calls by agent and processing mode.",
    ["agent_id", "mode"],
)

# -- get_policy() -------------------------------------------------------------

policy_total: Counter = Counter(
    "imprint_policy_total",
    "Total get_policy() calls received by the server.",
    ["agent_id"],
)

policy_latency: Histogram = Histogram(
    "imprint_policy_latency_seconds",
    "Wall-clock time for get_policy() calls. cached=true when served from Redis.",
    ["agent_id", "cached"],
)

policy_errors: Counter = Counter(
    "imprint_policy_errors_total",
    "Failed get_policy() calls by agent.",
    ["agent_id"],
)

policy_cache_hits: Counter = Counter(
    "imprint_policy_cache_hits_total",
    "Policy responses served from the Redis cache.",
    ["agent_id"],
)

policy_cache_misses: Counter = Counter(
    "imprint_policy_cache_misses_total",
    "Policy compilations that were not found in the Redis cache.",
    ["agent_id"],
)

policy_memories_retrieved: Histogram = Histogram(
    "imprint_policy_memories_retrieved",
    "Number of memories included in each compiled policy.",
    ["agent_id"],
    buckets=[0, 1, 2, 5, 10, 20, 50],
)

policy_memories_dropped: Histogram = Histogram(
    "imprint_policy_memories_dropped",
    "Number of memories truncated by the token budget on each policy call.",
    ["agent_id"],
    buckets=[0, 1, 2, 5, 10, 20, 50],
)

# -- Redis cache --------------------------------------------------------------

redis_invalidations: Counter = Counter(
    "imprint_redis_invalidations_total",
    "Redis policy cache invalidations triggered by memory writes.",
    ["agent_id"],
)

# -- consolidation ------------------------------------------------------------

consolidation_pruned: Counter = Counter(
    "imprint_consolidation_pruned_total",
    "Total memories pruned across all consolidation runs.",
    ["agent_id"],
)

# -- sessions -----------------------------------------------------------------

session_total: Counter = Counter(
    "imprint_session_total",
    "Total HTTP sessions opened.",
    ["agent_id"],
)

# -- scheduler ----------------------------------------------------------------

scheduler_job_total: Counter = Counter(
    "imprint_scheduler_job_total",
    "Scheduler jobs executed, by type and outcome.",
    ["job_type", "status"],
)

# -- extended gauges (IMPRINT_METRICS_EXTENDED=true) --------------------------

memories_active: Gauge = Gauge(
    "imprint_memories_active",
    "Number of active memories per agent across all users. "
    "Refreshed in background when IMPRINT_METRICS_EXTENDED=true.",
    ["agent_id"],
)

bandit_alpha_estimate: Gauge = Gauge(
    "imprint_bandit_alpha_estimate",
    "Current retrieval alpha estimate from the alpha tuner. "
    "For BanditAlphaTuner: expected value of the preferred arm. "
    "For StaticAlphaTuner: the fixed alpha. "
    "Refreshed in background when IMPRINT_METRICS_EXTENDED=true.",
    ["agent_id"],
)
