"""Prometheus metrics for imprint-server.

All agent-scoped metrics carry an agent_id label so dashboards can track
individual agents. Import the collectors directly from this module and call
.labels(agent_id=...).inc() / .observe() in route handlers.

/metrics is exposed by api/health.py using prometheus_client.generate_latest().
"""

from prometheus_client import Counter, Histogram

# -- observe() ----------------------------------------------------------------

observe_total: Counter = Counter(
    "imprint_observe_total",
    "Total observe() calls received by the server.",
    ["agent_id"],
)

observe_duration: Histogram = Histogram(
    "imprint_observe_duration_seconds",
    "Wall-clock time for observe() calls (including LLM latency in balanced/eager mode).",
    ["agent_id"],
)

# -- get_policy() -------------------------------------------------------------

policy_total: Counter = Counter(
    "imprint_policy_total",
    "Total get_policy() calls received by the server.",
    ["agent_id"],
)

policy_duration: Histogram = Histogram(
    "imprint_policy_duration_seconds",
    "Wall-clock time for get_policy() calls (including LLM latency on cache miss).",
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
