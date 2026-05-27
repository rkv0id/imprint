# imprint

**Behavioral memory for AI agents.**

Detect, distill, compile. Most memory systems store what was said -- imprint
learns what to do differently.

[![PyPI](https://img.shields.io/pypi/v/imprint-mem?color=0d9488&label=imprint-mem)](https://pypi.org/project/imprint-mem/)
[![PyPI server](https://img.shields.io/pypi/v/imprint-server?color=0d9488&label=imprint-server)](https://pypi.org/project/imprint-server/)
[![License](https://img.shields.io/badge/license-Apache%202.0-0d9488)](https://github.com/rkv0id/imprint/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/imprint-mem?color=0d9488)](https://pypi.org/project/imprint-mem/)

---

## What is imprint?

It watches agent-user interactions, extracts typed behavioral memories (rules,
preferences, facts, decisions), consolidates them as new ones arrive, and compiles
a behavioral policy the agent injects into its system prompt. The policy is the
output -- not a database the agent queries.

![imprint flow diagram](media/flow-diagram.svg)

## The learning loop

What makes imprint different is the **online learning** path -- the feedback
cycle shown at the bottom of the diagram above. Every time a session closes with
an outcome signal, two things happen:

1. **FSRS decay update** -- memories that were recalled during the session
   get a stability boost. Those never recalled gradually decay toward pruning.
2. **Bandit feedback** -- the `BanditAlphaTuner` learns whether sparse BM25 or
   dense vector retrieval produced better outcomes for this agent, and adjusts
   the alpha blend accordingly.

Over time, retrieval quality improves without any explicit configuration.
Frequently recalled memories that lead to positive outcomes persist and become
easier to retrieve. Irrelevant or incorrect memories decay and are pruned by
consolidation.

## Two packages

=== "imprint-mem"

    The core library. Embed directly in your agent process.

    ```sh
    pip install imprint-mem
    ```

    ```python
    from imprint import Imprint

    imprint = Imprint(
        agent_id="assistant",
        model="anthropic:claude-haiku-4-5-20251001",
        processing_mode="balanced",   # frugal | balanced | eager
    )
    await imprint.connect()

    # Record each agent turn -- nothing stored if no signal detected
    await imprint.observe(
        user_id="alice",
        agent_output="Here is a bullet list.",
        user_response="Please use prose instead.",
    )

    # Compile a behavioral policy -- inject into system prompt
    policy = await imprint.get_policy(user_id="alice")
    print(policy.text)
    # → "Write responses in prose rather than bullet points."
    ```

    [Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
    [API reference](library/api.md){ .md-button }

=== "imprint-server"

    The networked service. REST API + MCP SSE.

    ```sh
    pip install imprint-server
    imprint-server serve
    ```

    ```python
    from imprint.client import ImprintClient

    async with ImprintClient("http://localhost:8000") as client:
        policy = await client.get_policy("my-agent", "alice")
        await client.observe("my-agent", "alice",
            agent_output="Here is a list.",
            user_response="Prose please.")
    ```

    [Server docs](server/overview.md){ .md-button .md-button--primary }
    [REST API](server/rest-api.md){ .md-button }

## Key features

<div class="grid cards" markdown>

-   **Three processing modes**

    `frugal` -- heuristics only, zero LLM cost per observation. `balanced` --
    LLM as fallback when heuristics are silent. `eager` -- always LLM, maximum
    recall. Choose per agent.

-   **FSRS-inspired memory decay**

    Every memory has a stability score that rises on recall and decays over
    time. Consolidation prunes low-stability memories automatically. With
    `imprint-mem[online]`, `FSRSGradientDecay` learns per-agent decay rates
    from session outcomes.

-   **Contextual bandit retrieval**

    `BanditAlphaTuner` learns the optimal blend between BM25 keyword search
    and dense vector search from session outcome signals. No manual tuning.
    The alpha estimate improves with every closed session.

-   **Hybrid BM25 + dense retrieval**

    FTS5 full-text search fused with pgvector or sqlite-vec via Reciprocal
    Rank Fusion. Falls back to pure BM25 when no embedder is configured.

-   **MCP native**

    Mount imprint as an MCP SSE server. Eight tools: begin session, get
    policy, observe, recall, direct, end session, correct, reinforce.
    Supports Claude Code, Cursor, and Continue.

-   **Production ready**

    Postgres + pgvector, Redis distributed cache, rate limiting, versioned
    schema migrations with checksum verification, Prometheus metrics,
    Docker image.

</div>

## How does it compare?

The table below reflects publicly documented features as of May 2026. The space
moves quickly -- verify against each project's current docs before deciding.

| | imprint | Mem0 | Letta | Zep | LangMem |
|---|:---:|:---:|:---:|:---:|:---:|
| Per-user memory | ✓ | ✓ | ✓ | ✓ | ✓ |
| Typed memories (RULE, FACT...) | ✓ | | | | partial |
| Compile to behavioral policy text | ✓ | | | | |
| FSRS-style stability + pruning | ✓ | | | | |
| Recency-aware decay | ✓ | ✓ | | ✓ | |
| Bandit-tuned retrieval alpha | ✓ | | | | |
| Online learning from session outcomes | ✓ | | | | |
| Zero LLM cost observation mode | ✓ | | | | |
| MCP SSE endpoint | ✓ | | | | |
| Embedded library (no server needed) | ✓ | ✓ | | | ✓ |
| Temporal knowledge graph | | | | ✓ | |
| Agent self-manages own memory | | | ✓ | | |
| Hosted cloud API | | ✓ | ✓ | ✓ | |

Mem0 focuses on ease of integration -- "add memory in three lines of code" --
with vector + optional graph storage and a recency re-ranking decay. Letta
(formerly MemGPT) treats memory as the agent's own editable state, with
agents actively managing their context window via tool calls. Zep centers on
a temporal knowledge graph that tracks how facts change over time. LangMem is
LangChain's open-source SDK for episodic, semantic, and procedural memory
types.

Imprint's distinct focus is the compile-to-policy abstraction and the adaptive
learning loop -- the memory system improves retrieval quality over time from
session outcomes, without manual tuning.

## License

[Apache 2.0](https://github.com/rkv0id/imprint/blob/main/LICENSE).
