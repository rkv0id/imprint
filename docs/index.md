# imprint

<div class="hero" markdown>
<img src="media/mark-light.svg" class="hero-logo only-light" alt="imprint logo"/>
<img src="media/mark-dark.svg" class="hero-logo only-dark" alt="imprint logo"/>

**Behavioral memory for AI agents.**

Detect, distill, compile.
{ .hero-sub }

<div class="hero-badges">
  [![PyPI](https://img.shields.io/pypi/v/imprint-mem?color=0d9488&label=imprint-mem)](https://pypi.org/project/imprint-mem/)
  [![PyPI server](https://img.shields.io/pypi/v/imprint-server?color=0d9488&label=imprint-server)](https://pypi.org/project/imprint-server/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-0d9488)](https://github.com/rkv0id/imprint/blob/main/LICENSE)
  [![Python](https://img.shields.io/pypi/pyversions/imprint-mem?color=0d9488)](https://pypi.org/project/imprint-mem/)
</div>
</div>

---

## What is imprint?

Most memory systems store what was said. **Imprint learns what to do differently.**

It watches agent-user interactions, extracts typed behavioral memories (rules,
preferences, facts, decisions), consolidates them as new ones arrive, and compiles
a behavioral policy the agent injects into its system prompt. The policy is the
output -- not a database the agent queries.

```
observe() -> detect -> derive -> persist -> consolidate
get_policy() -> filter -> rank -> compile -> cache
```

## Two packages

=== "imprint-mem"

    The core library. Use it directly in your agent.

    ```sh
    pip install imprint-mem
    ```

    ```python
    from imprint import Imprint

    imprint = Imprint(agent_id="assistant", model="anthropic:claude-haiku-4-5-20251001")
    await imprint.connect()

    await imprint.observe(user_id="alice",
        agent_output="Here is a bullet list.",
        user_response="Please use prose instead.")

    policy = await imprint.get_policy(user_id="alice")
    print(policy.text)
    # -> "Write responses in prose rather than bullet points."
    ```

    [Get started with imprint-mem](getting-started/quickstart.md){ .md-button .md-button--primary }
    [API reference](library/api.md){ .md-button }

=== "imprint-server"

    The networked service. Exposes imprint-mem over HTTP/REST and MCP SSE.

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

    [Get started with imprint-server](server/overview.md){ .md-button .md-button--primary }
    [REST API reference](server/rest-api.md){ .md-button }

## Key features

<div class="grid cards" markdown>

- **Per-user behavioral memory** -- Each user gets their own isolated memory namespace. The agent learns different behaviors for different users without cross-contamination.

- **Three processing modes** -- `frugal` (heuristics only, zero LLM cost), `balanced` (heuristics + LLM fallback), `eager` (always LLM, highest recall). Pick the right cost/quality tradeoff per agent.

- **Online learning** -- FSRS-inspired memory decay, contextual bandit alpha tuning, and session-level outcome signals let imprint improve retrieval quality over time.

- **Hybrid retrieval** -- BM25 full-text search fused with dense vector search via Reciprocal Rank Fusion. Falls back to list order without an embedder configured.

- **Policy cache** -- Compiled policies are cached (locally or in Redis). The cache invalidates automatically when memories change. Zero redundant LLM calls.

- **MCP native** -- Mount imprint as an MCP SSE server. Claude Code, Cursor, and Continue can use the eight imprint tools directly.

- **Framework integrations** -- LangChain callback handler, LlamaIndex event handler, PydanticAI tool factory. Drop into existing agent code.

- **Production ready** -- Postgres + pgvector for multi-instance deployments, Redis for distributed cache and rate limiting, Docker Compose stacks included.

</div>

## Architecture

```
                    observe()                    get_policy()
                       |                              |
              +--------v--------+          +----------v----------+
              |   Detection     |          |   Scope inference   |
              | heuristics/LLM  |          |   embed/LLM/all     |
              +--------+--------+          +----------+----------+
                       |                             |
              +--------v--------+          +----------v----------+
              |   Derivation    |          |  Hybrid retrieval   |
              |  type/content/  |          |  BM25 + dense RRF   |
              |    scope        |          +----------+----------+
              +--------+--------+                     |
                       |                  +----------v----------+
              +--------v--------+          |   LLM compile      |
              |  Persistence    |          |   + cache store     |
              | FTS + vector    |          +----------+----------+
              +--------+--------+                     |
                       |                  +----------v----------+
              +--------v--------+          |   policy.text      |
              | Consolidation   |          | (inject as system  |
              | merge/contradict|          |   prompt)          |
              +--------+--------+          +---------------------+
                       |
              +--------v--------+
              |  Online learning|
              | bandit / decay  |
              +-----------------+
```

## License

[Apache 2.0](https://github.com/rkv0id/imprint/blob/main/LICENSE).
