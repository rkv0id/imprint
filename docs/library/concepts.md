# Concepts

## The observe-compile loop

Imprint sits between the agent and the user. On every agent turn, `observe()`
runs. Before every agent response, `get_policy()` runs.

```
User message
    |
    v
  Agent generates response
    |
    +----> observe(agent_output, user_response)
    |             |
    |          detect -> derive -> persist -> consolidate
    |
    v
  get_policy(user_id)
    |
  filter -> rank -> compile -> inject into system prompt
    |
    v
  Agent generates next response
```

Nothing changes in how you write the agent. The memory loop is a thin wrapper
around the existing observe-respond cycle.

## Memory types

Every stored memory has a type:

| Type | Meaning | Example |
|---|---|---|
| `RULE` | Behavioral rule the agent must follow | "Write in prose, not bullet points." |
| `PREFERENCE` | User preference that should influence tone or style | "Alice prefers concise answers." |
| `FACT` | Factual information about the user or context | "Bob works in the Pacific timezone." |
| `DECISION` | A decision that was made and should be remembered | "We agreed to use async/await throughout." |
| `CONTEXT` | Contextual information relevant to future responses | "Current project uses Python 3.12." |

The type is selected by the LLM during derivation (or inferred from heuristics
in frugal mode). It influences how memories are ranked during policy compilation.

## Detection

Detection decides whether an agent-user exchange carries a learnable signal.
The vast majority of interactions are pure dialogue -- no preference, no
correction, no directive -- and observation produces nothing.

**Heuristic detection** runs first (always). Patterns like "don't do X",
"always Y", explicit preference language, corrections, and confirmations fire
without any LLM call.

**LLM detection** runs as fallback in `balanced` mode when heuristics are
silent. In `eager` mode it always runs regardless of heuristic results.

Detection is the primary cost lever:

| Mode | LLM calls per observe() |
|---|---|
| `frugal` | 0 (heuristics only) |
| `balanced` | 0 or 1 (LLM only when heuristics are silent) |
| `eager` | 1 (always) |

## Derivation

When detection finds a signal, derivation converts it into a memory:

1. **Type** -- what kind of memory (RULE, PREFERENCE, FACT, etc.)
2. **Content** -- canonical third-person phrasing ("The user prefers...")
3. **Scope** -- which context this memory applies to (see [Scopes](#scopes))

Derivation always calls the LLM in balanced and eager modes. In frugal mode,
a simplified heuristic derivation runs without LLM.

## Consolidation

New memories are compared against existing ones. One of four actions is taken:

| Action | When | Effect |
|---|---|---|
| `distinct` | Memories are unrelated | Both stay active |
| `merge` | New memory is redundant with existing | New memory discarded |
| `contradict` | New memory contradicts existing | Old memory deactivated, new stored |
| `scope_override` | Conflict is scope-specific | Both stay active, scope wins at compile time |

Deactivated memories stay in the store for lineage tracking. They never appear
in policy compilation.

## Scopes

Scopes partition memories by context. A memory tagged with `"project:alpha"`
only appears in policies requested with that scope. The `"global"` scope is
always included.

Declare candidate scopes on construction:

```python
imprint = Imprint(
    agent_id="reviewer",
    scopes=["project:alpha", "project:beta", "role:senior"],
)
```

With `dynamic_scopes=True`, imprint can create new scope names on the fly
during derivation. Near-duplicates are collapsed to existing names.

## Memory decay and stability

Every memory has a **stability** score between 0 and 1. Stability increases
on recall (when the memory appears in a compiled policy) and decays over time
using an FSRS-inspired formula.

Consolidation uses stability to prune: memories below `prune_threshold`
(default 0.5) are deactivated. Pinned memories are never pruned.

```python
# Prune memories with stability below 0.3
pruned = await imprint.consolidate("alice", prune_threshold=0.3)
```

With `imprint-mem[online]`, a learned decay model (`FSRSGradientDecay`) replaces
the static formula. See [Memory decay guide](../guides/decay.md).

## Policy compilation

`get_policy()` assembles the behavioral policy:

1. **Scope inference** -- if `scopes=` is not provided, infer from `context=`
2. **Retrieval** -- list active memories for the requested scopes
3. **Hybrid ranking** -- BM25 + dense vector search fused via RRF (if embedder configured), otherwise list order
4. **Token budget** -- truncate to `max_input_tokens`, always keeping pinned memories
5. **LLM compile** -- compile selected memories into a concise instruction block
6. **Cache** -- cache the result; invalidate when memories change

In frugal mode, step 5 is skipped for empty memory lists (returns an empty string).
The policy text is designed to be injected directly into the system prompt.

## Memory loops

A `MemoryLoop` tracks one agent turn end-to-end and carries the outcome signal
back to the learning system. Outcome signals update the bandit alpha tuner and
the decay model.

```python
loop = await imprint.open_loop(user_id="alice", context="code review")
policy = await imprint.get_policy(user_id="alice", loop=loop)

# ... agent generates response ...

loop.set_outcome(0.9)  # 0 = correction, 0.5 = neutral, 1 = ideal
await imprint.finalize_loop(loop)
```

See the [Online learning guide](../guides/online-learning.md) for how outcomes
drive adaptation.

## Storage

| Backend | When to use |
|---|---|
| `SQLiteMemoryStore` (default) | Single-process, local dev, embedded |
| `PostgresMemoryStore` | Multi-instance server, production |

For dense retrieval:

| Backend | When to use |
|---|---|
| `SQLiteVecStore` | Single-process, local dev |
| `PostgresVectorStore` | Multi-instance, pgvector |

Storage is configured either via constructor arguments or via environment
variables (`IMPRINT_STORE`, `IMPRINT_VECTOR_STORE`).
