# Memory decay

## What is memory stability?

Every memory has a `stability` score between 0 and 1. Stability starts at 1.0
when a memory is created and decays over time. Stability increases when the
memory is recalled in a policy compilation.

```python
memory.stability    # float 0-1
memory.recall_count # int -- how many times it appeared in a policy
```

## The decay formula

The default model (`FSRSStaticDecay`) uses a static FSRS-inspired exponential:

```
stability(t) = stability_0 * exp(-decay_rate * elapsed_days)
```

`decay_rate` is calibrated at 0.3 (approximately 5% decay per week without recall).
A memory recalled once per week stays at approximately 80% stability.

## Pruning

`consolidate()` deactivates memories below `prune_threshold`:

```python
# Prune memories below 0.3 stability
pruned = await imprint.consolidate("alice", prune_threshold=0.3)
print(f"Pruned {pruned} memories")
```

Deactivated memories stay in the store for lineage tracking. They never appear
in `list_memories()` (default `active_only=True`) or policy compilation.

Pinned memories are never pruned regardless of stability:

```python
await imprint.pin_memory(memory_id)
```

## Visualizing stability

In the Memory Browser panel of the admin dashboard, each memory shows a
stability bar -- a horizontal bar proportional to current stability, with
the numeric value beside it.

```
[################----] 0.823  rc=3  active
[##########----------] 0.502  rc=1  active
[####----------------] 0.201  rc=0  active  <- will be pruned at threshold=0.25
```

## Scheduled consolidation

imprint-server runs consolidation in the background on a configurable schedule.
It also detects memory contradiction rate spikes (high `confusion` events) and
triggers early consolidation automatically.

## Learned decay (imprint-mem[online])

The default static model uses fixed parameters. For production deployments where
the agent operates over weeks or months, the gradient decay model adapts
per-agent parameters from feedback signals:

```python
from imprint import FSRSGradientDecay

imprint = Imprint(
    agent_id="assistant",
    decay_model=FSRSGradientDecay(),
    # IMPRINT_DECAY_MODEL=gradient via env
)
```

The gradient model uses River's online linear regressor. After enough sessions
with varied outcomes, it learns that frequently-recalled memories in high-outcome
sessions should decay slower, while memories that are consistently not helpful
should decay faster.

State is persisted in the store and survives restarts.

## Interaction with online learning

Stability and outcomes interact:

1. Memory is created with `stability=1.0`
2. Session opens, memory is retrieved for policy compilation, `recall_count += 1`
3. Session closes with `outcome=0.9` -- positive signal
4. Stability increases (exact amount depends on decay model and recall history)
5. Future consolidations with low `prune_threshold` will skip this memory
6. If outcome is negative or memory is never recalled, stability decays and the memory may be pruned

This creates a natural selection pressure: useful memories persist, irrelevant
or incorrect ones fade.
