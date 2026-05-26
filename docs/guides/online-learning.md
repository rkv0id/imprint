# Online learning

Imprint improves retrieval quality over time using two mechanisms:
a contextual bandit that learns the optimal sparse/dense retrieval balance,
and an FSRS-inspired decay model that weights memory stability by recall history.

## How outcomes drive learning

Every `MemoryLoop` carries an outcome signal set at close time:

```python
loop.set_outcome(0.9)   # 0 = correction, 0.5 = neutral, 1 = ideal
await imprint.finalize_loop(loop)
```

Or via imprint-server:

```sh
# Close a session with an outcome
curl -X POST /v1/agents/{agent_id}/sessions/{id}/close \
  -d '{"outcome": 0.9}'

# Or: explicit correction (outcome = -1.0)
curl -X POST /v1/agents/{agent_id}/correct/{user_id} \
  -d '{"content": "Do not use bullet points."}'

# Or: explicit positive signal
curl -X POST /v1/agents/{agent_id}/reinforce/{user_id} \
  -d '{"session_id": "sess_..."}'
```

## BanditAlphaTuner

The `BanditAlphaTuner` learns the optimal alpha parameter for hybrid retrieval:

```
alpha=0 -> full BM25 (exact keyword match)
alpha=1 -> full dense vector (semantic)
```

After each session closes, the bandit observes the outcome and updates its
alpha estimate using a Thompson Sampling approach. Over time it converges to
the alpha that produces the best outcomes for the agent's retrieval patterns.

The current alpha estimate is exposed as a Prometheus gauge:

```
imprint_bandit_alpha_estimate{agent_id="..."} 0.67
```

To see it, enable extended metrics:

```sh
IMPRINT_METRICS_EXTENDED=true imprint-server serve
```

## FSRS decay model

Memory stability follows an FSRS-inspired decay formula:

```
stability(t) = stability_0 * exp(-decay_rate * elapsed_days)
```

On recall (when a memory appears in a compiled policy), stability increases.
On consolidation, memories below `prune_threshold` are deactivated.

This means frequently recalled memories stay stable, while memories that are
never used gradually decay and are pruned.

## Gradient decay (imprint-mem[online])

Install the `online` extra for a learned decay model:

```sh
pip install imprint-mem[online]
```

```python
from imprint import FSRSGradientDecay

imprint = Imprint(
    agent_id="assistant",
    decay_model=FSRSGradientDecay(),
)
```

`FSRSGradientDecay` uses a [River](https://riverml.xyz/) online linear regressor
to learn per-agent decay parameters from feedback. State persists across restarts.

The static model uses fixed decay parameters calibrated on general usage.
The gradient model adapts to each agent's specific retrieval patterns.

## Observing the effect

After several sessions with varied outcomes, query alice's memories to see
stability evolve:

```python
memories = await imprint.list_memories("alice")
for m in sorted(memories, key=lambda x: x.stability, reverse=True):
    print(f"{m.stability:.3f}  rc={m.recall_count}  {m.content[:50]}")
```

Memories recalled in high-outcome sessions will have higher stability than
those recalled in low-outcome sessions.

In imprint-server, use the Memory Browser in the admin dashboard to visualize
stability bars and recall counts.

## Session-level learning in imprint-server

The server demo (`just demo`) seeds 9 sessions with varied outcomes -- alice
gets three sessions with outcomes 0.9, 0.6, and 0.85. The stability readout
in the seed output shows the effect:

```
[################----] 0.823  rc=3  [active]  "Always use Markdown tables..."
[##############------] 0.741  rc=3  [active]  "User prefers quarterly..."
```

Memories recalled more consistently in high-outcome sessions accumulate higher
stability than those retrieved in mixed-outcome sessions.
