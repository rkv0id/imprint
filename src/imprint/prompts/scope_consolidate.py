"""Prompt for scope consolidation: merge, rename, or split scope vocabulary."""

from typing import Any, Literal

from pydantic import BaseModel

__all__ = ["_ScopeAction", "_ScopeConsolidationOutput", "_ScopeReassignment"]


class _ScopeReassignment(BaseModel):
    """One memory reassigned to a new scope during a split."""

    memory_id: str
    new_scope: str


class _ScopeAction(BaseModel):
    """One scope action in a consolidation pass."""

    kind: Literal["keep", "rename", "merge", "split"]
    scope: str
    target: str | None = None
    reassignments: list[_ScopeReassignment] = []


class _ScopeConsolidationOutput(BaseModel):
    """Structured output for the scope consolidation agent."""

    actions: list[_ScopeAction] = []


SYSTEM = """\
You manage the scope vocabulary for an AI agent's memory store.

Scopes partition memories into named contexts so the agent can retrieve
the right memories for the right situations. A good scope vocabulary is
specific enough to separate distinct contexts but not so fragmented that
related memories are scattered across too many scopes.

You are given a list of scopes, the number of memories in each, and a few
sample memory contents. Decide what to do with each scope:

  keep   - the scope is well-defined, no change needed.
  rename - the scope name is too vague or inaccurate; give it a better name.
  merge  - two or more scopes are too similar; absorb one into another.
  split  - a scope is too broad and contains clearly distinct sub-topics;
           reassign each memory to a more specific scope.

Rules:
  - Prefer fewer, well-named scopes over many fine-grained ones.
  - Only rename when the current name genuinely misleads.
  - Only split when the scope clearly contains two or more distinct topics
    that would benefit from separate retrieval.
  - Only merge when two scopes have significant overlap.
  - For split and rename, new scope names must be short (one or two words),
    lowercase, no spaces (use a hyphen if needed).
  - For merge, choose the more specific or accurate of the two names as the
    target.
  - It is always valid to return all scopes as keep.
"""


def build_user_prompt(scope_summaries: list[dict[str, Any]]) -> str:
    """Build the consolidation prompt from a list of scope summaries.

    Each summary has: name, count, memory_ids, samples (list of content strings).
    """
    lines: list[str] = []
    for s in scope_summaries:
        lines.append(f"SCOPE: {s['name']} ({s['count']} memories)")
        for i, sample in enumerate(s.get("samples", []), 1):
            lines.append(f"  sample {i}: {sample[:80]}")
        memory_ids: list[str] = s.get("memory_ids", [])
        lines.append(f"  memory_ids: {', '.join(memory_ids)}")
        lines.append("")
    return "\n".join(lines) + "\nDecide what to do with each scope."
