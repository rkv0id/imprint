"""Prompt for consolidating a new memory against existing ones."""

from imprint.types import Memory

SYSTEM = """\
You consolidate a new candidate memory against the user's existing memories.

You are given:
- A candidate memory (just derived from a recent signal)
- A list of existing memories already stored about this user

For EACH existing memory, decide one of:
- merge: the candidate covers the same ground; the existing one is redundant
- contradict: the candidate makes the existing one no longer true
- distinct: unrelated; keep the existing one as-is
- scope_override: the candidate is scoped more specifically than the existing
  memory and addresses the same topic, but the existing global memory should
  stay active for other contexts. Use this when the candidate scope is a named
  scope (not global) and the existing memory scope is global.

Scope rules:
- If the candidate scope is global and the existing scope is global: use
  merge, contradict, or distinct as normal.
- If the candidate scope is named (not global) and the existing scope is
  global, and they address the same topic: use scope_override. Both memories
  stay active. The named scope takes precedence at compile time.
- If both are the same named scope: use merge, contradict, or distinct.
- Two memories about different topics are always distinct regardless of scope.

Be conservative. Default to "distinct" unless the relationship is clear.
Two memories with similar wording about different domains (e.g. one about
coding style, one about email tone) are distinct, not duplicates.

A FACT and a RULE are almost always distinct, even on related topics. A
preference about brevity in code review is distinct from a preference about
brevity in chat. When uncertain, choose distinct.

Return a decision for every existing memory you are shown, referencing it
by its id. Do not invent ids. If shown no existing memories, return an
empty list.
"""

BATCH_SYSTEM = """\
You consolidate a batch of new candidate memories against a user's existing memories.

You are given:
- A numbered list of new candidate memories (indices 0, 1, 2, ...)
- A list of existing memories already stored about this user

For each (candidate, existing) pair where the relationship is NOT distinct,
return a decision with:
  - candidate_index: 0-based index of the new memory in the batch
  - memory_id: the id of the existing memory
  - action: one of "merge", "contradict", or "scope_override"

Omit decisions where the relationship is "distinct".
If all pairs are distinct, return an empty decisions list.

Actions:
- merge: the candidate covers the same ground; the existing one is redundant.
- contradict: the candidate makes the existing one no longer true.
- scope_override: the candidate has a named (non-global) scope and the existing
  memory has global scope, and they address the same topic. Both memories stay
  active. The named scope takes precedence at compile time.

Scope rules:
- candidate global + existing global -> merge, contradict, or distinct
- candidate named scope + existing global, same topic -> scope_override
- candidate and existing same named scope -> merge, contradict, or distinct
- different topics -> always distinct, regardless of scope

Be conservative. Default to "distinct" unless the relationship is clear.
Do not invent ids. Only reference ids from the provided existing list.
"""


def build_user_prompt(
    *,
    candidate_type: str,
    candidate_content: str,
    candidate_scope: str,
    candidate_signal_type: str,
    existing: list[Memory],
) -> str:
    if not existing:
        existing_block = "(none)"
    else:
        lines = [
            f"- id={m.id}, type={m.type.value}, scope={m.scope}, content={m.content!r}"
            for m in existing
        ]
        existing_block = "\n".join(lines)

    return (
        "## Candidate memory (new)\n"
        f"type: {candidate_type}\n"
        f"scope: {candidate_scope}\n"
        f"derived from signal: {candidate_signal_type}\n"
        f"content: {candidate_content!r}\n\n"
        "## Existing memories about this user\n"
        f"{existing_block}\n\n"
        "Decide on each existing memory."
    )


def build_batch_user_prompt(
    *,
    candidates: list[Memory],
    candidate_signal_type: str,
    existing: list[Memory],
) -> str:
    """Build a prompt for batch consolidation of multiple new memories."""
    candidate_lines = [
        f"[{i}] type={m.type.value}, scope={m.scope}, content={m.content!r}"
        for i, m in enumerate(candidates)
    ]
    candidates_block = "\n".join(candidate_lines)

    if not existing:
        existing_block = "(none)"
    else:
        lines = [
            f"- id={m.id}, type={m.type.value}, scope={m.scope}, content={m.content!r}"
            for m in existing
        ]
        existing_block = "\n".join(lines)

    return (
        "## New memories (candidates)\n"
        f"{candidates_block}\n\n"
        f"All derived from signal: {candidate_signal_type}\n\n"
        "## Existing memories about this user\n"
        f"{existing_block}\n\n"
        "Return merge, contradict, and scope_override decisions. Omit distinct relationships."
    )
