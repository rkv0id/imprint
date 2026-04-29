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

Be conservative. Default to "distinct" unless the relationship is clear. \
Two memories with similar wording about different domains (e.g. one about \
coding style, one about email tone) are distinct, not duplicates.

A FACT and a RULE are almost always distinct, even on related topics. A \
preference about brevity in code review is distinct from a preference about \
brevity in chat. When uncertain, choose distinct.

Return a decision for every existing memory you are shown, referencing it \
by its id. Do not invent ids. If shown no existing memories, return an \
empty list.
"""


def build_user_prompt(
    *,
    candidate_type: str,
    candidate_content: str,
    candidate_signal_type: str,
    existing: list[Memory],
) -> str:
    if not existing:
        existing_block = "(none)"
    else:
        lines = [f"- id={m.id}, type={m.type.value}, content={m.content!r}" for m in existing]
        existing_block = "\n".join(lines)

    return (
        "## Candidate memory (new)\n"
        f"type: {candidate_type}\n"
        f"derived from signal: {candidate_signal_type}\n"
        f"content: {candidate_content!r}\n\n"
        "## Existing memories about this user\n"
        f"{existing_block}\n\n"
        "Decide on each existing memory."
    )
