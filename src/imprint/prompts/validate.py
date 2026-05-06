"""Prompt for validating a batch of candidate directions (eager mode)."""

from typing import Literal

from pydantic import BaseModel

__all__ = ["_DirectionVerdict", "_ValidationOutput"]


class _DirectionVerdict(BaseModel):
    """One verdict in an eager direction validation pass."""

    verdict: Literal["directive", "hedge", "contradiction", "non-directive"]


class _ValidationOutput(BaseModel):
    """Structured output for the direction validation agent."""

    verdicts: list[_DirectionVerdict] = []


SYSTEM = """\
You validate a list of candidate directions that a user wants to give an AI agent.

For each input, classify it as one of:
- directive: a clear, actionable instruction the agent should follow
- hedge: a vague or conditional statement that is not an actionable instruction
- contradiction: conflicts with a previously established instruction or is self-contradictory
- non-directive: not an instruction at all (a question, statement, opinion, etc.)

Be strict. Only classify as directive if the input is a clear, unambiguous instruction.
"""


def build_user_prompt(*, directions: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(directions))
    return (
        f"Validate each of the following candidate directions:\n\n"
        f"{numbered}\n\n"
        "Return a verdict for each input in order."
    )
