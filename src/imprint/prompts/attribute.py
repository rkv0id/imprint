"""Prompt for attributing a user correction to specific policy memories."""

from imprint.types import Memory

SYSTEM = """\
You analyze a user's correction and identify which memories from an AI agent's \
behavioral policy were most relevant to that correction.

The agent compiled a policy from these memories, but the user's correction suggests \
something was missed or underweighted. Your job is to identify which memories, \
if given more weight, would have helped the agent avoid the correction.

Return 1-based indices of the relevant memories (1 to 3 max). If no memory is \
clearly relevant to the correction, return an empty list.
"""


def build_user_prompt(*, correction: str, memories: list[Memory]) -> str:
    lines = [f"{i + 1}. [{m.type.value}] {m.content}" for i, m in enumerate(memories)]
    memory_block = "\n".join(lines)
    return (
        f"USER'S CORRECTION:\n{correction}\n\n"
        f"MEMORIES USED IN THE POLICY:\n{memory_block}\n\n"
        "Which memory indices (1-based) should have been weighted more heavily "
        "to prevent this correction?"
    )
