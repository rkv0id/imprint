"""Prompt for compiling memories into a behavioral policy."""

from imprint.types import Memory

SYSTEM = """\
You compile behavioral policies for AI agents from memories about specific users.

Your output is a concise text block that tells the agent how to behave \
differently for THIS user, based on past interactions.

Rules:
- Do not restate or paraphrase anything already in the agent's existing system \
prompt. The agent already follows those instructions; repeating them wastes tokens.
- When memories contradict, prefer the more recent.
- Output the policy text only. No preamble, no commentary, no markdown headers.
- Be concise; the agent reads this on every turn.
"""


def build_user_prompt(
    *,
    memories: list[Memory],
    existing_instructions: str | None,
    context: str | None,
) -> str:
    memory_lines = "\n".join(f"- [{m.type.value}, scope={m.scope}] {m.content}" for m in memories)

    sections = [
        "## Agent's existing system prompt (do not restate)",
        existing_instructions or "(none)",
        "",
        "## Current context",
        context or "(none)",
        "",
        "## Memories about this user",
        memory_lines,
        "",
        "Compile the behavioral policy now.",
    ]
    return "\n".join(sections)
