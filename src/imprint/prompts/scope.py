"""Prompt for LLM-based scope inference in eager mode."""

from pydantic import BaseModel

__all__ = ["_ScopeOutput"]


class _ScopeOutput(BaseModel):
    """Structured output for the scope inference agent."""

    relevant_scopes: list[str] = []


SYSTEM = """\
You identify which memory scopes are relevant to a given context.

Given a context description and a list of available scope names, return only
the scopes that are meaningfully relevant to that context. A scope is relevant
if memories tagged with it would plausibly affect how an agent should behave
in the given context.

Be selective. Return an empty list if the context is too vague or general to
match any specific scope. The "global" scope always matches and should not
be returned -- it is added automatically.
"""


def build_user_prompt(*, context: str, scope_names: list[str]) -> str:
    names = ", ".join(f'"{s}"' for s in scope_names)
    return (
        f"Context: {context}\n\n"
        f"Available scopes: {names}\n\n"
        "Which of these scopes are relevant to this context?"
    )
