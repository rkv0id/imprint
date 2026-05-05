"""Budget management for Imprint.

HeuristicTokenCounter is the default zero-dep token counter that ships in
core. It uses tiktoken when installed (opportunistic), falling back to the
chars/4 estimate otherwise.

truncate_to_budget trims a memory list to fit within a token budget,
dropping the least-stable non-pinned memories first. Pinned memories are
never dropped. Raises BudgetExceededError when the budget cannot be met.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from imprint.prompts import policy as policy_prompt
from imprint.protocols import DecayModel, TokenCounter
from imprint.types import BudgetExceededError, Memory, MemoryType

_tiktoken_enc: Any = None
try:
    import tiktoken as _tiktoken  # type: ignore[import-untyped]

    _tiktoken_enc = _tiktoken.get_encoding("o200k_base")  # type: ignore[unknown-member-type]
except Exception:
    pass


class HeuristicTokenCounter:
    """Estimates token count, preferring tiktoken when available.

    With tiktoken installed: uses o200k_base encoding (OpenAI gpt-4o and
    most current models). Accurate for English and code.

    Without tiktoken: falls back to ceil(chars / 4). Accurate to within
    ~10% on plain English; wider error on code or multilingual text.

    For exact counts, use AnthropicAPITokenCounter or OpenAITokenCounter.
    Swap in at construction time via the token_counter= parameter on Imprint.
    """

    def count(self, text: str) -> int:
        if _tiktoken_enc is not None:
            return len(_tiktoken_enc.encode(text))
        return math.ceil(len(text) / 4)


def truncate_to_budget(
    *,
    memories: list[Memory],
    max_input_tokens: int,
    on_budget_exceeded: Literal["truncate", "error"],
    decay_model: DecayModel,
    counter: TokenCounter,
    context: str | None,
    existing_instructions: str | None,
    agent_description: str | None,
    now: datetime,
) -> tuple[list[Memory], list[Memory]]:
    """Trim the memory list to fit within max_input_tokens.

    Drops the least-stable non-pinned memories first. Pinned memories are
    never dropped. Returns (kept, dropped).

    Raises BudgetExceededError if on_budget_exceeded='error' and the list
    does not fit, or if no further dropping is possible.
    """

    def _prompt_tokens(mems: list[Memory]) -> int:
        prompt = policy_prompt.build_user_prompt(
            memories=mems,
            existing_instructions=existing_instructions,
            context=context,
            agent_description=agent_description,
        )
        return counter.count(prompt)

    if _prompt_tokens(memories) <= max_input_tokens:
        return memories, []

    if on_budget_exceeded == "error":
        raise BudgetExceededError(f"memory prompt exceeds max_input_tokens={max_input_tokens}")

    pinned = [m for m in memories if m.pinned]
    droppable = [m for m in memories if not m.pinned]

    droppable.sort(
        key=lambda m: (
            m.type != MemoryType.CONTEXT,
            decay_model.effective_stability(m, now),
            m.created_at,
        )
    )

    dropped: list[Memory] = []
    while droppable and _prompt_tokens(pinned + droppable) > max_input_tokens:
        if len(pinned) + len(droppable) == 1:
            raise BudgetExceededError(
                f"cannot reduce memory set below 1 entry within max_input_tokens={max_input_tokens}"
            )
        dropped.append(droppable.pop(0))

    return pinned + droppable, dropped
