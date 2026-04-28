"""Signal detection.

Two functions: a pattern-based heuristic that costs nothing and catches
high-confidence cases, and an LLM-based detector for the rest.

The heuristic is intentionally conservative — false negatives are cheap
(`balanced` mode falls through to the LLM; `eager` always uses the LLM)
but false positives lock in via `frugal` mode and pollute the signal table.
"""

import re

from imprint.llm import LLMProvider
from imprint.prompts import signal as signal_prompt
from imprint.types import SignalType

# Pattern → SignalType. First match wins. Order: most specific first within
# each category; categories ordered by typical frequency.
_PATTERNS: list[tuple[re.Pattern[str], SignalType]] = [
    # Corrections — explicit negation or correction markers
    (re.compile(r"^\s*no[,.!]", re.IGNORECASE), SignalType.CORRECTION),
    (re.compile(r"\bactually,?\s", re.IGNORECASE), SignalType.CORRECTION),
    (re.compile(r"\bthat'?s not\b|\bthat is not\b", re.IGNORECASE), SignalType.CORRECTION),
    (re.compile(r"\b(wrong|incorrect)\b", re.IGNORECASE), SignalType.CORRECTION),
    # Directions — imperative instructions
    (re.compile(r"\b(don'?t|do not)\b", re.IGNORECASE), SignalType.DIRECTION),
    (re.compile(r"^\s*(always|never)\b", re.IGNORECASE), SignalType.DIRECTION),
    (re.compile(r"\bfrom now on\b", re.IGNORECASE), SignalType.DIRECTION),
    # Preferences — explicit I-statements
    (
        re.compile(r"\bi (prefer|like|want|need|expect)\b", re.IGNORECASE),
        SignalType.PREFERENCE,
    ),
    (
        re.compile(r"\bi'?d rather\b|\bi would rather\b", re.IGNORECASE),
        SignalType.PREFERENCE,
    ),
    # Facts — identity / background statements
    (re.compile(r"\bmy name is\b", re.IGNORECASE), SignalType.FACT),
    (re.compile(r"\bi work (at|for|in|on)\b", re.IGNORECASE), SignalType.FACT),
    (re.compile(r"\bi live (in|at)\b", re.IGNORECASE), SignalType.FACT),
    # Reinforcements — strong positive confirmation
    (re.compile(r"\b(perfect|exactly)\b", re.IGNORECASE), SignalType.REINFORCEMENT),
    (
        re.compile(r"\b(great|nice|excellent)\s+(work|job)\b", re.IGNORECASE),
        SignalType.REINFORCEMENT,
    ),
    (re.compile(r"^\s*(yes|yeah|yep)[!]", re.IGNORECASE), SignalType.REINFORCEMENT),
]


def detect_signal_heuristic(user_response: str) -> SignalType | None:
    """Return the matched SignalType for known patterns, or None.

    Conservative by design — only fires on high-confidence patterns.
    """
    for pattern, signal_type in _PATTERNS:
        if pattern.search(user_response):
            return signal_type
    return None


_VALID_TOKENS = {t.value.upper(): t for t in SignalType if t != SignalType.IMPLICIT}


async def detect_signal_llm(
    llm: LLMProvider,
    *,
    agent_output: str,
    user_response: str,
) -> SignalType | None:
    """Ask the LLM to classify the response into a SignalType or NONE."""
    prompt = signal_prompt.build_user_prompt(agent_output=agent_output, user_response=user_response)
    response = await llm.complete(
        prompt,
        system=signal_prompt.SYSTEM,
        max_tokens=10,
        temperature=0.0,
    )

    token = response.text.strip().upper().rstrip(".,;:!?")
    if not token or token == "NONE":
        return None
    return _VALID_TOKENS.get(token)
