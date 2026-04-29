"""Pattern-based signal detection.

LLM-based detection is owned by the Imprint facade (it has the agent).
This module is the cheap, deterministic heuristic layer used by `frugal`
mode and as the first pass for `balanced` mode.

The patterns are intentionally conservative - false negatives are cheap
(`balanced` falls through to the LLM; `eager` always uses the LLM) but
false positives lock in via `frugal` mode and pollute the signal table.
"""

import re

from imprint.types import SignalType

# Pattern -> SignalType. First match wins. Order: most specific first within
# each category; categories ordered by typical frequency.
_PATTERNS: list[tuple[re.Pattern[str], SignalType]] = [
    # Corrections - explicit negation or correction markers
    (re.compile(r"^\s*no[,.!]", re.IGNORECASE), SignalType.CORRECTION),
    (re.compile(r"\bactually,?\s", re.IGNORECASE), SignalType.CORRECTION),
    (re.compile(r"\bthat'?s not\b|\bthat is not\b", re.IGNORECASE), SignalType.CORRECTION),
    (re.compile(r"\b(wrong|incorrect)\b", re.IGNORECASE), SignalType.CORRECTION),
    # Directions - imperative instructions
    (re.compile(r"\b(don'?t|do not)\b", re.IGNORECASE), SignalType.DIRECTION),
    (re.compile(r"^\s*(always|never)\b", re.IGNORECASE), SignalType.DIRECTION),
    (re.compile(r"\bfrom now on\b", re.IGNORECASE), SignalType.DIRECTION),
    # Preferences - explicit I-statements
    (
        re.compile(r"\bi (prefer|like|want|need|expect)\b", re.IGNORECASE),
        SignalType.PREFERENCE,
    ),
    (
        re.compile(r"\bi'?d rather\b|\bi would rather\b", re.IGNORECASE),
        SignalType.PREFERENCE,
    ),
    # Facts - identity / background statements
    (re.compile(r"\bmy name is\b", re.IGNORECASE), SignalType.FACT),
    (re.compile(r"\bi work (at|for|in|on)\b", re.IGNORECASE), SignalType.FACT),
    (re.compile(r"\bi live (in|at)\b", re.IGNORECASE), SignalType.FACT),
    # Reinforcements - strong positive confirmation
    (re.compile(r"\b(perfect|exactly)\b", re.IGNORECASE), SignalType.REINFORCEMENT),
    (
        re.compile(r"\b(great|nice|excellent)\s+(work|job)\b", re.IGNORECASE),
        SignalType.REINFORCEMENT,
    ),
    (re.compile(r"^\s*(yes|yeah|yep)[!]", re.IGNORECASE), SignalType.REINFORCEMENT),
]


def detect_signal_heuristic(user_response: str) -> SignalType | None:
    """Return the matched SignalType for known patterns, or None.

    Conservative by design - only fires on high-confidence patterns.
    """
    for pattern, signal_type in _PATTERNS:
        if pattern.search(user_response):
            return signal_type
    return None
