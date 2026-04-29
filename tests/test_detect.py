import pytest

from imprint.detect import detect_signal_heuristic
from imprint.types import SignalType


@pytest.mark.parametrize(
    "text,expected",
    [
        ("No, write in paragraphs.", SignalType.CORRECTION),
        ("no. that's not what I meant", SignalType.CORRECTION),
        ("Actually, I changed my mind", SignalType.CORRECTION),
        ("That's not right", SignalType.CORRECTION),
        ("That is not what I asked for", SignalType.CORRECTION),
        ("This is wrong", SignalType.CORRECTION),
        ("The output is incorrect", SignalType.CORRECTION),
        ("Don't use bullet points", SignalType.DIRECTION),
        ("Do not include explanations", SignalType.DIRECTION),
        ("Always use 4-space indentation", SignalType.DIRECTION),
        ("Always cite sources with URLs", SignalType.DIRECTION),
        ("Never respond in all caps", SignalType.DIRECTION),
        ("From now on, be more concise", SignalType.DIRECTION),
        ("I prefer paragraphs", SignalType.PREFERENCE),
        ("I like detailed explanations", SignalType.PREFERENCE),
        ("I want shorter responses", SignalType.PREFERENCE),
        ("I'd rather see code than prose", SignalType.PREFERENCE),
        ("My name is Rami", SignalType.FACT),
        ("I work at Peripheral", SignalType.FACT),
        ("I live in Amsterdam", SignalType.FACT),
        ("Perfect, that's exactly what I needed", SignalType.REINFORCEMENT),
        ("Exactly!", SignalType.REINFORCEMENT),
        ("Great work on that one", SignalType.REINFORCEMENT),
        ("Yes! that worked", SignalType.REINFORCEMENT),
    ],
)
def test_heuristic_catches_known_patterns(text: str, expected: SignalType) -> None:
    assert detect_signal_heuristic(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "ok",
        "y",
        "thanks",
        "",
        "Tell me about Python",
        "What's the weather?",
        "Can you elaborate?",
        "Sure",
        "Got it",
        "Hmm",
        "I always wear blue",
    ],
)
def test_heuristic_returns_none_for_continuation(text: str) -> None:
    assert detect_signal_heuristic(text) is None
