from typing import Any

import pytest

from imprint import LLMResponse
from imprint.detect import detect_signal_heuristic, detect_signal_llm
from imprint.types import SignalType


class _CannedLLM:
    """LLM stub that returns a fixed response and records calls."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return LLMResponse(text=self.response_text, input_tokens=5, output_tokens=2)


# ---------- heuristic ------------------------------------------------------


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


# ---------- LLM detection ---------------------------------------------------


async def test_llm_detection_returns_signal_for_valid_token() -> None:
    llm = _CannedLLM("CORRECTION")
    result = await detect_signal_llm(llm, agent_output="x", user_response="actually no")
    assert result == SignalType.CORRECTION


async def test_llm_detection_returns_none_for_none_token() -> None:
    llm = _CannedLLM("NONE")
    result = await detect_signal_llm(llm, agent_output="x", user_response="ok")
    assert result is None


async def test_llm_detection_strips_punctuation_and_whitespace() -> None:
    llm = _CannedLLM("  PREFERENCE.  ")
    result = await detect_signal_llm(llm, agent_output="x", user_response="y")
    assert result == SignalType.PREFERENCE


async def test_llm_detection_returns_none_for_unknown_token() -> None:
    llm = _CannedLLM("MAYBE")
    result = await detect_signal_llm(llm, agent_output="x", user_response="y")
    assert result is None


async def test_llm_detection_rejects_implicit_token() -> None:
    """IMPLICIT is the v0.1.0 catch-all; the detector must not produce it."""
    llm = _CannedLLM("IMPLICIT")
    result = await detect_signal_llm(llm, agent_output="x", user_response="y")
    assert result is None


async def test_llm_detection_passes_required_args() -> None:
    llm = _CannedLLM("NONE")
    await detect_signal_llm(llm, agent_output="agent said this", user_response="user said that")

    call = llm.calls[0]
    assert "agent said this" in call["prompt"]
    assert "user said that" in call["prompt"]
    assert call["system"] is not None
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 10
