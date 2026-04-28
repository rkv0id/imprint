import os
from typing import Any

import pytest

from imprint import Imprint, LLMResponse


class _MockLLM:
    """Test double satisfying the LLMProvider Protocol structurally.

    Optionally branches by the system prompt: signal detection prompts get
    `signal_response`, compile prompts get `compile_response`. This lets a
    single mock serve both LLM calls in an end-to-end observe → compile flow.
    """

    def __init__(
        self,
        compile_response: str = "(mock policy)",
        signal_response: str = "NONE",
    ) -> None:
        self.compile_response = compile_response
        self.signal_response = signal_response
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
        text = (
            self.signal_response if system and "signal" in system.lower() else self.compile_response
        )
        return LLMResponse(text=text, input_tokens=10, output_tokens=5)


# ---------- compile pipeline -------------------------------------------------


async def test_get_policy_calls_llm_with_memory_in_prompt() -> None:
    llm = _MockLLM(compile_response="compiled output")
    imprint = Imprint(agent_id="reviewer", llm=llm, store=":memory:", detection_mode="frugal")
    await imprint.connect()

    await imprint.observe(
        user_id="rami",
        agent_output="I suggest using bullet points.",
        user_response="No, write in paragraphs.",
    )

    policy = await imprint.get_policy(user_id="rami")

    assert policy.text == "compiled output"
    assert len(policy.memories) == 1
    assert policy.memories[0].content == "No, write in paragraphs."
    compile_calls = [c for c in llm.calls if "policy" in (c["system"] or "").lower()]
    assert len(compile_calls) == 1
    assert "No, write in paragraphs." in compile_calls[0]["prompt"]
    await imprint.close()


async def test_get_policy_skips_llm_when_no_memories() -> None:
    llm = _MockLLM()
    imprint = Imprint(agent_id="empty", llm=llm, store=":memory:", detection_mode="frugal")
    await imprint.connect()

    policy = await imprint.get_policy(user_id="someone")

    assert policy.text == ""
    assert policy.memories == []
    assert llm.calls == []
    await imprint.close()


async def test_compile_uses_temperature_zero() -> None:
    llm = _MockLLM()
    imprint = Imprint(agent_id="a", llm=llm, store=":memory:", detection_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    await imprint.get_policy(user_id="u")

    compile_call = next(c for c in llm.calls if "policy" in (c["system"] or "").lower())
    assert compile_call["temperature"] == 0.0
    await imprint.close()


async def test_compile_passes_max_tokens_through() -> None:
    llm = _MockLLM()
    imprint = Imprint(agent_id="a", llm=llm, store=":memory:", detection_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    await imprint.get_policy(user_id="u", max_tokens=123)

    compile_call = next(c for c in llm.calls if "policy" in (c["system"] or "").lower())
    assert compile_call["max_tokens"] == 123
    await imprint.close()


async def test_existing_instructions_reach_the_prompt() -> None:
    llm = _MockLLM()
    imprint = Imprint(agent_id="a", llm=llm, store=":memory:", detection_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer terse output")
    await imprint.get_policy(
        user_id="u",
        existing_instructions="Always be terse.",
    )

    compile_call = next(c for c in llm.calls if "policy" in (c["system"] or "").lower())
    assert "Always be terse." in compile_call["prompt"]
    assert "do not restate" in compile_call["system"].lower()
    await imprint.close()


async def test_context_reaches_the_prompt() -> None:
    llm = _MockLLM()
    imprint = Imprint(agent_id="a", llm=llm, store=":memory:", detection_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer terse output")
    await imprint.get_policy(user_id="u", context="code review session")

    compile_call = next(c for c in llm.calls if "policy" in (c["system"] or "").lower())
    assert "code review session" in compile_call["prompt"]
    await imprint.close()


# ---------- per-user scoping ------------------------------------------------


async def test_memories_are_scoped_per_user_in_compile_prompt() -> None:
    llm = _MockLLM()
    imprint = Imprint(agent_id="agent", llm=llm, store=":memory:", detection_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="alice", agent_output="x", user_response="I prefer brevity")
    await imprint.observe(user_id="bob", agent_output="x", user_response="I want detail")

    await imprint.get_policy(user_id="alice")
    alice_call = next(c for c in reversed(llm.calls) if "policy" in (c["system"] or "").lower())
    assert "I prefer brevity" in alice_call["prompt"]
    assert "I want detail" not in alice_call["prompt"]

    await imprint.get_policy(user_id="bob")
    bob_call = next(c for c in reversed(llm.calls) if "policy" in (c["system"] or "").lower())
    assert "I want detail" in bob_call["prompt"]
    assert "I prefer brevity" not in bob_call["prompt"]
    await imprint.close()


# ---------- detection modes -------------------------------------------------


async def test_observe_skips_storage_when_no_signal_detected_frugal() -> None:
    """Frugal mode + non-signal response = nothing stored, no LLM call."""
    llm = _MockLLM()
    imprint = Imprint(agent_id="a", llm=llm, store=":memory:", detection_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="ok")

    # Nothing got stored — get_policy short-circuits (no memories) → no LLM call.
    policy = await imprint.get_policy(user_id="u")
    assert policy.memories == []
    assert llm.calls == []
    await imprint.close()


async def test_observe_stores_when_heuristic_matches_frugal() -> None:
    llm = _MockLLM()
    imprint = Imprint(agent_id="a", llm=llm, store=":memory:", detection_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="No, do it differently")

    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    # No signal-detection LLM call — heuristic caught it.
    detection_calls = [c for c in llm.calls if "signal" in (c["system"] or "").lower()]
    assert detection_calls == []
    await imprint.close()


async def test_balanced_falls_through_to_llm_when_heuristic_silent() -> None:
    """Balanced mode: heuristic doesn't match, LLM gets asked; LLM says CORRECTION."""
    llm = _MockLLM(signal_response="CORRECTION")
    imprint = Imprint(agent_id="a", llm=llm, store=":memory:", detection_mode="balanced")
    await imprint.connect()

    # "you misunderstood" doesn't match any heuristic pattern
    await imprint.observe(user_id="u", agent_output="x", user_response="you misunderstood")

    detection_calls = [c for c in llm.calls if "signal" in (c["system"] or "").lower()]
    assert len(detection_calls) == 1

    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    await imprint.close()


async def test_balanced_skips_llm_when_heuristic_matches() -> None:
    """Balanced mode: heuristic matches → no LLM call for detection."""
    llm = _MockLLM()
    imprint = Imprint(agent_id="a", llm=llm, store=":memory:", detection_mode="balanced")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="No, that's wrong")

    detection_calls = [c for c in llm.calls if "signal" in (c["system"] or "").lower()]
    assert detection_calls == []
    await imprint.close()


async def test_eager_always_calls_llm_for_detection() -> None:
    """Eager mode: LLM gets called even when heuristic would match."""
    llm = _MockLLM(signal_response="CORRECTION")
    imprint = Imprint(agent_id="a", llm=llm, store=":memory:", detection_mode="eager")
    await imprint.connect()

    # Heuristic would catch this as CORRECTION, but eager mode skips heuristic.
    await imprint.observe(user_id="u", agent_output="x", user_response="No, that's wrong")

    detection_calls = [c for c in llm.calls if "signal" in (c["system"] or "").lower()]
    assert len(detection_calls) == 1
    await imprint.close()


async def test_balanced_drops_observation_when_llm_says_none() -> None:
    """Balanced fallback: heuristic silent + LLM says NONE → nothing stored."""
    llm = _MockLLM(signal_response="NONE")
    imprint = Imprint(agent_id="a", llm=llm, store=":memory:", detection_mode="balanced")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="huh interesting")

    policy = await imprint.get_policy(user_id="u")
    assert policy.memories == []
    await imprint.close()


# ---------- store URL parsing -----------------------------------------------


async def test_memory_url_form_works() -> None:
    imprint = Imprint(
        agent_id="a",
        llm=_MockLLM(),
        store="sqlite:///:memory:",
        detection_mode="frugal",
    )
    await imprint.connect()
    policy = await imprint.get_policy(user_id="u")
    assert policy.text == ""
    await imprint.close()


# ---------- CLI -------------------------------------------------------------


def test_cli_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    from imprint.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0

    out = capsys.readouterr().out.lower()
    assert "imprint" in out


def test_cli_version_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    from imprint import __version__
    from imprint.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert __version__ in (captured.out + captured.err)


# ---------- live --------------------------------------------------------------


@pytest.mark.live
async def test_compile_via_anthropic_live() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from imprint.providers import AnthropicProvider

    imprint = Imprint(
        agent_id="live_compile",
        llm=AnthropicProvider(),
        store=":memory:",
        detection_mode="frugal",
    )
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="I prefer concise responses without bullet points",
    )
    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="Always cite sources with URLs when making factual claims",
    )

    policy = await imprint.get_policy(
        user_id="u",
        existing_instructions="You are a helpful assistant.",
        max_tokens=300,
    )

    assert policy.text.strip()
    assert len(policy.memories) == 2
    text_lower = policy.text.lower()
    assert any(word in text_lower for word in ["concise", "url", "source", "cite"])
    await imprint.close()


@pytest.mark.live
async def test_signal_detection_via_anthropic_live() -> None:
    """Real Anthropic call: balanced fallback for an ambiguous correction."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from imprint.providers import AnthropicProvider

    imprint = Imprint(
        agent_id="live_detect",
        llm=AnthropicProvider(),
        store=":memory:",
        detection_mode="balanced",
    )
    await imprint.connect()

    # Phrasing avoids heuristic patterns; LLM must classify it.
    await imprint.observe(
        user_id="u",
        agent_output="Here is a bulleted list of options.",
        user_response="you misunderstood my request entirely",
    )

    # Heuristic doesn't match "you misunderstood" → LLM was asked → it should
    # have flagged this as a signal (likely CORRECTION) → memory stored.
    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    await imprint.close()
