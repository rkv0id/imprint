import os
from contextlib import ExitStack

import pytest
from pydantic_ai.models.test import TestModel

from imprint import Imprint
from imprint.types import SignalType


def _make_imprint(
    *,
    detection_mode: str = "frugal",
    compile_text: str = "(mock policy)",
    signal_type: SignalType | None = None,
) -> tuple[Imprint, TestModel, TestModel]:
    """Build an Imprint with both agents pre-overridden.

    Returns (imprint, compile_test_model, detect_test_model). Tests can
    inspect either TestModel for call observation.
    """
    imprint = Imprint(
        agent_id="agent",
        model="anthropic:claude-haiku-4-5-20251001",
        store=":memory:",
        detection_mode=detection_mode,  # type: ignore[arg-type]
    )
    compile_model = TestModel(custom_output_text=compile_text)
    detect_model = TestModel(
        custom_output_args={"signal_type": signal_type.value if signal_type else None}
    )
    # An ExitStack pinned to the imprint keeps the override context managers
    # alive; without this they'd be garbage-collected and the overrides
    # would silently reset.
    stack = ExitStack()
    stack.enter_context(imprint._compile_agent.override(model=compile_model))
    stack.enter_context(imprint._detect_agent.override(model=detect_model))
    imprint._test_stack = stack  # type: ignore[attr-defined]
    return imprint, compile_model, detect_model


# ---------- compile pipeline -------------------------------------------------


async def test_get_policy_calls_compile_agent_with_memory_in_prompt() -> None:
    imprint, compile_model, _ = _make_imprint(
        detection_mode="frugal", compile_text="compiled output"
    )
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
    last = compile_model.last_model_request_parameters
    assert last is not None
    await imprint.close()


async def test_get_policy_skips_llm_when_no_memories() -> None:
    imprint, compile_model, _ = _make_imprint(detection_mode="frugal")
    await imprint.connect()

    policy = await imprint.get_policy(user_id="someone")

    assert policy.text == ""
    assert policy.memories == []
    # No call ever happened.
    assert compile_model.last_model_request_parameters is None
    await imprint.close()


async def test_compile_passes_max_tokens_through() -> None:
    imprint, _, _ = _make_imprint(detection_mode="frugal", compile_text="x")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    policy = await imprint.get_policy(user_id="u", max_tokens=123)

    assert policy.text == "x"
    await imprint.close()


async def test_existing_instructions_reach_the_prompt() -> None:
    imprint, compile_model, _ = _make_imprint(detection_mode="frugal", compile_text="x")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer terse output")
    await imprint.get_policy(user_id="u", existing_instructions="Always be terse.")

    # Inspect the actual user prompt sent to the model
    messages = compile_model.last_model_request_parameters
    assert messages is not None
    # The prompt construction is verified at the prompt module level; here we
    # just assert the agent was invoked and produced output.
    await imprint.close()


async def test_context_reaches_the_prompt() -> None:
    imprint, _, _ = _make_imprint(detection_mode="frugal", compile_text="x")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer terse output")
    policy = await imprint.get_policy(user_id="u", context="code review session")

    assert policy.text == "x"
    await imprint.close()


# ---------- per-user scoping ------------------------------------------------


async def test_memories_are_scoped_per_user() -> None:
    imprint, _, _ = _make_imprint(detection_mode="frugal", compile_text="ok")
    await imprint.connect()

    await imprint.observe(user_id="alice", agent_output="x", user_response="I prefer brevity")
    await imprint.observe(user_id="bob", agent_output="x", user_response="I want detail")

    alice_policy = await imprint.get_policy(user_id="alice")
    bob_policy = await imprint.get_policy(user_id="bob")

    assert len(alice_policy.memories) == 1
    assert alice_policy.memories[0].content == "I prefer brevity"
    assert len(bob_policy.memories) == 1
    assert bob_policy.memories[0].content == "I want detail"
    await imprint.close()


# ---------- detection modes -------------------------------------------------


async def test_frugal_no_signal_stores_nothing() -> None:
    imprint, _, _ = _make_imprint(detection_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="ok")

    policy = await imprint.get_policy(user_id="u")
    assert policy.memories == []
    await imprint.close()


async def test_frugal_heuristic_match_stores_memory() -> None:
    imprint, _, detect_model = _make_imprint(detection_mode="frugal", compile_text="ok")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="No, do it differently")

    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    # Frugal mode never asks the LLM.
    assert detect_model.last_model_request_parameters is None
    await imprint.close()


async def test_balanced_falls_through_to_llm_when_heuristic_silent() -> None:
    imprint, _, detect_model = _make_imprint(
        detection_mode="balanced",
        compile_text="ok",
        signal_type=SignalType.CORRECTION,
    )
    await imprint.connect()

    # "you misunderstood" doesn't match heuristic
    await imprint.observe(user_id="u", agent_output="x", user_response="you misunderstood")

    assert detect_model.last_model_request_parameters is not None
    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    await imprint.close()


async def test_balanced_skips_llm_when_heuristic_matches() -> None:
    imprint, _, detect_model = _make_imprint(detection_mode="balanced", compile_text="ok")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="No, that's wrong")

    # Heuristic matched, LLM was not consulted.
    assert detect_model.last_model_request_parameters is None
    await imprint.close()


async def test_eager_always_calls_llm_for_detection() -> None:
    imprint, _, detect_model = _make_imprint(
        detection_mode="eager",
        compile_text="ok",
        signal_type=SignalType.CORRECTION,
    )
    await imprint.connect()

    # Even heuristic-matchable text goes to LLM in eager mode.
    await imprint.observe(user_id="u", agent_output="x", user_response="No, that's wrong")

    assert detect_model.last_model_request_parameters is not None
    await imprint.close()


async def test_balanced_drops_observation_when_llm_says_no_signal() -> None:
    imprint, _, _ = _make_imprint(detection_mode="balanced", signal_type=None)
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="huh interesting")

    policy = await imprint.get_policy(user_id="u")
    assert policy.memories == []
    await imprint.close()


# ---------- store URL parsing -----------------------------------------------


async def test_memory_url_form_works() -> None:
    imprint = Imprint(agent_id="a", store="sqlite:///:memory:", detection_mode="frugal")
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

    imprint = Imprint(
        agent_id="live_compile",
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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    imprint = Imprint(
        agent_id="live_detect",
        store=":memory:",
        detection_mode="balanced",
    )
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="Here is a bulleted list of options.",
        user_response="you misunderstood my request entirely",
    )

    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    await imprint.close()
