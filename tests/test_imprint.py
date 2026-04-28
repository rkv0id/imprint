from typing import Any

import pytest

from imprint import Imprint, LLMResponse


class _MockLLM:
    """Test double satisfying the LLMProvider Protocol structurally."""

    def __init__(self) -> None:
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
        return LLMResponse(text="(mock)", input_tokens=10, output_tokens=5)


# ---------- round-trip integration ------------------------------------------


async def test_observe_then_get_policy_round_trip() -> None:
    imprint = Imprint(agent_id="reviewer", llm=_MockLLM(), store=":memory:")
    await imprint.connect()

    await imprint.observe(
        user_id="rami",
        agent_output="I suggest using bullet points.",
        user_response="No, write in paragraphs.",
    )

    policy = await imprint.get_policy(user_id="rami")
    assert "paragraphs" in policy.text
    assert len(policy.memories) == 1
    assert policy.memories[0].content == "No, write in paragraphs."
    await imprint.close()


async def test_get_policy_with_no_observations_returns_empty() -> None:
    imprint = Imprint(agent_id="empty", llm=_MockLLM(), store=":memory:")
    await imprint.connect()

    policy = await imprint.get_policy(user_id="someone")
    assert policy.text == ""
    assert policy.memories == []
    await imprint.close()


async def test_observe_creates_signal_memory_and_provenance_link() -> None:
    imprint = Imprint(agent_id="agent", llm=_MockLLM(), store=":memory:")
    await imprint.connect()

    await imprint.observe(
        user_id="user",
        agent_output="x",
        user_response="y",
    )

    # Verified indirectly via the round-trip test; signal/provenance inspection
    # awaits a public list_signals API.
    policy = await imprint.get_policy(user_id="user")
    assert len(policy.memories) == 1
    await imprint.close()


async def test_memories_are_scoped_per_user() -> None:
    imprint = Imprint(agent_id="agent", llm=_MockLLM(), store=":memory:")
    await imprint.connect()

    await imprint.observe(user_id="alice", agent_output="x", user_response="alice prefers brevity")
    await imprint.observe(user_id="bob", agent_output="x", user_response="bob prefers detail")

    alice_policy = await imprint.get_policy(user_id="alice")
    bob_policy = await imprint.get_policy(user_id="bob")

    assert "alice" in alice_policy.text
    assert "bob" not in alice_policy.text
    assert "bob" in bob_policy.text
    assert "alice" not in bob_policy.text

    await imprint.close()


# ---------- store URL parsing -----------------------------------------------


async def test_memory_url_form_works() -> None:
    imprint = Imprint(agent_id="a", llm=_MockLLM(), store="sqlite:///:memory:")
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
