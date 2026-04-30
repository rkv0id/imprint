from pathlib import Path

import pytest
from helpers import _make_imprint

from imprint import Imprint


async def test_memory_url_form_works() -> None:
    imprint = Imprint(agent_id="a", store="sqlite:///:memory:", processing_mode="frugal")
    await imprint.connect()
    policy = await imprint.get_policy(user_id="u")
    assert policy.text == ""
    await imprint.close()


def test_empty_store_url_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Imprint(agent_id="a", store="")


def test_unsupported_scheme_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported store URL scheme"):
        Imprint(agent_id="a", store="postgres://localhost/db")


async def test_agent_config_scopes_persist_across_reconnect(tmp_path: Path) -> None:
    db = str(tmp_path / "test.db")

    first = Imprint(
        agent_id="agent", store=db, scopes=["code", "personal"], processing_mode="eager"
    )
    await first.connect()
    await first.close()

    second = Imprint(agent_id="agent", store=db)
    await second.connect()

    assert second.scopes == ["code", "personal"]
    assert second.processing_mode == "eager"
    await second.close()


async def test_agent_config_constructor_overrides_stored(tmp_path: Path) -> None:
    db = str(tmp_path / "test.db")

    first = Imprint(agent_id="agent", store=db, scopes=["X"], processing_mode="frugal")
    await first.connect()
    await first.close()

    second = Imprint(agent_id="agent", store=db, scopes=["Y"], processing_mode="eager")
    await second.connect()

    assert second.scopes == ["Y"]
    assert second.processing_mode == "eager"
    await second.close()

    third = Imprint(agent_id="agent", store=db)
    await third.connect()

    assert third.scopes == ["Y"]
    assert third.processing_mode == "eager"
    await third.close()


async def test_agent_config_defaults_when_no_stored_config() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint()
    await imprint.connect()

    assert imprint.processing_mode == "frugal"
    assert imprint.scopes == []


async def test_processing_mode_persists_across_reconnect(tmp_path: Path) -> None:
    db = str(tmp_path / "test.db")
    first = Imprint(agent_id="agent", store=db, processing_mode="eager")
    await first.connect()
    await first.close()

    second = Imprint(agent_id="agent", store=db)
    await second.connect()

    assert second.processing_mode == "eager"
    await second.close()


async def test_agent_description_persists_across_reconnect(
    tmp_path: pytest.TempPathFactory,
) -> None:
    db = str(tmp_path / "test.db")  # type: ignore[operator]

    first = Imprint(
        agent_id="agent",
        model="anthropic:claude-haiku-4-5-20251001",
        store=db,
        agent_description="A helpful coding assistant.",
    )
    await first.connect()
    await first.close()

    second = Imprint(
        agent_id="agent",
        model="anthropic:claude-haiku-4-5-20251001",
        store=db,
    )
    await second.connect()
    assert second.agent_description == "A helpful coding assistant."
    await second.close()


async def test_bandit_state_persists_across_reconnect(tmp_path: pytest.TempPathFactory) -> None:
    from imprint import BanditAlphaTuner

    db = str(tmp_path / "test.db")  # type: ignore[operator]

    tuner = BanditAlphaTuner()
    await tuner.update(0.3, 1.0)
    await tuner.update(0.7, 0.0)
    expected_state = tuner.get_state()

    first = Imprint(
        agent_id="agent",
        model="anthropic:claude-haiku-4-5-20251001",
        store=db,
        alpha_tuner=tuner,
    )
    await first.connect()
    import json as _json

    await first._store.put_alpha_tuner_state("agent", _json.dumps(expected_state))
    await first.close()

    tuner2 = BanditAlphaTuner()
    second = Imprint(
        agent_id="agent",
        model="anthropic:claude-haiku-4-5-20251001",
        store=db,
        alpha_tuner=tuner2,
    )
    await second.connect()
    assert tuner2.get_state() == expected_state
    await second.close()
