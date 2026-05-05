"""Tests for dynamic scope creation (dynamic_scopes=True)."""

import pytest
import pytest_asyncio

from imprint import Imprint
from imprint._core import _SCOPE_RE, _levenshtein

# ---------------------------------------------------------------------------
# Unit tests for helpers (no I/O)
# ---------------------------------------------------------------------------


def test_scope_regex_valid() -> None:
    valid = [
        "lang:python",
        "topic:auth",
        "project:checkout",
        "domain:finance",
        "lang:type-script",
        "domain:finance-api",
        "x:y",
    ]
    for s in valid:
        assert _SCOPE_RE.match(s), f"expected valid: {s}"


def test_scope_regex_invalid() -> None:
    invalid = [
        "python",  # no colon
        "Lang:Python",  # uppercase
        "lang:python:extra",  # two colons
        "lang_python",  # underscore instead of colon
        ":value",  # empty category
        "lang:",  # empty value
        "lang: python",  # space
        "LANG:PYTHON",  # all uppercase
    ]
    for s in invalid:
        assert not _SCOPE_RE.match(s), f"expected invalid: {s}"


def test_levenshtein_basics() -> None:
    assert _levenshtein("", "") == 0
    assert _levenshtein("abc", "") == 3
    assert _levenshtein("", "abc") == 3
    assert _levenshtein("abc", "abc") == 0
    assert _levenshtein("lang:python", "lang:python") == 0
    assert _levenshtein("lang:pythn", "lang:python") == 1  # one insertion
    assert _levenshtein("lang:pyton", "lang:python") == 1  # one insertion
    assert _levenshtein("lan:python", "lang:python") == 1  # one insertion


# ---------------------------------------------------------------------------
# Integration tests using a live in-memory Imprint instance
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def dynamic_imprint() -> Imprint:  # type: ignore[misc]
    imp = Imprint(
        agent_id="test_dynamic",
        store=":memory:",
        processing_mode="balanced",
        dynamic_scopes=True,
    )
    await imp.connect()
    return imp


@pytest.mark.asyncio
async def test_is_valid_scope(dynamic_imprint: Imprint) -> None:
    assert dynamic_imprint._is_valid_scope("lang:python")
    assert dynamic_imprint._is_valid_scope("topic:auth")
    assert not dynamic_imprint._is_valid_scope("python")
    assert not dynamic_imprint._is_valid_scope("Lang:Python")


@pytest.mark.asyncio
async def test_accept_scope_global_passthrough(dynamic_imprint: Imprint) -> None:
    result = await dynamic_imprint._accept_scope("global")
    assert result == "global"


@pytest.mark.asyncio
async def test_accept_scope_empty_falls_back_to_global(dynamic_imprint: Imprint) -> None:
    result = await dynamic_imprint._accept_scope("")
    assert result == "global"


@pytest.mark.asyncio
async def test_accept_scope_invalid_format_falls_back_to_global(
    dynamic_imprint: Imprint,
) -> None:
    # No colon at all -> invalid, falls back to global.
    result = await dynamic_imprint._accept_scope("python")
    assert result == "global"

    # Uppercase is normalized to lowercase before validation, so it is accepted.
    result = await dynamic_imprint._accept_scope("Lang:Python")
    assert result == "lang:python"

    # Two colons -> invalid, falls back to global.
    result = await dynamic_imprint._accept_scope("lang:python:extra")
    assert result == "global"


@pytest.mark.asyncio
async def test_accept_scope_new_scope_registered(dynamic_imprint: Imprint) -> None:
    assert "lang:python" not in dynamic_imprint.scopes
    result = await dynamic_imprint._accept_scope("lang:python")
    assert result == "lang:python"
    assert "lang:python" in dynamic_imprint.scopes


@pytest.mark.asyncio
async def test_accept_scope_existing_scope_returned_as_is(dynamic_imprint: Imprint) -> None:
    # Register once.
    await dynamic_imprint._accept_scope("lang:python")
    count_before = len(dynamic_imprint.scopes)

    # Accept again -- should not add a duplicate.
    result = await dynamic_imprint._accept_scope("lang:python")
    assert result == "lang:python"
    assert len(dynamic_imprint.scopes) == count_before


@pytest.mark.asyncio
async def test_accept_scope_near_duplicate_collapsed(dynamic_imprint: Imprint) -> None:
    # Register the canonical form.
    await dynamic_imprint._accept_scope("lang:python")

    # Propose a one-char typo -- should collapse to existing.
    result = await dynamic_imprint._accept_scope("lang:pythn")
    assert result == "lang:python"
    # No new scope should have been added.
    assert dynamic_imprint.scopes.count("lang:pythn") == 0


@pytest.mark.asyncio
async def test_accept_scope_persisted_to_agent_config(dynamic_imprint: Imprint) -> None:
    await dynamic_imprint._accept_scope("lang:python")
    await dynamic_imprint._accept_scope("lang:typescript")

    # Create a second instance with the same store -- scopes should reload.
    imp2 = Imprint(
        agent_id="test_dynamic",
        store=dynamic_imprint._store,  # share the store
        dynamic_scopes=True,
    )
    await imp2.connect()
    assert "lang:python" in imp2.scopes
    assert "lang:typescript" in imp2.scopes
    await imp2.close()


@pytest.mark.asyncio
async def test_register_scope_deduplication(dynamic_imprint: Imprint) -> None:
    await dynamic_imprint._register_scope("lang:python")
    await dynamic_imprint._register_scope("lang:python")
    assert dynamic_imprint.scopes.count("lang:python") == 1


@pytest.mark.asyncio
async def test_register_scope_global_ignored(dynamic_imprint: Imprint) -> None:
    count_before = len(dynamic_imprint.scopes)
    await dynamic_imprint._register_scope("global")
    assert len(dynamic_imprint.scopes) == count_before


@pytest.mark.asyncio
async def test_find_canonical_scope_exact_match(dynamic_imprint: Imprint) -> None:
    dynamic_imprint.scopes.append("lang:python")
    result = dynamic_imprint._find_canonical_scope("lang:python")
    assert result == "lang:python"


@pytest.mark.asyncio
async def test_find_canonical_scope_near_miss(dynamic_imprint: Imprint) -> None:
    dynamic_imprint.scopes.append("lang:python")
    # Distance 1 typo.
    result = dynamic_imprint._find_canonical_scope("lang:pythn")
    assert result == "lang:python"


@pytest.mark.asyncio
async def test_find_canonical_scope_no_match(dynamic_imprint: Imprint) -> None:
    dynamic_imprint.scopes.append("lang:python")
    # "lang:rust" is distance 4 from "lang:python" -- no near-duplicate.
    result = dynamic_imprint._find_canonical_scope("lang:rust")
    assert result is None


@pytest.mark.asyncio
async def test_multiple_distinct_scopes_coexist(dynamic_imprint: Imprint) -> None:
    for scope in ["lang:python", "lang:typescript", "lang:rust", "topic:auth"]:
        await dynamic_imprint._accept_scope(scope)
    for scope in ["lang:python", "lang:typescript", "lang:rust", "topic:auth"]:
        assert scope in dynamic_imprint.scopes
