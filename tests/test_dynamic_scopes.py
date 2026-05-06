"""Tests for dynamic scope creation (dynamic_scopes=True)."""

import pytest
import pytest_asyncio

from imprint import Imprint
from imprint._utils import _MAX_SCOPE_LEN, _levenshtein

# ---------------------------------------------------------------------------
# Unit tests for helpers (no I/O)
# ---------------------------------------------------------------------------


def test_levenshtein_basics() -> None:
    assert _levenshtein("", "") == 0
    assert _levenshtein("abc", "") == 3
    assert _levenshtein("", "abc") == 3
    assert _levenshtein("abc", "abc") == 0
    assert _levenshtein("python", "python") == 0
    assert _levenshtein("pythn", "python") == 1
    assert _levenshtein("pyton", "python") == 1


def test_max_scope_len_is_reasonable() -> None:
    assert _MAX_SCOPE_LEN >= 10
    assert _MAX_SCOPE_LEN <= 100


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
async def test_is_valid_scope_non_empty(dynamic_imprint: Imprint) -> None:
    # Any non-empty, non-global string within length limit is valid.
    assert dynamic_imprint._is_valid_scope("python")
    assert dynamic_imprint._is_valid_scope("fiction")
    assert dynamic_imprint._is_valid_scope("technical-writing")
    assert dynamic_imprint._is_valid_scope("project-alpha")
    assert dynamic_imprint._is_valid_scope("billing")
    # Format is unconstrained -- colons, mixed case after normalization, etc.
    assert dynamic_imprint._is_valid_scope("lang:python")


@pytest.mark.asyncio
async def test_is_valid_scope_invalid(dynamic_imprint: Imprint) -> None:
    assert not dynamic_imprint._is_valid_scope("")
    assert not dynamic_imprint._is_valid_scope("global")
    assert not dynamic_imprint._is_valid_scope("x" * (_MAX_SCOPE_LEN + 1))


@pytest.mark.asyncio
async def test_accept_scope_global_passthrough(dynamic_imprint: Imprint) -> None:
    result = await dynamic_imprint._accept_scope("global")
    assert result == "global"


@pytest.mark.asyncio
async def test_accept_scope_empty_falls_back_to_global(dynamic_imprint: Imprint) -> None:
    result = await dynamic_imprint._accept_scope("")
    assert result == "global"


@pytest.mark.asyncio
async def test_accept_scope_too_long_falls_back_to_global(dynamic_imprint: Imprint) -> None:
    result = await dynamic_imprint._accept_scope("x" * (_MAX_SCOPE_LEN + 1))
    assert result == "global"


@pytest.mark.asyncio
async def test_accept_scope_new_scope_registered(dynamic_imprint: Imprint) -> None:
    assert "python" not in dynamic_imprint.scopes
    result = await dynamic_imprint._accept_scope("python")
    assert result == "python"
    assert "python" in dynamic_imprint.scopes


@pytest.mark.asyncio
async def test_accept_scope_normalizes_to_lowercase(dynamic_imprint: Imprint) -> None:
    # Uppercase gets normalized before validation and registration.
    result = await dynamic_imprint._accept_scope("Python")
    assert result == "python"
    assert "python" in dynamic_imprint.scopes


@pytest.mark.asyncio
async def test_accept_scope_existing_scope_no_duplicate(dynamic_imprint: Imprint) -> None:
    await dynamic_imprint._accept_scope("python")
    count_before = len(dynamic_imprint.scopes)
    result = await dynamic_imprint._accept_scope("python")
    assert result == "python"
    assert len(dynamic_imprint.scopes) == count_before


@pytest.mark.asyncio
async def test_accept_scope_near_duplicate_collapsed(dynamic_imprint: Imprint) -> None:
    await dynamic_imprint._accept_scope("python")
    # One-char typo should collapse to existing.
    result = await dynamic_imprint._accept_scope("pythn")
    assert result == "python"
    assert dynamic_imprint.scopes.count("pythn") == 0


@pytest.mark.asyncio
async def test_accept_scope_persisted_to_agent_config(dynamic_imprint: Imprint) -> None:
    await dynamic_imprint._accept_scope("python")
    await dynamic_imprint._accept_scope("typescript")

    imp2 = Imprint(
        agent_id="test_dynamic",
        store=dynamic_imprint._store,
        dynamic_scopes=True,
    )
    await imp2.connect()
    assert "python" in imp2.scopes
    assert "typescript" in imp2.scopes
    await imp2.close()


@pytest.mark.asyncio
async def test_register_scope_deduplication(dynamic_imprint: Imprint) -> None:
    await dynamic_imprint._register_scope("python")
    await dynamic_imprint._register_scope("python")
    assert dynamic_imprint.scopes.count("python") == 1


@pytest.mark.asyncio
async def test_register_scope_global_ignored(dynamic_imprint: Imprint) -> None:
    count_before = len(dynamic_imprint.scopes)
    await dynamic_imprint._register_scope("global")
    assert len(dynamic_imprint.scopes) == count_before


@pytest.mark.asyncio
async def test_find_canonical_scope_exact_match(dynamic_imprint: Imprint) -> None:
    dynamic_imprint.scopes.append("python")
    result = dynamic_imprint._find_canonical_scope("python")
    assert result == "python"


@pytest.mark.asyncio
async def test_find_canonical_scope_near_miss(dynamic_imprint: Imprint) -> None:
    dynamic_imprint.scopes.append("python")
    result = dynamic_imprint._find_canonical_scope("pythn")
    assert result == "python"


@pytest.mark.asyncio
async def test_find_canonical_scope_no_match(dynamic_imprint: Imprint) -> None:
    dynamic_imprint.scopes.append("python")
    # "billing" is far from "python".
    result = dynamic_imprint._find_canonical_scope("billing")
    assert result is None


@pytest.mark.asyncio
async def test_multiple_distinct_scopes_coexist(dynamic_imprint: Imprint) -> None:
    for scope in ["python", "typescript", "rust", "billing"]:
        await dynamic_imprint._accept_scope(scope)
    for scope in ["python", "typescript", "rust", "billing"]:
        assert scope in dynamic_imprint.scopes
