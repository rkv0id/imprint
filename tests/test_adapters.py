import importlib.util
import os
from typing import cast

import pytest
from helpers import _ConstantEmbedder, _InMemoryVectorStore, _make_imprint

from imprint import SQLiteMemoryStore


async def test_merge_increases_stability() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    known_id = "mem_decay_merge"
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="new",
        consolidation_decisions=[{"memory_id": known_id, "action": "merge"}],
    )
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    existing = Memory(
        id=known_id,
        agent_id="agent",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="old rule",
        source=MemorySource.DETECTED,
        stability=5.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(existing)

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    mems = await store.list_memories("agent", "u", active_only=False)
    merged = next(m for m in mems if m.id == known_id)
    assert merged.stability == 6.0


async def test_contradict_reduces_stability() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    known_id = "mem_decay_contradict"
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="new",
        consolidation_decisions=[{"memory_id": known_id, "action": "contradict"}],
    )
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    existing = Memory(
        id=known_id,
        agent_id="agent",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="old rule",
        source=MemorySource.DETECTED,
        stability=5.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(existing)

    await imprint.observe(user_id="u", agent_output="x", user_response="actually do the opposite")

    mems = await store.list_memories("agent", "u", active_only=False)
    contradicted = next(m for m in mems if m.id == known_id)
    assert contradicted.stability == 0.5


async def test_recall_increments_count() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(derived_content="rule", compile_text="be direct")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    await imprint.get_policy(user_id="u")
    await imprint.get_policy(user_id="u")

    store = cast(SQLiteMemoryStore, imprint._store)
    mems = await store.list_memories("agent", "u")
    assert mems[0].recall_count == 2


async def test_fsrs_static_decay_merge_cap() -> None:
    from datetime import UTC, datetime

    from imprint import FSRSStaticDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSStaticDecay()
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=99.5,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    assert decay.update_on_merge(m) == 100.0


async def test_fsrs_static_decay_contradict_floor() -> None:
    from datetime import UTC, datetime

    from imprint import FSRSStaticDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSStaticDecay()
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=0.5,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    assert decay.update_on_contradict(m) == 0.1


async def test_observe_stores_embedding_when_configured() -> None:
    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder([1.0, 0.0, 0.0])

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="balanced", derived_content="rule")
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    assert len(vec_store._store) == 1
    stored_id = next(iter(vec_store._store))
    mem_id = (await cast(SQLiteMemoryStore, imprint._store).list_memories("agent", "u"))[0].id
    assert stored_id == mem_id


async def test_balanced_prefilter_limits_candidates() -> None:
    """Balanced consolidation with vectors only processes similar memories."""
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    same_vec = [1.0, 0.0, 0.0]
    diff_vec = [0.0, 0.0, 1.0]

    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder(same_vec)

    similar_id = "mem_similar"
    dissimilar_id = "mem_dissimilar"

    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="new rule",
        consolidation_decisions=[{"memory_id": similar_id, "action": "merge"}],
    )
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    for mid, vec, content in [
        (similar_id, same_vec, "similar rule"),
        (dissimilar_id, diff_vec, "unrelated rule"),
    ]:
        await store.insert_memory(
            Memory(
                id=mid,
                agent_id="agent",
                user_id="u",
                type=MemoryType.RULE,
                scope="global",
                content=content,
                source=MemorySource.DETECTED,
                valid_from=now,
                created_at=now,
                updated_at=now,
            )
        )
        await vec_store.upsert(mid, vec)

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    all_mems = await store.list_memories("agent", "u", active_only=False)
    by_id = {m.id: m for m in all_mems}

    assert by_id[similar_id].active is False
    assert by_id[dissimilar_id].active is True


async def test_frugal_vector_consolidation_merges_similar() -> None:
    """Frugal mode with a vector store merges memories above the similarity threshold."""
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    same_vec = [1.0, 0.0, 0.0]
    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder(same_vec)

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    existing_id = "mem_existing"
    existing = Memory(
        id=existing_id,
        agent_id="agent",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="old rule",
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(existing)
    await vec_store.upsert(existing_id, same_vec)

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    all_mems = await store.list_memories("agent", "u", active_only=False)
    merged = next(m for m in all_mems if m.id == existing_id)
    assert merged.active is False


@pytest.mark.live
async def test_voyage_embedder_live() -> None:
    """VoyageEmbedder returns 1024-dim vectors and similar texts are close."""
    if importlib.util.find_spec("voyageai") is None:
        pytest.skip("voyageai not installed (pip install imprint[voyage])")
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    from imprint import VoyageEmbedder

    embedder = VoyageEmbedder(model="voyage-3.5-lite", dim=1024)

    v1 = await embedder.embed("The user prefers concise responses.")
    v2 = await embedder.embed("Keep answers brief and to the point.")
    v3 = await embedder.embed("The capital of France is Paris.")

    assert len(v1) == 1024
    assert len(v2) == 1024
    assert len(v3) == 1024

    import math

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))

    assert cosine(v1, v2) > cosine(v1, v3), (
        "semantically similar texts should be closer than dissimilar ones"
    )


@pytest.mark.live
async def test_voyage_embedder_batch_live() -> None:
    """embed_batch returns one embedding per input, same dim as embed."""
    if importlib.util.find_spec("voyageai") is None:
        pytest.skip("voyageai not installed (pip install imprint[voyage])")
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    from imprint import VoyageEmbedder

    embedder = VoyageEmbedder(model="voyage-3.5-lite", dim=1024)
    texts = ["first text", "second text", "third text"]
    batch = await embedder.embed_batch(texts)

    assert len(batch) == 3
    assert all(len(v) == 1024 for v in batch)


@pytest.mark.live
async def test_anthropic_token_counter_live() -> None:
    """AnthropicAPITokenCounter returns a positive integer for a short string."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from imprint import AnthropicAPITokenCounter

    counter = AnthropicAPITokenCounter()
    count = counter.count("Hello, how can I help you today?")

    assert isinstance(count, int)
    assert count > 0
    assert count < 50


@pytest.mark.live
async def test_anthropic_token_counter_longer_text_live() -> None:
    """Longer text produces more tokens than shorter text."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from imprint import AnthropicAPITokenCounter

    counter = AnthropicAPITokenCounter()
    short = counter.count("Hi.")
    long_text = (
        "This is a much longer piece of text that should produce significantly more tokens"
        " than the short greeting above, because it contains more words and more information."
    )
    long = counter.count(long_text)

    assert long > short


@pytest.mark.live
async def test_voyage_token_counter_live() -> None:
    """VoyageTokenCounter counts tokens locally -- no API call after tokenizer download."""
    if importlib.util.find_spec("voyageai") is None:
        pytest.skip("voyageai not installed (pip install imprint[voyage])")
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    from imprint import VoyageTokenCounter

    counter = VoyageTokenCounter(model="voyage-3.5-lite")

    short = counter.count("Hello.")
    long = counter.count(
        "This is a longer sentence that should produce more tokens than the short one above."
    )

    assert isinstance(short, int)
    assert short > 0
    assert long > short


async def test_sqlite_vec_store_raises_on_missing_dep() -> None:
    """SQLiteVecStore gives a clear ImportError when sqlite-vec is not installed."""
    import sys
    from unittest.mock import patch

    store = cast(SQLiteMemoryStore, _make_imprint()[0]._store)
    await store.connect()
    await store.init_schema()

    import aiosqlite

    from imprint.vector import SQLiteVecStore

    vec_store = SQLiteVecStore(cast(aiosqlite.Connection, store._conn), dim=3)

    with (
        patch.dict(sys.modules, {"sqlite_vec": None}),
        pytest.raises(ImportError, match="sqlite-vec is required"),
    ):
        await vec_store.upsert("m1", [1.0, 0.0, 0.0])

    await store.close()


async def test_voyage_embedder_raises_on_missing_dep() -> None:
    """VoyageEmbedder gives a clear ImportError when voyageai is not installed."""
    import sys
    from unittest.mock import patch

    from imprint import VoyageEmbedder

    embedder = VoyageEmbedder()
    with (
        patch.dict(sys.modules, {"voyageai": None}),
        pytest.raises(ImportError, match="voyageai is required"),
    ):
        await embedder.embed("hello")


async def test_voyage_token_counter_raises_on_missing_dep() -> None:
    """VoyageTokenCounter gives a clear ImportError when voyageai is not installed."""
    import sys
    from unittest.mock import patch

    from imprint import VoyageTokenCounter

    counter = VoyageTokenCounter()
    with (
        patch.dict(sys.modules, {"voyageai": None}),
        pytest.raises(ImportError, match="voyageai is required"),
    ):
        counter.count("hello")


async def test_anthropic_token_counter_raises_on_missing_dep() -> None:
    """AnthropicAPITokenCounter gives a clear ImportError when anthropic is not installed."""
    import sys
    from unittest.mock import patch

    from imprint import AnthropicAPITokenCounter

    counter = AnthropicAPITokenCounter()
    with (
        patch.dict(sys.modules, {"anthropic": None}),
        pytest.raises(ImportError, match="anthropic is required"),
    ):
        counter.count("hello")


async def test_fsrs_gradient_decay_learn_and_predict() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    from datetime import UTC, datetime

    from imprint.online import FSRSGradientDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSGradientDecay(learning_rate=0.1)
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=5.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )

    # Before any learning, effective_stability should be a positive float
    s1 = decay.effective_stability(m, now)
    assert s1 >= 0.1

    # After a positive learning signal, prediction should increase
    for _ in range(20):
        decay.learn(m, now, 1.0)

    s2 = decay.effective_stability(m, now)
    assert s2 > s1


async def test_fsrs_gradient_decay_state_roundtrip() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    from datetime import UTC, datetime

    from imprint.online import FSRSGradientDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSGradientDecay()
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=5.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    for _ in range(5):
        decay.learn(m, now, 0.7)

    state = decay.get_state()
    assert isinstance(state, str)

    decay2 = FSRSGradientDecay()
    decay2.set_state(state)

    pred1 = decay.effective_stability(m, now)
    pred2 = decay2.effective_stability(m, now)
    assert abs(pred1 - pred2) < 0.001


async def test_fsrs_gradient_decay_raises_without_river() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    import sys
    from unittest.mock import patch

    from imprint.online import FSRSGradientDecay

    decay = FSRSGradientDecay()
    decay._model = None  # type: ignore[assignment]

    with (
        patch.dict(sys.modules, {"river": None}),
        pytest.raises(ImportError, match="river is required"),
    ):
        decay._model = decay._build_model()


async def test_observe_feedback_with_gradient_decay() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    from imprint.online import FSRSGradientDecay

    decay = FSRSGradientDecay()
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal", compile_text="ok", derived_content="rule"
    )
    imprint._decay_model = decay
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m1",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="always be concise",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    await imprint.get_policy(user_id="u")
    initial_state = decay.get_state()

    loop = await imprint.open_loop(user_id="u")
    await imprint.get_policy(user_id="u", loop=loop)
    await loop.close(outcome=1.0)
    await imprint.drain()

    final_state = decay.get_state()
    assert initial_state != final_state


async def test_fsrs_static_decay_initial_stability() -> None:
    from datetime import UTC, datetime

    from imprint import FSRSStaticDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSStaticDecay()
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    assert decay.initial_stability(m) == 5.0


async def test_fsrs_static_decay_recall_boosts_stability() -> None:
    from datetime import UTC, datetime

    from imprint import FSRSStaticDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSStaticDecay()
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=7.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    # update_on_recall applies a 5% passive boost.
    result = decay.update_on_recall(m)
    assert result == pytest.approx(7.0 * 1.05, rel=1e-6)


async def test_fsrs_gradient_decay_negative_signal_reduces_prediction() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    from datetime import UTC, datetime

    from imprint.online import FSRSGradientDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSGradientDecay(learning_rate=0.5)
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=5.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )

    for _ in range(30):
        decay.learn(m, now, 1.0)
    high = decay.effective_stability(m, now)

    decay2 = FSRSGradientDecay(learning_rate=0.5)
    for _ in range(30):
        decay2.learn(m, now, -1.0)
    low = decay2.effective_stability(m, now)

    assert high > low


async def test_fsrs_gradient_decay_initial_stability() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    from datetime import UTC, datetime

    from imprint.online import FSRSGradientDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSGradientDecay()
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    assert decay.initial_stability(m) == 5.0


async def test_fsrs_gradient_decay_merge_and_contradict_delegates() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    from datetime import UTC, datetime

    from imprint.online import FSRSGradientDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSGradientDecay()
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=5.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    assert decay.update_on_merge(m) == 6.0
    assert decay.update_on_contradict(m) == 0.5


async def test_fsrs_gradient_decay_corrupted_state_resets_gracefully() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    from imprint.online import FSRSGradientDecay

    decay = FSRSGradientDecay()
    decay.set_state("this is not valid base64 or pickle data !!!")
    # model should remain functional (contextlib.suppress swallowed the error)
    assert decay._model is not None


def test_llm_compiler_has_compile_method() -> None:
    from imprint import LLMCompiler

    compiler = LLMCompiler("anthropic:claude-haiku-4-5-20251001")
    assert hasattr(compiler, "compile")
    assert callable(compiler.compile)


async def test_default_compiler_is_llm_compiler() -> None:
    from imprint import Imprint, LLMCompiler

    imp = Imprint(agent_id="x", store=":memory:")
    assert isinstance(imp._compiler, LLMCompiler)


async def test_compile_agent_shortcut_accessible_without_custom_compiler() -> None:
    from imprint import Imprint

    # _compile_agent must exist so test model overrides still work when no
    # custom compiler is injected.
    imp = Imprint(agent_id="x", store=":memory:")
    assert hasattr(imp, "_compile_agent")


async def test_compiler_param_in_constructor_sets_compiler() -> None:
    from imprint import Imprint
    from imprint.types import Memory

    class _NoopCompiler:
        async def compile(
            self,
            *,
            memories: list[Memory],
            agent_description: str | None,
            context: str | None,
            existing_instructions: str | None,
            max_tokens: int,
        ) -> str:
            return "noop"

    compiler = _NoopCompiler()
    imp = Imprint(agent_id="x", store=":memory:", compiler=compiler)
    assert imp._compiler is compiler
    # _compile_agent should not exist when a custom compiler is provided.
    assert not hasattr(imp, "_compile_agent")


async def test_custom_compiler_output_used_in_policy() -> None:
    from helpers import _make_imprint

    from imprint.types import Memory

    class _FixedCompiler:
        async def compile(
            self,
            *,
            memories: list[Memory],
            agent_description: str | None,
            context: str | None,
            existing_instructions: str | None,
            max_tokens: int,
        ) -> str:
            return "injected policy text"

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    imprint._compiler = _FixedCompiler()
    await imprint.connect()
    await imprint.observe_directions(user_id="u", directions=["be concise"])
    policy = await imprint.get_policy(user_id="u")
    assert policy.text == "injected policy text"


async def test_openai_token_counter_counts_without_api_call() -> None:
    """OpenAITokenCounter counts tokens locally via tiktoken, no API call."""
    if importlib.util.find_spec("tiktoken") is None:
        pytest.skip("tiktoken not installed (pip install imprint[openai])")

    from imprint import OpenAITokenCounter

    counter = OpenAITokenCounter(model="gpt-4o")
    n = counter.count("hello world")
    assert isinstance(n, int)
    assert n > 0
    # tiktoken counts "hello world" as 2 tokens for gpt-4o
    assert n == 2


async def test_openai_token_counter_async() -> None:
    """count_async runs without blocking."""
    if importlib.util.find_spec("tiktoken") is None:
        pytest.skip("tiktoken not installed (pip install imprint[openai])")

    from imprint import OpenAITokenCounter

    counter = OpenAITokenCounter(model="gpt-4o")
    n = await counter.count_async("hello world")
    assert n == 2


async def test_openai_token_counter_unknown_model_falls_back() -> None:
    """Unknown model name falls back to o200k_base encoding without raising."""
    if importlib.util.find_spec("tiktoken") is None:
        pytest.skip("tiktoken not installed (pip install imprint[openai])")

    from imprint import OpenAITokenCounter

    counter = OpenAITokenCounter(model="gpt-nonexistent-model")
    n = counter.count("hello")
    assert n > 0


async def test_openai_embedder_raises_without_dep() -> None:
    """OpenAIEmbedder raises ImportError if openai is not installed."""
    import sys
    from unittest.mock import patch

    from imprint import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    with (
        patch.dict(sys.modules, {"openai": None}),
        pytest.raises(ImportError, match="openai is required"),
    ):
        embedder._client = None
        embedder._get_client()


async def test_openai_embedder_dim_property() -> None:
    """dim reflects the configured or model-default dimension."""
    from imprint import OpenAIEmbedder

    e1 = OpenAIEmbedder(model="text-embedding-3-small")
    assert e1.dim == 1536

    e2 = OpenAIEmbedder(model="text-embedding-3-large")
    assert e2.dim == 3072

    e3 = OpenAIEmbedder(model="text-embedding-3-small", dimensions=512)
    assert e3.dim == 512


async def test_heuristic_token_counter_uses_tiktoken_when_available() -> None:
    """HeuristicTokenCounter uses tiktoken when installed."""
    from imprint import HeuristicTokenCounter
    from imprint import budget as _budget

    counter = HeuristicTokenCounter()
    n = counter.count("hello world")
    assert isinstance(n, int)
    assert n > 0
    # with tiktoken: 2 tokens; without: ceil(11/4) = 3
    if _budget._tiktoken_enc is not None:
        assert n == 2
    else:
        assert n == 3


@pytest.mark.live
async def test_openai_embedder_live() -> None:
    """OpenAIEmbedder returns correct-dim vectors and similar texts are close."""
    import os

    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai not installed (pip install imprint[openai])")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from imprint import OpenAIEmbedder

    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    vec = await embedder.embed("hello world")
    assert len(vec) == 1536
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.live
async def test_openai_embedder_batch_live() -> None:
    """embed_batch returns one vector per input in the correct order."""
    import os

    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai not installed (pip install imprint[openai])")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from imprint import OpenAIEmbedder

    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    texts = ["apple", "banana", "cherry"]
    vecs = await embedder.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 1536 for v in vecs)


@pytest.mark.live
async def test_openai_embedder_dimensions_reduction_live() -> None:
    """dimensions= parameter reduces output size."""
    import os

    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai not installed (pip install imprint[openai])")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from imprint import OpenAIEmbedder

    embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=256)
    vec = await embedder.embed("test")
    assert len(vec) == 256
