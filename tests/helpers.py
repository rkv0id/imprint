from contextlib import ExitStack

from pydantic_ai.models.test import TestModel

from imprint import Imprint
from imprint.types import SignalType

__all__ = ["_ConstantEmbedder", "_InMemoryVectorStore", "_make_imprint"]


def _make_imprint(
    *,
    processing_mode: str = "frugal",
    compile_text: str = "(mock policy)",
    signal_type: SignalType | None = None,
    derived_type: str = "rule",
    derived_content: str = "(derived content)",
    derived_scope: str = "global",
    consolidation_decisions: list[dict[str, str]] | None = None,
    validation_verdicts: list[dict[str, str]] | None = None,
    scopes: list[str] | None = None,
    feedback_timeout: int = 3600,
) -> tuple[Imprint, TestModel, TestModel, TestModel, TestModel, TestModel]:
    """Build an Imprint with all five agents pre-overridden.

    Returns (imprint, compile_model, detect_model, derive_model,
             consolidate_model, validate_model).
    """
    imprint = Imprint(
        agent_id="agent",
        model="anthropic:claude-haiku-4-5-20251001",
        store=":memory:",
        processing_mode=processing_mode,  # type: ignore[arg-type]
        scopes=scopes,
        feedback_timeout=feedback_timeout,
    )
    compile_model = TestModel(custom_output_text=compile_text)
    detect_model = TestModel(
        custom_output_args={"signal_type": signal_type.value if signal_type else None}
    )
    derive_model = TestModel(
        custom_output_args={
            "memory_type": derived_type,
            "content": derived_content,
            "scope": derived_scope,
        }
    )
    consolidate_model = TestModel(custom_output_args={"decisions": consolidation_decisions or []})
    validate_model = TestModel(custom_output_args={"verdicts": validation_verdicts or []})
    stack = ExitStack()
    stack.enter_context(imprint._compile_agent.override(model=compile_model))
    stack.enter_context(imprint._detect_agent.override(model=detect_model))
    stack.enter_context(imprint._derive_agent.override(model=derive_model))
    stack.enter_context(imprint._consolidate_agent.override(model=consolidate_model))
    stack.enter_context(imprint._validate_agent.override(model=validate_model))
    stack.enter_context(imprint._attribute_agent.override(model=validate_model))
    imprint._test_stack = stack  # type: ignore[attr-defined]
    return imprint, compile_model, detect_model, derive_model, consolidate_model, validate_model


class _InMemoryVectorStore:
    """Exact cosine similarity vector store for testing."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    async def upsert(self, id: str, embedding: list[float]) -> None:
        self._store[id] = embedding

    async def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        import math

        def cosine_distance(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 1.0
            return 1.0 - dot / (na * nb)

        results = [(id, cosine_distance(embedding, vec)) for id, vec in self._store.items()]
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    async def delete(self, id: str) -> None:
        self._store.pop(id, None)


class _ConstantEmbedder:
    """Test embedder that returns a fixed vector for any input."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    @property
    def dim(self) -> int:
        return len(self._vector)

    async def embed(self, text: str) -> list[float]:
        return self._vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]
