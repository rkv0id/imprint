"""Imprint facade: top-level SDK entry point."""

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, cast

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from imprint.budget import HeuristicTokenCounter
from imprint.decay import FSRSStaticDecay
from imprint.detect import detect_signal_heuristic
from imprint.prompts import consolidate as consolidate_prompt
from imprint.prompts import memory as memory_prompt
from imprint.prompts import policy as policy_prompt
from imprint.prompts import signal as signal_prompt
from imprint.protocols import (
    DecayModel,
    Embedder,
    EventLogger,
    MemoryStore,
    TokenCounter,
    VectorStore,
)
from imprint.store import NullEventLogger, SQLiteEventLogger, SQLiteMemoryStore
from imprint.types import (
    BudgetExceededError,
    Memory,
    MemorySource,
    MemoryType,
    Signal,
    SignalType,
)

ProcessingMode = Literal["frugal", "balanced", "eager"]

DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

_VALID_PROCESSING_MODES: frozenset[str] = frozenset({"frugal", "balanced", "eager"})


class _SignalDetection(BaseModel):
    """Structured output for the signal-detection agent."""

    signal_type: SignalType | None = None


class _DerivedMemory(BaseModel):
    """Structured output for the memory-derivation agent."""

    memory_type: MemoryType
    content: str
    scope: str = "global"


class _ConsolidationDecision(BaseModel):
    """One decision in a consolidation pass: what to do with one existing memory."""

    memory_id: str
    action: Literal["merge", "contradict", "distinct"]


class _ConsolidationOutput(BaseModel):
    """Structured output for the consolidation agent."""

    decisions: list[_ConsolidationDecision] = []


@dataclass(slots=True)
class Policy:
    text: str
    memories: list[Memory] = field(default_factory=list[Memory])
    dropped_memories: list[Memory] = field(default_factory=list[Memory])
    compiled_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class Imprint:
    def __init__(
        self,
        *,
        agent_id: str,
        model: str | Model = DEFAULT_MODEL,
        store: str | MemoryStore = "sqlite:///~/.imprint/imprint.db",
        event_logger: EventLogger | None = None,
        decay_model: DecayModel | None = None,
        token_counter: TokenCounter | None = None,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        agent_description: str | None = None,
        processing_mode: ProcessingMode | None = None,
        scopes: list[str] | None = None,
    ) -> None:
        self.agent_id = agent_id

        self._ctor_processing_mode = processing_mode
        self._ctor_agent_description = agent_description
        self._ctor_scopes = scopes

        self.processing_mode: ProcessingMode = (
            processing_mode if processing_mode is not None else "balanced"
        )
        self.agent_description: str | None = agent_description

        if scopes is not None:
            seen: set[str] = set()
            deduped: list[str] = []
            for s in scopes:
                if s == "global" or s in seen:
                    continue
                seen.add(s)
                deduped.append(s)
            self.scopes: list[str] = deduped
        else:
            self.scopes = []

        if isinstance(store, str):
            self._store: MemoryStore = SQLiteMemoryStore(_parse_store_url(store))
            self._owns_store = True
        else:
            self._store = store
            self._owns_store = False

        self._event_logger: EventLogger | None = event_logger
        self._decay_model: DecayModel = (
            decay_model if decay_model is not None else FSRSStaticDecay()
        )
        self._token_counter: TokenCounter = (
            token_counter if token_counter is not None else HeuristicTokenCounter()
        )
        self._vector_store: VectorStore | None = vector_store
        self._embedder: Embedder | None = embedder

        self._compile_agent: Agent[None, str] = Agent(
            model,
            instructions=policy_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )
        self._detect_agent: Agent[None, _SignalDetection] = Agent(
            model,
            output_type=_SignalDetection,
            instructions=signal_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )
        self._derive_agent: Agent[None, _DerivedMemory] = Agent(
            model,
            output_type=_DerivedMemory,
            instructions=memory_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )
        self._consolidate_agent: Agent[None, _ConsolidationOutput] = Agent(
            model,
            output_type=_ConsolidationOutput,
            instructions=consolidate_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )

    async def connect(self) -> None:
        await self._store.connect()
        await self._store.init_schema()
        await self._sync_agent_config()
        if self._event_logger is None:
            if isinstance(self._store, SQLiteMemoryStore):
                self._event_logger = SQLiteEventLogger(self._store)
            else:
                self._event_logger = NullEventLogger()

    async def _sync_agent_config(self) -> None:
        stored = await self._store.get_agent_config(self.agent_id)

        if self._ctor_processing_mode is not None:
            self.processing_mode = self._ctor_processing_mode  # pyright: ignore[reportAttributeAccessIssue]
        elif stored is not None and stored.processing_mode in _VALID_PROCESSING_MODES:
            self.processing_mode = cast(ProcessingMode, stored.processing_mode)
        else:
            self.processing_mode = "balanced"

        if self._ctor_agent_description is not None:
            self.agent_description = self._ctor_agent_description
        elif stored is not None:
            self.agent_description = stored.agent_description

        if self._ctor_scopes is not None:
            pass
        elif stored is not None and stored.scopes is not None:
            seen: set[str] = set()
            deduped: list[str] = []
            for s in stored.scopes:
                if s == "global" or s in seen:
                    continue
                seen.add(s)
                deduped.append(s)
            self.scopes = deduped

        await self._store.put_agent_config(
            agent_id=self.agent_id,
            processing_mode=self.processing_mode,
            agent_description=self.agent_description,
            scopes=self.scopes,
        )

    async def close(self) -> None:
        if self._owns_store:
            await self._store.close()

    async def observe(
        self,
        *,
        user_id: str,
        agent_output: str,
        user_response: str,
        context: str | None = None,
        session_id: str | None = None,
        scope: str | None = None,
    ) -> None:
        del session_id

        signal_type = await self._detect_signal(
            agent_output=agent_output, user_response=user_response
        )
        if signal_type is None:
            return

        derived = await self._derive_memory(
            agent_output=agent_output,
            user_response=user_response,
            signal_type=signal_type,
        )

        existing = await self._store.list_memories(self.agent_id, user_id)

        chosen_scope = scope if scope is not None else derived.scope

        now = datetime.now(UTC)
        memory = Memory(
            id=_new_id("mem"),
            agent_id=self.agent_id,
            user_id=user_id,
            type=derived.memory_type,
            scope=_resolve_scope(chosen_scope, self.scopes),
            content=derived.content,
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
        signal = Signal(
            id=_new_id("sig"),
            agent_id=self.agent_id,
            user_id=user_id,
            signal_type=signal_type,
            content=user_response,
            context=context,
            created_at=now,
        )

        await self._store.invalidate_cached_policies(self.agent_id, user_id)

        await self._store.insert_signal(signal)
        await self._store.insert_memory(memory)
        await self._store.link_signal_to_memory(memory_id=memory.id, signal_id=signal.id)

        if self._embedder is not None and self._vector_store is not None:
            embedding = await self._embedder.embed(memory.content)
            await self._vector_store.upsert(memory.id, embedding)

        await self._consolidate_against_existing(
            candidate=memory,
            candidate_signal_type=signal_type,
            existing=existing,
        )

    async def observe_directions(
        self,
        *,
        user_id: str,
        directions: list[str],
        context: str | None = None,
    ) -> None:
        del user_id, directions, context
        raise NotImplementedError("observe_directions() is not implemented yet")

    async def get_policy(
        self,
        *,
        user_id: str,
        context: str | None = None,
        existing_instructions: str | None = None,
        max_input_tokens: int = 8000,
        max_output_tokens: int = 3000,
        on_budget_exceeded: Literal["truncate", "error"] = "truncate",
        scopes: list[str] | None = None,
    ) -> Policy:
        all_memories = await self._store.list_memories(self.agent_id, user_id, scopes=scopes)
        if not all_memories:
            return Policy(text="", memories=[], dropped_memories=[])

        now = datetime.now(UTC)
        kept, dropped = _truncate_to_budget(
            memories=all_memories,
            max_input_tokens=max_input_tokens,
            on_budget_exceeded=on_budget_exceeded,
            decay_model=self._decay_model,
            counter=self._token_counter,
            context=context,
            existing_instructions=existing_instructions,
            agent_description=self.agent_description,
            now=now,
        )

        cache_key = _policy_cache_key(
            agent_id=self.agent_id,
            user_id=user_id,
            memories=kept,
            context=context,
            existing_instructions=existing_instructions,
            max_output_tokens=max_output_tokens,
            scopes=scopes,
        )
        cached = await self._store.get_cached_policy(cache_key)
        if cached is not None:
            cached_text, cached_at = cached
            await self._apply_recall(kept)
            return Policy(
                text=cached_text,
                memories=kept,
                dropped_memories=dropped,
                compiled_at=cached_at,
            )

        user_prompt = policy_prompt.build_user_prompt(
            memories=kept,
            existing_instructions=existing_instructions,
            context=context,
            agent_description=self.agent_description,
        )
        result = await self._compile_agent.run(
            user_prompt,
            model_settings={"temperature": 0.0, "max_tokens": max_output_tokens},
        )
        compiled_at = datetime.now(UTC)
        await self._store.put_cached_policy(
            cache_key=cache_key,
            agent_id=self.agent_id,
            user_id=user_id,
            policy_text=result.output,
            compiled_at=compiled_at,
        )
        await self._apply_recall(kept)
        return Policy(
            text=result.output,
            memories=kept,
            dropped_memories=dropped,
            compiled_at=compiled_at,
        )

    async def _prefilter_candidates(
        self,
        *,
        candidate: Memory,
        existing: list[Memory],
        top_k: int,
        threshold: float,
    ) -> list[Memory]:
        assert self._embedder is not None and self._vector_store is not None
        embedding = await self._embedder.embed(candidate.content)
        hits = await self._vector_store.search(embedding, top_k=top_k)
        existing_by_id = {m.id: m for m in existing}
        result: list[Memory] = []
        for hit_id, distance in hits:
            if hit_id not in existing_by_id:
                continue
            similarity = 1.0 - distance
            if similarity >= threshold:
                result.append(existing_by_id[hit_id])
        return result

    async def _consolidate_frugal_vector(
        self,
        *,
        candidate: Memory,
        existing: list[Memory],
    ) -> None:
        assert self._embedder is not None and self._vector_store is not None
        embedding = await self._embedder.embed(candidate.content)
        hits = await self._vector_store.search(embedding, top_k=5)
        existing_by_id = {m.id: m for m in existing}
        for hit_id, distance in hits:
            if hit_id not in existing_by_id:
                continue
            similarity = 1.0 - distance
            if similarity >= 0.85:
                existing_mem = existing_by_id[hit_id]
                new_stability = self._decay_model.update_on_merge(existing_mem)
                await self._store.update_memory_stability(hit_id, new_stability)
                await self._store.deactivate_memory(hit_id, superseded_by=candidate.id)
                if self._event_logger is not None:
                    await self._event_logger.log(hit_id, "merge", {"superseded_by": candidate.id})

    async def _apply_recall(self, memories: list[Memory]) -> None:
        for m in memories:
            await self._store.increment_recall_count(m.id)
            new_stability = self._decay_model.update_on_recall(m)
            if new_stability != m.stability:
                await self._store.update_memory_stability(m.id, new_stability)
            if self._event_logger is not None:
                await self._event_logger.log(m.id, "recall")

    async def _detect_signal(self, *, agent_output: str, user_response: str) -> SignalType | None:
        if self.processing_mode == "eager":
            return await self._detect_signal_llm(
                agent_output=agent_output, user_response=user_response
            )

        heuristic = detect_signal_heuristic(user_response)
        if self.processing_mode == "frugal":
            return heuristic
        if heuristic is not None:
            return heuristic
        return await self._detect_signal_llm(agent_output=agent_output, user_response=user_response)

    async def _detect_signal_llm(
        self, *, agent_output: str, user_response: str
    ) -> SignalType | None:
        prompt = signal_prompt.build_user_prompt(
            agent_output=agent_output, user_response=user_response
        )
        result = await self._detect_agent.run(prompt)
        return result.output.signal_type

    async def _derive_memory(
        self,
        *,
        agent_output: str,
        user_response: str,
        signal_type: SignalType,
    ) -> _DerivedMemory:
        if self.processing_mode == "frugal":
            return _derive_memory_frugal(user_response=user_response, signal_type=signal_type)

        prompt = memory_prompt.build_user_prompt(
            agent_output=agent_output,
            user_response=user_response,
            signal_type=signal_type.value,
            available_scopes=self.scopes,
        )
        result = await self._derive_agent.run(prompt)
        return result.output

    async def _consolidate_against_existing(
        self,
        *,
        candidate: Memory,
        candidate_signal_type: SignalType,
        existing: list[Memory],
    ) -> None:
        if not existing:
            return

        if self.processing_mode == "frugal":
            if self._embedder is not None and self._vector_store is not None:
                await self._consolidate_frugal_vector(candidate=candidate, existing=existing)
            return

        candidates = existing
        if (
            self.processing_mode == "balanced"
            and self._embedder is not None
            and self._vector_store is not None
        ):
            candidates = await self._prefilter_candidates(
                candidate=candidate, existing=existing, top_k=10, threshold=0.5
            )
        if not candidates:
            return

        prompt = consolidate_prompt.build_user_prompt(
            candidate_type=candidate.type.value,
            candidate_content=candidate.content,
            candidate_signal_type=candidate_signal_type.value,
            existing=candidates,
        )
        result = await self._consolidate_agent.run(prompt)

        existing_by_id = {m.id: m for m in candidates}
        existing_ids = existing_by_id.keys()
        now = datetime.now(UTC)
        for decision in result.output.decisions:
            if decision.memory_id not in existing_ids:
                continue
            existing_mem = existing_by_id[decision.memory_id]
            if decision.action == "merge":
                new_stability = self._decay_model.update_on_merge(existing_mem)
                await self._store.update_memory_stability(decision.memory_id, new_stability)
                await self._store.deactivate_memory(decision.memory_id, superseded_by=candidate.id)
                if self._event_logger is not None:
                    await self._event_logger.log(
                        decision.memory_id,
                        "merge",
                        {"superseded_by": candidate.id},
                    )
            elif decision.action == "contradict":
                new_stability = self._decay_model.update_on_contradict(existing_mem)
                await self._store.update_memory_stability(decision.memory_id, new_stability)
                await self._store.deactivate_memory(
                    decision.memory_id,
                    superseded_by=candidate.id,
                    valid_until=now,
                )
                await self._store.mark_signals_contradicted(decision.memory_id)
                if self._event_logger is not None:
                    await self._event_logger.log(
                        decision.memory_id,
                        "contradict",
                        {"superseded_by": candidate.id},
                    )
            elif decision.action == "distinct":
                if self._event_logger is not None:
                    await self._event_logger.log(decision.memory_id, "distinct")


def _truncate_to_budget(
    *,
    memories: list[Memory],
    max_input_tokens: int,
    on_budget_exceeded: Literal["truncate", "error"],
    decay_model: DecayModel,
    counter: TokenCounter,
    context: str | None,
    existing_instructions: str | None,
    agent_description: str | None,
    now: datetime,
) -> tuple[list[Memory], list[Memory]]:
    def _prompt_tokens(mems: list[Memory]) -> int:
        prompt = policy_prompt.build_user_prompt(
            memories=mems,
            existing_instructions=existing_instructions,
            context=context,
            agent_description=agent_description,
        )
        return counter.count(prompt)

    if _prompt_tokens(memories) <= max_input_tokens:
        return memories, []

    if on_budget_exceeded == "error":
        raise BudgetExceededError(f"memory prompt exceeds max_input_tokens={max_input_tokens}")

    pinned = [m for m in memories if m.pinned]
    droppable = [m for m in memories if not m.pinned]

    droppable.sort(
        key=lambda m: (
            m.type != MemoryType.CONTEXT,
            decay_model.effective_stability(m, now),
            m.created_at,
        )
    )

    dropped: list[Memory] = []
    while droppable and _prompt_tokens(pinned + droppable) > max_input_tokens:
        if len(pinned) + len(droppable) == 1:
            raise BudgetExceededError(
                f"cannot reduce memory set below 1 entry within max_input_tokens={max_input_tokens}"
            )
        dropped.append(droppable.pop(0))

    return pinned + droppable, dropped


def _derive_memory_frugal(*, user_response: str, signal_type: SignalType) -> _DerivedMemory:
    _TYPE_MAP: dict[SignalType, MemoryType] = {
        SignalType.CORRECTION: MemoryType.RULE,
        SignalType.DIRECTION: MemoryType.RULE,
        SignalType.PREFERENCE: MemoryType.RULE,
        SignalType.FACT: MemoryType.FACT,
        SignalType.REINFORCEMENT: MemoryType.CONTEXT,
    }
    memory_type = _TYPE_MAP[signal_type]
    content = " ".join(user_response.split())
    return _DerivedMemory(memory_type=memory_type, content=content, scope="global")


def _parse_store_url(url: str) -> str:
    """Parse a store URL into a SQLite path. Accepts:

    - `sqlite:///abs/path` -> /abs/path
    - `sqlite:///:memory:` -> :memory:
    - `:memory:` -> :memory:
    - bare absolute or relative path -> path (with ~ expansion)

    Rejects empty strings and non-sqlite URL schemes.
    """
    if not url:
        raise ValueError("store URL must be non-empty")
    if "://" in url and not url.startswith("sqlite://"):
        scheme = url.split("://", 1)[0]
        raise ValueError(f"unsupported store URL scheme: {scheme!r} (expected 'sqlite')")
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///") :]
        if path == ":memory:":
            return ":memory:"
        return os.path.expanduser(path)
    return os.path.expanduser(url)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _resolve_scope(requested: str | None, declared: list[str]) -> str:
    if requested is None:
        return "global"
    if requested == "global" or requested in declared:
        return requested
    return "global"


def _policy_cache_key(
    *,
    agent_id: str,
    user_id: str,
    memories: list[Memory],
    context: str | None,
    existing_instructions: str | None,
    max_output_tokens: int,
    scopes: list[str] | None,
) -> str:
    h = hashlib.sha256()
    h.update(b"agent\x00")
    h.update(agent_id.encode("utf-8"))
    h.update(b"\x00user\x00")
    h.update(user_id.encode("utf-8"))
    h.update(b"\x00mem\x00")
    for m in sorted(memories, key=lambda x: x.id):
        h.update(m.id.encode("utf-8"))
        h.update(b"|")
        h.update(m.updated_at.isoformat().encode("utf-8"))
        h.update(b"\x00")
    h.update(b"\x00ctx\x00")
    h.update((context or "").encode("utf-8"))
    h.update(b"\x00inst\x00")
    h.update((existing_instructions or "").encode("utf-8"))
    h.update(b"\x00max\x00")
    h.update(str(max_output_tokens).encode("utf-8"))
    h.update(b"\x00scopes\x00")
    if scopes is None:
        h.update(b"<none>")
    else:
        for s in sorted(scopes):
            h.update(s.encode("utf-8"))
            h.update(b"|")
    return h.hexdigest()
