"""Imprint facade: top-level SDK entry point."""

import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from imprint.detect import detect_signal_heuristic
from imprint.prompts import consolidate as consolidate_prompt
from imprint.prompts import memory as memory_prompt
from imprint.prompts import policy as policy_prompt
from imprint.prompts import signal as signal_prompt
from imprint.store import Store
from imprint.types import (
    Memory,
    MemorySource,
    MemoryType,
    Signal,
    SignalType,
)

DetectionMode = Literal["frugal", "balanced", "eager"]

_DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"


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
    compiled_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class Imprint:
    def __init__(
        self,
        *,
        agent_id: str,
        model: str | Model = _DEFAULT_MODEL,
        store: str = "sqlite:///~/.imprint/imprint.db",
        agent_description: str | None = None,
        detection_mode: DetectionMode = "balanced",
        scopes: list[str] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_description = agent_description
        self.detection_mode: DetectionMode = detection_mode
        # 'global' is implicit and always available; we don't require callers
        # to include it in their declared list.
        self.scopes: list[str] = list(scopes) if scopes else []

        self._store = Store(_parse_store_url(store))

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

    async def close(self) -> None:
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
        # Detect first; if no signal, store nothing.
        # Scope inference is deferred to its own slice; for now, callers can
        # pass `scope=` explicitly or get the default of "global".
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

        # Capture existing memories before inserting the candidate.
        existing = await self._store.list_memories(self.agent_id, user_id)

        # Caller-passed scope wins; otherwise use the LLM-derived one.
        # Both go through _resolve_scope, which guards against undeclared
        # scopes (caller typos, LLM hallucinations) by falling back to "global".
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

        await self._store.insert_signal(signal)
        await self._store.insert_memory(memory)
        await self._store.link_signal_to_memory(memory_id=memory.id, signal_id=signal.id)

        # Consolidate against the pre-existing set. The candidate is now in the
        # store, so superseded_by foreign keys to it resolve correctly.
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
        max_tokens: int = 400,
        scopes: list[str] | None = None,
    ) -> Policy:
        memories = await self._store.list_memories(self.agent_id, user_id, scopes=scopes)
        if not memories:
            return Policy(text="", memories=memories)

        user_prompt = policy_prompt.build_user_prompt(
            memories=memories,
            existing_instructions=existing_instructions,
            context=context,
        )
        result = await self._compile_agent.run(
            user_prompt,
            model_settings={"temperature": 0.0, "max_tokens": max_tokens},
        )
        return Policy(text=result.output, memories=memories)

    async def _detect_signal(self, *, agent_output: str, user_response: str) -> SignalType | None:
        if self.detection_mode == "eager":
            return await self._detect_signal_llm(
                agent_output=agent_output, user_response=user_response
            )

        heuristic = detect_signal_heuristic(user_response)
        if self.detection_mode == "frugal":
            return heuristic
        # balanced: heuristic first; fall through to LLM if heuristic is silent
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
        """Decide what to do with each existing memory given this new candidate.

        Mutates the store: deactivates existing memories that the LLM judges
        merged or contradicted by the candidate. Caller is responsible for
        passing the pre-candidate-insertion list of existing memories.
        """
        if not existing:
            return

        prompt = consolidate_prompt.build_user_prompt(
            candidate_type=candidate.type.value,
            candidate_content=candidate.content,
            candidate_signal_type=candidate_signal_type.value,
            existing=existing,
        )
        result = await self._consolidate_agent.run(prompt)

        existing_ids = {m.id for m in existing}
        now = datetime.now(UTC)
        for decision in result.output.decisions:
            # Defensive: ignore decisions referencing ids not in the input set.
            if decision.memory_id not in existing_ids:
                continue
            if decision.action == "merge":
                await self._store.deactivate_memory(decision.memory_id, superseded_by=candidate.id)
            elif decision.action == "contradict":
                await self._store.deactivate_memory(
                    decision.memory_id,
                    superseded_by=candidate.id,
                    valid_until=now,
                )
            # "distinct": no action


def _parse_store_url(url: str) -> str:
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///") :]
        if path == ":memory:":
            return ":memory:"
        return os.path.expanduser(path)
    return url


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _resolve_scope(requested: str | None, declared: list[str]) -> str:
    """Validate a caller-provided scope hint against the declared candidate set.

    Falls back to 'global' when no scope is requested or the requested scope
    is not in the declared set. The fallback prevents an LLM-driven inference
    layer (slice J2) from poisoning storage with hallucinated scope strings.
    """
    if requested is None:
        return "global"
    if requested == "global" or requested in declared:
        return requested
    return "global"
