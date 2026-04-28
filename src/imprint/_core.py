"""Imprint facade: top-level SDK entry point."""

import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from imprint.detect import detect_signal_heuristic, detect_signal_llm
from imprint.llm import LLMProvider
from imprint.prompts import policy as policy_prompt
from imprint.store import Store
from imprint.types import (
    Memory,
    MemorySource,
    MemoryType,
    Signal,
    SignalType,
)

DetectionMode = Literal["frugal", "balanced", "eager"]


@dataclass
class Policy:
    text: str
    memories: list[Memory] = field(default_factory=lambda: [])
    compiled_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _parse_store_url(url: str) -> str:
    path = url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else url
    if path == ":memory:":
        return ":memory:"
    return os.path.expanduser(path)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Imprint:
    def __init__(
        self,
        *,
        agent_id: str,
        llm: LLMProvider,
        store: str = "sqlite:///~/.imprint/imprint.db",
        agent_description: str | None = None,
        detection_mode: DetectionMode = "balanced",
    ) -> None:
        self.agent_id = agent_id
        self.agent_description = agent_description
        self.detection_mode = detection_mode
        self.llm = llm
        self._store = Store(_parse_store_url(store))

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
    ) -> None:
        # Slice G: real signal detection. If no signal, store nothing.
        # Memory derivation is still hard-coded — RULE type, verbatim content —
        # and gets replaced in slice H.
        del session_id

        signal_type = await self._detect_signal(
            agent_output=agent_output, user_response=user_response
        )
        if signal_type is None:
            return

        now = datetime.now(UTC)
        signal = Signal(
            id=_new_id("sig"),
            agent_id=self.agent_id,
            user_id=user_id,
            signal_type=signal_type,
            content=user_response,
            context=context,
            created_at=now,
        )
        memory = Memory(
            id=_new_id("mem"),
            agent_id=self.agent_id,
            user_id=user_id,
            type=MemoryType.RULE,
            scope="global",
            content=user_response,
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )

        await self._store.insert_signal(signal)
        await self._store.insert_memory(memory)
        await self._store.link_signal_to_memory(memory_id=memory.id, signal_id=signal.id)

    async def _detect_signal(self, *, agent_output: str, user_response: str) -> SignalType | None:
        if self.detection_mode == "eager":
            return await detect_signal_llm(
                self.llm, agent_output=agent_output, user_response=user_response
            )

        heuristic = detect_signal_heuristic(user_response)
        if self.detection_mode == "frugal":
            return heuristic
        # balanced: heuristic first; fall through to LLM if heuristic is silent
        if heuristic is not None:
            return heuristic
        return await detect_signal_llm(
            self.llm, agent_output=agent_output, user_response=user_response
        )

    async def get_policy(
        self,
        *,
        user_id: str,
        context: str | None = None,
        existing_instructions: str | None = None,
        max_tokens: int = 400,
        scopes: list[str] | None = None,
    ) -> Policy:
        # `scopes` is accepted for forward compatibility; scope filtering is
        # plumbed through Store.list_memories already, but threading user-supplied
        # scopes through is its own slice.
        del scopes

        memories = await self._store.list_memories(self.agent_id, user_id)
        if not memories:
            return Policy(text="", memories=memories)

        user_prompt = policy_prompt.build_user_prompt(
            memories=memories,
            existing_instructions=existing_instructions,
            context=context,
        )
        response = await self.llm.complete(
            user_prompt,
            system=policy_prompt.SYSTEM,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return Policy(text=response.text, memories=memories)
