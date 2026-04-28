"""Imprint facade: top-level SDK entry point."""

import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from imprint.llm import LLMProvider
from imprint.store import Store
from imprint.types import (
    Memory,
    MemorySource,
    MemoryType,
    Signal,
    SignalType,
)


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
        detection_mode: str = "balanced",
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
        # v0.1.0 stub: no signal detection, no memory derivation. The user_response
        # becomes a rule-typed memory verbatim, with one supporting signal. Replaced
        # in slice E when real prediction-error detection lands.
        del agent_output, session_id
        now = datetime.now(UTC)

        signal = Signal(
            id=_new_id("sig"),
            agent_id=self.agent_id,
            user_id=user_id,
            signal_type=SignalType.IMPLICIT,
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

    async def get_policy(
        self,
        *,
        user_id: str,
        context: str | None = None,
        existing_instructions: str | None = None,
        max_tokens: int = 400,
        scopes: list[str] | None = None,
    ) -> Policy:
        # v0.1.0 stub: no compilation, no dedup, no budgeting. Memory contents
        # are concatenated verbatim. The unused params are accepted to keep the
        # API surface stable across slices.
        del context, existing_instructions, max_tokens, scopes

        memories = await self._store.list_memories(self.agent_id, user_id)
        text = "\n".join(f"- {m.content}" for m in memories)
        return Policy(text=text, memories=memories)
