"""LlamaIndex event handler for Imprint.

ImprintEventHandler hooks into LlamaIndex's Instrumentation dispatcher
and calls imprint.observe() at the end of each query turn.

How it works:

  Events are matched by class name to avoid hard imports of untyped
  llama_index internals. QueryStartEvent captures the user's query.
  QueryEndEvent fires observe():
    agent_output = the engine's response text
    user_response = the query that triggered it

This is an approximation. For exact turn-level control, call
imprint.observe() directly.

Requires: pip install imprint-mem[llamaindex]

Usage:

    from llama_index.core.instrumentation import get_dispatcher
    from imprint.integrations.llamaindex import ImprintEventHandler

    handler = ImprintEventHandler(
        imprint=imprint_instance,
        user_id="rami",
        loop=loop,          # optional MemoryLoop
        context="qa",       # optional scope context
    )

    dispatcher = get_dispatcher()
    dispatcher.add_event_handler(handler)

    # Now any query engine call will feed into imprint automatically.
    response = await query_engine.aquery("What is the capital of France?")
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imprint._core import Imprint, MemoryLoop


def _require_llamaindex(name: str) -> None:
    try:
        import llama_index.core.instrumentation as _lli  # type: ignore[import-untyped]

        _ = _lli
    except ImportError as e:
        raise ImportError(
            f"llama-index-core is required for {name}; "
            "install it with: pip install imprint-mem[llamaindex]"
        ) from e


class ImprintEventHandler:
    """LlamaIndex event handler that feeds query turns into imprint.

    Register with the LlamaIndex instrumentation dispatcher via:
        dispatcher.add_event_handler(handler)

    Each completed query fires an observe() call so imprint can detect
    corrections, preferences, and facts from the Q&A turn.

    user_id identifies whose memory store to update.
    loop is an optional MemoryLoop for learning signal propagation.
    context is passed to observe() for scope inference.
    """

    def __init__(
        self,
        *,
        imprint: Imprint,
        user_id: str,
        loop: MemoryLoop | None = None,
        context: str | None = None,
    ) -> None:
        _require_llamaindex("ImprintEventHandler")
        self._imprint = imprint
        self._user_id = user_id
        self._loop = loop
        self._context = context
        self._pending_query: str = ""
        self._pending_tasks: set[asyncio.Task[None]] = set()

    @classmethod
    def class_name(cls) -> str:
        return "ImprintEventHandler"

    def handle(self, event: Any, **kwargs: Any) -> None:
        """Dispatch LlamaIndex events to imprint.

        Uses class name matching rather than isinstance to avoid hard imports
        of untyped llama_index internals and to stay stable across LlamaIndex
        version changes.
        """
        event_type = type(event).__name__

        if event_type == "QueryStartEvent":
            query = getattr(event, "query", None)
            if isinstance(query, str) and query:
                self._pending_query = query
            elif query is not None:
                qs = getattr(query, "query_str", None)
                if isinstance(qs, str):
                    self._pending_query = qs

        elif event_type == "QueryEndEvent":
            if not self._pending_query:
                return

            response = getattr(event, "response", None)
            if response is None:
                self._pending_query = ""
                return

            response_text = getattr(response, "response", None) or str(response)
            if not isinstance(response_text, str) or not response_text:
                self._pending_query = ""
                return

            query_text = self._pending_query
            self._pending_query = ""

            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(
                    self._imprint.observe(
                        user_id=self._user_id,
                        agent_output=response_text,
                        user_response=query_text,
                        context=self._context,
                    )
                )
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)
            except RuntimeError:
                pass

    async def flush(self) -> None:
        """Await all pending observe() tasks."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
