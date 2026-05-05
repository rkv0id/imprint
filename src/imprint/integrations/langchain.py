"""LangChain callback handler for Imprint.

ImprintCallbackHandler hooks into LangChain's callback system and calls
imprint.observe() at the end of each agent turn.

How it works:

  on_chain_start captures the user's input for this invocation.
  on_llm_end captures the last raw LLM generation (intermediate reasoning).
  on_agent_finish pairs them and fires observe():
    agent_output = last LLM generation
    user_response = user input captured from on_chain_start

This is an approximation. The observe() call detects memory signals from
whatever text is in those two fields. For exact turn-level control (e.g.
in multi-user apps or agentic loops with tool calls), call imprint.observe()
directly and do not use this handler.

Requires: pip install imprint-mem[langchain]

Usage:

    from imprint.integrations.langchain import ImprintCallbackHandler

    handler = ImprintCallbackHandler(
        imprint=imprint_instance,
        user_id="rami",
        loop=loop,            # optional MemoryLoop
        context="coding",     # optional scope context
    )

    chain = your_chain.with_config(callbacks=[handler])
    result = await chain.ainvoke({"input": user_message})
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imprint._core import Imprint, MemoryLoop


def _require_langchain(name: str) -> Any:
    try:
        import langchain_core  # type: ignore[import-untyped]

        return langchain_core
    except ImportError as e:
        raise ImportError(
            f"langchain-core is required for {name}; "
            "install it with: pip install imprint-mem[langchain]"
        ) from e


class ImprintCallbackHandler:
    """LangChain BaseCallbackHandler that feeds agent turns into imprint.

    Attach to any LangChain chain or agent via the callbacks= parameter.
    Each agent finish triggers an observe() call so imprint can detect
    corrections, preferences, and facts from the conversation.

    user_id identifies whose memory store to update.
    loop is an optional MemoryLoop opened before this invocation; when
    provided, correct() and reinforce() tools can close it with the right
    signal after the agent responds.
    context is passed to observe() and get_policy() for scope inference.
    """

    def __init__(
        self,
        *,
        imprint: Imprint,
        user_id: str,
        loop: MemoryLoop | None = None,
        context: str | None = None,
    ) -> None:
        _require_langchain("ImprintCallbackHandler")
        self._imprint = imprint
        self._user_id = user_id
        self._loop = loop
        self._context = context
        self._last_generation: str = ""
        self._chain_input: str = ""
        self._pending_tasks: set[asyncio.Task[None]] = set()

    # --- LangChain callback protocol ---

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Capture user input at chain invocation time."""
        for key in ("input", "human_input", "question", "query", "message"):
            val = inputs.get(key)
            if isinstance(val, str) and val:
                self._chain_input = val
                return

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Capture the last LLM generation text."""
        try:
            gen = response.generations[0][0]
            text = getattr(gen, "text", None) or getattr(gen, "message", None)
            if text is None and hasattr(gen, "message"):
                text = getattr(gen.message, "content", None)
            if isinstance(text, str) and text:
                self._last_generation = text
        except (IndexError, AttributeError, TypeError):
            pass

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        """Fire observe() when the agent completes a turn."""
        try:
            final_output = (
                finish.return_values.get("output", "")
                if hasattr(finish, "return_values")
                else str(finish)
            )
        except Exception:
            final_output = ""

        agent_output = self._last_generation or final_output
        user_response = self._chain_input or final_output

        if not agent_output or not user_response:
            return

        # Schedule observe() as a fire-and-forget task. We do not await here
        # because on_agent_finish is synchronous in LangChain's callback protocol.
        # The developer can call imprint.drain() if they need to ensure completion.
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._imprint.observe(
                    user_id=self._user_id,
                    agent_output=agent_output,
                    user_response=user_response,
                    context=self._context,
                )
            )
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
        except RuntimeError:
            # No running event loop -- caller is in a sync context.
            # The developer must call asyncio.run(handler.flush()) manually.
            pass

        # Reset per-turn state.
        self._last_generation = ""
        self._chain_input = ""

    async def flush(self) -> None:
        """Await all pending observe() tasks. Useful in tests and sync contexts."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
