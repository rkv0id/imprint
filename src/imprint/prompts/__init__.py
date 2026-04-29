"""Prompt modules used by Imprint's internal LLM agents.

Each module exposes a SYSTEM constant (the agent instructions) and a
build_user_prompt() function that assembles the per-call user message
from runtime inputs. One module per agent: signal, memory, consolidate,
policy.
"""
