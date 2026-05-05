"""Framework integration adapters for Imprint.

langchain.py  -- ImprintCallbackHandler for LangChain agents and chains.
llamaindex.py -- ImprintEventHandler for LlamaIndex query engines and agents.

Each integration is a thin wrapper that hooks into the framework's
observability system and calls imprint.observe() at the right points.
"""
