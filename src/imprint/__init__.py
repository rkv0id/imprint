from __future__ import annotations

from typing import TYPE_CHECKING

from imprint._core import Imprint, LLMCompiler, MemoryLoop, Policy
from imprint.budget import HeuristicTokenCounter
from imprint.decay import FSRSStaticDecay
from imprint.integrations.tools import make_anthropic_tools, make_pydantic_ai_tools
from imprint.protocols import (
    AlphaTuner,
    Compiler,
    DecayModel,
    Deriver,
    Detector,
    Embedder,
    EventLogger,
    MemoryStore,
    TokenCounter,
    VectorStore,
)
from imprint.providers.anthropic import AnthropicAPITokenCounter
from imprint.providers.openai import OpenAIEmbedder, OpenAITokenCounter
from imprint.providers.voyage import VoyageEmbedder, VoyageTokenCounter
from imprint.retrieval import BanditAlphaTuner, StaticAlphaTuner
from imprint.stores.postgres import PostgresMemoryStore, PostgresVectorStore
from imprint.stores.sqlite import SQLiteMemoryStore
from imprint.stores.vector import SQLiteVecStore
from imprint.types import (
    BudgetExceededError,
    Memory,
    MemoryEvent,
    MemoryHealth,
    MemoryLineage,
    MemorySource,
    MemoryType,
    Signal,
    SignalType,
)

if TYPE_CHECKING:
    from imprint.online import FSRSGradientDecay


__version__ = "0.5.1"


def __getattr__(name: str) -> object:
    if name == "FSRSGradientDecay":
        from imprint.online import FSRSGradientDecay

        return FSRSGradientDecay
    raise AttributeError(f"module 'imprint' has no attribute {name!r}")


__all__ = [
    "AlphaTuner",
    "AnthropicAPITokenCounter",
    "BanditAlphaTuner",
    "BudgetExceededError",
    "Compiler",
    "DecayModel",
    "Deriver",
    "Detector",
    "Embedder",
    "EventLogger",
    "FSRSGradientDecay",
    "FSRSStaticDecay",
    "HeuristicTokenCounter",
    "Imprint",
    "LLMCompiler",
    "Memory",
    "MemoryEvent",
    "MemoryHealth",
    "MemoryLineage",
    "MemoryLoop",
    "MemorySource",
    "MemoryStore",
    "MemoryType",
    "OpenAIEmbedder",
    "OpenAITokenCounter",
    "Policy",
    "PostgresMemoryStore",
    "PostgresVectorStore",
    "SQLiteMemoryStore",
    "SQLiteVecStore",
    "Signal",
    "SignalType",
    "StaticAlphaTuner",
    "TokenCounter",
    "VectorStore",
    "VoyageEmbedder",
    "VoyageTokenCounter",
    "__version__",
    "make_anthropic_tools",
    "make_pydantic_ai_tools",
]
