from __future__ import annotations

from typing import TYPE_CHECKING

from imprint._core import Imprint, LLMCompiler, MemoryLoop, Policy
from imprint.anthropic import AnthropicAPITokenCounter
from imprint.budget import HeuristicTokenCounter
from imprint.decay import FSRSStaticDecay
from imprint.openai import OpenAIEmbedder, OpenAITokenCounter
from imprint.postgres import PostgresMemoryStore, PostgresVectorStore
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
from imprint.retrieval import BanditAlphaTuner, StaticAlphaTuner
from imprint.store import SQLiteMemoryStore
from imprint.tools import make_anthropic_tools, make_pydantic_ai_tools
from imprint.turso import TursoMemoryStore
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
from imprint.vector import SQLiteVecStore
from imprint.voyage import VoyageEmbedder, VoyageTokenCounter

if TYPE_CHECKING:
    from imprint.online import FSRSGradientDecay


__version__ = "0.4.2"


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
    "TursoMemoryStore",
    "VectorStore",
    "VoyageEmbedder",
    "VoyageTokenCounter",
    "__version__",
    "make_anthropic_tools",
    "make_pydantic_ai_tools",
]
