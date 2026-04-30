from __future__ import annotations

from typing import TYPE_CHECKING

from imprint._core import Imprint, Policy
from imprint.budget import HeuristicTokenCounter
from imprint.decay import FSRSStaticDecay
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
from imprint.store import NullEventLogger, SQLiteEventLogger, SQLiteMemoryStore
from imprint.tokens import AnthropicAPITokenCounter
from imprint.types import (
    BudgetExceededError,
    ContextStat,
    Memory,
    MemorySource,
    MemoryType,
    Signal,
    SignalType,
)
from imprint.vector import SQLiteVecStore
from imprint.voyage import VoyageEmbedder, VoyageTokenCounter

if TYPE_CHECKING:
    from imprint.online import FSRSGradientDecay


__version__ = "0.1.0.post4"


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
    "ContextStat",
    "DecayModel",
    "Deriver",
    "Detector",
    "Embedder",
    "EventLogger",
    "FSRSGradientDecay",
    "FSRSStaticDecay",
    "HeuristicTokenCounter",
    "Imprint",
    "Memory",
    "MemorySource",
    "MemoryStore",
    "MemoryType",
    "NullEventLogger",
    "Policy",
    "SQLiteEventLogger",
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
]
