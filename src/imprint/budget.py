"""Token counting implementations for Imprint."""

import math


class HeuristicTokenCounter:
    """Estimates token count as ceil(chars / 4).

    Accurate to within ~10% on plain English. Wider error on code,
    multilingual text, or heavily symbolic content. Sufficient for
    budget enforcement; swap for TiktokenTokenCounter or
    AnthropicAPITokenCounter when precision matters.
    """

    def count(self, text: str) -> int:
        return math.ceil(len(text) / 4)
