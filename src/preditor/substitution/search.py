"""This module implements nodes in the implicit graph of sentence variants."""

import dataclasses
from typing import Callable, Optional

from preditor import caching


@dataclasses.dataclass(frozen=True)
class SearchNode:
    """A node in the implicit graph of sentence variants."""

    text: str
    nlp: float
    num_forms: int
    cache: Optional[caching.LazyCache] = None

    @property
    def cache_len(self) -> int:
        return caching.cache_len(self.cache)


ScoreKey = Callable[[SearchNode], float]


def nlp_key(node: SearchNode) -> float:
    """Straightforward scoring with NLP."""
    return node.nlp


def lp_key(node: SearchNode, alpha: float = 0.5) -> float:
    """Normalize score by a function of length."""
    factor = (5 + 1)**alpha / (5 + node.num_forms)**alpha
    return node.nlp * factor
