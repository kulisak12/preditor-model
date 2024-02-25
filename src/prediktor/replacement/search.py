import dataclasses
from typing import Callable, Optional

from prediktor.replacement import caching


@dataclasses.dataclass(frozen=True)
class SearchNode:
    text: str
    nlp: float
    num_forms: int
    cache: Optional[caching.Cache] = None

    @property
    def cache_len(self) -> int:
        if self.cache is None:
            return 0
        return self.cache[0][0].shape[2]


ScoreKey = Callable[[SearchNode], float]


def nlp_key(node: SearchNode) -> float:
    """Straightforward scoring with NLP."""
    return node.nlp


def lp_key(node: SearchNode, alpha: float = 0.5) -> float:
    """Normalize score by a function of length."""
    factor = (5 + 1)**alpha / (5 + node.num_forms)**alpha
    return node.nlp * factor
