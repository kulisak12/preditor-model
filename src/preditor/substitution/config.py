"""This module provides configuration for the substitution algorithms."""

import pydantic

from preditor.substitution.search import ScoreKey, lp_key


class SubstitutionConfig(pydantic.BaseModel):
    """Configuration for the substitution algorithms.

    min_variants: The minimum number of variants to add in each iteration.
    relax_count: The number of nodes to relax in a batch.
    pool_factor: What multiple of relax_count to use as the pool size
        for node selection.
    lp_alpha: The exponent in the length penalty function.
    """

    min_variants: int = pydantic.Field(2, ge=2)
    relax_count: int = pydantic.Field(8, ge=1)
    pool_factor: int = pydantic.Field(5, ge=1)
    # no need to select score key, 0.0 yields same behavior as nlp_key
    lp_alpha: float = pydantic.Field(0.0, ge=0.0, le=1.0)

    @property
    def score_key(self) -> ScoreKey:
        return lambda x: lp_key(x, self.lp_alpha)

    @property
    def pool_size(self) -> int:
        return self.relax_count * self.pool_factor
