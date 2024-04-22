import pydantic

from preditor.model.model import Model
from preditor.substitution import dijkstra, search
from preditor.substitution.variants import ReplacementVariantsGenerator


class SubstitutionConfig(pydantic.BaseModel):
    min_variants: int = pydantic.Field(2, ge=2)
    relax_count: int = pydantic.Field(8, ge=1)
    pool_factor: int = pydantic.Field(1, ge=1)
    # no need to select score key, 0.0 yields same behavior as nlp_key
    lp_alpha: float = pydantic.Field(0.0, ge=0.0, le=1.0)


def replace(
    model: Model,
    before_old: str, old: str, after_old: str, replacement: str,
    config: SubstitutionConfig,
) -> str:
    """Replace part of the text and modify the rest to match."""
    rvg = ReplacementVariantsGenerator(before_old, old, after_old, replacement)
    return dijkstra.replace_with_cache(
        model, rvg,
        min_variants=config.min_variants,
        relax_count=config.relax_count,
        pool_size=config.relax_count * config.pool_factor,
        score_key=lambda x: search.lp_key(x, config.lp_alpha),
    )
