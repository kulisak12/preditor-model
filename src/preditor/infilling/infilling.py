from typing import Callable, List

import pydantic

from preditor.infilling import blank, end, selection
from preditor.model.model import Model

InfillGenerateFunc = Callable[[str, str], List[str]]
InfillSelectFunc = Callable[[List[str], str, str], str]


class InfillingConfig(pydantic.BaseModel):
    max_length: int = pydantic.Field(8, ge=1)
    num_variants: int = pydantic.Field(10, ge=1)


def infill(
    model: Model, before_cursor: str, after_cursor: str, config: InfillingConfig
) -> str:
    """Generate an infill between the two texts."""
    variants = blank.generate_infills(
        model, before_cursor, after_cursor, config.max_length, config.num_variants
    )
    selected = selection.select_by_score(
        variants, model, before_cursor, after_cursor
    )
    return selected

