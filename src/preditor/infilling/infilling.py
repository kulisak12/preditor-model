from typing import Callable, List

import pydantic

from preditor.infilling import blank, end, selection
from preditor.model.model import Model

InfillGenerateFunc = Callable[[str, str], List[str]]
InfillSelectFunc = Callable[[List[str], str, str], str]


class InfillingConfig(pydantic.BaseModel):
    max_length: int = pydantic.Field(8, ge=1, le=30)
    num_variants: int = pydantic.Field(10, ge=1, le=30)


def infill(model: Model, text: str, cursor_pos: int, config: InfillingConfig) -> str:
    """Generate an infill at the given position."""
    def generate_func(before: str, after: str): return blank.generate_infills(
        model, before, after, config.max_length, config.num_variants
    )
    def select_func(variants: List[str], before: str, after: str): return selection.select_by_score(
        variants, model, before, after
    )
    return _infill(generate_func, select_func, text, cursor_pos)


def _infill(
    generate_func: InfillGenerateFunc, select_func: InfillSelectFunc,
    text: str, cursor_pos: int
) -> str:
    """Generate an infill at the given position using the given functions."""
    before_cursor = text[:cursor_pos]
    after_cursor = text[cursor_pos:]
    variants = generate_func(before_cursor, after_cursor)
    selected = select_func(variants, before_cursor, after_cursor)
    return selected
