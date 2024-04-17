from typing import Callable, List

from preditor.infilling import blank, end, selection
from preditor.model.model import Model

InfillGenerateFunc = Callable[[Model, str, str], List[str]]
InfillSelectFunc = Callable[[List[str], Model, str, str], str]


def infill(model: Model, text: str, cursor_pos: int) -> str:
    """Generate an infill at the given position."""
    return _infill(
        model,
        blank.generate_infills,
        selection.select_by_score,
        text, cursor_pos
    )


def _infill(
    model: Model,
    generate_func: InfillGenerateFunc, select_func: InfillSelectFunc,
    text: str, cursor_pos: int
) -> str:
    """Generate an infill at the given position using the given function."""
    before_cursor = text[:cursor_pos]
    after_cursor = text[cursor_pos:]
    variants = generate_func(model, before_cursor, after_cursor)
    selected = select_func(variants, model, before_cursor, after_cursor)
    return selected
