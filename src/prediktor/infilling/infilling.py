from typing import Callable

from prediktor.infilling import blank, end
from prediktor.model.model import Model

InfillFunc = Callable[[Model, str, str], str]


def infill(model: Model, text: str, cursor_pos: int) -> str:
    """Generate an infill at the given position."""
    return _infill(model, blank.infill_between, text, cursor_pos)


def _infill(model: Model, func: InfillFunc, text: str, cursor_pos: int) -> str:
    """Generate an infill at the given position using the given function."""
    before_cursor = text[:cursor_pos]
    after_cursor = text[cursor_pos:]
    output = func(model, before_cursor, after_cursor)
    return output
