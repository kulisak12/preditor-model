from typing import Callable

from prediktor.infilling import blank, end


class Infiller:
    def __init__(self, func: Callable[[str, str], str]) -> None:
        self.func = func

    def infill(self, text: str, cursor_pos: int) -> str:
        """Generate an infill at the given position."""
        before_cursor = text[:cursor_pos].rstrip()
        after_cursor = text[cursor_pos:].lstrip()
        output = self.func(before_cursor, after_cursor)
        if len(before_cursor) < cursor_pos:
            output = output.lstrip()
        return output


blank_infiller = Infiller(blank.infill_between)
end_infiller = Infiller(end.infill_between)
