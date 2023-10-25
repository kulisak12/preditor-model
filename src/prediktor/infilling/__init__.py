from prediktor.infilling.blank import infill_between
from prediktor.prediction import predict


def infill(text: str, cursor_pos: int) -> str:
    """Generate an infill at the given position."""
    if cursor_pos >= len(text.rstrip()):
        return predict(text)

    before_cursor = text[:cursor_pos].rstrip()
    after_cursor = text[cursor_pos:].lstrip()
    output = infill_between(before_cursor, after_cursor)
    if len(before_cursor) < cursor_pos:
        output = output.lstrip()
    return output
