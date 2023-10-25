from prediktor import prediction
from prediktor.infilling import blank


def infill(text: str, cursor_pos: int) -> str:
    """Generate an infill at the given position."""
    if cursor_pos >= len(text.rstrip()):
        return prediction.predict(text)

    before_cursor = text[:cursor_pos].rstrip()
    after_cursor = text[cursor_pos:].lstrip()
    output = blank.infill_between(before_cursor, after_cursor)
    if len(before_cursor) < cursor_pos:
        output = output.lstrip()
    return output
