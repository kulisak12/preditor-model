from typing import List

from preditor.infilling import infilling
from preditor.model.model import Model
from preditor.prediction import prediction


def suggest(
    model: Model,
    before_cursor: str, after_cursor: str,
    prediction_config: prediction.PredictionConfig,
    infilling_config: infilling.InfillingConfig,
) -> str:
    """Get a suggestion for the given position in the text.

    Choose between prediction and infilling tasks.
    """
    lines_before = before_cursor.splitlines()
    lines_after = after_cursor.splitlines()
    lines_after = _get_first_paragraph(lines_after)
    if not lines_after:
        # prediction
        # converting hard wrapped text to one line
        return prediction.predict(model, " ".join(lines_before), prediction_config)
    # infilling
    lines_before = _get_last_paragraph(lines_before)
    return infilling.infill(
        model, " ".join(lines_before), " ".join(lines_after), infilling_config
    )


def _get_first_paragraph(lines: List[str]) -> List[str]:
    """Get the lines before the first empty line."""
    paragraph: List[str] = []
    for i, line in enumerate(lines):
        if line.strip():
            paragraph.append(line)
        elif i == 0:
            # the first line could be the end of a hard-wrapped line
            continue
        else:
            break
    return paragraph


def _get_last_paragraph(lines: List[str]) -> List[str]:
    """Get the lines after the last empty line."""
    last = _get_first_paragraph(list(reversed(lines)))
    last.reverse()
    return last
