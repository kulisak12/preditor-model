"""This module is the entry point for suggestion.

Suggestion combines prediction and infilling tasks.
"""

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
    lines_before = _get_last_paragraph(before_cursor.splitlines())
    lines_after = _get_first_paragraph(after_cursor.splitlines())
    # converting hard wrapped text to one line
    joined_before = " ".join(lines_before)
    joined_after = " ".join(lines_after)
    if joined_after:
        return infilling.infill(
            model, joined_before, joined_after, infilling_config
        )
    else:
        return prediction.predict(model, joined_before, prediction_config)


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
