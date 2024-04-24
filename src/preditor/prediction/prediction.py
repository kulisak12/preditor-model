import re
from typing import Callable

from preditor.model.model import Model
from preditor.prediction import confidence
from preditor.prediction.config import PredictionConfig

TERMINATORS = ".!?:;"
PredictFunc = Callable[[Model, str, PredictionConfig], str]


def predict(
    model: Model, text: str, config: PredictionConfig,
    func: PredictFunc = confidence.generate,
) -> str:
    """Generate continuation for the given text."""
    # the model doesn't like whitespace at the end of the text
    text_stripped = text.rstrip()
    had_trailing_space = text != text_stripped
    prediction = func(model, text_stripped, config)
    if had_trailing_space:
        prediction = prediction.lstrip()

    # only use one sentence
    return _first_sentence(prediction)


def _first_sentence(text: str) -> str:
    """Return the first sentence of the given text."""
    first_terminator = _find_first_occurence(text, TERMINATORS)
    if first_terminator != -1:
        return text[:first_terminator + 1]
    return text


def _find_first_occurence(text: str, chars: str) -> int:
    """Return the index of the first occurence of any of the given chars."""
    pattern = f"[{re.escape(chars)}]"
    match = re.search(pattern, text)
    return match.start() if match else -1
