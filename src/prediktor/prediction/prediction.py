import re
from typing import Callable

from prediktor.model.model import Model
from prediktor.prediction import confidence

TERMINATORS = ".!?:;"
PredictFunc = Callable[[Model, str], str]


def predict(model: Model, text: str) -> str:
    """Generate continuation for the given text."""
    return _predict(model, confidence.generate, text)


def _predict(model: Model, func: PredictFunc, text: str) -> str:
    """Generate continuation for the given text using the given function."""
    # the model doesn't like whitespace at the after
    trimmed_text = text.rstrip()
    stripped_suffix = text[len(trimmed_text):]
    prediction = func(model, trimmed_text)
    if stripped_suffix:
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
