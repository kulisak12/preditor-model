import re
from typing import Callable

import pydantic

from preditor.model.model import Model
from preditor.prediction import confidence

TERMINATORS = ".!?:;"
PredictFunc = Callable[[str], str]


class PredictionConfig(pydantic.BaseModel):
    max_length: int = pydantic.Field(30, ge=1)
    confidence: float = pydantic.Field(5.0, ge=1.0, )


def predict(model: Model, text: str, config: PredictionConfig) -> str:
    """Generate continuation for the given text."""
    def func(text: str): return confidence.generate(
        model, text, config.max_length, config.confidence
    )
    return _predict(func, text)


def _predict(func: PredictFunc, text: str) -> str:
    """Generate continuation for the given text using the given function."""
    # the model doesn't like whitespace at the after
    trimmed_text = text.rstrip()
    stripped_suffix = text[len(trimmed_text):]
    prediction = func(trimmed_text)
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
