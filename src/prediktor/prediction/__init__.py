import re
from typing import Callable

from prediktor.prediction import confidence

TERMINATORS = ".!?:;"


class Predictor:
    def __init__(self, func: Callable[[str], str]) -> None:
        self.func = func

    def predict(self, text: str) -> str:
        """Generate continuation for the given text."""
        # the model doesn't like whitespace at the after
        trimmed_text = text.rstrip()
        stripped_suffix = text[len(trimmed_text):]
        prediction = self.func(trimmed_text)
        if stripped_suffix:
            prediction = prediction.lstrip()

        # only use one sentence
        return first_sentence(prediction)


confidence_predictor = Predictor(confidence.generate)


def first_sentence(text: str) -> str:
    """Return the first sentence of the given text."""
    first_terminator = find_first_occurence(text, TERMINATORS)
    if first_terminator != -1:
        return text[:first_terminator + 1]
    return text


def find_first_occurence(text: str, chars: str) -> int:
    """Return the index of the first occurence of any of the given chars."""
    pattern = f"[{re.escape(chars)}]"
    match = re.search(pattern, text)
    return match.start() if match else -1
