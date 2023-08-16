import re

from prediktor.model import generate

TERMINATORS = ".!?:;"


def predict(text: str) -> str:
    # the model doesn't like whitespace at the end
    trimmed_text = text.rstrip()
    stripped_suffix = text[len(trimmed_text):]
    prediction = generate(trimmed_text, confidence=2.5)
    if stripped_suffix:
        prediction = prediction.lstrip()

    # only use one sentence
    return first_sentence(prediction)


def first_sentence(text: str) -> str:
    first_terminator = find_first_occurence(text, TERMINATORS)
    if first_terminator != -1:
        return text[:first_terminator + 1]
    return text


def find_first_occurence(text: str, chars: str) -> int:
    pattern = f"[{re.escape(chars)}]"
    match = re.search(pattern, text)
    return match.start() if match else -1
