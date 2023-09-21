import re

from prediktor.causal.model import generate
from prediktor.config import Config

TERMINATORS = ".!?:;"
MASK = "..."


def predict(text: str) -> str:
    """Generate continuation for the given text."""
    # the model doesn't like whitespace at the end
    trimmed_text = text.rstrip()
    stripped_suffix = text[len(trimmed_text):]
    prediction = generate(trimmed_text, Config.confidence)
    if stripped_suffix:
        prediction = prediction.lstrip()

    # only use one sentence
    return first_sentence(prediction)


def infill(text: str, infill_pos: int) -> str:
    """Generate an infill at the given position."""
    if infill_pos == len(text):
        return predict(text)

    prompt = format_infill_prompt(text, infill_pos)
    prediction = generate(prompt, Config.confidence)
    return prediction


def format_infill_prompt(text: str, cursor_pos: int) -> str:
    """Create the prompt for the infill generation."""
    before_cursor = text[:cursor_pos].rstrip()
    after_cursor = text[cursor_pos:].lstrip()
    with_mask = f"{before_cursor} {MASK} {after_cursor}"
    prompt = f"The previous message with {MASK} filled in:"
    return f"{with_mask}\n{prompt}\n{before_cursor}"


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
