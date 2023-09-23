import re
from typing import List

from prediktor.causal.model import beam_search, generate
from prediktor.config import Config

TERMINATORS = ".!?:;"
PROMPT = """\
### Instruction:
{}

### Input:
{}

### Output:
{}"""
INFILL_INSTRUCTION = "Fill in the blank marked by [...]"

bad_words = ["[...]", "[", "...", "[…]", "…"]
bad_words.extend("#" * i for i in range(1, 5))
bad_words.extend(" " * i for i in range(2, 10))
bad_words.extend("_" * i for i in range(2, 10))


def predict(text: str) -> str:
    """Generate continuation for the given text."""
    # the model doesn't like whitespace at the after
    trimmed_text = text.rstrip()
    stripped_suffix = text[len(trimmed_text):]
    prediction = generate(trimmed_text, Config.confidence)
    if stripped_suffix:
        prediction = prediction.lstrip()

    # only use one sentence
    return first_sentence(prediction)


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


def infill(text: str, cursor_pos: int) -> str:
    """Generate an infill at the given position."""
    if cursor_pos >= len(text.rstrip()):
        return predict(text)

    before_cursor = text[:cursor_pos].rstrip()
    after_cursor = text[cursor_pos:].lstrip()
    prompt = format_infill_prompt(before_cursor, after_cursor)
    decoded = beam_search(prompt, bad_words, after_cursor)
    outputs = [extract_output(text, before_cursor) for text in decoded]
    best_output = get_best_output(outputs, after_cursor).rstrip()
    if len(before_cursor) < cursor_pos:
        best_output = best_output.lstrip()
    return best_output


def format_infill_prompt(before: str, after: str) -> str:
    """Create the prompt for the infill generation."""
    return PROMPT.format(
        INFILL_INSTRUCTION,
        before + " [...] " + after,
        before
    )


def extract_output(text: str, before_cursor: str) -> str:
    start = "### Output:\n" + before_cursor
    filled = text[text.find(start) + len(start):]
    return filled[:filled.find("\n")]


def get_best_output(outputs: List[str], after: str) -> str:
    """Return the output that best matches the text after cursor.

    Favor outputs in the beginning of the list.
    """
    # exact match
    for output in outputs:
        if output.endswith(after):
            return output[:len(output) - len(after)]
    # partial match
    end_first_word = after.split()[0]
    for output in outputs:
        pos = output.find(end_first_word)
        if pos != -1:
            return output[:pos]
    # no match
    for output in outputs:
        splits = output.split()
        if len(splits) > 1:
            return splits[0]
    # no output
    return ""
