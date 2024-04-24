from typing import List

from preditor.infilling import generation
from preditor.infilling.config import InfillingConfig
from preditor.model.model import Model

# this code is very similar to the end strategy
# it is intentionally not refactored
# because other strategies may not follow the same structure


INSTRUCTIONS = {
    "en": "Fill in the blank marked by ___",
    "cs": "Vyplň mezeru označenou ___"
}
PROMPT = "{0}\n{1} ___ {2}\n{1}"

MAX_TOKEN_LENGTH = 80
bad_words: List[str] = []
bad_words.extend("#" * i for i in range(1, MAX_TOKEN_LENGTH + 1))
bad_words.extend("_" * i for i in range(1, MAX_TOKEN_LENGTH + 1))
bad_words.extend(" " * i for i in range(2, MAX_TOKEN_LENGTH + 1))


def generate_infills(
    model: Model, before_cursor: str, after_cursor: str,
    config: InfillingConfig, lang: str = "en"
) -> List[str]:
    """Generate possible infills between the given strings.

    Instruct the model to fill in the position with a blank marker.
    """
    before_stripped = before_cursor.rstrip()
    after_stripped = after_cursor.lstrip()
    had_trailing_space = before_cursor != before_stripped
    input_text = _format_input(before_stripped, after_stripped, lang)
    decoded = generation.beam_search(
        model, input_text, had_trailing_space, bad_words,
        config.max_length, config.num_variants
    )
    return generation.process_decoded(decoded, had_trailing_space)


def _format_input(before: str, after: str, lang: str) -> str:
    """Create the input for the infill generation."""
    if lang not in INSTRUCTIONS:
        lang = "en"
    instruction = INSTRUCTIONS[lang]
    return PROMPT.format(instruction, before, after)
