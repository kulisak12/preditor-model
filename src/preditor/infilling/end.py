from typing import List

from preditor.infilling.config import InfillingConfig
from preditor.model.model import Model
from preditor.suggestion import generation

# this code is very similar to the blank strategy
# it is intentionally not refactored
# because other strategies may not follow the same structure

INSTRUCTIONS = {
    "en": "Write a sentence such that it ends with:",
    "cs": "Napiš větu tak, aby končila na:"
}
PROMPT = "{0}: {2}\n{1}"

def generate_infills(
    model: Model, before_cursor: str, after_cursor: str,
    config: InfillingConfig, lang: str = "en"
) -> List[str]:
    """Generate possible infills between the given strings.

    Instruct the model to continue the text before the cursor
    such that it ends with the text after the cursor.
    """
    before_stripped = before_cursor.rstrip()
    after_stripped = after_cursor.lstrip()
    had_trailing_space = before_cursor != before_stripped
    input_text = _format_input(before_stripped, after_stripped, lang)
    decoded = generation.beam_search(
        model, input_text, had_trailing_space, [],
        config.max_length, config.num_variants
    )
    return [generation.trim_decoded(d, had_trailing_space) for d in decoded]


def _format_input(before: str, after: str, lang: str) -> str:
    """Create the input for the infill generation."""
    if lang not in INSTRUCTIONS:
        lang = "en"
    instruction = INSTRUCTIONS[lang]
    return PROMPT.format(instruction, before, after)
