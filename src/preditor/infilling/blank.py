"""This module implements the blank strategy.

It generates infills by instructing the model to fill in the blank marker.
"""

import functools
from typing import List

from transformers import PreTrainedTokenizer

from preditor.infilling.config import InfillingConfig
from preditor.model.model import Model
from preditor.suggestion import generation

# this code is very similar to the end strategy
# it is intentionally not refactored
# because other strategies may not follow the same structure


INSTRUCTIONS = {
    "en": "Fill in the blank marked by ___",
    "cs": "Vyplň mezeru označenou ___"
}
PROMPT = "{0}\n{1} ___ {2}\n{1}"


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
    blank_tokens = _get_blank_tokens(model.tokenizer)
    decoded = generation.beam_search(
        model, input_text, had_trailing_space, blank_tokens,
        config.max_length, config.num_variants
    )
    return [generation.trim_decoded(d, had_trailing_space) for d in decoded]


def _format_input(before: str, after: str, lang: str) -> str:
    """Create the input for the infill generation."""
    if lang not in INSTRUCTIONS:
        lang = "en"
    instruction = INSTRUCTIONS[lang]
    return PROMPT.format(instruction, before, after)


@functools.lru_cache(maxsize=None)
def _get_blank_tokens(tokenizer: PreTrainedTokenizer) -> List[int]:
    """Return a list of tokens that resemble the blank marker.

    Such tokens should be suppressed from the output.
    """
    token_ids = tokenizer.get_vocab().values()
    result: List[int] = []
    for token_id in token_ids:
        token = tokenizer.decode(token_id)
        if "_" in token or ".." in token:
            result.append(token_id)
    return result
