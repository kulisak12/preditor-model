"""This module implements the simple strategy.

It generates a continuation and returns its first sentence.
"""

import re

from preditor.model.model import Model
from preditor.prediction.config import PredictionConfig
from preditor.suggestion import generation

TERMINATORS = ".!?:;"


def generate(model: Model, input_text: str, config: PredictionConfig) -> str:
    """Generate a continuation of the input text."""
    text_stripped = input_text.rstrip()
    had_trailing_space = input_text != text_stripped
    input_ids = generation.encode_with_eos(model, text_stripped).to(model.device)
    processors = generation.get_suppress_processors(
        model.tokenizer, had_trailing_space, len(input_ids[0]), []
    )
    output_ids = model.model.generate(
        input_ids,
        logits_processor=processors,
        generation_config=model.config,
        max_new_tokens=config.max_length,
    )
    gen_ids = output_ids[0][len(input_ids[0]):]
    decoded_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True)
    trimmed = generation.trim_decoded(decoded_text, had_trailing_space)
    return _first_sentence(trimmed)


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
