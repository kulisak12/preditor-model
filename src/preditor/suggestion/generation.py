import functools
from typing import Any, Iterable, List

from transformers import (LogitsProcessorList, PreTrainedTokenizer, SuppressTokensAtBeginLogitsProcessor,
                          SuppressTokensLogitsProcessor)

from preditor.model.model import Model


def beam_search(
    model: Model,
    input_text: str,
    should_start_with_space: bool,
    suppress_tokens: List[int],
    max_length: int,
    num_variants: int
) -> List[str]:
    """Generate continuations using beam search."""
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    input_len = len(input_ids[0])
    processors = get_suppress_processors(
        model.tokenizer, should_start_with_space, input_len, suppress_tokens
    )

    gen_ids = model.model.generate(
        input_ids,
        logits_processor=processors,
        max_new_tokens=max_length,
        num_return_sequences=num_variants * 2,
        num_beams=num_variants * 2,
        num_beam_groups=num_variants,
        diversity_penalty=20.0,
        pad_token_id=model.tokenizer.eos_token_id
    )
    infills_ids = gen_ids[:, input_len:]
    decoded_infills = model.tokenizer.batch_decode(infills_ids, skip_special_tokens=True)
    return decoded_infills


def get_suppress_processors(
    tokenizer: PreTrainedTokenizer, should_start_with_space: bool, input_len: int,
    suppress_tokens: Iterable[int]
) -> LogitsProcessorList:
    """Get the processors for suppressing tokens in the generation."""
    processors = LogitsProcessorList()
    if should_start_with_space:
        space_tokens = _get_tokens_with_prefix_space(tokenizer)
        space_processor = SuppressTokensAtBeginLogitsProcessor(space_tokens, input_len)
        processors.append(space_processor)
    if suppress_tokens:
        suppress_processor = SuppressTokensLogitsProcessor(suppress_tokens)
        processors.append(suppress_processor)
    return processors


@functools.lru_cache(maxsize=None)
def _get_tokens_with_prefix_space(tokenizer: PreTrainedTokenizer) -> List[int]:
    """Get the token ids that are preceded by a space in the tokenizer."""
    token_ids = tokenizer.get_vocab().values()
    return [
        token_id for token_id in token_ids
        if tokenizer.decode([token_id])[0].isspace()
    ]


def trim_decoded(decoded: str, had_trailing_space: bool) -> str:
    """Trim the decoded text to the first line,
    and remove the leading space if the input had a trailing space.
    """
    output = _first_line(decoded)
    if had_trailing_space:
        return output.lstrip()
    return output


def _first_line(text: str) -> str:
    """Return the first line of the text."""
    return text[:text.find("\n")]
