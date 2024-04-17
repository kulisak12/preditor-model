import functools
from typing import List, Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizer

from preditor.config import Config
from preditor.model.model import Model


class FirstTokenLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that constrains the first generated token to a list of ids."""

    def __init__(self, prompt_length_to_skip: int, token_ids: List[int]):
        self.prompt_length_to_skip = prompt_length_to_skip
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        # if it's the first token to be generated
        if input_ids.shape[-1] - self.prompt_length_to_skip == 0:
            # set the scores of all tokens that are not in the list to -inf
            mask = torch.ones_like(scores) * -float('inf')
            mask[:, self.token_ids] = 0
            scores = scores + mask
        return scores


PROMPT_EN = "Write a sentence such that it ends with:"
PROMPT_CS = "Napiš větu tak, aby končila na:"


def generate_infills(
    model: Model, before_cursor: str, after_cursor: str, prompt: Optional[str],
    max_length: int, num_variants: int
) -> List[str]:
    """Generate possible infills between the given strings."""
    had_trailing_space = bool(before_cursor) and before_cursor[-1].isspace()
    before_cursor = before_cursor.rstrip()
    after_cursor = after_cursor.lstrip()
    input_text = _format_input(before_cursor, after_cursor, prompt)
    decoded = _beam_search(
        model, input_text, had_trailing_space,
        max_length, num_variants
    )
    return decoded


def _format_input(before: str, after: str, prompt: Optional[str]) -> str:
    """Create the input for the infill generation."""
    if prompt is None:
        return before
    return prompt + " " + after + "\n" + before


def _beam_search(
    model: Model,
    input_text: str,
    should_start_with_space: bool,
    max_length: int,
    num_variants: int
) -> List[str]:
    """Generate continuations using beam search."""
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    space_tokens = _get_tokens_with_prefix_space(model.tokenizer)
    space_processor = FirstTokenLogitsProcessor(len(input_ids[0]), space_tokens)
    processor_list = LogitsProcessorList([space_processor])

    gen_ids = model.model.generate(
        input_ids,
        logits_processor=processor_list if should_start_with_space else None,
        max_new_tokens=max_length,
        num_return_sequences=num_variants,
        num_beams=num_variants,
        num_beam_groups=(num_variants + 1) // 2,
        diversity_penalty=20.0,
        pad_token_id=model.tokenizer.eos_token_id
    )
    infills_ids = gen_ids[:, input_ids.shape[-1] :]
    decoded_infills = model.tokenizer.batch_decode(infills_ids, skip_special_tokens=True)
    return decoded_infills


@functools.lru_cache(maxsize=None)
def _get_tokens_with_prefix_space(tokenizer: PreTrainedTokenizer) -> List[int]:
    """Get the token ids that are preceded by a space in the tokenizer."""
    token_ids = tokenizer.get_vocab().values()
    return [
        token_id for token_id in token_ids
        if tokenizer.decode([token_id])[0].isspace()
    ]
