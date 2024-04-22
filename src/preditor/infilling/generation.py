import functools
from typing import Any, Iterable, List

import torch
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizer

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


def beam_search(
    model: Model,
    input_text: str,
    should_start_with_space: bool,
    bad_words: Iterable[str],
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
        bad_words_ids=_get_bad_tokens(model, bad_words) if bad_words else None,
        max_new_tokens=max_length,
        num_return_sequences=num_variants * 2,
        num_beams=num_variants * 2,
        num_beam_groups=num_variants,
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


def _get_bad_tokens(model: Model, bad_words: Iterable[str]) -> List[Any]:
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in bad_words:
        # some model.tokenizers accept the prefix space,
        # some need the special tokenizer
        tokenized_word = model.prefix_space_tokenizer(
            [" " + word], add_special_tokens=False
        ).input_ids[0]
        tokens_list.append(tokenized_word)
        tokenized_word = model.tokenizer([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list


def process_decoded(decoded: List[str], had_trailing_space: bool) -> List[str]:
    outputs = [_first_line(text) for text in decoded]
    if had_trailing_space:
        outputs = [output.lstrip() for output in outputs]
    return outputs


def _first_line(text: str) -> str:
    return text[:text.find("\n")]
