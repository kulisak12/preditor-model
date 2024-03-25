import functools
from typing import Iterable, List, Optional, Set

import torch
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizer

from prediktor import nlp
from prediktor.config import Config
from prediktor.model.model import Model


class FirstTokenLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that constrains the first generated token to a list of ids."""

    def __init__(self, prompt_length_to_skip: int, token_ids: List[int]):
        self.prompt_length_to_skip = prompt_length_to_skip
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # if it's the first token to be generated
        if input_ids.shape[-1] - self.prompt_length_to_skip == 0:
            # set the scores of all tokens that are not in the list to -inf
            mask = torch.ones_like(scores) * -float('inf')
            mask[:, self.token_ids] = 0
            scores = scores + mask
        return scores


PROMPT_EN = "Write a sentence such that it ends with:"
PROMPT_CS = "Napiš větu tak, aby končila na:"


def infill_between(
    model: Model, before_cursor: str, after_cursor: str, prompt: Optional[str]
) -> str:
    """Generate an infill between the given strings."""
    has_trailing_space = before_cursor and before_cursor[-1].isspace()
    before_cursor = before_cursor.rstrip()
    after_cursor = after_cursor.lstrip()
    input_text = _format_input(before_cursor, after_cursor, prompt)
    decoded = _beam_search(model, input_text, has_trailing_space)
    variants = list(_expand_prefixes(decoded))
    final = [
        _format_final_sentence(before_cursor, variant, after_cursor)
        for variant in variants
    ]
    nlps = nlp.infer_nlp(model, final)
    argmin = min(range(len(nlps)), key=nlps.__getitem__)
    return variants[argmin]


def _format_input(before: str, after: str, prompt: Optional[str]) -> str:
    """Create the input for the infill generation."""
    if prompt is None:
        return before
    return prompt + " " + after + "\n" + before


def _format_final_sentence(before: str, infill: str, after: str) -> str:
    """Create the final sentence from the infill and the context."""
    return before + infill + " " + after


def _expand_prefixes(infill_texts: Iterable[str]) -> Set[str]:
    """Generate all prefixes of the generated texts.

    The prefix always ends with a word boundary.
    The last word is not included since it is not certain that it is complete.
    """
    result: Set[str] = set()
    for text in infill_texts:
        words = text.split()
        for i in range(1, len(words)):
            result.add(" ".join(words[:i]))
    return result


def _beam_search(
    model: Model,
    input_text: str,
    has_trailing_space: bool,
) -> List[str]:
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    space_tokens = _get_tokens_with_prefix_space(model.tokenizer)
    space_processor = FirstTokenLogitsProcessor(len(input_ids[0]), space_tokens)
    processor_list = LogitsProcessorList([space_processor])

    gen_ids = model.model.generate(
        input_ids,
        logits_processor=processor_list if has_trailing_space else None,
        max_new_tokens=Config.max_infill_length,
        num_return_sequences=Config.num_beams,
        num_beams=Config.num_beams,
        num_beam_groups=(Config.num_beams + 1) // 2,
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
