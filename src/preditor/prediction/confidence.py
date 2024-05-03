"""This module implements the confidence strategy.

It finds the optimal place to cut off the generated text.
"""

import math
from typing import List, Tuple

import torch

from preditor import nlp
from preditor.model.model import Model
from preditor.prediction.config import PredictionConfig
from preditor.suggestion import generation


def generate(model: Model, input_text: str, config: PredictionConfig) -> str:
    """Generate a continuation of the input text.

    Trim the generated text to only include the tokens
    where the model is confident enough.
    The higher the confidence parameter, the longer the output.
    """
    text_stripped = input_text.rstrip()
    had_trailing_space = input_text != text_stripped
    gen_ids, logits = _get_model_outputs(model, text_stripped, had_trailing_space, config)
    nlps = nlp.infer_nlps_from_logits(gen_ids, logits).tolist()
    # expected[i] is the expected usefulness of the prefix of length i
    expected = _calculate_expected_usefulness(nlps, config.confidence)
    best = max(range(len(expected)), key=expected.__getitem__)
    output_ids = gen_ids[:best+1]
    decoded_text = model.tokenizer.decode(output_ids, skip_special_tokens=True)
    trimmed = generation.trim_decoded(decoded_text, had_trailing_space)
    return trimmed


def _get_model_outputs(
    model: Model, text_stripped: str, had_trailing_space: bool,
    config: PredictionConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate the continuation ids and logits."""
    input_ids = generation.encode_with_eos(model, text_stripped).to(model.device)
    processors = generation.get_suppress_processors(
        model.tokenizer, had_trailing_space, len(input_ids[0]), []
    )
    output = model.model.generate(
        input_ids,
        logits_processor=processors,
        generation_config=model.config,
        max_new_tokens=config.max_length,
        output_scores=True,
        return_dict_in_generate=True,
    )
    gen_ids = output.sequences[0][len(input_ids[0]):]
    logits = torch.stack(output.scores).squeeze(1).to(torch.float64)
    return gen_ids, logits


def _calculate_expected_usefulness(
    nlps: List[float], confidence: float
) -> List[float]:
    """Calculate the expected usefulness for each prefix of the generated text.

    The expected usefulness is the length of the prefix
    times the probability of the prefix being what the user wants.
    Higher confidence discounts longer sequences more.
    """
    prefix_nlps = [sum(nlps[:i]) for i in range(len(nlps) + 1)]
    expected = [
        length * math.exp(-nlp / confidence)
        for length, nlp in enumerate(prefix_nlps)
    ]
    # remove the empty prefix
    return expected[1:]
