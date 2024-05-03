"""This module provides functions that calculate the sentence score."""

from typing import List, Optional, Tuple

import torch

from preditor import caching
from preditor.model.model import Model


def infer_nlp_single(model: Model, text: str) -> float:
    """Infer the negative log probability of the text."""
    return infer_nlp(model, [text])[0]


def infer_nlp(model: Model, texts: List[str]) -> List[float]:
    """Infer the negative log probability of the texts.

    The texts are processed in a batch, which is faster than processing
    them one by one.
    """
    input_ids_batch = [_encode_with_eos(model, text)[0] for text in texts]
    trimmed_batch = _trim_and_pad(input_ids_batch)
    with torch.no_grad():
        logits_batch = model.model(trimmed_batch).logits
    return [
        _get_nlp_of_input(input_ids, logits)
        for input_ids, logits in zip(input_ids_batch, logits_batch)
    ]


def infer_nlp_with_cache(
    model: Model,
    texts: List[str],
    in_caches: List[Optional[caching.LazyCache]],
) -> Tuple[List[float], List[caching.LazyCache]]:
    """Infer the negative log probability of the texts. Use the cache.

    The texts are processed in a batch, which is faster than processing
    them one by one.
    """
    input_ids_batch = [_encode_with_eos(model, text)[0] for text in texts]
    trimmed_batch = _trim_and_pad(input_ids_batch)
    logits_batch, caches = _get_outputs_with_cache(model, trimmed_batch, in_caches)
    logits_shift = trimmed_batch.shape[1] - logits_batch.shape[1]
    starts = [caching.cache_len(cache) for cache in in_caches]
    nlps = [
        _get_nlp_of_input(input_ids[start:], logits[start - logits_shift:])
        for input_ids, logits, start in zip(input_ids_batch, logits_batch, starts)
    ]
    out_caches = [
        caching.LazyCache(cache, len(input_ids) - 1)
        for input_ids, cache in zip(input_ids_batch, caches)
    ]
    return nlps, out_caches


def infer_nlps_from_logits(
    text_ids: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    """Calculate the negative log probability for each token in the text.

    The i-th logits predict the i-th token.
    """
    softmax = torch.softmax(logits, dim=-1)
    probs = softmax[torch.arange(len(text_ids)), text_ids]
    return -torch.log(probs)


def _encode_with_eos(model: Model, text: str) -> torch.Tensor:
    """Encode text with EOS token."""
    input_ids = model.tokenizer.encode(text, return_tensors="pt")
    eos_tensor = torch.tensor(model.tokenizer.eos_token_id).reshape(1, 1)
    return torch.cat([eos_tensor, input_ids], dim=-1).to(model.device)


def _trim_and_pad(input_ids: List[torch.Tensor]) -> torch.Tensor:
    """Create the input ids tensor for the model.

    Trim the last token, we don't need its logits for scoring.
    Pad the input ids to the same length.
    """
    trimmed = [ids[:-1] for ids in input_ids]
    return torch.nn.utils.rnn.pad_sequence(trimmed, batch_first=True)


def _get_nlp_of_input(input_ids: torch.Tensor, logits: torch.Tensor) -> float:
    """Calculate the nlp for one item in batch.

    The i-th logits are derived from the i-th token.
    The logits tensor does not need to be trimmed to match the input_ids.
    """
    # prediction for a token is in the previous position
    text_ids = input_ids[1:]
    nlps = infer_nlps_from_logits(text_ids, logits)
    return nlps.sum().item()


def _get_outputs_with_cache(
    model: Model,
    input_ids: torch.Tensor,
    caches: List[Optional[caching.LazyCache]]
) -> Tuple[torch.Tensor, List[caching.Cache]]:
    """Prepare inputs and get outputs from the model. Use the cache."""
    cache_batch = caching.join_caches_optional(caches)
    model_kwargs = {
        "use_cache": True,
        "past_key_values": cache_batch,
    }
    model_inputs = model.model.prepare_inputs_for_generation(
        input_ids, **model_kwargs
    )
    with torch.no_grad():
        outputs = model.model(**model_inputs, return_dict=True)
    return outputs.logits, caching.split_cache(outputs.past_key_values)
