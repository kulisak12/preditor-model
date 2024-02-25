from typing import List, Optional, Tuple

import torch

from prediktor import model
from prediktor.replacement import caching


def infer_nlp_single(text: str) -> float:
    """Infer the negative log probability of the text."""
    return infer_nlp([text])[0]


def infer_nlp(texts: List[str]) -> List[float]:
    """Infer the negative log probability of the texts.

    The texts are processed in a batch, which is faster than processing
    them one by one.
    """
    input_ids_batch = [model.encode_with_eos(text)[0] for text in texts]
    trimmed_batch = _trim_and_pad(input_ids_batch)
    with torch.no_grad():
        logits_batch = model.model(trimmed_batch).logits
    return [
        _nlp_from_logits(input_ids, logits)
        for input_ids, logits in zip(input_ids_batch, logits_batch)
    ]


def infer_nlp_with_cache(
    texts: List[str],
    in_caches: List[Optional[caching.Cache]],
) -> Tuple[List[float], List[caching.Cache]]:
    """Infer the negative log probability of the texts. Use the cache.

    The texts are processed in a batch, which is faster than processing
    them one by one.
    """
    input_ids_batch = [model.encode_with_eos(text)[0] for text in texts]
    trimmed_batch = _trim_and_pad(input_ids_batch)
    logits_batch, caches = _get_outputs_with_cache(trimmed_batch, in_caches)
    input_start = trimmed_batch.shape[1] - logits_batch.shape[1]
    nlps = [
        _nlp_from_logits(input_ids[input_start:], logits)
        for input_ids, logits in zip(input_ids_batch, logits_batch)
    ]
    out_caches = [
        caching.trim_cache(cache, len(input_ids) - 1)
        for input_ids, cache in zip(input_ids_batch, caches)
    ]
    return nlps, out_caches


def _trim_and_pad(input_ids: List[torch.Tensor]) -> torch.Tensor:
    """Create the input ids tensor for the model.

    Trim the last token, we don't need its logits for scoring.
    Pad the input ids to the same length.
    """
    trimmed = [ids[:-1] for ids in input_ids]
    return torch.nn.utils.rnn.pad_sequence(trimmed, batch_first=True)


def _nlp_from_logits(input_ids: torch.Tensor, logits: torch.Tensor) -> float:
    """Calculate the nlp for one item in batch.

    The logits tensor does not need to be trimmed to match the input_ids.
    """
    # prediction for a token is in the previous position
    text_ids = input_ids[1:]
    # casting to avoid different results in bfloat16 and float
    softmax = torch.softmax(logits.to(float), dim=-1)
    probs = softmax[torch.arange(len(text_ids)), text_ids]
    return -torch.log(probs).sum().item()


def _get_outputs_with_cache(
    input_ids: torch.Tensor, caches: List[Optional[caching.Cache]]
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
