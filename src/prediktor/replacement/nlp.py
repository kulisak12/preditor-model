from typing import List, Optional, Tuple

import torch

from prediktor import model
from prediktor.replacement import caching
from prediktor.replacement.search import SearchNode


def infer_nlp(text: str) -> float:
    """Infer the negative log probability of the text."""
    return infer_nlp_batch([text])[0]


def infer_nlp_batch(texts: List[str]) -> List[float]:
    """Infer the negative log probability of the texts.

    The texts are processed in a batch, which is faster than processing
    them one by one.
    """
    input_ids_batch = [model.encode_with_eos(text)[0] for text in texts]
    input_ids_padded = _trim_and_pad(input_ids_batch)
    with torch.no_grad():
        logits_batch = model.model(input_ids_padded).logits
    return [
        _nlp_from_logits(input_ids, logits)
        for input_ids, logits in zip(input_ids_batch, logits_batch)
    ]


def infer_nlp_cache(in_node: SearchNode) -> SearchNode:
    """Infer the negative log probability of the text, using the cache."""
    input_ids = [model.encode_with_eos(in_node.text)[0]]
    trimmed_ids = _trim_and_pad(input_ids)
    logits, caches = _get_outputs_cache(trimmed_ids, [in_node.cache])
    input_start = trimmed_ids.shape[1] - logits.shape[1]
    nlp_diff = _nlp_from_logits(input_ids[0][input_start:], logits[0])
    return SearchNode(
        in_node.text, in_node.nlp + nlp_diff,
        in_node.num_forms, len(input_ids[0]), caches[0]
    )


def infer_nlp_batch_cache(in_nodes: List[SearchNode]) -> List[SearchNode]:
    """Infer the negative log probability of the texts, using the cache.

    The texts are processed in a batch, which is faster than processing
    them one by one.
    """
    input_ids_batch = [model.encode_with_eos(node.text)[0] for node in in_nodes]
    trimmed_batch = _trim_and_pad(input_ids_batch)
    in_caches = [node.cache for node in in_nodes]
    logits_batch, caches = _get_outputs_cache(trimmed_batch, in_caches)
    input_start = trimmed_batch.shape[1] - logits_batch.shape[1]
    return [
        SearchNode(
            node.text,
            node.nlp + _nlp_from_logits(input_ids[input_start:], logits),
            node.num_forms, len(input_ids),
            caching.trim_cache(cache, len(input_ids) - 1)
        )
        for node, input_ids, logits, cache in zip(
            in_nodes, input_ids_batch, logits_batch, caches
        )
    ]


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


def _get_outputs_cache(
    input_ids: torch.Tensor, caches: List[Optional[caching.Cache]]
) -> Tuple[torch.Tensor, List[caching.Cache]]:
    """Prepare inputs and get outputs from the model, using the cache."""
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
