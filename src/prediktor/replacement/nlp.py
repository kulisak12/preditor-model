from typing import List

import torch

from prediktor import model


def infer_nlp(text: str) -> float:
    """Infer the negative log probability of the text."""
    input_ids = model.encode_with_eos(text)
    with torch.no_grad():
        logits = model.model(input_ids).logits
    # batch size is 1
    return _nlp_from_logits(input_ids[0], logits[0])


def infer_nlp_batch(texts: List[str]) -> List[float]:
    """Infer the negative log probability of the texts.

    The texts are processed in a batch, which is faster than processing
    them one by one.
    """
    input_ids_batch = [model.encode_with_eos(text)[0] for text in texts]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_batch, batch_first=True
    )
    with torch.no_grad():
        logits_batch = model.model(input_ids_padded).logits
    return [
        _nlp_from_logits(input_ids, logits[:len(input_ids)])
        for input_ids, logits in zip(input_ids_batch, logits_batch)
    ]


def _nlp_from_logits(input_ids: torch.Tensor, logits: torch.Tensor) -> float:
    """Calculate the nlp for one item in batch."""
    # prediction for a token is in the previous position
    text_ids = input_ids[1:]
    text_logits = logits[:-1]
    softmax = torch.softmax(text_logits, dim=-1)
    probs = softmax[torch.arange(len(text_ids)), text_ids]
    return -torch.log(probs).sum().item()
