from typing import List

import torch

from prediktor import model


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
    softmax = torch.softmax(logits, dim=-1)
    probs = softmax[torch.arange(len(text_ids)), text_ids]
    return -torch.log(probs).sum().item()
