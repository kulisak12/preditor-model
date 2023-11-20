import dataclasses

import torch

from prediktor import model
from prediktor.replacement.variants import ReplacementVariantsGenerator


def infer_continuation_nlp(
    input_ids: torch.Tensor, continuation_start: int
) -> float:
    """Infer the negative log probability of continuation."""
    assert continuation_start > 0, "continuation must start after BOS token"
    with torch.no_grad():
        logits = model.model(input_ids).logits[0]
    # prediction is in the previous position
    continuation_logits = logits[continuation_start-1:-1]
    softmax = torch.softmax(continuation_logits, dim=-1)
    continuation_ids = input_ids[0, continuation_start:]
    probs = softmax[torch.arange(len(continuation_ids)), continuation_ids]
    return -torch.log(probs).sum().item()


@dataclasses.dataclass(frozen=True)
class SearchNode:
    text: str
    nlp: float
    num_forms: int


def score(node: SearchNode) -> float:
    """Score of a search node."""
    # ? favor longer texts?
    return node.nlp


def replace_dijkstra(rvg: ReplacementVariantsGenerator) -> str:
    """Find best replacement using Dijkstra-inspired approach."""
    start_node = SearchNode("", 0, 0)
    open_nodes = [start_node]

    while True:
        current = min(open_nodes, key=score)
        open_nodes.remove(current)
        variants, num_variant_forms = rvg.get_variants(current.num_forms)
        if num_variant_forms == 0:
            return current.text

        for variant in variants:
            input_ids = model.encode_with_eos(current.text + variant)
            variant_len = len(model.tokenizer.encode(variant))
            nlp = infer_continuation_nlp(input_ids, variant_len)
            node = SearchNode(
                current.text + variant, nlp,
                current.num_forms + num_variant_forms
            )
            open_nodes.append(node)
