import copy
import dataclasses
from typing import List

import torch

from prediktor import model
from prediktor.replacement.variants import ReplacementVariantsGenerator


def infer_token_nlp(input_ids: List[int], next_token: int) -> float:
    """Infer the negative log-probability of the next token."""
    with torch.no_grad():
        input_tensor = torch.tensor([input_ids]).to(model.device)
        logits = model.model(input_tensor).logits[0, -1, :]
        prob = torch.softmax(logits, dim=-1)[next_token]
        return -torch.log(prob).item()


def infer_continuation_nlp(
    input_ids: List[int], continuation: List[int]
):
    """Infer the negative log-probability of the continuation."""
    processed_ids = copy.deepcopy(input_ids)
    nlp = 0.0
    for token in continuation:
        processed_ids.append(token)
        nlp += infer_token_nlp(processed_ids, token)
    return nlp


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

        current_ids = model.tokenizer.encode(current.text)
        for variant in variants:
            variant_ids = model.tokenizer.encode(variant)
            nlp = infer_continuation_nlp(current_ids, variant_ids)
            node = SearchNode(
                current.text + variant, nlp,
                current.num_forms + num_variant_forms
            )
            open_nodes.append(node)
