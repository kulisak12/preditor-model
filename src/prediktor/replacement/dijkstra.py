import dataclasses

from prediktor.replacement import nlp
from prediktor.replacement.variants import ReplacementVariantsGenerator


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
        new_num_forms = current.num_forms + num_variant_forms

        for variant in variants:
            new_text = current.text + variant
            new_nlp = nlp.infer_nlp(new_text)
            node = SearchNode(new_text, new_nlp, new_num_forms)
            open_nodes.append(node)
