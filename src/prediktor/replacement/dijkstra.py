import dataclasses
from typing import List

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

        new_texts = [current.text + variant for variant in variants]
        new_nlps = nlp.infer_nlp_batch(new_texts)
        for new_text, new_nlp in zip(new_texts, new_nlps):
            node = SearchNode(new_text, new_nlp, new_num_forms)
            open_nodes.append(node)


def replace_dijkstra_batch(
    rvg: ReplacementVariantsGenerator,
    relax_count: int = 8,
) -> str:
    """Find best replacement using Dijkstra-inspired approach.

    Relaxes multiple nodes at once to speed up the search.
    """
    start_node = SearchNode("", 0, 0)
    open_nodes = [start_node]

    while True:
        new_texts: List[str] = []
        num_forms: List[int] = []
        for i in range(relax_count):
            if not open_nodes:
                break
            current = min(open_nodes, key=score)
            variants, num_variant_forms = rvg.get_variants(current.num_forms)
            if num_variant_forms == 0:
                if i == 0:
                    return current.text
                # can't return early, need to relax other nodes
                break
            open_nodes.remove(current)
            new_texts.extend(
                current.text + variant
                for variant in variants
            )
            num_forms.extend(
                current.num_forms + num_variant_forms
                for _ in variants
            )

        new_nlps = nlp.infer_nlp_batch(new_texts)
        for new_text, new_nlp, forms in zip(new_texts, new_nlps, num_forms):
            node = SearchNode(new_text, new_nlp, forms)
            open_nodes.append(node)
