import dataclasses
from typing import Callable, List

from prediktor.replacement import nlp
from prediktor.replacement.variants import ReplacementVariantsGenerator


@dataclasses.dataclass(frozen=True)
class SearchNode:
    text: str
    nlp: float
    num_forms: int


def nlp_key(node: SearchNode) -> float:
    """Straightforward scoring with NLP."""
    return node.nlp


def lp_key(node: SearchNode, alpha: float = 0.5) -> float:
    """Normalize score by a function of length."""
    factor = (5 + 1)**alpha / (5 + node.num_forms)**alpha
    return node.nlp * factor


def replace_dijkstra_simple(rvg: ReplacementVariantsGenerator) -> str:
    """Find best replacement using Dijkstra-inspired approach.

    A simplified version of replace_dijkstra without speedups.
    Useful for testing.
    """
    start_node = SearchNode("", 0, 0)
    open_nodes = [start_node]

    while True:
        current = min(open_nodes, key=nlp_key)
        open_nodes.remove(current)
        variants, num_variant_forms = rvg.get_variants(current.num_forms, 1)
        if num_variant_forms == 0:
            return current.text
        new_num_forms = current.num_forms + num_variant_forms

        new_texts = [current.text + variant for variant in variants]
        new_nlps = nlp.infer_nlp_batch(new_texts)
        for new_text, new_nlp in zip(new_texts, new_nlps):
            node = SearchNode(new_text, new_nlp, new_num_forms)
            open_nodes.append(node)


def replace_dijkstra(
    rvg: ReplacementVariantsGenerator,
    min_variants: int = 1,
    relax_count: int = 8,
    score_key: Callable[[SearchNode], float] = nlp_key,
) -> str:
    """Find best replacement using Dijkstra-inspired approach.

    Scores many texts at once to speed up the search.

    Args:
        min_variants: Generate at least this many variants for each
            node, possible extending the text by more than one word.
        relax_count: Number of nodes to relax at once.
        score_key: Function to use for scoring nodes.
    """
    start_node = SearchNode("", 0, 0)
    open_nodes = [start_node]

    while True:
        new_texts: List[str] = []
        num_forms: List[int] = []
        for i in range(relax_count):
            if not open_nodes:
                break
            current = min(open_nodes, key=score_key)
            variants, num_variant_forms = rvg.get_variants(
                current.num_forms, min_variants
            )
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
