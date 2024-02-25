import heapq
from typing import List

from prediktor import nlp
from prediktor.replacement.search import ScoreKey, SearchNode, nlp_key
from prediktor.replacement.variants import ReplacementVariantsGenerator


def replace_dijkstra_simple(rvg: ReplacementVariantsGenerator) -> str:
    """Find best replacement using Dijkstra-inspired approach.

    A simplified version of replace_dijkstra without speedups.
    Useful for testing.
    """
    start_node = SearchNode("", 0, 0)
    open_nodes = {start_node}

    while True:
        current = min(open_nodes, key=nlp_key)
        open_nodes.remove(current)
        if current.num_forms == rvg.num_forms:
            return current.text
        relaxed = _relax_nodes([current], rvg)
        open_nodes.update(relaxed)


def replace_dijkstra(
    rvg: ReplacementVariantsGenerator,
    min_variants: int = 2,
    relax_count: int = 8,
    score_key: ScoreKey = nlp_key,
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
    open_nodes = {start_node}

    while True:
        best = min(open_nodes, key=score_key)
        if best.num_forms == rvg.num_forms:
            return best.text
        unfinished = (
            node for node in open_nodes
            if node.num_forms < rvg.num_forms
        )
        to_relax = heapq.nsmallest(relax_count, unfinished, key=score_key)
        open_nodes.difference_update(to_relax)
        relaxed = _relax_nodes(to_relax, rvg, min_variants)
        open_nodes.update(relaxed)


def replace_dijkstra_baseline(rvg: ReplacementVariantsGenerator) -> str:
    """Find best replacement using Dijkstra-inspired approach.

    Keep track of the best NLP for each word. Calculate the score as the
    difference between the best NLP and the current NLP.

    Does not work well.
    """
    def update_baselines(
        old_num_forms: int, new_num_forms: int, nlp_diff: float
    ) -> None:
        baseline_diff = baselines[new_num_forms] - baselines[old_num_forms]
        reduce_amount = baseline_diff - nlp_diff
        if reduce_amount > 0:
            for i in range(new_num_forms, len(baselines)):
                baselines[i] -= reduce_amount

    def baseline_key(node: SearchNode) -> float:
        return node.nlp - baselines[node.num_forms]

    start_node = SearchNode("", 0, 0)
    open_nodes = {start_node}
    baselines = [0.0]
    for _ in range(rvg.num_forms):
        baselines.append(baselines[-1] + 1000.0)
    finished_nodes: List[SearchNode] = []

    while open_nodes and len(finished_nodes) < 10:
        current = min(open_nodes, key=baseline_key)
        open_nodes.remove(current)
        if current.num_forms == rvg.num_forms:
            finished_nodes.append(current)
            continue

        extensions, extension_end = rvg.get_extensions(current.num_forms)
        new_texts = [current.text + extension for extension in extensions]
        new_nlps = nlp.infer_nlp(new_texts)
        for new_text, new_nlp in zip(new_texts, new_nlps):
            node = SearchNode(new_text, new_nlp, extension_end)
            open_nodes.add(node)
        best_nlp_diff = min(new_nlps) - current.nlp
        update_baselines(current.num_forms, extension_end, best_nlp_diff)

    return min(finished_nodes, key=baseline_key).text


def replace_dijkstra_with_cache(
    rvg: ReplacementVariantsGenerator,
    min_variants: int = 2,
) -> str:
    """Find best replacement using Dijkstra-inspired approach.

    Caches the NLP scores to avoid redundant calculations.
    """
    start_node = SearchNode("", 0, 0, None)
    open_nodes = {start_node}

    while True:
        current = min(open_nodes, key=nlp_key)
        open_nodes.remove(current)
        if current.num_forms == rvg.num_forms:
            return current.text
        relaxed = _relax_nodes_with_cache([current], rvg, min_variants)
        open_nodes.update(relaxed)


def _relax_nodes(
    nodes: List[SearchNode],
    rvg: ReplacementVariantsGenerator,
    min_variants: int = 2,
) -> List[SearchNode]:
    """Relax nodes by scoring their extensions."""
    to_score = _create_nodes_to_score(nodes, rvg, min_variants)
    new_nlps = nlp.infer_nlp([node.text for node in to_score])
    return [
        SearchNode(node.text, new_nlp, node.num_forms)
        for node, new_nlp in zip(to_score, new_nlps)
    ]


def _relax_nodes_with_cache(
    nodes: List[SearchNode],
    rvg: ReplacementVariantsGenerator,
    min_variants: int = 2,
) -> List[SearchNode]:
    """Relax nodes by scoring their extensions. Use the cache."""
    to_score = _create_nodes_to_score(nodes, rvg, min_variants)
    nlp_diffs, caches = nlp.infer_nlp_with_cache(
        [node.text for node in to_score],
        [node.cache for node in to_score],
    )
    return [
        SearchNode(node.text, node.nlp + nlp_diff, node.num_forms, cache)
        for node, nlp_diff, cache in zip(to_score, nlp_diffs, caches)
    ]


def _create_nodes_to_score(
    nodes: List[SearchNode],
    rvg: ReplacementVariantsGenerator,
    min_variants: int = 2,
) -> List[SearchNode]:
    """Create nodes to score by extending the given nodes."""
    to_score: List[SearchNode] = []
    for node in nodes:
        extensions, extension_end = rvg.get_extensions(
            node.num_forms, min_variants
        )
        to_score.extend(
            SearchNode(node.text + extension, node.nlp, extension_end, node.cache)
            for extension in extensions
        )
    return to_score
