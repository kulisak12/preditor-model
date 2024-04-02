import heapq
from typing import Iterable, List

from preditor import nlp
from preditor.model.model import Model
from preditor.replacement.search import ScoreKey, SearchNode, nlp_key
from preditor.replacement.variants import ReplacementVariantsGenerator


def replace(
    model: Model,
    rvg: ReplacementVariantsGenerator,
    min_variants: int = 2,
    relax_count: int = 8,
    score_key: ScoreKey = nlp_key,
) -> str:
    """Find best replacement using Dijkstra-inspired approach.

    Scores many texts at once to speed up the search.

    min_variants: Generate at least this many variants for each node,
        possibly extending the text by more than one word.
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
        relaxed = _relax_nodes(model, to_relax, rvg, min_variants)
        open_nodes.update(relaxed)


def replace_baseline(
    model: Model,
    rvg: ReplacementVariantsGenerator
) -> str:
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
        new_nlps = nlp.infer_nlp(model, new_texts)
        for new_text, new_nlp in zip(new_texts, new_nlps):
            node = SearchNode(new_text, new_nlp, extension_end)
            open_nodes.add(node)
        best_nlp_diff = min(new_nlps) - current.nlp
        update_baselines(current.num_forms, extension_end, best_nlp_diff)

    return min(finished_nodes, key=baseline_key).text


def replace_with_cache(
    model: Model,
    rvg: ReplacementVariantsGenerator,
    min_variants: int = 2,
    relax_count: int = 8,
    pool_size: int = 40,
    score_key: ScoreKey = nlp_key,
) -> str:
    """Find best replacement using Dijkstra-inspired approach.

    Caches the NLP scores to avoid redundant calculations.
    """
    start_node = SearchNode("", 0, 0, None)
    open_nodes = {start_node}

    while True:
        best = min(open_nodes, key=score_key)
        if best.num_forms == rvg.num_forms:
            return best.text
        unfinished = (
            node for node in open_nodes
            if node.num_forms < rvg.num_forms
        )
        to_relax = _select_nodes_to_relax_with_cache(
            best, unfinished, relax_count, pool_size, score_key
        )
        open_nodes.difference_update(to_relax)
        relaxed = _relax_nodes_with_cache(model, to_relax, rvg, min_variants)
        open_nodes.update(relaxed)


def _select_nodes_to_relax_with_cache(
    best: SearchNode,
    unfinished: Iterable[SearchNode],
    relax_count: int,
    pool_size: int,
    score_key: ScoreKey,
) -> List[SearchNode]:
    """Select the best nodes to relax.

    Always include the best node.
    Then, create a pool of the best nodes and select a subset of them
    such that the lengths of their caches are as similar as possible.
    """
    def sort_pool(pool: List[SearchNode]) -> List[SearchNode]:
        """Sort the pool by cache length.

        If equal, move nodes with a better score closer to the best node.
        """
        shorter = [node for node in pool if node.cache_len <= best.cache_len]
        longer = [node for node in pool if node.cache_len > best.cache_len]
        shorter.sort(key=lambda node: (node.cache_len, -score_key(node)))
        longer.sort(key=lambda node: (node.cache_len, score_key(node)))
        return shorter + longer

    def find_most_similar_subarray(array: List[int], length: int) -> int:
        """Find a subarray such that the sum of its elements differs the least
        from its minimum multiplied by the length.
        """
        min_diff = sum(array)
        start_index = 0
        for i in range(len(array) - length + 1):
            subarray = array[i:i+length]
            subarray_sum = sum(subarray)
            min_element = min(subarray)
            diff = abs(subarray_sum - min_element * length)
            if diff < min_diff:
                min_diff = diff
                start_index = i
        return start_index

    pool = heapq.nsmallest(pool_size, unfinished, key=score_key)
    if len(pool) <= relax_count:
        return pool
    pool = sort_pool(pool)
    best_index = pool.index(best)
    # ensure that the best node is included
    pool = pool[max(0, best_index - relax_count + 1):best_index + relax_count]
    lengths = [node.cache_len for node in pool]
    start = find_most_similar_subarray(lengths, relax_count)
    return pool[start:start+relax_count]


def _relax_nodes(
    model: Model,
    nodes: List[SearchNode],
    rvg: ReplacementVariantsGenerator,
    min_variants,
) -> List[SearchNode]:
    """Relax nodes by scoring their extensions."""
    to_score = _create_nodes_to_score(nodes, rvg, min_variants)
    new_nlps = nlp.infer_nlp(model, [node.text for node in to_score])
    return [
        SearchNode(node.text, new_nlp, node.num_forms)
        for node, new_nlp in zip(to_score, new_nlps)
    ]


def _relax_nodes_with_cache(
    model: Model,
    nodes: List[SearchNode],
    rvg: ReplacementVariantsGenerator,
    min_variants,
) -> List[SearchNode]:
    """Relax nodes by scoring their extensions. Use the cache."""
    to_score = _create_nodes_to_score(nodes, rvg, min_variants)
    nlp_diffs, caches = nlp.infer_nlp_with_cache(
        model,
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
    min_variants,
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
