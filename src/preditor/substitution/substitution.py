"""This module is the entry point for the substitution task."""

import itertools
from typing import Callable, Tuple

from preditor import tags
from preditor.model.model import Model
from preditor.substitution import dijkstra
from preditor.substitution.config import SubstitutionConfig
from preditor.substitution.variants import ReplacementVariantsGenerator

SubstituteFunc = Callable[[Model, ReplacementVariantsGenerator, SubstitutionConfig], str]


def replace(
    model: Model,
    before_old: str, old: str, after_old: str, replacement: str,
    config: SubstitutionConfig,
    func: SubstituteFunc = dijkstra.replace_with_cache,
) -> str:
    """Replace part of the sentence and modify the rest to match."""
    previous_sentences, _, next_sentences = _find_sentence_with_old(
        before_old, old, after_old
    )
    replaced_sentence = _replace_one_sentence(
        model,
        before_old[len(previous_sentences):],
        old,
        after_old[:len(after_old)-len(next_sentences)],
        replacement,
        config,
        func,
    )
    return previous_sentences + replaced_sentence + next_sentences


def _find_sentence_with_old(
    before_old: str, old: str, after_old: str
) -> Tuple[str, str, str]:
    """Find the sentence containing the old part.

    Return the previous sentences, the sentence containing the old part,
    and the next sentences.
    """
    text = before_old + old + after_old
    sentences = tags.split_sentences(text)
    sentence_ends = itertools.accumulate(map(len, sentences))
    target_index = next(
        i for i, end in enumerate(sentence_ends)
        if end > len(before_old)
    )
    previous_sentences = "".join(sentences[:target_index])
    sentence = sentences[target_index]
    next_sentences = "".join(sentences[target_index + 1:])
    return previous_sentences, sentence, next_sentences


def _replace_one_sentence(
    model: Model,
    before_old: str, old: str, after_old: str, replacement: str,
    config: SubstitutionConfig,
    func: SubstituteFunc = dijkstra.replace_with_cache,
) -> str:
    """Replace part of the sentence and modify the rest to match."""
    rvg = ReplacementVariantsGenerator(before_old, old, after_old, replacement)
    return func(model, rvg, config)
