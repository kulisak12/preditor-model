import itertools
from typing import Tuple

import pydantic

from preditor import tags
from preditor.model.model import Model
from preditor.substitution import dijkstra, search
from preditor.substitution.variants import ReplacementVariantsGenerator


class SubstitutionConfig(pydantic.BaseModel):
    min_variants: int = pydantic.Field(2, ge=2)
    relax_count: int = pydantic.Field(8, ge=1)
    pool_factor: int = pydantic.Field(1, ge=1)
    # no need to select score key, 0.0 yields same behavior as nlp_key
    lp_alpha: float = pydantic.Field(0.0, ge=0.0, le=1.0)


def replace(
    model: Model,
    before_old: str, old: str, after_old: str, replacement: str,
    config: SubstitutionConfig,
) -> str:
    """Replace part of the sentence containing the old part with the replacement
    and modify the rest of the sentence to match.
    """
    previous_sentences, _, next_sentences = _find_sentence_with_old(
        before_old, old, after_old
    )
    replaced_sentence = _replace_one_sentence(
        model,
        before_old[len(previous_sentences):],
        old,
        after_old[:-len(next_sentences)],
        replacement,
        config,
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
) -> str:
    """Replace part of the sentence and modify the rest to match."""
    rvg = ReplacementVariantsGenerator(before_old, old, after_old, replacement)
    return dijkstra.replace_with_cache(
        model, rvg,
        min_variants=config.min_variants,
        relax_count=config.relax_count,
        pool_size=config.relax_count * config.pool_factor,
        score_key=lambda x: search.lp_key(x, config.lp_alpha),
    )
