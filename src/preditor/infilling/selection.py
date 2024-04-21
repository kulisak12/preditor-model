from typing import Iterable, List, Set

from preditor import nlp
from preditor.model.model import Model


def select_by_match(
    variants: List[str],
    model: Model, before_cursor: str, after_cursor: str,
) -> str:
    """Return the variant that best matches the text after cursor.

    Favor variants in the beginning of the list.
    """
    # exact match
    for variant in variants:
        if variant.endswith(after_cursor):
            return variant[:len(variant) - len(after_cursor)]
    # partial match
    end_first_word = after_cursor.split()[0]
    for variant in variants:
        pos = variant.find(end_first_word)
        if pos != -1:
            return variant[:pos]
    # no match
    for variant in variants:
        splits = variant.split()
        if len(splits) > 1:
            return splits[0]
    # no variant
    return ""


def select_by_score(
    variants: List[str],
    model: Model, before_cursor: str, after_cursor: str,
) -> str:
    """Select the best infill from given variants.

    Score the sentence filled with all possible prefixes of the variants.
    Return the prefix that yields the best score.
    """
    variants = list(_expand_prefixes(variants))
    if after_cursor.lstrip() == after_cursor:
        variants = [variant + " " for variant in variants]
    final = [
        before_cursor + variant + after_cursor
        for variant in variants
    ]
    nlps = nlp.infer_nlp(model, final)
    argmin = min(range(len(nlps)), key=nlps.__getitem__)
    return variants[argmin]


def _expand_prefixes(infill_texts: Iterable[str]) -> Set[str]:
    """Generate all prefixes of the generated texts.

    The prefix always ends with a word boundary.
    The last word is not included since it is not certain that it is complete.
    """
    result: Set[str] = set()
    for text in infill_texts:
        words = text.split()
        for i in range(1, len(words)):
            result.add(" ".join(words[:i]))
    return result
