import re
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
    # try prefixes of after_cursor from longest to shortest
    for i in range(len(after_cursor), 0, -1):
        for variant in variants:
            pos = variant.find(after_cursor[:i])
            if pos == -1:
                continue
            infill = variant[:pos]
            # avoid empty infills
            if infill.strip():
                return infill
    # no match
    return variants[0]


def get_number_of_tokens(model: Model, text: str) -> int:
    """Get the number of tokens if the text were tokenized.

    It is recommended to increase max_tokens by the number of tokens
    in the text after cursor if using the select_by_match strategy.
    """
    return len(model.tokenizer.encode(text))


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
        boundaries = [match.end() for match in re.finditer(r'\b', text)]
        for boundary in boundaries[:-1]:
            prefix = text[:boundary].rstrip()
            result.add(prefix)
    return result
