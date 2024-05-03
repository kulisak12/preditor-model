"""This module is the entry point for the infilling task."""

from typing import Callable, List

from preditor import language
from preditor.infilling import blank, end, selection
from preditor.infilling.config import InfillingConfig
from preditor.model.model import Model

GenerateFunc = Callable[[Model, str, str, InfillingConfig, str], List[str]]
SelectFunc = Callable[[List[str], Model, str, str], str]
LANGS = ["en", "cs"]

def infill(
    model: Model, before_cursor: str, after_cursor: str,
    config: InfillingConfig,
    generate_func: GenerateFunc = end.generate_infills,
    select_func: SelectFunc = selection.select_by_score
) -> str:
    """Generate an infill between the two texts."""
    lang = language.estimate_language(before_cursor + " " + after_cursor, LANGS)
    variants = generate_func(
        model, before_cursor, after_cursor, config, lang
    )
    # filter out empty strings
    variants = [v for v in variants if v]
    if not variants:
        return ""
    selected = select_func(
        variants, model, before_cursor, after_cursor
    )
    return selected

