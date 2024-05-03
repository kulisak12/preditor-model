"""This module provides functions for estimating the language of a text."""

from typing import Any, Iterable, Tuple

import fasttext

from preditor.config import Config

model = fasttext.load_model(Config.fasttext_path)


def estimate_language(text: str, choices: Iterable[str]) -> str:
    """Estimate the most likely language of the text.

    Only consider the languages given as choices.
    """
    labels, probs = model.predict(text, k=len(model.labels))
    most_likely = max(choices, key=lambda lang: _get_prob(labels, probs, lang))
    return most_likely


def _get_prob(labels: Tuple, probs: Any, lang: str) -> float:
    label = "__label__" + lang
    if label not in labels:
        return 0.0
    label_index = labels.index("__label__" + lang)
    return probs[label_index]
