"""This module is the entry point for the prediction task."""

from typing import Callable

from preditor.model.model import Model
from preditor.prediction import confidence
from preditor.prediction.config import PredictionConfig

PredictFunc = Callable[[Model, str, PredictionConfig], str]


def predict(
    model: Model, text: str, config: PredictionConfig,
    func: PredictFunc = confidence.generate,
) -> str:
    """Generate continuation for the given text."""
    return func(model, text, config)
