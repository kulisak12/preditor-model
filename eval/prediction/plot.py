from typing import List, Tuple

import matplotlib.pyplot as plt

from preditor import nlp
from preditor.model.model import Model
from preditor.prediction import confidence
from preditor.prediction.config import PredictionConfig
from preditor.server import model


def save_plot(
    text: str, max_length: int, confidences: List[float],
    filename: str, figsize: Tuple[int, int] = (5, 4)
) -> None:
    tokens, usefulnesses = get_usefulness_data(model, text, max_length, confidences)
    plt.figure(figsize=figsize)
    for conf, usefulness in zip(confidences, usefulnesses):
        line, = plt.plot(usefulness, label=f"c={conf}")
        max_index = max(range(len(usefulness)), key=usefulness.__getitem__)
        plt.scatter(max_index, usefulness[max_index], color=line.get_color())
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.ylabel("E[u]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


def get_usefulness_data(
    model: Model, input_text: str, max_length: int, confidences: List[float]
) -> Tuple[List[str], List[List[float]]]:
    text_stripped = input_text.rstrip()
    had_trailing_space = input_text != text_stripped
    config = PredictionConfig(max_length=max_length)
    gen_ids, logits = confidence._get_model_outputs(model, text_stripped, had_trailing_space, config)
    tokens = [model.tokenizer.decode(x) for x in gen_ids]
    nlps = nlp.infer_nlps_from_logits(gen_ids, logits).tolist()
    # expected[i] is the expected usefulness of the prefix of length i
    usefulnesses = [
        confidence._calculate_expected_usefulness(nlps, c)
        for c in confidences
    ]
    return tokens, usefulnesses
