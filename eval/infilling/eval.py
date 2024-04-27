#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import re
import time
from typing import Any, Dict, List, TextIO

from preditor.infilling import blank, end, infilling, selection
from preditor.model.model import Model
from preditor.prediction import simple
from preditor.server import model


@dataclasses.dataclass(frozen=True)
class Example:
    before_cursor: str
    expected: str
    after_cursor: str


def generate_infills_with_prediction(
    model: Model, before_cursor: str, after_cursor: str,
    config: infilling.InfillingConfig, lang: str
) -> List[str]:
    """Baseline for infilling that uses prediction."""
    prediction_config = simple.PredictionConfig(max_length=config.max_length)
    result = simple.generate(model, before_cursor, prediction_config)
    return [result]


GENERATE_FUNCS: Dict[str, infilling.GenerateFunc] = {
    "blank": blank.generate_infills,
    "end": end.generate_infills,
    "predict": generate_infills_with_prediction,
}
SELECT_FUNCS: Dict[str, infilling.SelectFunc] = {
    "match": selection.select_by_match,
    "score": selection.select_by_score,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate(
        args.dataset, args.generate, args.select,
        args.max_length, args.num_variants, args.debug,
    )


def evaluate(
    dataset: str, generate_funcname: str, select_funcname: str,
    max_length: int, num_variants: int, is_debug: bool = False
) -> None:
    summary = args_summary(locals())
    config = infilling.InfillingConfig(max_length=max_length, num_variants=num_variants)
    generate_func = GENERATE_FUNCS[generate_funcname]
    select_func = SELECT_FUNCS[select_funcname]
    with open(dataset) as file:
        examples = read_examples(file)

    correct = 0
    start = time.time()
    for example in examples:
        actual = infilling.infill(
            model, example.before_cursor, example.after_cursor,
            config, generate_func, select_func,
        )
        if actual == example.expected:
            correct += 1
        if is_debug:
            print(f"{example.before_cursor}|{actual}|{example.after_cursor}")
    end = time.time()
    avg_time = (end - start) / len(examples)

    print(f"{correct}/{len(examples)}  avg {avg_time:.3f}s  {summary}")


def read_examples(file: TextIO) -> List[Example]:
    reader = csv.DictReader(file, delimiter="|")
    return [Example(**row) for row in reader]


def args_summary(args: Dict[str, Any]) -> str:
    return ",".join(
        "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
        for k, v in args.items()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--generate", choices=GENERATE_FUNCS.keys(), required=True)
    parser.add_argument("--select", choices=SELECT_FUNCS.keys(), required=True)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--num-variants", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    main()
