#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import time
from typing import Any, Dict, List, TextIO

from preditor.infilling import blank, end, infilling, selection
from preditor.model.model import Model
from preditor.prediction import simple


@dataclasses.dataclass(frozen=True)
class Example:
    before_cursor: str
    expected: str
    after_cursor: str


@dataclasses.dataclass(frozen=True)
class Result:
    before_cursor: str
    infill: str
    after_cursor: str
    expected: str
    time: float

    @classmethod
    def from_dict(cls, data: dict) -> 'Result':
        data["time"] = float(data["time"])
        return cls(**data)


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
    if not args.results_only:
        run(
            args.dataset, args.output,
            args.generate, args.select,
            args.max_length, args.num_variants,
            args.progress,
        )
    eval(args.output)


def run(
    dataset: str, out_filename: str,
    generate_funcname: str, select_funcname: str,
    max_length: int, num_variants: int,
    show_progress: bool = False
) -> None:
    from preditor.server import model
    config = infilling.InfillingConfig(max_length=max_length, num_variants=num_variants)
    generate_func = GENERATE_FUNCS[generate_funcname]
    select_func = SELECT_FUNCS[select_funcname]
    with open(dataset) as in_file:
        examples = read_examples(in_file)

    with open(out_filename, "w") as out_file:
        fieldnames = [f.name for f in dataclasses.fields(Result)]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()

        for i, example in enumerate(examples, start=1):
            if show_progress:
                print(f"{i}/{len(examples)}")
            start = time.time()
            infill = infilling.infill(
                model, example.before_cursor, example.after_cursor,
                config, generate_func, select_func,
            )
            end = time.time()
            result = Result(infill=infill, time=end - start, **dataclasses.asdict(example))
            writer.writerow(dataclasses.asdict(result))


def eval(results_filename: str) -> None:
    with open(results_filename) as file:
        reader = csv.DictReader(file, delimiter="|")
        results = [Result.from_dict(row) for row in reader]

    avg_time = sum(result.time for result in results) / len(results)
    total_correct = sum(result.infill == result.expected for result in results)
    print(f"Average time: {avg_time:.3f}s")
    print(f"Total correct: {total_correct}/{len(results)}")


def read_examples(file: TextIO) -> List[Example]:
    reader = csv.DictReader(file, delimiter="|")
    return [Example(**row) for row in reader]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--generate", choices=GENERATE_FUNCS.keys(), required=True)
    parser.add_argument("--select", choices=SELECT_FUNCS.keys(), required=True)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--num-variants", type=int, default=4)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    return parser


if __name__ == "__main__":
    main()
