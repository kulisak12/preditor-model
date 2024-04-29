#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import enum
import time
from typing import Dict, Iterable, List, TextIO

from preditor.model.model import Model
from preditor.substitution import dijkstra, substitution


@dataclasses.dataclass(frozen=True)
class Example:
    before_old: str
    old: str
    after_old: str
    replacement: str
    expected: str


@dataclasses.dataclass(frozen=True)
class Result:
    original: str
    replaced: str
    expected: str
    time: float

    @classmethod
    def from_dict(cls, data: dict) -> 'Result':
        data["time"] = float(data["time"])
        return cls(**data)


class ChangeType(enum.Enum):
    NONE = 1
    GOOD = 2
    BAD = 3
    MISSED = 4


SUBSTITUTE_FUNCS: Dict[str, substitution.SubstituteFunc] = {
    "simple": dijkstra.replace,
    "cache": dijkstra.replace_with_cache,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.results_only:
        run(
            args.dataset, args.output, args.strategy,
            args.min_variants, args.relax_count,
            args.pool_factor, args.lp_alpha,
            args.progress
        )
    eval(args.output)


def run(
    dataset: str, out_filename: str,
    substitute_funcname: str,
    min_variants: int, relax_count: int,
    pool_factor: int, lp_alpha: float,
    show_progress: bool = False
) -> None:
    from preditor.server import model
    config = substitution.SubstitutionConfig(
        min_variants=min_variants, relax_count=relax_count,
        pool_factor=pool_factor, lp_alpha=lp_alpha,
    )
    substitute_func = SUBSTITUTE_FUNCS[substitute_funcname]
    with open(dataset) as in_file:
        examples = read_examples(in_file)

    with open(out_filename, "w") as out_file:
        fieldnames = [f.name for f in dataclasses.fields(Result)]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        warm_up(model, config, substitute_func)

        for i, example in enumerate(examples, start=1):
            if show_progress:
                print(f"{i}/{len(examples)}")
            start = time.time()
            replaced = substitution.replace(
                model, example.before_old, example.old, example.after_old,
                example.replacement, config, substitute_func
            )
            end = time.time()
            original = example.before_old + example.old + example.after_old
            result = Result(
                replaced=replaced, time=end - start,
                original=original, expected=example.expected
            )
            writer.writerow(dataclasses.asdict(result))


def eval(results_filename: str) -> None:
    with open(results_filename) as file:
        reader = csv.DictReader(file, delimiter="|")
        results = [Result.from_dict(row) for row in reader]

    avg_time = sum(result.time for result in results) / len(results)
    total_correct = sum(result.replaced == result.expected for result in results)
    print(f"Average time: {avg_time:.2f}s")
    print(f"Total correct: {total_correct}/{len(results)}")
    print(f"Total good changes: {count_changes(results, ChangeType.GOOD) - len(results)}")
    print(f"Total bad changes: {count_changes(results, ChangeType.BAD)}")
    print(f"Total missed changes: {count_changes(results, ChangeType.MISSED)}")


def count_changes(results: Iterable[Result], change_type: ChangeType) -> int:
    count = 0
    for result in results:
        original_words = result.original.split()
        replaced_words = result.replaced.split()
        expected_words = result.expected.split()
        for original, replaced, expected in zip(original_words, replaced_words, expected_words):
            if get_change_type(original, replaced, expected) == change_type:
                count += 1
    return count


def get_change_type(original: str, replaced: str, expected: str) -> ChangeType:
    if replaced == expected and original == expected:
        return ChangeType.NONE
    if replaced == expected:
        return ChangeType.GOOD
    if original == expected:
        return ChangeType.BAD
    return ChangeType.MISSED


def read_examples(file: TextIO) -> List[Example]:
    reader = csv.DictReader(file, delimiter="|")
    return [Example(**row) for row in reader]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--strategy", choices=SUBSTITUTE_FUNCS.keys(), required=True)
    parser.add_argument("--min-variants", type=int, default=2)
    parser.add_argument("--relax-count", type=int, default=8)
    parser.add_argument("--pool-factor", type=int, default=5)
    parser.add_argument("--lp-alpha", type=float, default=0.0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    return parser


def warm_up(
    model: Model, config: substitution.SubstitutionConfig,
    substitute_func: substitution.SubstituteFunc
) -> None:
    """Run a warm-up to load the data into cache."""
    substitution.replace(
        model, "Modrá ", "barva", ".", "světlo",
        config, substitute_func
    )


if __name__ == "__main__":
    main()
