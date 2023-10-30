#!/usr/bin/env python3

import dataclasses
import sys
from typing import Dict, List, TextIO

from prediktor.infilling import blank, end, infilling
from prediktor.server import model


@dataclasses.dataclass(frozen=True)
class Example:
    before_cursor: str
    after_cursor: str
    expected: str


INFILLERS: Dict[str, infilling.InfillFunc] = {
    "Blank": blank.infill_between,
    "End": lambda m, b, a: end.infill_between(m, b, a, end.PROMPT_CS),
}


def main() -> None:
    is_debug = "--debug" in sys.argv
    with open("sentences.txt") as file:
        examples = read_examples(file)
    for name, infiller in INFILLERS.items():
        correct = evaluate(infiller, examples, is_debug)
        total = len(examples)
        print(f"{name}: {correct}/{total}")


def evaluate(
    infiller: infilling.InfillFunc,
    examples: List[Example],
    is_debug: bool = False
) -> int:
    correct = 0
    for example in examples:
        actual = infilling._infill(
            model, infiller,
            example.before_cursor + example.after_cursor,
            len(example.before_cursor)
        )
        if actual == example.expected:
            correct += 1
        if is_debug:
            print(f"{example.before_cursor}<{actual}>{example.after_cursor}")
    return correct


def read_examples(file: TextIO) -> List[Example]:
    examples: List[Example] = []
    for line in file:
        line = line.strip()
        if line:
            examples.append(make_example(line))
    return examples


def make_example(line: str) -> Example:
    blank_start = line.find("<")
    blank_end = line.find(">")
    assert blank_start != -1 and blank_end != -1
    before_cursor = line[:blank_start]
    after_cursor = line[blank_end + 1:]
    expected = line[blank_start + 1:blank_end]
    return Example(before_cursor, after_cursor, expected)


if __name__ == "__main__":
    main()
