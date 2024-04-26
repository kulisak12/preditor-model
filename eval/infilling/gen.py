#!/usr/bin/env python3
import csv
import dataclasses
import random
import re
import sys
from typing import Optional


@dataclasses.dataclass(frozen=True)
class Example:
    before_cursor: str
    expected: str
    after_cursor: str


def main() -> None:
    random.seed(42)
    fieldnames = [f.name for f in dataclasses.fields(Example)]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="|")
    writer.writeheader()
    for line in sys.stdin:
        line = line.strip()
        result = split_line(line)
        if result:
            writer.writerow(dataclasses.asdict(result))


def split_line(line: str) -> Optional[Example]:
    words = line.split()
    if len(words) < 8:
        return None
    skipped = random.randint(1, 3)
    skip_start = random.randrange(0, len(words) - skipped)
    before_cursor = " ".join(words[:skip_start]) + " "
    expected = " ".join(words[skip_start:skip_start + skipped])
    after_cursor = " " + " ".join(words[skip_start + skipped:])
    return Example(before_cursor.lstrip(), expected, after_cursor.rstrip())


if __name__ == "__main__":
    main()
