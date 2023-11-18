from typing import Callable

from prediktor.replacement import dijkstra
from prediktor.replacement.variants import ReplacementVariantsGenerator


class Replacer:
    def __init__(
        self,
        func: Callable[[ReplacementVariantsGenerator], str],
    ) -> None:
        self.func = func

    def replace(
        self, text: str,
        start: int, length: int, replacement: str
    ) -> str:
        """Replace part of the text and modify the rest to match."""
        rvg = ReplacementVariantsGenerator(text, start, length, replacement)
        return self.func(rvg)


dijkstra_replacer = Replacer(dijkstra.replace_dijkstra)
