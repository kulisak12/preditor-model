from prediktor.model.model import Model
from prediktor.replacement import dijkstra
from prediktor.replacement.variants import ReplacementVariantsGenerator


def replace(
    model: Model,
    text: str, start: int, length: int, replacement: str
) -> str:
    """Replace part of the text and modify the rest to match."""
    rvg = ReplacementVariantsGenerator(text, start, length, replacement)
    return dijkstra.replace_with_cache(model, rvg)
