from preditor.model.model import Model
from preditor.substitution import dijkstra
from preditor.substitution.variants import ReplacementVariantsGenerator


def replace(
    model: Model,
    text: str, start: int, length: int, replacement: str
) -> str:
    """Replace part of the text and modify the rest to match."""
    rvg = ReplacementVariantsGenerator(text, start, length, replacement)
    return dijkstra.replace_with_cache(model, rvg)
