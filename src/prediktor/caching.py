import dataclasses
from typing import Iterable, List, Optional, Tuple

import torch

# each cache is a tuple of layers
# each layer is a tuple of tensors (keys, values)
# each tensor has shape [batch_size, num_heads, seq_len, head_dim]
CacheLayer = Tuple[torch.Tensor, torch.Tensor]
Cache = Tuple[CacheLayer, ...]


@dataclasses.dataclass(frozen=True)
class LazyCache:
    """A cache that has not yet been trimmed."""
    cache: Cache
    length: int


def cache_len(cache: Optional[LazyCache]) -> int:
    """Return the length of the cache."""
    if cache is None:
        return 0
    return cache.length


def join_caches_optional(caches: List[Optional[LazyCache]]) -> Optional[Cache]:
    """Join the caches along the batch dimension.

    If any of the caches is None, the result is None.
    """
    # a bit convoluted to satisfy mypy
    # https://github.com/python/mypy/issues/4573
    not_none = [cache for cache in caches if cache is not None]
    if len(not_none) == len(caches):
        return join_caches(not_none)
    return None


def join_caches(lazy_caches: Iterable[LazyCache]) -> Cache:
    """Join the caches along the batch dimension."""
    length = min(cache.length for cache in lazy_caches)
    caches = (cache.cache for cache in lazy_caches)
    return tuple(
        _join_layers(layers, length)
        # one layer for each of the caches
        for layers in zip(*caches)
    )


def _join_layers(layers: Iterable[CacheLayer], length: int) -> CacheLayer:
    """Join the layers along the batch dimension.

    Truncate the layers to the given length.
    """
    keys = [layer[0] for layer in layers]
    values = [layer[1] for layer in layers]
    return _truncate_cat(keys, length), _truncate_cat(values, length)


def _truncate_cat(tensors: List[torch.Tensor], length: int) -> torch.Tensor:
    """Truncate the tensors and concatenate them along the batch dimension."""
    truncated = [tensor[:, :, :length] for tensor in tensors]
    return torch.cat(truncated, dim=0)


def split_cache(cache: Cache) -> List[Cache]:
    """Split the cache along the batch dimension."""
    caches_by_layer = (
        _split_layer(layer)
        for layer in cache
    )
    return list(zip(*caches_by_layer))


def _split_layer(layer: CacheLayer) -> List[CacheLayer]:
    """Split the layer along the batch dimension."""
    keys, values = layer
    return [
        (keys[i : i + 1], values[i : i + 1])  # keeps the batch dimension
        for i in range(keys.size(0))
    ]
