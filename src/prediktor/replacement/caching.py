from typing import Iterable, List, Optional, Tuple

import torch

# each cache is a tuple of layers
# each layer is a tuple of tensors (keys, values)
# each tensor has shape [batch_size, num_heads, seq_len, head_dim]
CacheLayer = Tuple[torch.Tensor, torch.Tensor]
Cache = Tuple[CacheLayer, ...]


def join_caches_optional(caches: List[Optional[Cache]]) -> Optional[Cache]:
    """Join the caches along the batch dimension.

    If any of the caches is None, the result is None.
    """
    # a bit convoluted to satisfy mypy
    # https://github.com/python/mypy/issues/4573
    not_none = [cache for cache in caches if cache is not None]
    if len(not_none) == len(caches):
        return join_caches(not_none)
    return None


def join_caches(caches: Iterable[Cache]) -> Cache:
    """Join the caches along the batch dimension."""
    return tuple(
        _join_layers(layers)
        # one layer for each of the caches
        for layers in zip(*caches)
    )


def _join_layers(layers: Iterable[CacheLayer]) -> CacheLayer:
    """Join the layers along the batch dimension."""
    keys = [layer[0] for layer in layers]
    values = [layer[1] for layer in layers]
    return torch.cat(keys, dim=0), torch.cat(values, dim=0)


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


def trim_cache(cache: Cache, length: int) -> Cache:
    """Trim the cache to the given length."""
    return tuple(
        _trim_layer(layer, length)
        for layer in cache
    )


def _trim_layer(layer: CacheLayer, length: int) -> CacheLayer:
    """Trim the layer to the given length."""
    keys, values = layer
    return keys[:, :, :length], values[:, :, :length]
