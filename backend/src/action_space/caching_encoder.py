"""Caching decorator for encoder backends.

Memoizes modifier estimation results keyed on (kind, name, description)
to avoid redundant LLM calls when the same action appears across ticks.
Uses an LRU eviction strategy with configurable max size.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from src.action_space.proposal import _ProposalBase
from src.shared.logging import get_logger

_log = get_logger(module="action_space.caching_encoder")


def _cache_key(proposal: _ProposalBase) -> tuple[str, str, str]:
    """Derive a stable cache key from proposal identity fields."""
    kind = getattr(proposal, "kind", "unknown")
    return (kind, proposal.name, proposal.description)


class CachingEncoderBackend:
    """LRU caching decorator over any EncoderBackend.

    Wraps an inner backend and caches results keyed on
    (kind, name, description). Thread-safety is not required
    because the simulator runs single-threaded per tick.
    """

    def __init__(
        self,
        inner: Any,
        max_size: int = 256,
    ) -> None:
        self._inner = inner
        self._max_size = max_size
        self._cache: OrderedDict[tuple[str, str, str], dict[str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def estimate(self, proposal: _ProposalBase) -> dict[str, float]:
        """Return cached modifiers or delegate to inner backend."""
        key = _cache_key(proposal)

        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return dict(self._cache[key])

        self._misses += 1
        modifiers = self._inner.estimate(proposal)
        self._cache[key] = dict(modifiers)

        if len(self._cache) > self._max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            _log.debug("cache_evicted", key=evicted_key)

        return dict(modifiers)

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss/size statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }

    def clear(self) -> None:
        """Flush the cache and reset counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
