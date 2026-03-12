"""Tests for caching encoder backend decorator."""

from src.action_space.caching_encoder import CachingEncoderBackend
from src.action_space.proposal import TextActionProposal, ToolActionProposal


class CountingBackend:
    """Tracks how many times estimate() is called."""

    def __init__(self, modifiers: dict[str, float] | None = None) -> None:
        self.call_count = 0
        self._modifiers = modifiers or {"O": 0.5}

    def estimate(self, proposal: object) -> dict[str, float]:
        """Return fixed modifiers and increment counter."""
        self.call_count += 1
        return dict(self._modifiers)


class TestCachingEncoderBackend:
    """CachingEncoderBackend memoizes inner backend calls."""

    def test_caches_identical_proposals(self) -> None:
        """Identical proposals hit the cache on second call."""
        inner = CountingBackend(modifiers={"O": 0.7})
        cached = CachingEncoderBackend(inner=inner, max_size=128)
        proposal = ToolActionProposal(
            name="search",
            description="search papers",
            tool_name="web_search",
            tool_args={"query": "test"},
        )
        r1 = cached.estimate(proposal)
        r2 = cached.estimate(proposal)
        assert r1 == r2
        assert inner.call_count == 1

    def test_different_proposals_miss_cache(self) -> None:
        """Different proposals each trigger a backend call."""
        inner = CountingBackend()
        cached = CachingEncoderBackend(inner=inner)
        p1 = ToolActionProposal(
            name="search",
            description="search papers",
            tool_name="web_search",
            tool_args={},
        )
        p2 = TextActionProposal(
            name="explain",
            description="explain concept",
            intent="explain",
        )
        cached.estimate(p1)
        cached.estimate(p2)
        assert inner.call_count == 2

    def test_same_name_different_description_misses(self) -> None:
        """Same name but different description is a cache miss."""
        inner = CountingBackend()
        cached = CachingEncoderBackend(inner=inner)
        p1 = TextActionProposal(
            name="explain",
            description="explain quantum physics",
            intent="explain",
        )
        p2 = TextActionProposal(
            name="explain",
            description="explain cooking",
            intent="explain",
        )
        cached.estimate(p1)
        cached.estimate(p2)
        assert inner.call_count == 2

    def test_evicts_when_full(self) -> None:
        """LRU eviction removes oldest entry when cache exceeds max_size."""
        inner = CountingBackend()
        cached = CachingEncoderBackend(inner=inner, max_size=2)
        proposals = [
            TextActionProposal(name=f"action_{i}", description=f"desc_{i}", intent="do")
            for i in range(3)
        ]
        for p in proposals:
            cached.estimate(p)
        assert inner.call_count == 3
        # Re-request the first (evicted) — should miss
        cached.estimate(proposals[0])
        assert inner.call_count == 4
        # Re-request the third (still cached) — should hit
        cached.estimate(proposals[2])
        assert inner.call_count == 4

    def test_cache_stats(self) -> None:
        """Stats reflect hits, misses, and current cache size."""
        inner = CountingBackend()
        cached = CachingEncoderBackend(inner=inner)
        proposal = ToolActionProposal(
            name="s",
            description="s",
            tool_name="t",
            tool_args={},
        )
        cached.estimate(proposal)
        cached.estimate(proposal)
        cached.estimate(proposal)
        stats = cached.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_clear_empties_cache(self) -> None:
        """Clearing the cache forces re-computation on next call."""
        inner = CountingBackend()
        cached = CachingEncoderBackend(inner=inner)
        proposal = ToolActionProposal(
            name="s",
            description="s",
            tool_name="t",
            tool_args={},
        )
        cached.estimate(proposal)
        cached.clear()
        cached.estimate(proposal)
        assert inner.call_count == 2
