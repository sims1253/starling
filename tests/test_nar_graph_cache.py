from __future__ import annotations

import pytest

from starling.nar.mega import _GraphCache


def test_graph_cache_captures_only_recurring_shapes():
    cache = _GraphCache(max_captures=2, capture_after=2)

    assert not cache.should_capture("short")
    assert cache.should_capture("short")

    entry = {"graph": object()}
    cache.put("short", entry)
    assert cache.get("short") is entry
    assert not cache.should_capture("short")


def test_graph_cache_stops_capturing_at_capacity():
    cache = _GraphCache(max_captures=1, capture_after=1)

    assert cache.should_capture("short")
    cache.put("short", {"graph": object()})

    assert not cache.should_capture("medium")
    assert cache.get("short") is not None
    assert cache.get("medium") is None
    with pytest.raises(RuntimeError, match="capacity"):
        cache.put("medium", {"graph": object()})


def test_graph_cache_bounds_shape_sightings():
    cache = _GraphCache(max_captures=1, capture_after=3, max_seen=2)

    assert not cache.should_capture("short")
    assert not cache.should_capture("medium")
    assert not cache.should_capture("long")

    assert list(cache._seen) == ["medium", "long"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_captures": -1}, "max_captures"),
        ({"max_captures": 1, "capture_after": 0}, "capture_after"),
        ({"max_captures": 1, "max_seen": 0}, "max_seen"),
    ],
)
def test_graph_cache_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _GraphCache(**kwargs)
