from __future__ import annotations

from contextlib import contextmanager

import pytest

from pyampp.gxbox import gx_fov2box


class _FakeCoord:
    def __init__(self, state):
        self._state = state

    def transform_to(self, _frame):
        return {"screened": bool(self._state.get("screen", False))}


class _FakeMap:
    def __init__(self):
        self.coordinate_frame = type("_Frame", (), {"observer": "earth"})()

    def submap(self, bottom_left, *, top_right):
        if bottom_left.get("screened") and top_right.get("screened"):
            return "ok"
        raise ValueError(
            "The provided input coordinates to ``submap`` when transformed "
            "to the target coordinate frame contain NaN values and cannot be used to crop the map. "
            "The most common reason for NaN values is transforming off-disk 2D coordinates without "
            "specifying an assumption (e.g., via sunpy.coordinates.SphericalScreen())."
        )


def test_submap_with_fov_safe_retries_with_spherical_screen(monkeypatch) -> None:
    state = {"screen": False, "entered": 0}

    @contextmanager
    def _fake_screen_ctx(_observer):
        state["entered"] += 1
        state["screen"] = True
        try:
            yield
        finally:
            state["screen"] = False

    monkeypatch.setattr(gx_fov2box, "_spherical_screen_context_for_observer", _fake_screen_ctx)

    result = gx_fov2box._submap_with_fov_safe(_FakeMap(), _FakeCoord(state), _FakeCoord(state))

    assert result == "ok"
    assert state["entered"] == 1


def test_submap_with_fov_safe_propagates_non_offdisk_errors(monkeypatch) -> None:
    class _BadMap(_FakeMap):
        def submap(self, _bottom_left, *, top_right):
            raise ValueError("some unrelated submap failure")

    called = {"count": 0}

    @contextmanager
    def _fake_screen_ctx(_observer):
        called["count"] += 1
        yield

    monkeypatch.setattr(gx_fov2box, "_spherical_screen_context_for_observer", _fake_screen_ctx)

    with pytest.raises(ValueError, match="unrelated"):
        gx_fov2box._submap_with_fov_safe(_BadMap(), _FakeCoord({"screen": False}), _FakeCoord({"screen": False}))

    assert called["count"] == 0
