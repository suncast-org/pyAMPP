"""Tests for header-only model geometry WCS scaffold (Context = none)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import Helioprojective, get_earth
from sunpy.map import Map, make_fitswcs_header

from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
from pyampp.gxbox.selector_api import BoxGeometrySelection, CoordMode, SelectorSessionInput

_MODEL_TIME = "2026-04-03T19:46:37.800"
_GEOMETRY = BoxGeometrySelection(CoordMode.HPC, 0.0, 225.83, 64, 64, 32, 1400.0)


def _model_base_wcs_header(*, date_obs: str = _MODEL_TIME, shape: tuple[int, int] = (128, 128)) -> str:
    obstime = Time(date_obs)
    earth = get_earth(obstime)
    center = SkyCoord(0 * u.arcsec, 225.83 * u.arcsec, frame=Helioprojective(observer=earth, obstime=obstime))
    header = make_fitswcs_header(
        np.zeros(shape, dtype=np.float32),
        center,
        scale=u.Quantity([6.0, 6.0], u.arcsec / u.pix),
    )
    header["DATE-OBS"] = obstime.isot
    return fits.Header(header).tostring(sep="\n", endcard=True)


def _make_widget(*, display_observer_key: str = "earth") -> MapBoxDisplayWidget:
    obstime = Time(_MODEL_TIME)
    earth = get_earth(obstime)
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._cache_lock = __import__("threading").Lock()
    widget._raw_map_cache = {}
    widget._observer_coord_cache = {"earth": earth}
    widget._observer_metadata_cache = {}
    widget._observer_warning_cache = set()
    widget._observer_source_b3d = lambda: {}
    widget._record_prepare_event = lambda _msg: None
    widget._ensure_cache_initialized = lambda: None
    widget._state = SimpleNamespace(
        session_input=SelectorSessionInput(
            time_iso=_MODEL_TIME,
            data_dir=".",
            geometry=_GEOMETRY,
            base_wcs_header=_model_base_wcs_header(),
            base_geometry=_GEOMETRY,
        ),
        geometry=_GEOMETRY,
        base_geometry=_GEOMETRY,
        base_wcs_header=_model_base_wcs_header(),
        geometry_definition_observer_key="earth",
        display_observer_key=display_observer_key,
        fov_definition_observer_key="earth",
        fov=None,
        fov_box=None,
        map_source_mode="embedded",
        selected_context_id=None,
        custom_observer_ephemeris=None,
    )
    return widget


def test_model_geometry_scaffold_earth_matches_model_time():
    widget = _make_widget(display_observer_key="earth")
    scaffold = widget._model_geometry_scaffold_map()
    assert scaffold is not None
    assert scaffold.date.isot == Time(_MODEL_TIME).isot
    assert np.all(np.isnan(np.asarray(scaffold.data)))


def test_model_geometry_scaffold_earth_uses_base_wcs_header_shape():
    widget = _make_widget(display_observer_key="earth")
    header = widget._model_geometry_earth_wcs_header()
    assert header is not None
    assert int(header["NAXIS1"]) == 128
    assert int(header["NAXIS2"]) == 128


def test_model_geometry_scaffold_stereo_builds_direct_observer_canvas():
    widget = _make_widget(display_observer_key="stereo-a")
    earth_header = widget._model_geometry_earth_wcs_header()
    earth_map = widget._header_only_map_from_wcs(earth_header)
    canvas = SimpleNamespace(date=Time(_MODEL_TIME), data=np.full((64, 64), np.nan))

    with patch(
        "pyampp.gxbox.box_view2d.MapBoxDisplayWidget._empty_observer_scaffold_from_geometry",
        return_value=canvas,
    ) as scaffold_mock, patch.object(
        MapBoxDisplayWidget,
        "_reproject_map_for_display_observer",
    ) as reproj_mock:
        out = widget._model_geometry_scaffold_map()

    assert out is canvas
    scaffold_mock.assert_called_once()
    reproj_mock.assert_not_called()


def test_default_context_id_none_when_only_bottom_maps():
    session = SelectorSessionInput(
        time_iso=_MODEL_TIME,
        data_dir=".",
        geometry=_GEOMETRY,
        map_ids=("Bx", "By", "Bz"),
        base_maps={"bx": np.zeros((2, 2)), "by": np.zeros((2, 2)), "bz": np.zeros((2, 2))},
    )
    assert MapBoxDisplayWidget._available_context_map_ids(session) == ["Bz"]
    assert MapBoxDisplayWidget._default_context_id(session) == "Bz"

    bottom_only = SelectorSessionInput(
        time_iso=_MODEL_TIME,
        data_dir=".",
        geometry=_GEOMETRY,
        map_ids=("Bx", "By"),
        base_maps={"bx": np.zeros((2, 2)), "by": np.zeros((2, 2))},
    )
    assert MapBoxDisplayWidget._available_context_map_ids(bottom_only) == []
    assert MapBoxDisplayWidget._default_context_id(bottom_only) is None


def test_uses_geometry_scaffold_when_none_selected_or_no_context_maps():
    widget = _make_widget(display_observer_key="earth")
    widget._state.selected_context_id = None
    assert widget._uses_geometry_scaffold_for_context() is True
    assert widget._context_canvas_map() is not None

    widget._state.selected_context_id = "171"
    widget._state.session_input = SelectorSessionInput(
        time_iso=_MODEL_TIME,
        data_dir=".",
        geometry=_GEOMETRY,
        map_ids=("Bx", "By"),
        base_maps={"bx": np.zeros((2, 2)), "by": np.zeros((2, 2))},
    )
    assert widget._uses_geometry_scaffold_for_context() is True

    with patch.object(
        MapBoxDisplayWidget, "_model_geometry_scaffold_map", return_value="scaffold"
    ) as scaffold_mock, patch.object(
        MapBoxDisplayWidget, "_selected_context_map", return_value="real-map"
    ) as selected_mock:
        assert widget._context_canvas_map() == "scaffold"
    scaffold_mock.assert_called_once()
    selected_mock.assert_not_called()


def test_model_geometry_scaffold_returns_none_without_geometry():
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        geometry=None,
        base_wcs_header=None,
        session_input=SimpleNamespace(time_iso=_MODEL_TIME),
    )
    assert widget._model_geometry_scaffold_map() is None
