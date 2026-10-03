"""Regression tests for embedded map display crop after observer/source switches."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import Helioprojective, get_earth
from sunpy.map import Map, make_fitswcs_header

from pyampp.gxbox.box_view2d import (
    MapBoxDisplayWidget,
    _CONTEXT_PREPARE_VARIANT_FOV_CROP,
    _EMBEDDED_REFMAP_FLAG,
    _MIN_DISPLAY_MAP_SIDE,
)
from pyampp.gxbox.boxutils import map_from_data_header_compat
from pyampp.gxbox.selector_api import DisplayFovSelection

_OBS_TIME = "2026-04-03T19:46:37.800"
_MODEL_FOV = DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64)


def _stereo_native_embedded_map(*, size: int = 256, scale: float = 4.0) -> Map:
    header = fits.Header()
    header["TELESCOP"] = "STEREO"
    header["INSTRUME"] = "SECCHI"
    header["PYALIGN"] = False
    header["NAXIS"] = 2
    header["NAXIS1"] = size
    header["NAXIS2"] = size
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = size / 2 + 0.5
    header["CRPIX2"] = size / 2 + 0.5
    header["CRVAL1"] = -1020.0
    header["CRVAL2"] = 110.0
    header["CDELT1"] = scale
    header["CDELT2"] = scale
    header["DATE-OBS"] = "2026-04-03T19:58:30.006"
    header["HGLN_OBS"] = 54.3710463167
    header["HGLT_OBS"] = -0.875478602937
    header["DSUN_OBS"] = 1.496e11
    smap = map_from_data_header_compat(np.ones((size, size), dtype=np.float32), header)
    smap.meta[_EMBEDDED_REFMAP_FLAG] = True
    return smap


def _make_widget(*, display_observer_key: str) -> MapBoxDisplayWidget:
    obstime = Time(_OBS_TIME)
    earth = get_earth(obstime)
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key=display_observer_key,
        fov_definition_observer_key="stereo-a",
        geometry_definition_observer_key="earth",
        fov=_MODEL_FOV,
        geometry=None,
        session_input=SimpleNamespace(time_iso=obstime.isot),
        custom_observer_ephemeris=None,
    )
    widget._status_callback = None
    widget._observer_coord_cache = {"earth": earth}
    widget._observer_metadata_cache = {}
    widget._observer_warning_cache = set()
    widget._observer_source_b3d = lambda: {}
    widget._record_prepare_event = lambda _msg: None
    widget._apply_display_scaling = lambda smap, _k: smap
    widget._view_mode = "box_fov"
    widget._observer_context = lambda key, _obstime: SimpleNamespace(
        observer_coordinate=earth if key == "earth" else "stereo-a",
        date=obstime,
    )
    return widget


def test_valid_map_array_rejects_degenerate_submaps():
    tiny = SimpleNamespace(data=np.zeros((6, 5)))
    ok = SimpleNamespace(data=np.zeros((_MIN_DISPLAY_MAP_SIDE, _MIN_DISPLAY_MAP_SIDE)))
    assert MapBoxDisplayWidget._valid_map_array(tiny) is False
    assert MapBoxDisplayWidget._valid_map_array(ok) is True


def test_prepare_embedded_stereo_map_keeps_usable_extent_for_earth_display():
    widget = _make_widget(display_observer_key="earth")
    smap = _stereo_native_embedded_map(size=256, scale=4.0)
    stereo_obs = smap.observer_coordinate
    widget._status_callback = None
    widget._observer_coord_cache["stereo-a"] = stereo_obs
    widget._observer_context = lambda key, obstime: SimpleNamespace(
        observer_coordinate=widget._observer_coord_cache["earth"] if key == "earth" else stereo_obs,
        date=obstime,
    )

    with patch.object(
        MapBoxDisplayWidget,
        "_reproject_map_for_display_observer",
        side_effect=lambda m, **kwargs: (m, None),
    ):
        out, _coverage = widget._prepare_context_map(
            "20260403_200000_195A",
            smap,
            prepare_variant=_CONTEXT_PREPARE_VARIANT_FOV_CROP,
        )

    assert out is not None
    shape = tuple(np.asarray(out.data).shape)
    assert shape[0] >= _MIN_DISPLAY_MAP_SIDE
    assert shape[1] >= _MIN_DISPLAY_MAP_SIDE


def test_submap_to_fov_selection_pixels_returns_original_when_crop_degenerate():
    obstime = Time(_OBS_TIME)
    earth = get_earth(obstime)
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=earth, obstime=obstime))
    header = make_fitswcs_header(
        np.zeros((128, 128), dtype=np.float32),
        center,
        scale=u.Quantity([4.0, 4.0], u.arcsec / u.pix),
    )
    smap = Map(np.zeros((128, 128), dtype=np.float32), header)
    widget = _make_widget(display_observer_key="earth")
    widget._projected_box_bbox_rect = None

    with patch.object(
        MapBoxDisplayWidget,
        "_fov_selection_to_pixel_rect",
        return_value=SimpleNamespace(get_x=lambda: 0.0, get_y=lambda: 0.0, get_width=lambda: 4.0, get_height=lambda: 5.0),
    ):
        out = widget._submap_to_fov_selection_pixels(
            smap,
            _MODEL_FOV,
            use_display_observer=True,
        )

    assert tuple(np.asarray(out.data).shape) == (128, 128)
