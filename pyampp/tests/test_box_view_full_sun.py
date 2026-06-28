"""Regression tests for Full Sun View viewport behavior.

Full Sun View must be map-source and observer agnostic:

1. Center on display-observer solar disk center (HPC 0, 0), not the FOV center.
2. Size the square viewport from ``RSUN_ARCSEC / map plate scale`` (primary path in
   ``_full_sun_disk_half_extent_pixels``).
3. Union projected red/blue overlay rects only to expand the square, never to replace
   disk sizing.
4. Do not fall back to ``_projected_box_bbox_rect`` when converting disk geometry for
   display-observer FOV rects (``use_display_observer=True``).
5. Apply viewport after ``smap.plot()`` and box outline (``_refresh_plot``), not before.
6. Embedded PYALIGN maps in ``full_sun`` mode pass a disk HPC ROI to reproject.

Do not remove or bypass these steps without updating this module and verifying Earth
filesystem AIA and embedded views manually.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from matplotlib.patches import Rectangle
from sunpy.coordinates import Helioprojective, get_earth
from sunpy.map import make_fitswcs_header, Map

from pyampp.gxbox.box_view2d import (
    MapBoxDisplayWidget,
    _FULL_SUN_DISK_EXTENT_PAD,
    _FULL_SUN_VIEWPORT_PAD,
)
from pyampp.gxbox.selector_api import DisplayFovSelection

_OBS_TIME = "2026-04-03T19:46:37.800"
_MODEL_FOV = DisplayFovSelection(-0.0, 225.83, 665.20, 665.20)
_AIA_PLATE_SCALE_ARCSEC = 0.6


def _make_earth_widget(*, smap=None) -> MapBoxDisplayWidget:
    obstime = Time(_OBS_TIME)
    earth = get_earth(obstime)
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="earth",
        fov_definition_observer_key="earth",
        fov=_MODEL_FOV,
        fov_box=None,
        geometry=None,
        session_input=SimpleNamespace(time_iso=obstime.isot),
        custom_observer_ephemeris=None,
    )
    widget._observer_coord_cache = {"earth": earth}
    widget._observer_metadata_cache = {}
    widget._observer_warning_cache = set()
    widget._observer_source_b3d = lambda: {}
    if smap is not None:
        widget._current_map = smap
    widget._current_axes = SimpleNamespace(
        get_xlim=lambda: (0.0, 4096.0),
        get_ylim=lambda: (0.0, 4096.0),
        set_xlim=lambda *_a, **_k: None,
        set_ylim=lambda *_a, **_k: None,
    )
    widget._canvas = SimpleNamespace(draw_idle=lambda: None)
    return widget


def _aia_like_map(*, shape: tuple[int, int] = (4096, 4096)) -> Map:
    obstime = Time(_OBS_TIME)
    earth = get_earth(obstime)
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=earth, obstime=obstime))
    header = make_fitswcs_header(
        np.zeros(shape, dtype=np.float32),
        center,
        scale=u.Quantity([_AIA_PLATE_SCALE_ARCSEC, _AIA_PLATE_SCALE_ARCSEC], u.arcsec / u.pix),
    )
    return Map(np.zeros(shape, dtype=np.float32), header)


def test_union_pixel_view_windows_merges_disjoint_extents():
    left = ((-10.0, 10.0), (-10.0, 10.0))
    right = ((50.0, 70.0), (-5.0, 5.0))
    out = MapBoxDisplayWidget._union_pixel_view_windows(left, right)
    assert out == ((-10.0, 70.0), (-10.0, 10.0))


def test_union_pixel_rect_bounds_merges_disjoint_rects():
    left = Rectangle((-10.0, -10.0), 20.0, 20.0, visible=False)
    right = Rectangle((50.0, -5.0), 20.0, 10.0, visible=False)
    out = MapBoxDisplayWidget._union_pixel_rect_bounds(left, right)
    assert out == (-10.0, 70.0, -10.0, 10.0)


def test_rsun_arcsec_from_observer_metadata_uses_dsun_when_map_lacks_it():
    obstime = Time(_OBS_TIME)
    earth = get_earth(obstime)
    metadata = {
        "observer_coordinate": earth,
        "dsun_cm": float(earth.radius.to_value(u.cm)),
        "rsun_cm": 6.96e10,
    }
    rsun = MapBoxDisplayWidget._rsun_arcsec_from_observer_metadata(metadata)
    assert rsun is not None
    assert rsun == pytest.approx(960.0, rel=0.05)


def test_full_sun_disk_rsun_prefers_observer_metadata_over_map_without_dsun():
    obstime = Time(_OBS_TIME)
    widget = _make_earth_widget()
    smap = SimpleNamespace(meta={}, rsun_meters=None, dsun=None)
    rsun = widget._full_sun_disk_rsun_arcsec(smap, obstime)
    assert rsun is not None
    assert rsun == pytest.approx(960.0, rel=0.05)


def test_full_sun_disk_half_extent_pixels_uses_rsun_and_map_scale():
    smap = _aia_like_map()
    widget = _make_earth_widget(smap=smap)

    half = widget._full_sun_disk_half_extent_pixels(smap, pad_factor=_FULL_SUN_DISK_EXTENT_PAD)
    assert half is not None
    rsun = widget._full_sun_disk_rsun_arcsec(smap, Time(_OBS_TIME))
    assert rsun is not None
    expected = float(rsun) * _FULL_SUN_DISK_EXTENT_PAD / _AIA_PLATE_SCALE_ARCSEC
    assert half == pytest.approx(expected, rel=0.02)


def test_full_sun_viewport_side_exceeds_fov_on_aia_like_earth_map():
    """Regression lock: Earth filesystem view must not stay near the 665 arcsec FOV width."""
    smap = _aia_like_map()
    widget = _make_earth_widget(smap=smap)
    widget._projected_box_bbox_rect = Rectangle((1900.0, 2100.0), 1100.0, 1100.0, visible=False)
    widget._overlay_bbox_rect = None
    calls: list[tuple[float, float, float, float]] = []
    widget._set_view_window = lambda cx, cy, width, height: calls.append((cx, cy, width, height))

    widget._set_view_to_full_sun_disk()
    assert len(calls) == 1
    _cx, _cy, width, height = calls[0]
    assert width == pytest.approx(height)
    fov_width_px = _MODEL_FOV.width_arcsec / _AIA_PLATE_SCALE_ARCSEC
    assert width > fov_width_px * 1.5
    rsun = widget._full_sun_disk_rsun_arcsec(smap, Time(_OBS_TIME))
    expected_side = (
        2.0 * float(rsun) * _FULL_SUN_DISK_EXTENT_PAD / _AIA_PLATE_SCALE_ARCSEC * _FULL_SUN_VIEWPORT_PAD
    )
    assert width == pytest.approx(expected_side, rel=0.03)


def test_set_view_to_full_sun_disk_centers_on_disk_not_fov():
    obstime = Time(_OBS_TIME)
    earth = get_earth(obstime)
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=earth, obstime=obstime))
    header = make_fitswcs_header(
        np.zeros((512, 512), dtype=np.float32),
        center,
        scale=u.Quantity([4.0, 4.0], u.arcsec / u.pix),
    )
    smap = Map(np.zeros((512, 512), dtype=np.float32), header)

    widget = _make_earth_widget(smap=smap)
    widget._projected_box_bbox_rect = Rectangle((180.0, 200.0), 120.0, 120.0, visible=False)
    widget._overlay_bbox_rect = None
    calls: list[tuple[float, float, float, float]] = []
    widget._set_view_window = lambda cx, cy, width, height: calls.append((cx, cy, width, height))

    widget._set_view_to_full_sun_disk()
    assert len(calls) == 1
    cx, cy, width, height = calls[0]
    disk_px, disk_py = smap.wcs.world_to_pixel(center)
    disk_px = float(np.asarray(disk_px).ravel()[0])
    disk_py = float(np.asarray(disk_py).ravel()[0])
    assert cx == pytest.approx(disk_px, rel=0.02)
    assert cy == pytest.approx(disk_py, rel=0.02)
    assert width == pytest.approx(height)
    assert width > float(widget._projected_box_bbox_rect.get_width()) * 2.0


def test_set_view_to_full_sun_disk_unions_disk_and_overlay_rects():
    widget = _make_earth_widget()
    widget._current_map = object()
    widget._projected_box_bbox_rect = Rectangle((40.0, 40.0), 20.0, 20.0, visible=False)
    widget._overlay_bbox_rect = None
    calls: list[tuple[float, float, float, float]] = []
    widget._set_view_window = lambda cx, cy, width, height: calls.append((cx, cy, width, height))

    with patch.object(MapBoxDisplayWidget, "_display_disk_center_pixel", return_value=(0.0, 0.0)), patch.object(
        MapBoxDisplayWidget,
        "_full_sun_disk_half_extent_pixels",
        return_value=100.0,
    ):
        widget._set_view_to_full_sun_disk()

    assert len(calls) == 1
    cx, cy, width, height = calls[0]
    assert width == pytest.approx(height)
    assert cx == pytest.approx(0.0)
    assert cy == pytest.approx(0.0)
    assert width == pytest.approx(200.0 * _FULL_SUN_VIEWPORT_PAD)


def test_fov_selection_to_pixel_rect_skips_box_fallback_for_display_observer():
    smap = _aia_like_map()
    widget = _make_earth_widget(smap=smap)
    widget._projected_box_bbox_rect = Rectangle((1.0, 2.0), 50.0, 60.0, visible=False)
    fov = DisplayFovSelection(0.0, 0.0, 2000.0, 2000.0)

    with patch.object(type(smap.wcs), "world_to_pixel", side_effect=RuntimeError("fail")):
        rect = widget._fov_selection_to_pixel_rect(smap, fov, use_display_observer=True)

    assert rect is None


def test_set_view_to_projected_fov_uses_pixel_rect_after_plot():
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._current_axes = SimpleNamespace(
        get_xlim=lambda: (0.0, 500.0),
        get_ylim=lambda: (0.0, 500.0),
        set_xlim=lambda *_a, **_k: None,
        set_ylim=lambda *_a, **_k: None,
    )
    widget._canvas = SimpleNamespace(draw_idle=lambda: None)
    widget._projected_box_bbox_rect = Rectangle((100.0, 120.0), 80.0, 60.0, visible=False)
    widget._overlay_bbox_rect = None
    calls: list[tuple[float, float, float, float]] = []
    widget._set_view_window = lambda cx, cy, width, height: calls.append((cx, cy, width, height))

    widget._set_view_to_projected_fov(pad_factor=1.10)
    assert len(calls) == 1
    cx, cy, width, height = calls[0]
    assert cx == pytest.approx(140.0)
    assert cy == pytest.approx(150.0)
    assert width == pytest.approx(80.0 * 1.10)
    assert height == pytest.approx(60.0 * 1.10)


def test_refresh_plot_applies_full_sun_viewport_after_draw():
    widget = _make_earth_widget()
    widget._view_mode = "full_sun"
    widget._state.selected_context_id = "171"
    widget._state.selected_bottom_id = None
    widget._state.session_input = SimpleNamespace(
        time_iso=Time(_OBS_TIME).isot,
        map_ids=("171",),
    )
    mock_ax = SimpleNamespace(
        set_facecolor=lambda *args, **kwargs: None,
        set_box_aspect=lambda *args, **kwargs: None,
        set_aspect=lambda *args, **kwargs: None,
        get_xlim=lambda: (0.0, 4096.0),
        get_ylim=lambda: (0.0, 4096.0),
        set_title=lambda *args, **kwargs: None,
        set_xlim=lambda *args, **kwargs: None,
        set_ylim=lambda *args, **kwargs: None,
        axis=lambda *args, **kwargs: None,
        text=lambda *args, **kwargs: None,
    )
    widget._fig = SimpleNamespace(
        clear=lambda: None,
        add_subplot=lambda *args, **kwargs: mock_ax,
        text=lambda *args, **kwargs: None,
        subplots_adjust=lambda *args, **kwargs: None,
        bbox=SimpleNamespace(x0=0, y0=0, width=800, height=600),
        subplotpars=SimpleNamespace(left=0.12, right=0.98, bottom=0.12, top=0.93),
    )
    widget._clear_drag_preview_artists = lambda: None
    widget._overlay_rect = None
    widget._overlay_bbox_rect = None
    widget._projected_box_bbox_rect = None
    widget._projected_box_fov = None
    widget._overlay_center_artist = None
    widget._overlay_corner_artists = []
    widget._overlay_line_artists = []
    widget._zoom_anchor_px = None
    widget._full_view_limits = None
    widget._pending_launch_margin_fix = False
    calls: list[str] = []
    widget._set_view_to_full_sun_disk = lambda *_a, **_k: calls.append("full_sun")
    widget._set_view_to_projected_fov = lambda *_a, **_k: calls.append("box_fov")

    smap = _aia_like_map()
    smap.plot = lambda *args, **kwargs: None
    smap.draw_grid = lambda *args, **kwargs: None
    smap.draw_limb = lambda *args, **kwargs: None
    with patch.object(MapBoxDisplayWidget, "_context_canvas_map", return_value=smap), patch.object(
        MapBoxDisplayWidget, "_uses_geometry_scaffold_for_context", return_value=False
    ), patch.object(
        MapBoxDisplayWidget, "_selected_bottom_map", return_value=None
    ), patch.object(MapBoxDisplayWidget, "_plot_box_outline"), patch.object(
        MapBoxDisplayWidget, "_emit_observer_info"
    ), patch.object(
        MapBoxDisplayWidget, "_display_map_label", return_value="171"
    ), patch.object(
        MapBoxDisplayWidget, "_observer_label_for_key", return_value="Earth"
    ), patch.object(
        MapBoxDisplayWidget, "_auto_adjust_axes_margins", return_value=False
    ), patch.object(MapBoxDisplayWidget, "_render_fieldlines"), patch.object(
        MapBoxDisplayWidget, "_update_cursor_for_mode"
    ), patch.object(MapBoxDisplayWidget, "_should_plot_bottom_overlay", return_value=False), patch.object(
        MapBoxDisplayWidget, "_canonical_map_key", side_effect=lambda map_id, **kwargs: map_id
    ):
        widget._refresh_plot()

    assert calls == ["full_sun"]


def test_prepare_context_map_full_sun_embedded_requests_disk_reproject():
    obstime = Time(_OBS_TIME)
    earth = get_earth(obstime)
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=earth, obstime=obstime))
    header = make_fitswcs_header(
        np.zeros((64, 64), dtype=np.float32),
        center,
        scale=u.Quantity([12.0, 12.0], u.arcsec / u.pix),
    )
    header["PYEMBED"] = True
    header["PYALIGN"] = True
    smap = Map(np.zeros((64, 64), dtype=np.float32), header)

    widget = _make_earth_widget()
    widget._view_mode = "full_sun"
    widget._record_prepare_event = lambda _msg: None
    widget._apply_display_scaling = lambda *_a, **_k: None
    widget._embedded_context_needs_display_crop = lambda _s: False
    widget._is_embedded_native_spacecraft_map = lambda _s: False

    captured: dict[str, object] = {}

    def _fake_reproject(smap_in, *, fov_override=None):
        captured["fov_override"] = fov_override
        return smap_in, fov_override

    disk_fov = DisplayFovSelection(0.0, 0.0, 2000.0, 2000.0)
    with patch.object(MapBoxDisplayWidget, "_is_embedded_pyalign_map", return_value=True), patch.object(
        MapBoxDisplayWidget,
        "_full_sun_disk_hpc_fov",
        return_value=disk_fov,
    ), patch.object(
        MapBoxDisplayWidget,
        "_fov_selection_projected_to_display_observer",
        return_value=disk_fov,
    ), patch.object(MapBoxDisplayWidget, "_reproject_map_for_display_observer", side_effect=_fake_reproject):
        widget._prepare_context_map("171", smap, prepare_variant="fov_crop")

    assert captured.get("fov_override") is disk_fov


def test_obstime_for_map_uses_model_fallback_when_date_obs_missing():
    from astropy.io import fits
    from pyampp.gxbox.boxutils import map_from_data_header_compat

    model_time = "2026-04-03T19:34:37.800"
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 4
    header["NAXIS2"] = 4
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CRPIX1"] = 2.5
    header["CRPIX2"] = 2.5
    header["CRVAL1"] = 0
    header["CRVAL2"] = 0
    header["CDELT1"] = 1
    header["CDELT2"] = 1
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    smap = map_from_data_header_compat(np.zeros((4, 4), dtype=np.float32), header)

    resolved = MapBoxDisplayWidget._obstime_for_map(smap, model_time)
    assert resolved is not None
    assert resolved.isot.startswith("2026-04-03T19:34:37")


def test_ensure_embedded_header_obstime_injects_model_time():
    from astropy.io import fits

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        session_input=SimpleNamespace(time_iso="2026-04-03T19:34:37.800"),
    )
    header = fits.Header()
    header["NAXIS"] = 2
    MapBoxDisplayWidget._ensure_embedded_header_obstime(widget, header)
    assert header["DATE-OBS"] == "2026-04-03T19:34:37.800"


def test_copy_observer_cards_preserves_embedded_date_obs():
    model_time = "2026-04-03T19:34:37.800"
    ref_time = "2026-06-28T03:44:26.967"
    obstime = Time(model_time)
    earth = get_earth(obstime)
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=earth, obstime=obstime))
    embedded_header = make_fitswcs_header(
        np.zeros((8, 8), dtype=np.float32),
        center,
        scale=u.Quantity([12.0, 12.0], u.arcsec / u.pix),
    )
    embedded_header["DATE-OBS"] = model_time
    embedded_header["PYALIGN"] = True

    ref_obstime = Time(ref_time)
    ref_earth = get_earth(ref_obstime)
    ref_center = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=Helioprojective(observer=ref_earth, obstime=ref_obstime),
    )
    ref_header = make_fitswcs_header(
        np.zeros((8, 8), dtype=np.float32),
        ref_center,
        scale=u.Quantity([12.0, 12.0], u.arcsec / u.pix),
    )
    ref_header["DATE-OBS"] = ref_time
    ref_map = Map(np.zeros((8, 8), dtype=np.float32), ref_header)

    MapBoxDisplayWidget._copy_observer_cards_from_map(embedded_header, ref_map)

    assert embedded_header["DATE-OBS"] == model_time


def test_format_time_delta_short_uses_compact_units():
    assert MapBoxDisplayWidget._format_time_delta_short(0.0) == "Δt=0"
    assert MapBoxDisplayWidget._format_time_delta_short(12.0) == "Δt=+12s"
    assert MapBoxDisplayWidget._format_time_delta_short(-45.0) == "Δt=-45s"
    assert MapBoxDisplayWidget._format_time_delta_short(138.0) == "Δt=+2.3min"
    assert MapBoxDisplayWidget._format_time_delta_short(8280.0) == "Δt=+2.3h"


def test_format_display_time_banner_includes_obs_and_delta():
    model_time = "2026-04-03T17:16:37.800"
    ref_time = "2026-04-03T19:46:37.800"
    banner = MapBoxDisplayWidget._format_display_time_banner(ref_time, model_time)
    assert banner.startswith("OBS 2026-04-03T19:46:37.800")
    assert "(Δt=+2.5h)" in banner


def test_normalize_observer_key_maps_sdo_to_earth():
    assert MapBoxDisplayWidget._normalize_observer_key("sdo") == "earth"
    assert MapBoxDisplayWidget._normalize_observer_key("SDO/AIA") == "earth"


def test_display_observer_options_exclude_sdo():
    from pyampp.gxbox import box_view2d as bv2d

    option_keys = {key for key, _label in bv2d._DISPLAY_OBSERVER_OPTIONS}
    assert "sdo" not in option_keys
    assert "earth" in option_keys
