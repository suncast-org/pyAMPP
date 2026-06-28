"""Regression tests for PYALIGN embedded refmap cross-observer reprojection.

Earth-view embedded refmaps (``PYALIGN=True``) are cropped to ~1.1x model FOV when
saved to H5. In spacecraft display views the context overlay must keep that margin
relative to the blue FOV box. That requires:

1. Skip display-time pixel crop for PYALIGN maps (already cropped at embed).
2. Project the 1.1x FOV from ``fov_definition_observer_key`` into display HPC.
3. Reproject with an ROI header (``[roi]``), not a full-disk header (``[full]``).

Do not remove or bypass these steps without updating this module and verifying the
STEREO-A / Earth visual parity manually.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import HeliographicStonyhurst, get_earth

from pyampp.gxbox.box_view2d import (
    MapBoxDisplayWidget,
    _CONTEXT_PREPARE_VARIANT_FOV_CROP,
    _EMBEDDED_REFMAP_FOV_PAD_FACTOR,
    _EMBEDDED_REFMAP_FLAG,
)
from pyampp.gxbox.boxutils import map_from_data_header_compat
from pyampp.gxbox.selector_api import DisplayFovSelection

_OBS_TIME = "2026-04-03T19:46:37.800"
_MODEL_FOV = DisplayFovSelection(-0.0, 225.83, 665.20, 665.20)
# Fixed STEREO-A ephemeris (matches selector regression FITS headers; no Horizons).
_STEREO_A_HGLN_DEG = 54.3710463167
_STEREO_A_HGLT_DEG = -0.875478602937
_STEREO_A_DSUN_CM = 1.496e11


def _stereo_a_observer(when: Time) -> SkyCoord:
    return SkyCoord(
        lon=_STEREO_A_HGLN_DEG * u.deg,
        lat=_STEREO_A_HGLT_DEG * u.deg,
        radius=_STEREO_A_DSUN_CM * u.cm,
        frame=HeliographicStonyhurst(obstime=when),
    )


def _earth_stereo_observer_context(observer_key: str, obstime):
    when = Time(obstime)
    if observer_key == "earth":
        return SimpleNamespace(observer_coordinate=get_earth(when), date=when)
    if observer_key == "stereo-a":
        return SimpleNamespace(observer_coordinate=_stereo_a_observer(when), date=when)
    return None


def _padded_model_fov() -> DisplayFovSelection:
    pad = _EMBEDDED_REFMAP_FOV_PAD_FACTOR
    return DisplayFovSelection(
        center_x_arcsec=_MODEL_FOV.center_x_arcsec,
        center_y_arcsec=_MODEL_FOV.center_y_arcsec,
        width_arcsec=_MODEL_FOV.width_arcsec * pad,
        height_arcsec=_MODEL_FOV.height_arcsec * pad,
    )


def _make_stereo_earth_widget() -> MapBoxDisplayWidget:
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        fov_definition_observer_key="earth",
        session_input=SimpleNamespace(time_iso=_OBS_TIME),
        fov=_MODEL_FOV,
    )
    widget._observer_coord_cache = {}
    widget._observer_metadata_cache = {}
    widget._observer_warning_cache = set()
    widget._last_status_text = ""
    widget._status_callback = None
    widget._record_prepare_event = lambda _msg: None
    widget._observer_source_b3d = lambda: {}
    return widget


def _make_pyalign_embedded_smap():
    header = fits.Header()
    header["TELESCOP"] = "EOVSA"
    header["PYEMBED"] = True
    header["PYALIGN"] = True
    header["NAXIS"] = 2
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 32.5
    header["CRPIX2"] = 32.5
    header["CRVAL1"] = float(_MODEL_FOV.center_x_arcsec)
    header["CRVAL2"] = float(_MODEL_FOV.center_y_arcsec)
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    header["DATE-OBS"] = _OBS_TIME
    smap = map_from_data_header_compat(np.ones((64, 64), dtype=np.float32), header)
    smap.meta[_EMBEDDED_REFMAP_FLAG] = True
    smap.meta["PYALIGN"] = True
    return smap


def test_embedded_refmap_fov_pad_factor_is_locked():
    assert _EMBEDDED_REFMAP_FOV_PAD_FACTOR == pytest.approx(1.10)


def test_projected_padded_fov_is_wider_than_one_x_fov_in_stereo_frame():
    widget = _make_stereo_earth_widget()
    padded = _padded_model_fov()
    with patch.object(
        MapBoxDisplayWidget,
        "_observer_context",
        side_effect=_earth_stereo_observer_context,
    ), patch.object(MapBoxDisplayWidget, "_observers_share_los", return_value=False):
        projected_pad = widget._fov_selection_projected_to_display_observer(padded, _OBS_TIME)
        projected_1x = widget._fov_selection_projected_to_display_observer(_MODEL_FOV, _OBS_TIME)

    assert projected_pad is not None
    assert projected_1x is not None
    assert projected_pad.width_arcsec > projected_1x.width_arcsec
    assert projected_pad.height_arcsec > projected_1x.height_arcsec
    # AABB after corner projection is not exactly 1.1x, but must keep visible margin.
    assert projected_pad.width_arcsec >= projected_1x.width_arcsec * 1.05
    assert projected_pad.height_arcsec >= projected_1x.height_arcsec * 1.05


def test_projected_fov_differs_from_source_frame_for_cross_observer():
    widget = _make_stereo_earth_widget()
    padded = _padded_model_fov()
    with patch.object(
        MapBoxDisplayWidget,
        "_observer_context",
        side_effect=_earth_stereo_observer_context,
    ), patch.object(MapBoxDisplayWidget, "_observers_share_los", return_value=False):
        projected = widget._fov_selection_projected_to_display_observer(padded, _OBS_TIME)

    assert projected is not None
    assert projected.center_x_arcsec != pytest.approx(padded.center_x_arcsec, abs=1.0)
    assert projected.center_y_arcsec != pytest.approx(padded.center_y_arcsec, abs=1.0)


def test_prepare_context_map_pyalign_uses_projected_fov_for_stereo_reproject():
    widget = _make_stereo_earth_widget()
    smap = SimpleNamespace(data=np.zeros((12, 12)), meta={"PYEMBED": True, "PYALIGN": True})
    projected_fov = DisplayFovSelection(-900.0, 120.0, 880.0, 880.0)
    rotated = SimpleNamespace(data=np.zeros((12, 12)))

    with patch.object(MapBoxDisplayWidget, "_build_native_crop", return_value=(smap, None)) as native_crop_mock, patch.object(MapBoxDisplayWidget, "_submap_to_fov_selection_pixels") as crop_mock, patch.object(
        MapBoxDisplayWidget,
        "_embedded_context_crop_fov",
        return_value=_padded_model_fov(),
    ), patch.object(
        MapBoxDisplayWidget,
        "_fov_selection_projected_to_display_observer",
        return_value=projected_fov,
    ) as project_mock, patch.object(
        MapBoxDisplayWidget,
        "_reproject_map_for_display_observer",
        return_value=(rotated, projected_fov),
    ) as reproj_mock, patch.object(
        MapBoxDisplayWidget, "_apply_display_scaling", side_effect=lambda m, _k: m
    ), patch.object(MapBoxDisplayWidget, "_is_non_earth_display_observer", return_value=True):
        out, coverage = widget._prepare_context_map(
            "EOVSA_f1.418GHz",
            smap,
            prepare_variant=_CONTEXT_PREPARE_VARIANT_FOV_CROP,
        )

    assert out is rotated
    native_crop_mock.assert_not_called()
    crop_mock.assert_not_called()
    project_mock.assert_called_once()
    reproj_mock.assert_called_once()
    assert reproj_mock.call_args.kwargs["fov_override"] is projected_fov
    assert coverage is projected_fov


def test_reproject_with_projected_fov_uses_roi_not_full_disk():
    widget = _make_stereo_earth_widget()
    smap = _make_pyalign_embedded_smap()
    projected_fov = DisplayFovSelection(-900.0, 120.0, 880.0, 880.0)
    events: list[str] = []
    widget._record_prepare_event = events.append
    widget._view_mode = "box_fov"

    with patch.object(
        MapBoxDisplayWidget,
        "_resolve_display_observer_coord",
        wraps=widget._resolve_display_observer_coord,
    ), patch.object(
        MapBoxDisplayWidget,
        "_display_observer_reproject_header_for_selection",
        wraps=widget._display_observer_reproject_header_for_selection,
    ) as roi_mock, patch.object(
        MapBoxDisplayWidget,
        "_solar_disk_center_for_observer",
    ) as disk_center_mock:
        widget._reproject_map_for_display_observer(smap, fov_override=projected_fov)

    roi_mock.assert_called_once()
    assert roi_mock.call_args.args[3] is projected_fov
    disk_center_mock.assert_not_called()
    assert any("[roi]" in event for event in events)
    assert not any("[full]" in event for event in events)


def test_reproject_without_fov_override_falls_back_to_full_disk_for_pyalign_shape():
    """Document the regression: omitting fov_override uses [full] and shrinks context."""
    widget = _make_stereo_earth_widget()
    smap = _make_pyalign_embedded_smap()
    events: list[str] = []
    widget._record_prepare_event = events.append
    widget._view_mode = "box_fov"

    with patch.object(
        MapBoxDisplayWidget,
        "_resolve_display_observer_coord",
        wraps=widget._resolve_display_observer_coord,
    ):
        widget._reproject_map_for_display_observer(smap, fov_override=None)

    assert any("[full]" in event for event in events)
    assert not any("[roi]" in event for event in events)


def test_native_stereo_embedded_still_crops_in_native_space_before_reproject():
    """PYALIGN=False native spacecraft maps crop in map-native LOS before display reproj."""
    header = fits.Header()
    header["TELESCOP"] = "STEREO"
    header["INSTRUME"] = "SECCHI"
    header["PYALIGN"] = False
    header["NAXIS"] = 2
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 32.5
    header["CRPIX2"] = 32.5
    header["CRVAL1"] = -1020.0
    header["CRVAL2"] = 110.0
    header["CDELT1"] = 12.0
    header["CDELT2"] = 12.0
    header["DATE-OBS"] = "2026-04-03T19:58:30.006"
    header["HGLN_OBS"] = 54.3710463167
    header["HGLT_OBS"] = -0.875478602937
    header["DSUN_OBS"] = 1.496e11
    smap = map_from_data_header_compat(np.ones((64, 64), dtype=np.float32), header)
    smap.meta[_EMBEDDED_REFMAP_FLAG] = True

    widget = _make_stereo_earth_widget()
    rotated = SimpleNamespace(data=np.zeros((12, 12)))
    cropped = SimpleNamespace(data=np.zeros((8, 8)))

    with patch.object(MapBoxDisplayWidget, "_build_native_crop", return_value=(cropped, DisplayFovSelection(-1.0, 2.0, 3.0, 4.0))) as native_crop_mock, patch.object(
        MapBoxDisplayWidget,
        "_fov_selection_projected_to_display_observer",
        return_value=DisplayFovSelection(-900.0, 120.0, 880.0, 880.0),
    ) as project_mock, patch.object(
        MapBoxDisplayWidget, "_reproject_map_for_display_observer", return_value=(rotated, None)
    ) as reproj_mock, patch.object(MapBoxDisplayWidget, "_apply_display_scaling", side_effect=lambda m, _k: m):
        out, _coverage = widget._prepare_context_map(
            "stereo171",
            smap,
            prepare_variant=_CONTEXT_PREPARE_VARIANT_FOV_CROP,
        )

    assert out is rotated
    native_crop_mock.assert_called_once()
    reproj_mock.assert_called_once()
    assert reproj_mock.call_args.kwargs["fov_override"] is None
    project_mock.assert_not_called()
