"""Tests for geometry-driven reference-map cropping."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from sunpy.coordinates import Heliocentric, Helioprojective, get_earth
from sunpy.map import Map, make_fitswcs_header

from pyampp.gxbox.box import Box
from pyampp.gxbox.boxutils import map_from_data_header_compat
from pyampp.io.refmap_crop import (
    apply_pad_factor_to_fov,
    composite_map_onto_canvas,
    compute_crop_fov_for_observer,
    compute_inscribing_fov_box_for_observer,
    crop_fov_at_model_times,
    crop_fov_xy_from_inscribing_box,
    crop_refmap_to_model_box,
    crop_refmap_to_model_box_after_pangle_rotation,
    display_observer_reproject_header_for_fov,
    full_disk_fov_for_map,
    make_empty_observer_fov_map,
    plot_inscribing_fov_box_on_axes,
    project_fov_between_observers,
    project_inscribing_xy_to_observer,
    reproject_map_to_target_observer_fov,
)
from pyampp.io.refmap_crop_plots import plot_refmap_crop_diagnostics


def _make_model_box() -> tuple[Box, Time, object]:
    obs_time = Time("2026-04-03T19:46:37.800")
    observer = get_earth(obs_time)
    frame_obs = Helioprojective(observer=observer, obstime=obs_time)
    box_origin = SkyCoord(Tx=0 * u.arcsec, Ty=0 * u.arcsec, distance=observer.radius, frame=frame_obs)
    frame_hcc = Heliocentric(observer=observer, obstime=obs_time)
    box_center = box_origin.transform_to(frame_hcc)
    box_center = SkyCoord(x=box_center.x, y=box_center.y, z=box_center.z + 20 * u.Mm, frame=box_center.frame)
    box = Box(
        frame_obs,
        box_origin,
        box_center,
        np.array([8, 6, 4]) * u.pix,
        np.array([5.0, 5.0, 10.0]) * u.Mm,
    )
    return box, obs_time, observer


def _earth_aia_map(*, size: int = 256, scale: float = 4.0, date_obs: str) -> Map:
    obstime = Time(date_obs)
    earth = get_earth(obstime)
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=earth, obstime=obstime))
    header = make_fitswcs_header(
        np.ones((size, size), dtype=np.float32),
        center,
        scale=u.Quantity([scale, scale], u.arcsec / u.pix),
        instrument="AIA",
        observatory="SDO",
    )
    header["TELESCOP"] = "SDO/AIA"
    header["INSTRUME"] = "AIA_3"
    header["WAVELNTH"] = 171
    header["WAVEUNIT"] = "angstrom"
    return Map(np.ones((size, size), dtype=np.float32), header)


def _stereo_map(*, size: int = 256, scale: float = 4.0, date_obs: str) -> Map:
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
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = scale
    header["CDELT2"] = scale
    header["DATE-OBS"] = date_obs
    header["HGLN_OBS"] = 54.3710463167
    header["HGLT_OBS"] = -0.875478602937
    header["DSUN_OBS"] = 1.496e11
    header["WAVELNTH"] = 171
    header["WAVEUNIT"] = "angstrom"
    return map_from_data_header_compat(np.ones((size, size), dtype=np.float32), header)


def test_composite_map_onto_canvas_preserves_sunpy_plot_settings():
    smap = _stereo_map(date_obs="2026-04-03T19:58:30.006")
    smap.plot_settings["cmap"] = "euvi284"
    canvas = make_empty_observer_fov_map(
        smap,
        observer=smap.coordinate_frame.observer,
        obstime=smap.date,
        fov=full_disk_fov_for_map(smap),
    )
    composite = composite_map_onto_canvas(canvas, smap)
    assert composite.plot_settings.get("cmap") == "euvi284"


def test_apply_pad_factor_scales_about_center():
    fov = {
        "xc_arcsec": 10.0,
        "yc_arcsec": -5.0,
        "xsize_arcsec": 100.0,
        "ysize_arcsec": 80.0,
    }
    out = apply_pad_factor_to_fov(fov, 1.1)
    assert out["xsize_arcsec"] == pytest.approx(110.0)
    assert out["ysize_arcsec"] == pytest.approx(88.0)
    assert out["xmin_arcsec"] == pytest.approx(10.0 - 55.0)
    assert out["ymax_arcsec"] == pytest.approx(-5.0 + 44.0)


def test_crop_refmap_to_model_box_reduces_earth_aia_extent():
    box, model_time, earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    smap = _earth_aia_map(date_obs=model_time.isot)
    result = crop_refmap_to_model_box(smap, world, pad=1.1, model_obstime=model_time)
    assert tuple(result.cropped_map.data.shape) != (256, 256)
    assert result.cropped_map.data.shape[0] >= 8
    assert result.cropped_map.data.shape[1] >= 8
    assert result.cropped_map.meta.get("TELESCOP") == "SDO/AIA"
    assert result.cropped_map.meta.get("PYALIGN") is False
    ny, nx = result.cropped_map.data.shape
    assert float(result.cropped_map.meta["CRPIX1"]) == pytest.approx(nx / 2.0 + 0.5)
    assert float(result.cropped_map.meta["CRPIX2"]) == pytest.approx(ny / 2.0 + 0.5)
    center = result.cropped_map.pixel_to_world(
        ((nx - 1) / 2.0) * u.pix,
        ((ny - 1) / 2.0) * u.pix,
    )
    assert float(result.cropped_map.meta["CRVAL1"]) == pytest.approx(
        float(center.Tx.to_value(u.arcsec)),
        abs=0.05,
    )
    assert float(result.cropped_map.meta["CRVAL2"]) == pytest.approx(
        float(center.Ty.to_value(u.arcsec)),
        abs=0.05,
    )
    assert float(result.cropped_map.meta["XCEN"]) == pytest.approx(
        float(result.cropped_map.meta["CRVAL1"]),
        abs=0.05,
    )
    assert float(result.cropped_map.meta["YCEN"]) == pytest.approx(
        float(result.cropped_map.meta["CRVAL2"]),
        abs=0.05,
    )


def test_earth_map_fov_matches_across_frames_when_times_equal():
    box, model_time, earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    smap = _earth_aia_map(date_obs=model_time.isot)
    result = crop_refmap_to_model_box(smap, world, pad=1.1, model_obstime=model_time)
    map_los, earth_los = crop_fov_at_model_times(
        world,
        map_observer=result.map_observer,
        model_obstime=model_time,
        pad=1.1,
    )
    assert map_los is not None and earth_los is not None
    for key in ("xc_arcsec", "yc_arcsec", "xsize_arcsec", "ysize_arcsec"):
        assert float(map_los[key]) == pytest.approx(float(earth_los[key]), rel=0.02, abs=2.0)
        assert float(result.crop_fov[key]) == pytest.approx(float(map_los[key]), rel=0.02, abs=2.0)


def test_stereo_crop_fov_differs_between_map_los_and_earth_at_model_time():
    box, model_time, _earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    map_time = Time("2026-04-03T19:58:30.006")
    smap = _stereo_map(date_obs=map_time.isot)
    result = crop_refmap_to_model_box(smap, world, pad=1.1, model_obstime=model_time)
    map_los, earth_los = crop_fov_at_model_times(
        world,
        map_observer=result.map_observer,
        model_obstime=model_time,
        pad=1.1,
    )
    assert map_los is not None and earth_los is not None
    center_sep = np.hypot(
        float(map_los["xc_arcsec"]) - float(earth_los["xc_arcsec"]),
        float(map_los["yc_arcsec"]) - float(earth_los["yc_arcsec"]),
    )
    assert center_sep > 10.0
    assert abs(float(result.cropped_map.meta["CRVAL1"])) == pytest.approx(
        abs(float(result.crop_fov["xc_arcsec"])),
        rel=0.05,
        abs=20.0,
    )
    assert float(result.cropped_map.meta["XCEN"]) == pytest.approx(
        float(result.cropped_map.meta["CRVAL1"]),
        abs=0.05,
    )


def test_inscribing_fov_box_is_smaller_than_crop_fov_rectangle():
    box, model_time, earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    pad = 1.1
    inscribing_box = compute_inscribing_fov_box_for_observer(
        world,
        observer=earth,
        obstime=model_time,
    )
    crop_fov = crop_fov_xy_from_inscribing_box(inscribing_box, pad)
    assert inscribing_box is not None and crop_fov is not None
    assert float(crop_fov["xsize_arcsec"]) == pytest.approx(
        float(inscribing_box["xsize_arcsec"]) * pad,
        rel=0.02,
    )
    assert float(crop_fov["ysize_arcsec"]) == pytest.approx(
        float(inscribing_box["ysize_arcsec"]) * pad,
        rel=0.02,
    )


def test_crop_after_pangle_rotation_reduces_map_extent():
    box, model_time, _earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    smap = _earth_aia_map(date_obs=model_time.isot)
    result = crop_refmap_to_model_box_after_pangle_rotation(
        smap,
        world,
        pad=1.1,
        model_obstime=model_time,
    )
    assert tuple(result.cropped_map.data.shape) != tuple(smap.data.shape)
    assert result.cropped_map.data.shape[0] >= 8
    assert result.cropped_map.data.shape[1] >= 8


def test_plot_inscribing_fov_box_on_axes_uses_plot_coord(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    box, model_time, earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    smap = _earth_aia_map(date_obs=model_time.isot)
    fov_box = compute_inscribing_fov_box_for_observer(
        world,
        observer=earth,
        obstime=model_time,
    )
    assert fov_box is not None

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=smap)
    smap.plot(axes=ax, annotate=False)
    artists = plot_inscribing_fov_box_on_axes(
        ax,
        fov_box,
        observer=earth,
        obstime=model_time,
    )
    assert artists
    fig.savefig(tmp_path / "inscribing_box_overlay.png")
    plt.close(fig)


def test_plot_refmap_crop_diagnostics_runs_for_earth_and_stereo(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    box, model_time, _earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None

    earth_map = _earth_aia_map(date_obs=model_time.isot)
    fig, _result = plot_refmap_crop_diagnostics(
        earth_map,
        world,
        pad=1.1,
        model_obstime=model_time,
        save_path=tmp_path / "earth_crop_diag.png",
    )
    assert (tmp_path / "earth_crop_diag.png").exists()
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close(fig)

    stereo_map = _stereo_map(date_obs="2026-04-03T19:58:30.006")
    fig2, _result2 = plot_refmap_crop_diagnostics(
        stereo_map,
        world,
        pad=1.1,
        model_obstime=model_time,
        save_path=tmp_path / "stereo_crop_diag.png",
    )
    assert (tmp_path / "stereo_crop_diag.png").exists()
    plt.close(fig2)


def test_compute_crop_fov_uses_map_time_not_model_time_for_stereo():
    box, model_time, _earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    map_time = Time("2026-04-03T07:46:37.800")
    smap = _stereo_map(date_obs=map_time.isot)
    fov_map_time = compute_crop_fov_for_observer(
        world,
        observer=smap.observer_coordinate,
        obstime=map_time,
        pad=1.1,
    )
    fov_wrong_time = compute_crop_fov_for_observer(
        world,
        observer=smap.observer_coordinate,
        obstime=model_time,
        pad=1.1,
    )
    assert fov_map_time is not None and fov_wrong_time is not None
    assert abs(float(fov_map_time["xc_arcsec"]) - float(fov_wrong_time["xc_arcsec"])) > 0.2


def test_reproject_map_to_display_observer_fov_uses_roi_header():
    box, model_time, earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    map_time = Time("2026-04-03T19:58:30.006")
    smap = _stereo_map(date_obs=map_time.isot)
    result = crop_refmap_to_model_box_after_pangle_rotation(
        smap,
        world,
        pad=1.1,
        model_obstime=model_time,
    )
    projected = project_fov_between_observers(
        result.crop_fov,
        source_observer=result.map_observer,
        source_obstime=result.map_obstime,
        target_observer=earth,
        target_obstime=model_time,
    )
    assert projected is not None
    header = display_observer_reproject_header_for_fov(
        result.cropped_map,
        observer=earth,
        obstime=model_time,
        fov=projected,
    )
    assert header is not None
    inscribing = compute_inscribing_fov_box_for_observer(
        world,
        observer=result.map_observer,
        obstime=result.map_obstime,
    )
    assert inscribing is not None
    earth_crop_fov = project_fov_between_observers(
        result.crop_fov,
        source_observer=result.map_observer,
        source_obstime=result.map_obstime,
        target_observer=earth,
        target_obstime=result.map_obstime,
    )
    assert earth_crop_fov is not None
    reprojected = reproject_map_to_target_observer_fov(
        result.cropped_map,
        target_fov=earth_crop_fov,
        target_observer=earth,
        target_obstime=result.map_obstime,
    )
    assert tuple(reprojected.data.shape) != tuple(result.cropped_map.data.shape)
    assert reprojected.data.shape[0] >= 8
    assert reprojected.data.shape[1] >= 8
