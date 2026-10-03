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
    crop_refmap_spatial,
    crop_refmap_to_model_box,
    crop_refmap_to_model_box_after_pangle_rotation,
    display_observer_reproject_header_for_fov,
    rotate_refmap_for_display,
    full_disk_fov_for_map,
    make_empty_observer_fov_map,
    plot_inscribing_fov_box_on_axes,
    project_fov_between_observers,
    project_inscribing_xy_to_observer,
    reproject_map_to_target_observer_fov,
)
from pyampp.io.refmap_crop_plots import (
    _compute_viewport,
    _finer_plate_scale_arcsec_per_pix,
    _rsun_arcsec_from_map,
    _viewport_sun_centered,
    plot_refmap_crop_diagnostics,
)


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


def _earth_aia_map(
    *,
    size: int = 256,
    scale: float | tuple[float, float] = 4.0,
    date_obs: str,
) -> Map:
    obstime = Time(date_obs)
    earth = get_earth(obstime)
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=earth, obstime=obstime))
    if isinstance(scale, (tuple, list)):
        sx, sy = float(scale[0]), float(scale[1])
    else:
        sx = sy = float(scale)
    header = make_fitswcs_header(
        np.ones((size, size), dtype=np.float32),
        center,
        scale=u.Quantity([sx, sy], u.arcsec / u.pix),
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


def test_plot_refmap_crop_diagnostics_unlinks_owned_temp_fits(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib
    import tempfile
    from pathlib import Path

    matplotlib.use("Agg")

    owned: list[Path] = []
    real_named_temporary_file = tempfile.NamedTemporaryFile

    def _named_temporary_file(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["dir"] = str(tmp_path)
        handle = real_named_temporary_file(*args, **kwargs)
        owned.append(Path(handle.name))
        return handle

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _named_temporary_file)

    box, model_time, _earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    earth_map = _earth_aia_map(date_obs=model_time.isot)
    fig, _result = plot_refmap_crop_diagnostics(
        earth_map,
        world,
        pad=1.1,
        model_obstime=model_time,
        save_path=tmp_path / "owned_temp_diag.png",
    )
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close(fig)

    assert owned
    assert all(not path.exists() for path in owned)


def test_plot_refmap_crop_diagnostics_keeps_caller_fits(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    box, model_time, _earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    earth_map = _earth_aia_map(date_obs=model_time.isot)
    caller_fits = tmp_path / "caller_crop.fits"
    fig, _result = plot_refmap_crop_diagnostics(
        earth_map,
        world,
        pad=1.1,
        model_obstime=model_time,
        crop_fits_path=caller_fits,
        save_path=tmp_path / "caller_fits_diag.png",
    )
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close(fig)
    assert caller_fits.exists()


def test_finer_plate_scale_selects_smaller_axis():
    smap = _earth_aia_map(date_obs="2026-04-03T19:46:37.800", scale=(2.0, 5.0))
    assert _finer_plate_scale_arcsec_per_pix(smap) == pytest.approx(2.0)


def test_viewport_sun_centered_covers_disk_on_anisotropic_scale():
    smap = _earth_aia_map(date_obs="2026-04-03T19:46:37.800", scale=(2.0, 5.0), size=2048)
    rsun = _rsun_arcsec_from_map(smap) or 960.0
    disk_pad = 1.05
    half_arcsec = rsun * disk_pad
    (x0, x1), (y0, y1) = _viewport_sun_centered(smap, disk_pad=disk_pad)
    half_pix = 0.5 * abs(x1 - x0)
    # Finer axis is 2"/pix: using coarser 5"/pix would under-size by 2.5×.
    assert half_pix == pytest.approx(half_arcsec / 2.0, rel=1e-6)
    assert 0.5 * abs(y1 - y0) == pytest.approx(half_pix, rel=1e-6)


def test_compute_viewport_covers_disk_on_anisotropic_scale():
    smap = _earth_aia_map(date_obs="2026-04-03T19:46:37.800", scale=(2.0, 5.0), size=2048)
    rsun = _rsun_arcsec_from_map(smap) or 960.0
    disk_pad = 1.05
    half_arcsec = rsun * disk_pad
    box, model_time, earth = _make_model_box()
    world = box.model_box_corners_world()
    assert world is not None
    scene = {
        "observer": earth,
        "obstime": model_time,
        "box_corners_world": world,
    }
    (x0, x1), (y0, y1) = _compute_viewport(smap, scene=scene, disk_pad=disk_pad)
    half_pix = 0.5 * abs(x1 - x0)
    # Disk half-extent (finer scale) is a lower bound; overlays may expand further.
    assert half_pix >= half_arcsec / 2.0 - 1e-6
    assert half_pix > half_arcsec / 5.0
    assert 0.5 * abs(y1 - y0) == pytest.approx(half_pix, rel=1e-6)


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


def test_cross_observer_limb_mask_keeps_ondisk_earth_fov():
    """STEREO-A pixels that land on the Earth disk must survive the limb mask.

    A spherical screen centered on the source spacecraft and applied to on-disk
    coordinates maps those pixels to Tx/Ty of order 1e5 arcsec, so the mask
    blanks the whole Earth FOV.
    """
    date_obs = "2012-07-12T04:46:15.006"
    smap = _stereo_map(size=64, scale=40.0, date_obs=date_obs)
    earth = get_earth(Time(date_obs))
    fov = {
        "xc_arcsec": 0.0,
        "yc_arcsec": 0.0,
        "xsize_arcsec": 400.0,
        "ysize_arcsec": 400.0,
    }
    masked = reproject_map_to_target_observer_fov(
        smap,
        target_fov=fov,
        target_observer=earth,
        target_obstime=smap.date,
        mask_off_limb=True,
        algorithm="interpolation",
    )
    finite = np.isfinite(np.asarray(masked.data, dtype=float))
    assert float(finite.mean()) > 0.5


def test_reproject_map_to_target_observer_fov_masks_off_limb_pixels():
    """mask_off_limb must strictly reduce finite pixels vs the unmasked reproject."""
    date_obs = "2012-07-12T04:46:15.006"
    smap = _stereo_map(size=64, scale=40.0, date_obs=date_obs)
    earth = get_earth(Time(date_obs))
    # Large Earth FOV so some reprojected pixels land off-limb in the STEREO frame.
    fov = {
        "xc_arcsec": 0.0,
        "yc_arcsec": 0.0,
        "xsize_arcsec": 2200.0,
        "ysize_arcsec": 2200.0,
    }
    unmasked = reproject_map_to_target_observer_fov(
        smap,
        target_fov=fov,
        target_observer=earth,
        target_obstime=smap.date,
        mask_off_limb=False,
        algorithm="interpolation",
    )
    masked = reproject_map_to_target_observer_fov(
        smap,
        target_fov=fov,
        target_observer=earth,
        target_obstime=smap.date,
        mask_off_limb=True,
        algorithm="interpolation",
    )
    n_unmasked = int(np.sum(np.isfinite(unmasked.data)))
    n_masked = int(np.sum(np.isfinite(masked.data)))
    assert n_unmasked > 0
    assert n_masked < n_unmasked
    assert np.any(~np.isfinite(masked.data))
    # On-disk Earth pixels must still survive (companion regression).
    assert float(np.mean(np.isfinite(masked.data))) > 0.3


def test_rotate_refmap_for_display_propagates_rotation_failure(monkeypatch):
    """Failed rotate must not be swallowed — callers clear CROTA after success."""
    smap = _earth_aia_map(date_obs="2026-04-03T19:46:37.800")
    smap.meta["CROTA2"] = 12.5

    def _boom(*_args, **_kwargs):
        raise RuntimeError("rotate failed")

    monkeypatch.setattr(type(smap), "rotate", _boom)
    with pytest.raises(RuntimeError, match="rotate failed"):
        rotate_refmap_for_display(smap)

    with pytest.raises(RuntimeError, match="rotate failed"):
        crop_refmap_to_model_box_after_pangle_rotation(
            smap,
            _make_model_box()[0].model_box_corners_world(),
            pad=1.1,
            model_obstime=Time("2026-04-03T19:46:37.800"),
        )


def test_reproject_map_to_target_observer_fov_propagates_header_failure(monkeypatch):
    """Header failure must raise so callers do not keep target FOV coverage."""
    import pyampp.io.refmap_crop as crop_mod

    date_obs = "2012-07-12T04:46:15.006"
    smap = _stereo_map(size=32, scale=40.0, date_obs=date_obs)
    earth = get_earth(Time(date_obs))
    fov = {
        "xc_arcsec": 0.0,
        "yc_arcsec": 0.0,
        "xsize_arcsec": 400.0,
        "ysize_arcsec": 400.0,
    }
    monkeypatch.setattr(crop_mod, "display_observer_reproject_header_for_fov", lambda *a, **k: None)
    with pytest.raises(ValueError, match="could not build target-observer ROI header"):
        reproject_map_to_target_observer_fov(
            smap,
            target_fov=fov,
            target_observer=earth,
            target_obstime=smap.date,
        )


def test_reproject_map_to_target_observer_fov_propagates_reproject_failure(monkeypatch):
    """Canvas reprojection failure must raise, not return the source map."""
    import pyampp.io.refmap_crop as crop_mod

    date_obs = "2012-07-12T04:46:15.006"
    smap = _stereo_map(size=32, scale=40.0, date_obs=date_obs)
    earth = get_earth(Time(date_obs))
    fov = {
        "xc_arcsec": 0.0,
        "yc_arcsec": 0.0,
        "xsize_arcsec": 400.0,
        "ysize_arcsec": 400.0,
    }

    def _boom(*_args, **_kwargs):
        raise RuntimeError("reproject failed")

    monkeypatch.setattr(crop_mod, "reproject_map_onto_canvas", _boom)
    with pytest.raises(RuntimeError, match="reproject failed"):
        reproject_map_to_target_observer_fov(
            smap,
            target_fov=fov,
            target_observer=earth,
            target_obstime=smap.date,
            algorithm="interpolation",
        )


def test_reproject_map_to_display_observer_fov_drops_coverage_on_failure(monkeypatch):
    """Compound API must return (smap, None) rather than claim target FOV."""
    import pyampp.io.refmap_crop as crop_mod

    date_obs = "2012-07-12T04:46:15.006"
    smap = _stereo_map(size=32, scale=40.0, date_obs=date_obs)
    earth = get_earth(Time(date_obs))
    stereo = smap.observer_coordinate
    fov = {
        "xc_arcsec": 0.0,
        "yc_arcsec": 0.0,
        "xsize_arcsec": 400.0,
        "ysize_arcsec": 400.0,
    }

    def _boom(*_args, **_kwargs):
        raise RuntimeError("reproject failed")

    monkeypatch.setattr(crop_mod, "reproject_map_to_target_observer_fov", _boom)
    out, coverage = crop_mod.reproject_map_to_display_observer_fov(
        smap,
        source_fov=fov,
        source_observer=stereo,
        source_obstime=smap.date,
        target_observer=earth,
        target_obstime=smap.date,
    )
    assert out is smap
    assert coverage is None


def test_crop_refmap_spatial_auto_skips_rotate_when_crota_near_zero(monkeypatch):
    """auto policy must read CROTA2 via meta mapping and honor pangle_rotated."""
    import pyampp.io.refmap_crop as crop_mod

    smap = _earth_aia_map(date_obs="2026-04-03T19:46:37.800")
    smap.meta["CROTA2"] = 0.05
    box, obs_time, _ = _make_model_box()
    corners = box.model_box_corners_world()

    calls = {"rotate_path": 0, "plain_path": 0}
    real_plain = crop_mod.crop_refmap_to_model_box
    real_rot = crop_mod.crop_refmap_to_model_box_after_pangle_rotation

    def _plain(*args, **kwargs):
        calls["plain_path"] += 1
        return real_plain(*args, **kwargs)

    def _rot(*args, **kwargs):
        calls["rotate_path"] += 1
        return real_rot(*args, **kwargs)

    monkeypatch.setattr(crop_mod, "crop_refmap_to_model_box", _plain)
    monkeypatch.setattr(crop_mod, "crop_refmap_to_model_box_after_pangle_rotation", _rot)

    result = crop_refmap_spatial(
        smap=smap,
        box_corners_world=corners,
        model_obstime=obs_time,
        pangle_policy="auto",
    )
    assert result.pangle_rotated is False
    assert result.pangle_value_deg == pytest.approx(0.05)
    assert calls["plain_path"] == 1
    assert calls["rotate_path"] == 0
    assert int(result.cropped_map.meta["pyampp_crop_pangle_rotated"]) == 0


def test_crop_refmap_spatial_auto_rotates_when_crota_nonzero(monkeypatch):
    import pyampp.io.refmap_crop as crop_mod

    smap = _earth_aia_map(date_obs="2026-04-03T19:46:37.800")
    smap.meta["CROTA2"] = 12.5
    box, obs_time, _ = _make_model_box()
    corners = box.model_box_corners_world()

    calls = {"rotate_path": 0}
    real_rot = crop_mod.crop_refmap_to_model_box_after_pangle_rotation

    def _rot(*args, **kwargs):
        calls["rotate_path"] += 1
        return real_rot(*args, **kwargs)

    monkeypatch.setattr(crop_mod, "crop_refmap_to_model_box_after_pangle_rotation", _rot)

    result = crop_refmap_spatial(
        smap=smap,
        box_corners_world=corners,
        model_obstime=obs_time,
        pangle_policy="auto",
    )
    assert result.pangle_rotated is True
    assert result.pangle_value_deg == pytest.approx(12.5)
    assert calls["rotate_path"] == 1
    assert int(result.cropped_map.meta["pyampp_crop_pangle_rotated"]) == 1


def test_crop_refmap_spatial_rejects_unknown_pangle_policy():
    smap = _earth_aia_map(date_obs="2026-04-03T19:46:37.800")
    box, obs_time, _ = _make_model_box()
    with pytest.raises(ValueError, match="Unsupported pangle_policy"):
        crop_refmap_spatial(
            smap=smap,
            box_corners_world=box.model_box_corners_world(),
            model_obstime=obs_time,
            pangle_policy="sometimes",
        )
