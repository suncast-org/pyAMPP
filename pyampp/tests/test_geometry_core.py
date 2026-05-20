from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import Heliocentric, Helioprojective, get_earth
from sunpy.map import Map, make_fitswcs_header

from pyampp.geometry import (
    build_fov_box_from_red_box_world,
    build_fov_box_from_user_hpc_and_red_box_world,
    compute_inscribing_fov_box_from_world,
    compute_inscribing_fov_from_world,
    local_cartesian_to_world,
    make_observer_wcs_header,
    observer_fov_box_to_world_corners,
    observer_rectangle_to_hpc_corners,
    project_box_front_face_to_observer_hpc,
    project_coordinate_edges_to_observer_hpc,
    project_world_to_observer_hpc,
    project_world_to_pixel,
)
from pyampp.gxbox.box import Box


def _make_test_box() -> tuple[Box, Time, object, object]:
    obs_time = Time("2024-05-12T16:00:00")
    observer = get_earth(obs_time)
    frame_obs = Helioprojective(observer=observer, obstime=obs_time)
    box_origin = SkyCoord(Tx=0 * u.arcsec, Ty=0 * u.arcsec, distance=observer.radius, frame=frame_obs)
    frame_hcc = Heliocentric(observer=observer, obstime=obs_time)
    box_center = box_origin.transform_to(frame_hcc)
    box_center = SkyCoord(x=box_center.x, y=box_center.y, z=box_center.z + 20 * u.Mm, frame=box_center.frame)
    box = Box(frame_obs, box_origin, box_center, np.array([8, 6, 4]) * u.pix, np.array([5.0, 5.0, 10.0]) * u.Mm)
    return box, obs_time, observer, frame_obs


def test_public_geometry_core_matches_box_wrapper_fov_results() -> None:
    box, _obs_time, _observer, frame_obs = _make_test_box()

    world = box.model_box_corners_world()
    assert world is not None

    fov_from_box = box.model_box_inscribing_fov()
    fov_box_from_box = box.model_box_inscribing_fov_box()
    assert fov_from_box is not None
    assert fov_box_from_box is not None

    fov_from_core = compute_inscribing_fov_from_world(world, frame_obs=frame_obs)
    fov_box_from_core = compute_inscribing_fov_box_from_world(world, frame_obs=frame_obs)

    assert fov_from_core is not None
    assert fov_box_from_core is not None
    for key in ("xc_arcsec", "yc_arcsec", "xsize_arcsec", "ysize_arcsec"):
        assert np.isclose(float(fov_from_box[key]), float(fov_from_core[key]))
        assert np.isclose(float(fov_box_from_box[key]), float(fov_box_from_core[key]))
    for key in ("zmin_mm", "zmax_mm"):
        assert np.isclose(float(fov_box_from_box[key]), float(fov_box_from_core[key]))


def test_public_geometry_core_projects_vectorized_polyline_to_pixels() -> None:
    obs_time = Time("2024-05-12T16:00:00")
    observer = get_earth(obs_time)
    frame_obs = Helioprojective(observer=observer, obstime=obs_time)
    origin = SkyCoord(Tx=0 * u.arcsec, Ty=0 * u.arcsec, distance=observer.radius, frame=frame_obs)
    frame_hcc = Heliocentric(observer=observer, obstime=obs_time)
    center = origin.transform_to(frame_hcc)

    coords = np.array(
        [
            [-10.0, -5.0, 0.0],
            [-5.0, -2.0, 5.0],
            [0.0, 0.0, 10.0],
            [5.0, 3.0, 12.0],
            [10.0, 6.0, 15.0],
        ],
        dtype=float,
    )
    world = local_cartesian_to_world(coords, frame=center.frame, z_base_mm=0.0)
    hpc = project_world_to_observer_hpc(world, observer=observer, obstime=obs_time)

    header = make_fitswcs_header(
        data=np.zeros((128, 128)),
        coordinate=SkyCoord(Tx=0 * u.arcsec, Ty=0 * u.arcsec, frame=frame_obs),
        scale=[2.0, 2.0] * (u.arcsec / u.pix),
        instrument="TEST",
        observatory="TEST",
        wavelength=171 * u.angstrom,
    )
    smap = Map(np.zeros((128, 128)), header)
    pixels = project_world_to_pixel(hpc, smap)

    assert world is not None
    assert hpc is not None
    assert pixels is not None
    assert pixels[0].shape == (5,)
    assert pixels[1].shape == (5,)
    assert np.all(np.isfinite(pixels[0]))
    assert np.all(np.isfinite(pixels[1]))


def test_make_observer_wcs_header_uses_explicit_iso_obstime_and_observer_cards() -> None:
    observer_time = Time("2024-05-12T16:00:00")
    header_time = Time("2024-05-12T18:30:00")
    observer = get_earth(observer_time)

    header = make_observer_wcs_header(
        nx=64,
        ny=48,
        xc_arcsec=12.5,
        yc_arcsec=-8.5,
        dx_arcsec=2.0,
        dy_arcsec=3.0,
        observer=observer,
        obs_time=header_time,
        bunit="DN s-1 pix-1",
        observer_name="earth",
    )

    assert header["DATE-OBS"] == header_time.isot
    assert header["DATE_OBS"] == header_time.isot
    assert header["BUNIT"] == "DN s-1 pix-1"
    assert header["OBSERVER"] == "Earth"
    assert header["NAXIS1"] == 64
    assert header["NAXIS2"] == 48
    assert header["CTYPE1"].startswith("HPLN")
    assert header["CTYPE2"].startswith("HPLT")
    for key in ("HGLN_OBS", "HGLT_OBS", "DSUN_OBS", "RSUN_REF", "RSUN_OBS"):
        assert key in header
        assert np.isfinite(float(header[key]))


def test_local_cartesian_to_world_rejects_nonfinite_rows() -> None:
    box, _obs_time, _observer, _frame_obs = _make_test_box()
    local = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 2.0]], dtype=float)

    world = local_cartesian_to_world(local, frame=getattr(getattr(box, "_center", None), "frame", None), z_base_mm=0.0)

    assert world is None


def test_public_geometry_core_reconstructs_saved_fov_box_corners() -> None:
    box, obs_time, observer, _frame_obs = _make_test_box()
    world = observer_fov_box_to_world_corners(
        xc_arcsec=20.0,
        yc_arcsec=-10.0,
        xsize_arcsec=40.0,
        ysize_arcsec=30.0,
        zmin_mm=-5.0,
        zmax_mm=15.0,
        observer=observer,
        obstime=obs_time,
        target_frame=getattr(getattr(box, "_center", None), "frame", None),
    )

    assert world is not None
    assert len(world) == 8


def test_public_geometry_core_projects_box_edges_to_observer_hpc() -> None:
    box, obs_time, observer, frame_obs = _make_test_box()
    world = observer_fov_box_to_world_corners(
        xc_arcsec=20.0,
        yc_arcsec=-10.0,
        xsize_arcsec=40.0,
        ysize_arcsec=30.0,
        zmin_mm=-5.0,
        zmax_mm=15.0,
        observer=observer,
        obstime=obs_time,
        target_frame=getattr(getattr(box, "_center", None), "frame", None),
    )

    edges = project_coordinate_edges_to_observer_hpc(
        world,
        edge_pairs=((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)),
        frame_obs=frame_obs,
    )

    assert edges is not None
    assert len(edges) == 12
    assert all(len(edge) == 2 for edge in edges)
    assert all(np.all(np.isfinite(edge.Tx.to_value(u.arcsec))) for edge in edges)
    assert all(np.all(np.isfinite(edge.Ty.to_value(u.arcsec))) for edge in edges)


def test_public_geometry_core_projects_front_face_and_rectangle_corners() -> None:
    box, obs_time, observer, frame_obs = _make_test_box()
    world = observer_fov_box_to_world_corners(
        xc_arcsec=20.0,
        yc_arcsec=-10.0,
        xsize_arcsec=40.0,
        ysize_arcsec=30.0,
        zmin_mm=-5.0,
        zmax_mm=15.0,
        observer=observer,
        obstime=obs_time,
        target_frame=getattr(getattr(box, "_center", None), "frame", None),
    )
    face = project_box_front_face_to_observer_hpc(world, frame_obs=frame_obs)
    rect = observer_rectangle_to_hpc_corners(
        xc_arcsec=20.0,
        yc_arcsec=-10.0,
        xsize_arcsec=40.0,
        ysize_arcsec=30.0,
        observer=observer,
        obstime=obs_time,
    )

    assert face is not None
    assert len(face) == 5
    assert np.all(np.isfinite(face.Tx.to_value(u.arcsec)))
    assert np.all(np.isfinite(face.Ty.to_value(u.arcsec)))
    assert rect is not None
    assert len(rect) == 4
    assert np.all(np.isfinite(rect.Tx.to_value(u.arcsec)))
    assert np.all(np.isfinite(rect.Ty.to_value(u.arcsec)))


def test_public_geometry_user_fov_box_constructor_keeps_inscribing_z() -> None:
    box, obs_time, observer, _frame_obs = _make_test_box()
    world = box.model_box_corners_world()
    assert world is not None

    auto_box = build_fov_box_from_red_box_world(world, observer=observer, obstime=obs_time)
    user_box = build_fov_box_from_user_hpc_and_red_box_world(
        world,
        xc_arcsec=120.0,
        yc_arcsec=-80.0,
        xsize_arcsec=300.0,
        ysize_arcsec=220.0,
        observer=observer,
        obstime=obs_time,
    )

    assert auto_box is not None
    assert user_box is not None
    assert np.isclose(float(user_box["zmin_mm"]), float(auto_box["zmin_mm"]))
    assert np.isclose(float(user_box["zmax_mm"]), float(auto_box["zmax_mm"]))
    assert np.isclose(float(user_box["xc_arcsec"]), 120.0)
    assert np.isclose(float(user_box["yc_arcsec"]), -80.0)
    assert np.isclose(float(user_box["xsize_arcsec"]), 300.0)
    assert np.isclose(float(user_box["ysize_arcsec"]), 220.0)
