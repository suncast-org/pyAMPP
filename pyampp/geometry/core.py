"""Public vectorized geometry primitives for pyAMPP.

The gxbox tools historically carried the proven geometry implementation inside
their application layer. This module promotes the reusable, headless pieces so
all 3D structures can flow through the same world/observer projection path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from sunpy.coordinates import Heliocentric, Helioprojective
from sunpy.coordinates import HeliographicCarrington, HeliographicStonyhurst
from sunpy.map.header_helper import make_fitswcs_header

if TYPE_CHECKING:  # pragma: no cover
    from pyampp.gxbox.box import Box, BoxGeometryMixin


def local_cartesian_to_world(
    local_points_mm: np.ndarray,
    *,
    frame,
    z_base_mm: float = 0.0,
) -> SkyCoord | None:
    """Convert model-local Cartesian points into world-frame coordinates."""
    if frame is None:
        return None
    try:
        rows = np.asarray(local_points_mm, dtype=float)
    except Exception:
        return None
    if rows.ndim != 2 or rows.shape[1] != 3 or rows.shape[0] == 0:
        return None
    if not np.all(np.isfinite(rows)):
        return None
    try:
        return SkyCoord(
            x=rows[:, 0] * u.Mm,
            y=rows[:, 1] * u.Mm,
            z=(rows[:, 2] + float(z_base_mm)) * u.Mm,
            frame=frame,
        )
    except Exception:
        return None


def project_world_to_observer_hpc(
    world: SkyCoord,
    *,
    observer=None,
    obstime=None,
    frame_obs=None,
) -> SkyCoord | None:
    """Project world coordinates into the observer helioprojective frame."""
    if world is None:
        return None
    world_frame = getattr(world, "frame", None)
    if observer is None and frame_obs is not None:
        observer = getattr(frame_obs, "observer", None)
    if observer is None and world_frame is not None:
        observer = getattr(world_frame, "observer", None)
    if obstime is None and frame_obs is not None:
        obstime = getattr(frame_obs, "obstime", None)
    if obstime is None and world_frame is not None:
        obstime = getattr(world_frame, "obstime", None)
    if observer is None:
        return None
    try:
        return world.transform_to(Helioprojective(observer=observer, obstime=obstime))
    except Exception:
        return None


def project_world_to_observer_hcc(
    world: SkyCoord,
    *,
    observer=None,
    obstime=None,
    frame_obs=None,
) -> SkyCoord | None:
    """Project world coordinates into the observer-centric heliocentric frame."""
    if world is None:
        return None
    world_frame = getattr(world, "frame", None)
    if observer is None and frame_obs is not None:
        observer = getattr(frame_obs, "observer", None)
    if observer is None and world_frame is not None:
        observer = getattr(world_frame, "observer", None)
    if obstime is None and frame_obs is not None:
        obstime = getattr(frame_obs, "obstime", None)
    if obstime is None and world_frame is not None:
        obstime = getattr(world_frame, "obstime", None)
    if observer is None:
        return None
    try:
        return world.transform_to(Heliocentric(observer=observer, obstime=obstime))
    except Exception:
        return None


def compute_inscribing_fov_from_hpc(
    coords_hpc: SkyCoord,
    *,
    pad_arcsec: float = 0.0,
) -> dict | None:
    """Compute the smallest axis-aligned HPC rectangle covering the inputs."""
    if coords_hpc is None:
        return None
    try:
        tx = np.asarray(coords_hpc.Tx.to_value(u.arcsec), dtype=float)
        ty = np.asarray(coords_hpc.Ty.to_value(u.arcsec), dtype=float)
    except Exception:
        return None
    if tx.size == 0 or ty.size == 0:
        return None
    finite = np.isfinite(tx) & np.isfinite(ty)
    if not np.any(finite):
        return None
    tx = tx[finite]
    ty = ty[finite]
    pad = max(float(pad_arcsec), 0.0)
    xmin = float(np.min(tx)) - pad
    xmax = float(np.max(tx)) + pad
    ymin = float(np.min(ty)) - pad
    ymax = float(np.max(ty)) + pad
    return {
        "xc_arcsec": 0.5 * (xmin + xmax),
        "yc_arcsec": 0.5 * (ymin + ymax),
        "xsize_arcsec": xmax - xmin,
        "ysize_arcsec": ymax - ymin,
        "xmin_arcsec": xmin,
        "xmax_arcsec": xmax,
        "ymin_arcsec": ymin,
        "ymax_arcsec": ymax,
        "corners_hpc": coords_hpc,
    }


def compute_inscribing_fov_from_world(
    world: SkyCoord,
    *,
    observer=None,
    obstime=None,
    frame_obs=None,
    pad_arcsec: float = 0.0,
) -> dict | None:
    """Project world coordinates and compute the enclosing 2D observer FOV."""
    coords_hpc = project_world_to_observer_hpc(
        world,
        observer=observer,
        obstime=obstime,
        frame_obs=frame_obs,
    )
    return compute_inscribing_fov_from_hpc(coords_hpc, pad_arcsec=pad_arcsec)


def compute_inscribing_fov_box_from_world(
    world: SkyCoord,
    *,
    observer=None,
    obstime=None,
    frame_obs=None,
    pad_xy_arcsec: float = 0.0,
    pad_z_frac: float = 0.10,
) -> dict | None:
    """Compute the observer-aligned 3D FOV box enclosing the inputs."""
    footprint = compute_inscribing_fov_from_world(
        world,
        observer=observer,
        obstime=obstime,
        frame_obs=frame_obs,
        pad_arcsec=pad_xy_arcsec,
    )
    if footprint is None:
        return None
    coords_hcc = project_world_to_observer_hcc(
        world,
        observer=observer,
        obstime=obstime,
        frame_obs=frame_obs,
    )
    if coords_hcc is None:
        return None
    try:
        z_vals = np.asarray(coords_hcc.z.to_value(u.Mm), dtype=float)
    except Exception:
        return None
    finite = np.isfinite(z_vals)
    if not np.any(finite):
        return None
    z_vals = z_vals[finite]
    z_min = float(np.nanmin(z_vals))
    z_max = float(np.nanmax(z_vals))
    z_span = max(1e-6, z_max - z_min)
    z_pad = max(0.0, float(pad_z_frac)) * z_span
    result = dict(footprint)
    result.update({
        "zmin_mm": z_min - z_pad,
        "zmax_mm": z_max + z_pad,
    })
    return result

def build_fov_box_from_red_box_world(
    world: SkyCoord,
    *,
    observer=None,
    obstime=None,
    frame_obs=None,
    pad_xy_arcsec: float = 0.0,
    pad_z_frac: float = 0.10,
) -> dict | None:
    """Build a full observer-aligned FOV box from red-box world corners."""
    return compute_inscribing_fov_box_from_world(
        world,
        observer=observer,
        obstime=obstime,
        frame_obs=frame_obs,
        pad_xy_arcsec=pad_xy_arcsec,
        pad_z_frac=pad_z_frac,
    )

def build_fov_box_from_user_hpc_and_red_box_world(
    world: SkyCoord,
    *,
    xc_arcsec: float,
    yc_arcsec: float,
    xsize_arcsec: float,
    ysize_arcsec: float,
    observer=None,
    obstime=None,
    frame_obs=None,
    pad_z_frac: float = 0.10,
) -> dict | None:
    """Build user-defined x/y FOV with LOS-safe z from the red-box inscribing FOV box."""
    base = build_fov_box_from_red_box_world(
        world,
        observer=observer,
        obstime=obstime,
        frame_obs=frame_obs,
        pad_xy_arcsec=0.0,
        pad_z_frac=pad_z_frac,
    )
    if base is None:
        return None
    result = dict(base)
    result.update(
        {
            "xc_arcsec": float(xc_arcsec),
            "yc_arcsec": float(yc_arcsec),
            "xsize_arcsec": max(float(xsize_arcsec), 1e-6),
            "ysize_arcsec": max(float(ysize_arcsec), 1e-6),
        }
    )
    return result


def observer_fov_box_to_world_corners(
    *,
    xc_arcsec: float,
    yc_arcsec: float,
    xsize_arcsec: float,
    ysize_arcsec: float,
    zmin_mm: float,
    zmax_mm: float,
    observer,
    obstime,
    target_frame,
) -> SkyCoord | None:
    """Reconstruct 8 world-space corners from saved observer `fov_box` metadata."""
    if observer is None or target_frame is None:
        return None
    try:
        dsun_mm = float(observer.radius.to_value(u.Mm))
    except Exception:
        return None
    half_w = 0.5 * max(float(xsize_arcsec), 1e-6)
    half_h = 0.5 * max(float(ysize_arcsec), 1e-6)
    frame_hpc = Helioprojective(observer=observer, obstime=obstime)
    points = np.asarray(
        [
            [float(xc_arcsec) - half_w, float(yc_arcsec) - half_h, float(zmin_mm)],
            [float(xc_arcsec) + half_w, float(yc_arcsec) - half_h, float(zmin_mm)],
            [float(xc_arcsec) - half_w, float(yc_arcsec) + half_h, float(zmin_mm)],
            [float(xc_arcsec) + half_w, float(yc_arcsec) + half_h, float(zmin_mm)],
            [float(xc_arcsec) - half_w, float(yc_arcsec) - half_h, float(zmax_mm)],
            [float(xc_arcsec) + half_w, float(yc_arcsec) - half_h, float(zmax_mm)],
            [float(xc_arcsec) - half_w, float(yc_arcsec) + half_h, float(zmax_mm)],
            [float(xc_arcsec) + half_w, float(yc_arcsec) + half_h, float(zmax_mm)],
        ],
        dtype=float,
    )
    distances = dsun_mm - points[:, 2]
    if not np.all(np.isfinite(distances)) or np.any(distances <= 0):
        return None
    try:
        corners_hpc = SkyCoord(
            Tx=points[:, 0] * u.arcsec,
            Ty=points[:, 1] * u.arcsec,
            distance=distances * u.Mm,
            frame=frame_hpc,
        )
        return corners_hpc.transform_to(target_frame)
    except Exception:
        return None


def observer_rectangle_to_hpc_corners(
    *,
    xc_arcsec: float,
    yc_arcsec: float,
    xsize_arcsec: float,
    ysize_arcsec: float,
    observer,
    obstime,
) -> SkyCoord | None:
    """Construct 4 helioprojective rectangle corners for an observer view."""
    if observer is None:
        return None
    half_w = 0.5 * max(float(xsize_arcsec), 1e-6)
    half_h = 0.5 * max(float(ysize_arcsec), 1e-6)
    try:
        return SkyCoord(
            Tx=np.asarray(
                [
                    float(xc_arcsec) - half_w,
                    float(xc_arcsec) + half_w,
                    float(xc_arcsec) - half_w,
                    float(xc_arcsec) + half_w,
                ],
                dtype=float,
            )
            * u.arcsec,
            Ty=np.asarray(
                [
                    float(yc_arcsec) - half_h,
                    float(yc_arcsec) - half_h,
                    float(yc_arcsec) + half_h,
                    float(yc_arcsec) + half_h,
                ],
                dtype=float,
            )
            * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
    except Exception:
        return None


def project_coordinate_edges_to_observer_hpc(
    coords,
    *,
    edge_pairs,
    observer=None,
    obstime=None,
    frame_obs=None,
) -> list[SkyCoord] | None:
    """Project coordinate edge pairs into one observer helioprojective frame."""
    if coords is None:
        return None
    try:
        edges = []
        for i, j in edge_pairs:
            edge = SkyCoord([coords[int(i)], coords[int(j)]])
            projected = project_world_to_observer_hpc(
                edge,
                observer=observer,
                obstime=obstime,
                frame_obs=frame_obs,
            )
            if projected is None:
                return None
            edges.append(projected)
        return edges
    except Exception:
        return None


def project_box_front_face_to_observer_hpc(
    world_corners: SkyCoord,
    *,
    observer=None,
    obstime=None,
    frame_obs=None,
    front_face=(0, 1, 3, 2),
    back_face=(4, 5, 7, 6),
) -> SkyCoord | None:
    """Project the nearer box face into the observer helioprojective frame."""
    if world_corners is None:
        return None
    corners_obs = project_world_to_observer_hpc(
        world_corners,
        observer=observer,
        obstime=obstime,
        frame_obs=frame_obs,
    )
    if corners_obs is None:
        return None

    def _mean_distance(indices) -> float:
        try:
            return float(np.nanmean(corners_obs[np.asarray(indices, dtype=int)].distance.to_value(u.Mm)))
        except Exception:
            return np.inf

    face = np.asarray(front_face if _mean_distance(front_face) <= _mean_distance(back_face) else back_face, dtype=int)
    try:
        ordered = [corners_obs[int(i)] for i in face] + [corners_obs[int(face[0])]]
        return SkyCoord(ordered)
    except Exception:
        return None


def project_world_to_pixel(world: SkyCoord, smap) -> tuple[np.ndarray, np.ndarray] | None:
    """Project world coordinates into a map's pixel plane in one vectorized call."""
    if world is None or smap is None:
        return None
    try:
        xpix, ypix = smap.world_to_pixel(world)
        x = np.asarray(xpix.value if hasattr(xpix, "value") else xpix, dtype=float)
        y = np.asarray(ypix.value if hasattr(ypix, "value") else ypix, dtype=float)
    except Exception:
        return None
    return x, y

def world_to_local_cartesian_mm(world: SkyCoord, *, z_base_mm: float = 0.0) -> np.ndarray | None:
    """Extract cartesian mm rows from world coords and shift z by z_base_mm."""
    if world is None:
        return None
    try:
        rows = np.column_stack(
            [
                np.asarray(world.x.to_value(u.Mm), dtype=float),
                np.asarray(world.y.to_value(u.Mm), dtype=float),
                np.asarray(world.z.to_value(u.Mm), dtype=float) - float(z_base_mm),
            ]
        )
    except Exception:
        return None
    if rows.ndim != 2 or rows.shape[1] != 3 or rows.shape[0] == 0:
        return None
    if not np.all(np.isfinite(rows)):
        return None
    return rows.astype(float)


def make_observer_wcs_header(
    *,
    nx: int,
    ny: int,
    xc_arcsec: float,
    yc_arcsec: float,
    dx_arcsec: float,
    dy_arcsec: float,
    observer: SkyCoord,
    obs_time,
    bunit: str,
    observer_name: str = "custom",
    rsun_ref_m: float | None = None,
    rsun_obs_arcsec: float | None = None,
) -> fits.Header:
    """Create a SunPy-compatible observer-aware WCS header."""
    ref_coord = SkyCoord(
        Tx=float(xc_arcsec) * u.arcsec,
        Ty=float(yc_arcsec) * u.arcsec,
        frame=Helioprojective(observer=observer, obstime=observer.obstime),
    )
    header = make_fitswcs_header(
        np.empty((int(ny), int(nx)), dtype=np.float32),
        ref_coord,
        scale=u.Quantity([float(dx_arcsec), float(dy_arcsec)], u.arcsec / u.pix),
    )

    observer_hgs = observer.transform_to(HeliographicStonyhurst(obstime=observer.obstime))
    l0_deg = float(observer_hgs.lon.to_value(u.deg))
    b0_deg = float(observer_hgs.lat.to_value(u.deg))
    dsun_cm = float(observer_hgs.radius.to_value(u.cm))

    rsun_ref_m_value = float(rsun_ref_m) if rsun_ref_m is not None else 6.957e8
    rsun_obs_arcsec_value = rsun_obs_arcsec
    if rsun_obs_arcsec_value is None or not np.isfinite(float(rsun_obs_arcsec_value)) or float(rsun_obs_arcsec_value) <= 0:
        ratio = min(1.0, rsun_ref_m_value / (dsun_cm / 100.0))
        rsun_obs_arcsec_value = float(np.degrees(np.arcsin(ratio)) * 3600.0)
    else:
        rsun_obs_arcsec_value = float(rsun_obs_arcsec_value)

    observer_label = str(observer_name or "custom").replace("-", " ").title()
    header["DATE-OBS"] = str(obs_time)
    header["BUNIT"] = str(bunit)
    header["OBSERVER"] = observer_label
    header["B0"] = b0_deg
    header["L0"] = l0_deg
    header["RSUN_ARC"] = float(rsun_obs_arcsec_value)
    header["SOLAR_B0"] = b0_deg
    header["SOLAR_L0"] = l0_deg
    header["HGLN_OBS"] = l0_deg
    header["HGLT_OBS"] = b0_deg
    header["DSUN_OBS"] = dsun_cm / 100.0
    header["RSUN_REF"] = rsun_ref_m_value
    header["RSUN_OBS"] = float(rsun_obs_arcsec_value)

    try:
        observer_hgc = observer.transform_to(HeliographicCarrington(obstime=observer.obstime, observer="self"))
        header["CRLN_OBS"] = float(observer_hgc.lon.to_value(u.deg))
        header["CRLT_OBS"] = float(observer_hgc.lat.to_value(u.deg))
    except Exception:
        pass
    return header


def __getattr__(name: str):
    if name in {"Box", "BoxGeometryMixin"}:
        from pyampp.gxbox.box import Box, BoxGeometryMixin

        exports = {
            "Box": Box,
            "BoxGeometryMixin": BoxGeometryMixin,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "local_cartesian_to_world",
    "project_world_to_observer_hpc",
    "project_world_to_observer_hcc",
    "compute_inscribing_fov_from_hpc",
    "compute_inscribing_fov_from_world",
    "compute_inscribing_fov_box_from_world",
    "observer_fov_box_to_world_corners",
    "observer_rectangle_to_hpc_corners",
    "project_coordinate_edges_to_observer_hpc",
    "project_box_front_face_to_observer_hpc",
    "project_world_to_pixel",
    "build_fov_box_from_red_box_world",
    "build_fov_box_from_user_hpc_and_red_box_world",
    "world_to_local_cartesian_mm",
    "make_observer_wcs_header",
    "Box",
    "BoxGeometryMixin",
]