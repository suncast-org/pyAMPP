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
from sunpy.coordinates import Heliocentric, Helioprojective

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
    "Box",
    "BoxGeometryMixin",
]