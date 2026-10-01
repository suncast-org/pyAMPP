"""Geometry-driven reference-map cropping for pyAMPP models.

Maps are cropped in their native observer frame at the map observation time.
No embed-time rotation or reprojection is applied; display alignment is deferred
to the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import Helioprojective, get_earth
from sunpy.map import Map, make_fitswcs_header

from pyampp.geometry.core import (
    compute_inscribing_fov_box_from_world,
    compute_inscribing_fov_from_world,
    make_observer_wcs_header,
    observer_rectangle_to_hpc_corners,
    project_coordinate_edges_to_observer_hpc,
)
from pyampp.gxbox.gx_fov2box import _spherical_screen_context_for_observer, _submap_with_fov_safe

_BOX_EDGE_INDEX_PAIRS = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

_HEADER_PRESERVE_KEYS = (
    "TELESCOP",
    "INSTRUME",
    "DETECTOR",
    "WAVELNTH",
    "WAVEUNIT",
    "CONTENT",
    "BUNIT",
    "T_OBS",
    "T_REC",
    "DATE",
    "DATE-OBS",
    "DATE_OBS",
    "EXPTIME",
    "LVL_NUM",
    "QUALITY",
    "CRVAL3",
    "CDELT3",
    "CUNIT3",
    "CTYPE3",
    "CRVAL4",
    "CDELT4",
    "CUNIT4",
    "CTYPE4",
    "HGLN_OBS",
    "HGLT_OBS",
    "DSUN_OBS",
    "RSUN_OBS",
    "RSUN_REF",
)


@dataclass(frozen=True)
class RefmapCropResult:
    """Output of :func:`crop_refmap_to_model_box`."""

    cropped_map: Map
    crop_fov: dict[str, float]
    map_observer: Any
    map_obstime: Time
    model_obstime: Time
    box_corners_world: SkyCoord
    pad_factor: float


def infer_model_obstime_from_box_corners(box_corners_world: SkyCoord) -> Time | None:
    """Return the model obstime carried by red-box world coordinates, if any."""
    frame = getattr(box_corners_world, "frame", None)
    obstime = getattr(frame, "obstime", None) if frame is not None else None
    if obstime is None:
        return None
    try:
        return Time(obstime)
    except Exception:
        return None


def apply_pad_factor_to_fov_box(fov_box: dict[str, float], pad: float) -> dict[str, float]:
    """Scale an inscribing FOV box by ``pad`` about its center."""
    footprint = apply_pad_factor_to_fov(fov_box, pad)
    zmin = float(fov_box["zmin_mm"])
    zmax = float(fov_box["zmax_mm"])
    zc = 0.5 * (zmin + zmax)
    half_z = 0.5 * max(zmax - zmin, 1e-6) * max(float(pad), 1.0)
    out = dict(footprint)
    out["zmin_mm"] = zc - half_z
    out["zmax_mm"] = zc + half_z
    return out


def compute_crop_fov_box_for_observer(
    box_corners_world: SkyCoord,
    *,
    observer,
    obstime,
    pad: float = 1.1,
    pad_z_frac: float = 0.10,
) -> dict[str, float] | None:
    """Project the red box and return a padded 3D inscribing FOV box dict."""
    base = compute_inscribing_fov_box_from_world(
        box_corners_world,
        observer=observer,
        obstime=obstime,
        pad_xy_arcsec=0.0,
        pad_z_frac=pad_z_frac,
    )
    if base is None:
        return None
    return apply_pad_factor_to_fov_box(base, pad)


def compute_inscribing_fov_box_for_observer(
    box_corners_world: SkyCoord,
    *,
    observer,
    obstime,
    pad_z_frac: float = 0.10,
) -> dict[str, float] | None:
    """Return the unpadded 3D inscribing FOV box in an observer frame.

    The 2D xy extent is ``1/pad`` times the padded crop FOV rectangle when
    ``pad`` is applied to :func:`compute_crop_fov_for_observer` (e.g. ~10% smaller at pad=1.1).
    """
    return compute_crop_fov_box_for_observer(
        box_corners_world,
        observer=observer,
        obstime=obstime,
        pad=1.0,
        pad_z_frac=pad_z_frac,
    )


def fov_box_corners_hpc(
    fov_box: dict[str, float],
    *,
    observer,
    obstime,
) -> SkyCoord | None:
    """Eight 3D helioprojective corners of an observer-aligned FOV box."""
    try:
        dsun_mm = float(observer.radius.to_value(u.Mm))
    except Exception:
        return None
    half_w = 0.5 * max(float(fov_box["xsize_arcsec"]), 1e-6)
    half_h = 0.5 * max(float(fov_box["ysize_arcsec"]), 1e-6)
    zmin = float(fov_box["zmin_mm"])
    zmax = float(fov_box["zmax_mm"])
    frame_hpc = Helioprojective(observer=observer, obstime=obstime)
    points = np.asarray(
        [
            [float(fov_box["xc_arcsec"]) - half_w, float(fov_box["yc_arcsec"]) - half_h, zmin],
            [float(fov_box["xc_arcsec"]) + half_w, float(fov_box["yc_arcsec"]) - half_h, zmin],
            [float(fov_box["xc_arcsec"]) - half_w, float(fov_box["yc_arcsec"]) + half_h, zmin],
            [float(fov_box["xc_arcsec"]) + half_w, float(fov_box["yc_arcsec"]) + half_h, zmin],
            [float(fov_box["xc_arcsec"]) - half_w, float(fov_box["yc_arcsec"]) - half_h, zmax],
            [float(fov_box["xc_arcsec"]) + half_w, float(fov_box["yc_arcsec"]) - half_h, zmax],
            [float(fov_box["xc_arcsec"]) - half_w, float(fov_box["yc_arcsec"]) + half_h, zmax],
            [float(fov_box["xc_arcsec"]) + half_w, float(fov_box["yc_arcsec"]) + half_h, zmax],
        ],
        dtype=float,
    )
    distances = dsun_mm - points[:, 2]
    if not np.all(np.isfinite(distances)) or np.any(distances <= 0):
        return None
    return SkyCoord(
        Tx=points[:, 0] * u.arcsec,
        Ty=points[:, 1] * u.arcsec,
        distance=distances * u.Mm,
        frame=frame_hpc,
    )


_FOV_BOX_EDGE_INDEX_PAIRS = _BOX_EDGE_INDEX_PAIRS


def project_fov_box_edges_to_observer_hpc(
    fov_box: dict[str, float],
    *,
    observer,
    obstime,
) -> list[SkyCoord]:
    corners = fov_box_corners_hpc(fov_box, observer=observer, obstime=obstime)
    if corners is None:
        return []
    edges = project_coordinate_edges_to_observer_hpc(
        corners,
        edge_pairs=_FOV_BOX_EDGE_INDEX_PAIRS,
        observer=observer,
        obstime=obstime,
    )
    return edges or []


def project_fov_box_edges_reprojected(
    fov_box: dict[str, float],
    *,
    source_observer,
    source_obstime,
    target_observer,
    target_obstime,
) -> list[SkyCoord]:
    """Project a source-frame FOV box wireframe into a target observer frame."""
    corners = fov_box_corners_hpc(
        fov_box,
        observer=source_observer,
        obstime=source_obstime,
    )
    if corners is None:
        return []
    edges = project_coordinate_edges_to_observer_hpc(
        corners,
        edge_pairs=_FOV_BOX_EDGE_INDEX_PAIRS,
        observer=target_observer,
        obstime=target_obstime,
    )
    return edges or []


def plot_inscribing_fov_box_on_axes(
    ax,
    fov_box: dict[str, float],
    *,
    observer,
    obstime,
    color: str = "blue",
    linewidth: float = 0.9,
    zorder: int = 21,
    label: str | None = "inscribing FOV box",
) -> list[Any]:
    """Draw a projected 3D inscribing FOV box wireframe on a sunpy map axes.

    Uses the same ``axes.plot_coord`` overlay path as
    :class:`pyampp.gxbox.box_view2d.MapBoxDisplayWidget`.
    """
    artists: list[Any] = []
    edges = project_fov_box_edges_to_observer_hpc(
        fov_box,
        observer=observer,
        obstime=obstime,
    )
    for i, edge in enumerate(edges):
        try:
            artists.extend(
                ax.plot_coord(
                    edge,
                    color=color,
                    ls="-",
                    marker="",
                    lw=linewidth,
                    zorder=zorder,
                    label=label if i == 0 and label else None,
                )
            )
        except Exception:
            continue
    return artists


def sample_world_edge(
    world: SkyCoord,
    start_idx: int,
    end_idx: int,
    *,
    samples: int = 16,
) -> SkyCoord | None:
    """Sample points along one 3D model-box edge in its native world frame."""
    try:
        p0 = world[int(start_idx)]
        p1 = world[int(end_idx)]
        frame = p0.frame
        t = np.linspace(0.0, 1.0, max(int(samples), 2))
        xs = np.interp(t, [0.0, 1.0], [p0.x.to_value(u.Mm), p1.x.to_value(u.Mm)])
        ys = np.interp(t, [0.0, 1.0], [p0.y.to_value(u.Mm), p1.y.to_value(u.Mm)])
        zs = np.interp(t, [0.0, 1.0], [p0.z.to_value(u.Mm), p1.z.to_value(u.Mm)])
        return SkyCoord(x=xs * u.Mm, y=ys * u.Mm, z=zs * u.Mm, frame=frame)
    except Exception:
        return None


def project_model_box_edges_to_observer_hpc(
    box_corners_world: SkyCoord,
    *,
    observer,
    obstime,
    samples_per_edge: int = 16,
) -> list[SkyCoord]:
    """Project sampled model-box edges into an observer helioprojective frame."""
    edges: list[SkyCoord] = []
    for i, j in _BOX_EDGE_INDEX_PAIRS:
        sampled = sample_world_edge(box_corners_world, i, j, samples=samples_per_edge)
        if sampled is None:
            continue
        projected = project_world_to_observer_hpc(
            sampled,
            observer=observer,
            obstime=obstime,
        )
        if projected is not None:
            edges.append(projected)
    return edges


def project_world_to_observer_hpc(
    world: SkyCoord,
    *,
    observer,
    obstime,
    frame_obs=None,
) -> SkyCoord | None:
    """Local wrapper around :func:`pyampp.geometry.core.project_world_to_observer_hpc`."""
    from pyampp.geometry.core import project_world_to_observer_hpc as _project

    return _project(world, observer=observer, obstime=obstime, frame_obs=frame_obs)


def apply_pad_factor_to_fov(fov: dict[str, float], pad: float) -> dict[str, float]:
    """Scale an inscribing FOV rectangle by ``pad`` about its center."""
    pad = max(float(pad), 1.0)
    xc = float(fov["xc_arcsec"])
    yc = float(fov["yc_arcsec"])
    half_w = 0.5 * max(float(fov["xsize_arcsec"]), 1e-6) * pad
    half_h = 0.5 * max(float(fov["ysize_arcsec"]), 1e-6) * pad
    return {
        "xc_arcsec": xc,
        "yc_arcsec": yc,
        "xsize_arcsec": 2.0 * half_w,
        "ysize_arcsec": 2.0 * half_h,
        "xmin_arcsec": xc - half_w,
        "xmax_arcsec": xc + half_w,
        "ymin_arcsec": yc - half_h,
        "ymax_arcsec": yc + half_h,
    }


def compute_crop_fov_for_observer(
    box_corners_world: SkyCoord,
    *,
    observer,
    obstime,
    pad: float = 1.1,
) -> dict[str, float] | None:
    """Project red-box corners and return the padded inscribing FOV dict."""
    base = compute_inscribing_fov_from_world(
        box_corners_world,
        observer=observer,
        obstime=obstime,
        pad_arcsec=0.0,
    )
    if base is None:
        return None
    return apply_pad_factor_to_fov(base, pad)


def fov_crop_sky_coords(
    fov: dict[str, float],
    *,
    observer,
    obstime,
) -> tuple[SkyCoord, SkyCoord]:
    """Bottom-left and top-right sky corners for a padded FOV rectangle."""
    half_w = 0.5 * max(float(fov["xsize_arcsec"]), 1e-6)
    half_h = 0.5 * max(float(fov["ysize_arcsec"]), 1e-6)
    xc = float(fov["xc_arcsec"])
    yc = float(fov["yc_arcsec"])
    frame = Helioprojective(observer=observer, obstime=obstime)
    bottom_left = SkyCoord(
        Tx=(xc - half_w) * u.arcsec,
        Ty=(yc - half_h) * u.arcsec,
        frame=frame,
    )
    top_right = SkyCoord(
        Tx=(xc + half_w) * u.arcsec,
        Ty=(yc + half_h) * u.arcsec,
        frame=frame,
    )
    return bottom_left, top_right


_ALT_WCS_KEY_PREFIXES = (
    "CRPIX",
    "CRVAL",
    "CDELT",
    "CTYPE",
    "CUNIT",
    "PC",
    "PV",
    "CRDER",
    "CSYER",
    "CD",
    "PROJP",
)


def _strip_stale_parent_wcs_keywords(meta: dict[str, Any]) -> dict[str, Any]:
    """Drop alternate full-disk WCS keywords copied from the parent map."""
    out = dict(meta)
    for key in list(out):
        key_upper = str(key).upper()
        if key_upper.endswith("A") and key_upper.startswith(_ALT_WCS_KEY_PREFIXES):
            out.pop(key, None)
    return out


def _sync_legacy_idl_center_keywords(meta: dict[str, Any], *, nx: int, ny: int) -> dict[str, Any]:
    """Keep SSW ``XCEN``/``YCEN`` consistent with FITS WCS (``get_fits_cen``).

    IDL map readers prefer explicit ``XCEN``/``YCEN`` over ``CRVAL`` when those
    tags are present and non-zero, so stale parent values must be overwritten.
    """
    out = dict(meta)

    def _meta_float(*names: str, default: float = 0.0) -> float:
        for name in names:
            value = out.get(name)
            if value is not None:
                return float(value)
        return float(default)

    crpix1 = _meta_float("CRPIX1", "crpix1", default=nx / 2.0 + 0.5)
    crpix2 = _meta_float("CRPIX2", "crpix2", default=ny / 2.0 + 0.5)
    crval1 = _meta_float("CRVAL1", "crval1")
    crval2 = _meta_float("CRVAL2", "crval2")
    cdelt1 = _meta_float("CDELT1", "cdelt1", default=1.0)
    cdelt2 = _meta_float("CDELT2", "cdelt2", default=1.0)
    xcen = crval1 + cdelt1 * ((nx + 1.0) / 2.0 - crpix1)
    ycen = crval2 + cdelt2 * ((ny + 1.0) / 2.0 - crpix2)
    out["XCEN"] = float(xcen)
    out["YCEN"] = float(ycen)
    out["xcen"] = float(xcen)
    out["ycen"] = float(ycen)
    return out


def _localize_submap_wcs(cropped: Map) -> Map:
    """Re-express a SunPy submap WCS with CRPIX at the array center.

    ``Map.submap`` keeps the parent reference pixel (often outside the cropped
    array) while preserving pixel-to-world transforms.  Many FITS consumers and
    mixed-projection plots expect CRPIX/CRVAL to describe the cropped array
    itself; updating those two keys leaves the transform unchanged.
    """
    ny, nx = np.asarray(cropped.data).shape
    crpix1 = nx / 2.0 + 0.5
    crpix2 = ny / 2.0 + 0.5
    center_px = ((nx - 1) / 2.0) * u.pix
    center_py = ((ny - 1) / 2.0) * u.pix
    ref_world = cropped.pixel_to_world(center_px, center_py)
    meta = dict(getattr(cropped, "meta", {}) or {})
    meta["CRPIX1"] = float(crpix1)
    meta["CRPIX2"] = float(crpix2)
    meta["CRVAL1"] = float(ref_world.Tx.to_value(u.arcsec))
    meta["CRVAL2"] = float(ref_world.Ty.to_value(u.arcsec))
    meta["crpix1"] = meta["CRPIX1"]
    meta["crpix2"] = meta["CRPIX2"]
    meta["crval1"] = meta["CRVAL1"]
    meta["crval2"] = meta["CRVAL2"]
    return Map(cropped.data, meta)


def _preserve_map_metadata(cropped: Map, original: Map) -> Map:
    """Keep instrument and observer metadata from the source map on the crop."""
    meta = dict(getattr(original, "meta", {}) or {})
    cropped_meta = dict(getattr(cropped, "meta", {}) or {})
    for key in _HEADER_PRESERVE_KEYS:
        value = meta.get(key)
        if value is not None and key not in cropped_meta:
            cropped_meta[key] = value
    cropped.meta.update(cropped_meta)
    cropped.meta["PYALIGN"] = False
    return cropped


def _finalize_cropped_submap(cropped: Map, source_map: Map, *, roll_baked_in: bool = False) -> Map:
    """Apply SunPy submap post-processing: localize WCS, keep instrument meta."""
    cropped = _localize_submap_wcs(cropped)
    cropped = _preserve_map_metadata(cropped, source_map)
    ny, nx = np.asarray(cropped.data).shape
    meta = _sync_legacy_idl_center_keywords(dict(cropped.meta), nx=nx, ny=ny)
    meta = _strip_stale_parent_wcs_keywords(meta)
    if roll_baked_in:
        for key in ("CROTA", "CROTA2", "crota", "crota2"):
            if key in meta:
                meta[key] = 0.0
    return Map(cropped.data, meta)


def crop_fov_xy_from_inscribing_box(
    inscribing_box: dict[str, float],
    pad: float = 1.1,
) -> dict[str, float]:
    """2D crop FOV rectangle parallel to the inscribing box xy footprint, scaled by ``pad``."""
    footprint = {
        "xc_arcsec": float(inscribing_box["xc_arcsec"]),
        "yc_arcsec": float(inscribing_box["yc_arcsec"]),
        "xsize_arcsec": float(inscribing_box["xsize_arcsec"]),
        "ysize_arcsec": float(inscribing_box["ysize_arcsec"]),
    }
    return apply_pad_factor_to_fov(footprint, pad)


def rotate_refmap_for_display(smap: Map) -> Map:
    """Rotate a map by P-angle (CROTA2) so solar north aligns with +y.

    Failures propagate. Callers that clear CROTA/CROTA2 after baking roll into
    the pixels (``roll_baked_in=True``) must only do so when this returns.
    """
    data = np.asarray(smap.data)
    if np.issubdtype(data.dtype, np.integer):
        fill = False if data.dtype == np.bool_ else 0
        return smap.rotate(order=3, missing=fill, clip=False)
    return smap.rotate(order=3)


def crop_refmap_to_model_box_after_pangle_rotation(
    smap: Map,
    box_corners_world: SkyCoord,
    *,
    pad: float = 1.1,
    model_obstime: str | Time | None = None,
) -> RefmapCropResult:
    """Crop a P-angle-rotated reference map using the inscribing FOV box footprint.

    The crop rectangle is axis-parallel to the 2D xy projection of the inscribing
    FOV box and larger by ``pad`` in all directions (default 1.1 = 10%).
    """
    if box_corners_world is None:
        raise ValueError("box_corners_world is required")

    map_observer = getattr(smap, "observer_coordinate", None)
    map_obstime = getattr(smap, "date", None)
    if map_observer is None or map_obstime is None:
        raise ValueError("reference map is missing observer metadata or DATE-OBS")

    if model_obstime is None:
        model_obstime = infer_model_obstime_from_box_corners(box_corners_world)
    if model_obstime is None:
        model_obstime = map_obstime
    model_obstime = Time(model_obstime)
    map_obstime = Time(map_obstime)

    display_map = rotate_refmap_for_display(smap)
    inscribing_box = compute_inscribing_fov_box_for_observer(
        box_corners_world,
        observer=map_observer,
        obstime=map_obstime,
    )
    if inscribing_box is None:
        raise ValueError("could not compute inscribing FOV box for reference map")

    crop_fov = crop_fov_xy_from_inscribing_box(inscribing_box, pad)
    bottom_left, top_right = fov_crop_sky_coords(
        crop_fov,
        observer=map_observer,
        obstime=map_obstime,
    )
    cropped = _submap_with_fov_safe(display_map, bottom_left, top_right)
    cropped = _finalize_cropped_submap(cropped, display_map, roll_baked_in=True)

    return RefmapCropResult(
        cropped_map=cropped,
        crop_fov=crop_fov,
        map_observer=map_observer,
        map_obstime=map_obstime,
        model_obstime=model_obstime,
        box_corners_world=box_corners_world,
        pad_factor=float(pad),
    )


def crop_refmap_to_model_box(
    smap: Map,
    box_corners_world: SkyCoord,
    *,
    pad: float = 1.1,
    model_obstime: str | Time | None = None,
) -> RefmapCropResult:
    """Crop a reference map to a padded inscribing FOV around the model red box.

    The crop is performed in the map's native observer frame at the map
    observation time. The map pixels and WCS are not rotated or reprojected.
    """
    if box_corners_world is None:
        raise ValueError("box_corners_world is required")

    map_observer = getattr(smap, "observer_coordinate", None)
    map_obstime = getattr(smap, "date", None)
    if map_observer is None or map_obstime is None:
        raise ValueError("reference map is missing observer metadata or DATE-OBS")

    if model_obstime is None:
        model_obstime = infer_model_obstime_from_box_corners(box_corners_world)
    if model_obstime is None:
        model_obstime = map_obstime
    model_obstime = Time(model_obstime)
    map_obstime = Time(map_obstime)

    crop_fov = compute_crop_fov_for_observer(
        box_corners_world,
        observer=map_observer,
        obstime=map_obstime,
        pad=pad,
    )
    if crop_fov is None:
        raise ValueError("could not compute inscribing crop FOV for reference map")

    bottom_left, top_right = fov_crop_sky_coords(
        crop_fov,
        observer=map_observer,
        obstime=map_obstime,
    )
    cropped = _submap_with_fov_safe(smap, bottom_left, top_right)
    cropped = _finalize_cropped_submap(cropped, smap)

    return RefmapCropResult(
        cropped_map=cropped,
        crop_fov=crop_fov,
        map_observer=map_observer,
        map_obstime=map_obstime,
        model_obstime=model_obstime,
        box_corners_world=box_corners_world,
        pad_factor=float(pad),
    )


def project_box_edges_to_observer_hpc(
    box_corners_world: SkyCoord,
    *,
    observer,
    obstime,
) -> list[SkyCoord]:
    """Project red-box wireframe edges into an observer helioprojective frame."""
    edges = project_coordinate_edges_to_observer_hpc(
        box_corners_world,
        edge_pairs=_BOX_EDGE_INDEX_PAIRS,
        observer=observer,
        obstime=obstime,
    )
    return edges or []


def fov_rectangle_corners_hpc(
    fov: dict[str, float],
    *,
    observer,
    obstime,
) -> SkyCoord | None:
    """Return four rectangle corners (closed loop available via first point repeat)."""
    corners = observer_rectangle_to_hpc_corners(
        xc_arcsec=float(fov["xc_arcsec"]),
        yc_arcsec=float(fov["yc_arcsec"]),
        xsize_arcsec=float(fov["xsize_arcsec"]),
        ysize_arcsec=float(fov["ysize_arcsec"]),
        observer=observer,
        obstime=obstime,
    )
    if corners is None:
        return None
    try:
        return SkyCoord([corners[0], corners[1], corners[3], corners[2], corners[0]])
    except Exception:
        return corners


def project_fov_between_observers(
    fov: dict[str, float],
    *,
    source_observer,
    source_obstime,
    target_observer,
    target_obstime,
) -> dict[str, float] | None:
    """Project an observer-aligned FOV rectangle between two observers.

    Mirrors :meth:`pyampp.gxbox.box_view2d.MapBoxDisplayWidget._project_fov_between_observers`.
    """
    base_corners = observer_rectangle_to_hpc_corners(
        xc_arcsec=float(fov["xc_arcsec"]),
        yc_arcsec=float(fov["yc_arcsec"]),
        xsize_arcsec=float(fov["xsize_arcsec"]),
        ysize_arcsec=float(fov["ysize_arcsec"]),
        observer=source_observer,
        obstime=source_obstime,
    )
    if base_corners is None:
        return None
    target_frame = Helioprojective(observer=target_observer, obstime=target_obstime)
    try:
        # Match gxbox-view2d: do not wrap in SphericalScreen for FOV corner projection.
        projected = base_corners.transform_to(target_frame)
        tx = np.asarray(projected.Tx.to_value(u.arcsec), dtype=float).ravel()
        ty = np.asarray(projected.Ty.to_value(u.arcsec), dtype=float).ravel()
        finite = np.isfinite(tx) & np.isfinite(ty)
        if not np.any(finite):
            return None
        tx = tx[finite]
        ty = ty[finite]
        xmin, xmax = float(np.min(tx)), float(np.max(tx))
        ymin, ymax = float(np.min(ty)), float(np.max(ty))
        return {
            "xc_arcsec": 0.5 * (xmin + xmax),
            "yc_arcsec": 0.5 * (ymin + ymax),
            "xsize_arcsec": max(xmax - xmin, 4.0),
            "ysize_arcsec": max(ymax - ymin, 4.0),
            "xmin_arcsec": xmin,
            "xmax_arcsec": xmax,
            "ymin_arcsec": ymin,
            "ymax_arcsec": ymax,
        }
    except Exception:
        return None


def display_observer_reproject_header_for_fov(
    smap: Map,
    *,
    observer,
    obstime,
    fov: dict[str, float],
):
    """Build a SunPy ROI header for cross-observer display reprojection.

    Mirrors
    :meth:`pyampp.gxbox.box_view2d.MapBoxDisplayWidget._display_observer_reproject_header_for_selection`.
    """
    try:
        scale_x = abs(float(smap.scale.axis1.to_value(u.arcsec / u.pix)))
        scale_y = abs(float(smap.scale.axis2.to_value(u.arcsec / u.pix)))
    except Exception:
        return None
    if not (np.isfinite(scale_x) and np.isfinite(scale_y) and scale_x > 0 and scale_y > 0):
        return None
    width = max(float(fov["xsize_arcsec"]), 4.0)
    height = max(float(fov["ysize_arcsec"]), 4.0)
    nx = max(32, int(np.ceil(width / scale_x)))
    ny = max(32, int(np.ceil(height / scale_y)))
    target_center = SkyCoord(
        Tx=float(fov["xc_arcsec"]) * u.arcsec,
        Ty=float(fov["yc_arcsec"]) * u.arcsec,
        frame=Helioprojective(observer=observer, obstime=obstime),
    )
    header = make_fitswcs_header(
        np.empty((ny, nx), dtype=np.float32),
        target_center,
        scale=u.Quantity([scale_x, scale_y], u.arcsec / u.pix),
    )
    try:
        header["rsun_ref"] = float(smap.rsun_meters.to_value(u.m))
    except Exception:
        pass
    return header


def project_inscribing_xy_to_observer(
    inscribing_box: dict[str, float],
    *,
    source_observer,
    source_obstime,
    target_observer,
    target_obstime,
) -> dict[str, float] | None:
    """Project the unpadded 2D xy footprint of an inscribing FOV box."""
    xy_fov = {
        "xc_arcsec": float(inscribing_box["xc_arcsec"]),
        "yc_arcsec": float(inscribing_box["yc_arcsec"]),
        "xsize_arcsec": float(inscribing_box["xsize_arcsec"]),
        "ysize_arcsec": float(inscribing_box["ysize_arcsec"]),
    }
    return project_fov_between_observers(
        xy_fov,
        source_observer=source_observer,
        source_obstime=source_obstime,
        target_observer=target_observer,
        target_obstime=target_obstime,
    )


def project_padded_inscribing_xy_to_observer(
    inscribing_box: dict[str, float],
    *,
    source_observer,
    source_obstime,
    target_observer,
    target_obstime,
    pad: float = 1.1,
) -> dict[str, float] | None:
    """Project an inscribing xy footprint then apply ``pad`` in the target frame.

    This matches the source-frame relationship ``pad(unpadded_xy)`` used for
    native crops, while using the gxbox-view2d cross-observer projection path.
    """
    projected = project_inscribing_xy_to_observer(
        inscribing_box,
        source_observer=source_observer,
        source_obstime=source_obstime,
        target_observer=target_observer,
        target_obstime=target_obstime,
    )
    if projected is None:
        return None
    return apply_pad_factor_to_fov(projected, pad)


def reproject_map_to_target_observer_fov(
    smap: Map,
    *,
    target_fov: dict[str, float],
    target_observer,
    target_obstime,
    algorithm: str = "adaptive",
    mask_off_limb: bool = True,
) -> Map:
    """Reproject ``smap`` into a precomputed target-observer ROI."""
    header = display_observer_reproject_header_for_fov(
        smap,
        observer=target_observer,
        obstime=target_obstime,
        fov=target_fov,
    )
    if header is None:
        return smap
    try:
        ny = int(header["NAXIS2"])
        nx = int(header["NAXIS1"])
        canvas = Map(np.full((ny, nx), np.nan, dtype=float), header)
        reprojected = reproject_map_onto_canvas(smap, canvas, algorithm=algorithm)
        if mask_off_limb:
            reprojected = mask_pixels_not_visible_from_source(smap, reprojected)
        return _copy_plot_settings(reprojected, smap)
    except Exception:
        return smap


def project_rectangle_corners_between_observers(
    fov: dict[str, float],
    *,
    source_observer,
    source_obstime,
    target_observer,
    target_obstime,
) -> SkyCoord | None:
    """Reproject four rectangle corners between observers (gxbox corner path)."""
    corners = observer_rectangle_to_hpc_corners(
        xc_arcsec=float(fov["xc_arcsec"]),
        yc_arcsec=float(fov["yc_arcsec"]),
        xsize_arcsec=float(fov["xsize_arcsec"]),
        ysize_arcsec=float(fov["ysize_arcsec"]),
        observer=source_observer,
        obstime=source_obstime,
    )
    if corners is None:
        return None
    target_frame = Helioprojective(observer=target_observer, obstime=target_obstime)
    try:
        with _spherical_screen_context_for_observer(target_observer):
            projected = corners.transform_to(target_frame)
        tx = np.asarray(projected.Tx.to_value(u.arcsec), dtype=float).ravel()
        ty = np.asarray(projected.Ty.to_value(u.arcsec), dtype=float).ravel()
        if tx.size < 4 or ty.size < 4:
            return None
        finite = np.isfinite(tx[:4]) & np.isfinite(ty[:4])
        if np.count_nonzero(finite) < 3:
            return None
        loop = SkyCoord(
            Tx=np.asarray([tx[0], tx[1], tx[3], tx[2], tx[0]], dtype=float) * u.arcsec,
            Ty=np.asarray([ty[0], ty[1], ty[3], ty[2], ty[0]], dtype=float) * u.arcsec,
            frame=target_frame,
        )
        return loop
    except Exception:
        return None


def project_padded_crop_bottom_face_corners_between_observers(
    inscribing_box: dict[str, float],
    *,
    pad: float,
    source_observer,
    source_obstime,
    target_observer,
    target_obstime,
) -> SkyCoord | None:
    """Reproject the padded crop bottom-face quad between observers.

    Uses the 3D bottom face (``z=zmin``) of the inscribing FOV box so the
    reprojected outline matches SunPy map pixels and the blue-box bottom face.
    """
    xy = apply_pad_factor_to_fov(
        {
            "xc_arcsec": float(inscribing_box["xc_arcsec"]),
            "yc_arcsec": float(inscribing_box["yc_arcsec"]),
            "xsize_arcsec": float(inscribing_box["xsize_arcsec"]),
            "ysize_arcsec": float(inscribing_box["ysize_arcsec"]),
        },
        pad,
    )
    try:
        dsun_mm = float(source_observer.radius.to_value(u.Mm))
    except Exception:
        return None
    zmin = float(inscribing_box["zmin_mm"])
    half_w = 0.5 * max(float(xy["xsize_arcsec"]), 1e-6)
    half_h = 0.5 * max(float(xy["ysize_arcsec"]), 1e-6)
    xc = float(xy["xc_arcsec"])
    yc = float(xy["yc_arcsec"])
    source_frame = Helioprojective(observer=source_observer, obstime=source_obstime)
    points = np.asarray(
        [
            [xc - half_w, yc - half_h],
            [xc + half_w, yc - half_h],
            [xc + half_w, yc + half_h],
            [xc - half_w, yc + half_h],
        ],
        dtype=float,
    )
    distance = (dsun_mm - zmin) * u.Mm
    corners = SkyCoord(
        Tx=points[:, 0] * u.arcsec,
        Ty=points[:, 1] * u.arcsec,
        distance=np.full(4, distance.value) * distance.unit,
        frame=source_frame,
    )
    target_frame = Helioprojective(observer=target_observer, obstime=target_obstime)
    try:
        with _spherical_screen_context_for_observer(target_observer):
            projected = corners.transform_to(target_frame)
        tx = np.asarray(projected.Tx.to_value(u.arcsec), dtype=float).ravel()
        ty = np.asarray(projected.Ty.to_value(u.arcsec), dtype=float).ravel()
        if tx.size < 4 or ty.size < 4:
            return None
        loop = SkyCoord(
            Tx=np.asarray([tx[0], tx[1], tx[2], tx[3], tx[0]], dtype=float) * u.arcsec,
            Ty=np.asarray([ty[0], ty[1], ty[2], ty[3], ty[0]], dtype=float) * u.arcsec,
            frame=target_frame,
        )
        return loop
    except Exception:
        return None


def reproject_map_to_display_observer_fov(
    smap: Map,
    *,
    source_fov: dict[str, float],
    source_observer,
    source_obstime,
    target_observer,
    target_obstime,
) -> tuple[Map, dict[str, float] | None]:
    """Reproject a native-LOS crop into a target observer ROI.

    This is the inverse of the gxbox-view2d path that projects an Earth crop
    into a spacecraft display frame: project the source FOV, build an ROI header
    at the target observer, then call ``Map.reproject_to``.
    """
    projected_fov = project_fov_between_observers(
        source_fov,
        source_observer=source_observer,
        source_obstime=source_obstime,
        target_observer=target_observer,
        target_obstime=target_obstime,
    )
    if projected_fov is None:
        return smap, None
    return (
        reproject_map_to_target_observer_fov(
            smap,
            target_fov=projected_fov,
            target_observer=target_observer,
            target_obstime=target_obstime,
        ),
        projected_fov,
    )


def _copy_plot_settings(target: Map, source: Map) -> Map:
    """Keep SunPy display defaults (e.g. ``euvi284``) when building derived maps."""
    source_settings = getattr(source, "plot_settings", None) or {}
    if source_settings:
        if not hasattr(target, "plot_settings") or target.plot_settings is None:
            target.plot_settings = {}
        target.plot_settings.update(source_settings)
    return target


def make_empty_observer_fov_map(
    reference_smap: Map,
    *,
    observer,
    obstime,
    fov: dict[str, float],
    fill: float = np.nan,
) -> Map:
    """Return an observer-aligned FOV map with uniform ``fill`` data."""
    try:
        header = display_observer_reproject_header_for_fov(
            reference_smap,
            observer=observer,
            obstime=obstime,
            fov=fov,
        )
        if header is None:
            raise ValueError("could not build observer FOV header")
        ny = int(header["NAXIS2"])
        nx = int(header["NAXIS1"])
        return Map(np.full((ny, nx), fill, dtype=float), header)
    except Exception as exc:
        raise ValueError("could not build empty observer FOV map") from exc


def full_disk_fov_for_map(smap: Map, *, pad: float = 1.05) -> dict[str, float]:
    """Axis-aligned full-disk FOV centered on helioprojective origin."""
    try:
        rsun_m = float(smap.rsun_meters.to_value(u.m))
        dsun_m = float(smap.dsun.to_value(u.m))
        if np.isfinite(rsun_m) and np.isfinite(dsun_m) and dsun_m > 0:
            rsun_arcsec = float(np.degrees(np.arcsin(min(1.0, rsun_m / dsun_m))) * 3600.0)
        else:
            rsun_arcsec = 960.0
    except Exception:
        rsun_arcsec = 960.0
    side = 2.0 * float(rsun_arcsec) * max(float(pad), 1.0)
    half = 0.5 * side
    return {
        "xc_arcsec": 0.0,
        "yc_arcsec": 0.0,
        "xsize_arcsec": side,
        "ysize_arcsec": side,
        "xmin_arcsec": -half,
        "xmax_arcsec": half,
        "ymin_arcsec": -half,
        "ymax_arcsec": half,
    }


def make_full_disk_observer_canvas(
    reference_smap: Map,
    *,
    observer,
    obstime,
    pad: float = 1.05,
) -> Map:
    """Empty full-disk WCS canvas for ``observer`` at ``obstime``."""
    return make_empty_observer_fov_map(
        reference_smap,
        observer=observer,
        obstime=obstime,
        fov=full_disk_fov_for_map(reference_smap, pad=pad),
    )


def reproject_map_onto_canvas(
    smap: Map,
    canvas: Map,
    *,
    algorithm: str = "adaptive",
) -> Map:
    """Reproject map pixels onto a pre-built observer canvas (IDL-style workflow)."""
    for algo in (algorithm, "interpolation", "adaptive", "exact"):
        try:
            reprojected = smap.reproject_to(
                canvas.wcs,
                algorithm=algo,
                roundtrip_coords=False,
            )
            data = np.asarray(reprojected.data, dtype=float)
            if np.any(np.isfinite(data)):
                return _copy_plot_settings(Map(data, canvas.meta), smap)
        except Exception:
            continue
    return canvas


def mask_pixels_not_visible_from_source(source_map: Map, display_map: Map) -> Map:
    """Mask display pixels that map off-disk in ``source_map`` observer frame.

    This is used to prevent off-limb interpolation artifacts when displaying a
    reprojected map in a different observer frame.
    """
    try:
        rsun_m = float(source_map.rsun_meters.to_value(u.m))
        dsun_m = float(source_map.dsun.to_value(u.m))
        if np.isfinite(rsun_m) and np.isfinite(dsun_m) and dsun_m > 0:
            rsun_arcsec = float(np.degrees(np.arcsin(min(1.0, rsun_m / dsun_m))) * 3600.0)
        else:
            rsun_arcsec = 960.0
    except Exception:
        rsun_arcsec = 960.0

    data = np.asarray(display_map.data, dtype=float)
    ny, nx = data.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    try:
        display_coords = display_map.pixel_to_world(xs * u.pix, ys * u.pix)
        source_frame = source_map.coordinate_frame
        source_observer = getattr(source_frame, "observer", None)
        with _spherical_screen_context_for_observer(source_observer):
            source_coords = display_coords.transform_to(source_frame)
        tx = np.asarray(source_coords.Tx.to_value(u.arcsec), dtype=float)
        ty = np.asarray(source_coords.Ty.to_value(u.arcsec), dtype=float)
    except Exception:
        return display_map

    radius = np.sqrt(tx**2 + ty**2)
    masked = data.copy()
    invalid = (~np.isfinite(radius)) | (radius > float(rsun_arcsec) * 1.02)
    masked[invalid] = np.nan
    out = Map(masked, display_map.meta)
    return _copy_plot_settings(out, source_map)


def reproject_refmap_to_observer(
    smap: Map,
    *,
    observer,
    obstime: str | Time | None = None,
    reference_smap: Map | None = None,
    fov: dict[str, float] | None = None,
    full_disk_pad: float = 1.05,
    algorithm: str = "adaptive",
    mask_off_limb: bool = True,
) -> Map:
    """Reproject a refmap to a target observer/time with optional source-limb masking.

    Parameters
    ----------
    smap : Map
        Input map to reproject (cropped or full disk).
    observer : Any
        Target observer coordinate (Earth or spacecraft).
    obstime : str | Time | None
        Target time. Defaults to ``smap.date``.
    reference_smap : Map | None
        Map used to build output canvas scale/FOV. Defaults to ``smap``.
    fov : dict[str, float] | None
        Explicit output FOV dict. If omitted, uses full disk from
        ``reference_smap`` with ``full_disk_pad``.
    full_disk_pad : float
        Padding factor for implicit full-disk canvas.
    algorithm : str
        Reprojection algorithm preference.
    mask_off_limb : bool
        If True, masks pixels not visible from the source observer after
        reprojection.
    """
    target_obstime = Time(obstime) if obstime is not None else Time(getattr(smap, "date", Time.now()))
    reference = reference_smap or smap
    target_fov = fov or full_disk_fov_for_map(reference, pad=full_disk_pad)

    canvas = make_empty_observer_fov_map(
        reference,
        observer=observer,
        obstime=target_obstime,
        fov=target_fov,
    )
    reprojected = reproject_map_onto_canvas(smap, canvas, algorithm=algorithm)
    if mask_off_limb:
        reprojected = mask_pixels_not_visible_from_source(smap, reprojected)
    return _copy_plot_settings(reprojected, smap)


def composite_map_onto_canvas(
    canvas: Map,
    patch: Map,
    *,
    positive_only: bool = True,
) -> Map:
    """Paint ``patch`` pixels onto ``canvas`` WCS without resampling the patch.

    Use this when ``patch`` already lives in the same observer frame as
    ``canvas`` (native crop on a full-disk grid).  Each finite patch pixel is
    mapped through world coordinates into canvas pixel indices.
    """
    out = np.full(np.asarray(canvas.data).shape, np.nan, dtype=float)
    patch_data = np.asarray(patch.data, dtype=float)
    ny, nx = patch_data.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    finite = np.isfinite(patch_data)
    if positive_only:
        finite &= patch_data > 0
    if not np.any(finite):
        return _copy_plot_settings(Map(out, canvas.meta), patch)

    coords = patch.pixel_to_world(xs[finite] * u.pix, ys[finite] * u.pix)
    observer = getattr(canvas.coordinate_frame, "observer", None)
    with _spherical_screen_context_for_observer(observer):
        cx, cy = canvas.world_to_pixel(coords)
    cx = np.asarray(cx.to_value(u.pix) if hasattr(cx, "to_value") else cx, dtype=float)
    cy = np.asarray(cy.to_value(u.pix) if hasattr(cy, "to_value") else cy, dtype=float)
    ix = np.rint(cx).astype(int)
    iy = np.rint(cy).astype(int)
    in_bounds = (ix >= 0) & (ix < out.shape[1]) & (iy >= 0) & (iy < out.shape[0])
    out[iy[in_bounds], ix[in_bounds]] = patch_data[finite][in_bounds]
    return _copy_plot_settings(Map(out, canvas.meta), patch)


def reproject_map_to_observer_fov(
    smap: Map,
    *,
    observer,
    obstime,
    fov: dict[str, float],
) -> Map:
    """Reproject ``smap`` into ``observer``/``obstime`` centered on ``fov``."""
    try:
        header = display_observer_reproject_header_for_fov(
            smap,
            observer=observer,
            obstime=obstime,
            fov=fov,
        )
        if header is None:
            return smap
        return smap.reproject_to(header, algorithm="adaptive", roundtrip_coords=False)
    except Exception:
        return smap


def reproject_map_corner_bounds_to_observer(
    smap: Map,
    *,
    observer,
    obstime,
    crop_fov: dict[str, float] | None = None,
    source_observer=None,
    source_obstime=None,
) -> Map:
    """Reproject ``smap`` using the gxbox-view2d cross-observer ROI workflow."""
    if (
        crop_fov is not None
        and source_observer is not None
        and source_obstime is not None
    ):
        reprojected, _projected_fov = reproject_map_to_display_observer_fov(
            smap,
            source_fov=crop_fov,
            source_observer=source_observer,
            source_obstime=source_obstime,
            target_observer=observer,
            target_obstime=obstime,
        )
        return reprojected

    corners = _map_pixel_corners_hpc(smap)
    if corners is None:
        return smap

    target_frame = Helioprojective(observer=observer, obstime=obstime)
    with _spherical_screen_context_for_observer(observer):
        projected = corners.transform_to(target_frame)
    tx = np.asarray(projected.Tx.to_value(u.arcsec), dtype=float)
    ty = np.asarray(projected.Ty.to_value(u.arcsec), dtype=float)
    finite = np.isfinite(tx) & np.isfinite(ty)
    if np.count_nonzero(finite) < 2:
        return smap
    tx = tx[finite]
    ty = ty[finite]
    xmin, xmax = float(np.min(tx)), float(np.max(tx))
    ymin, ymax = float(np.min(ty)), float(np.max(ty))
    fov = {
        "xc_arcsec": 0.5 * (xmin + xmax),
        "yc_arcsec": 0.5 * (ymin + ymax),
        "xsize_arcsec": max(xmax - xmin, 1e-6),
        "ysize_arcsec": max(ymax - ymin, 1e-6),
        "xmin_arcsec": xmin,
        "xmax_arcsec": xmax,
        "ymin_arcsec": ymin,
        "ymax_arcsec": ymax,
    }
    return reproject_map_to_observer_fov(smap, observer=observer, obstime=obstime, fov=fov)


def _map_pixel_corners_hpc(smap: Map) -> SkyCoord | None:
    ny, nx = np.asarray(smap.data).shape
    if nx <= 0 or ny <= 0:
        return None
    return smap.pixel_to_world(
        np.array([0.5, nx + 0.5, nx + 0.5, 0.5], dtype=float) * u.pix,
        np.array([0.5, 0.5, ny + 0.5, ny + 0.5], dtype=float) * u.pix,
    )


def crop_fov_boxes_at_model_times(
    box_corners_world: SkyCoord,
    *,
    map_observer,
    model_obstime,
    pad: float = 1.1,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """3D inscribing FOV boxes at model time in map LOS and Earth LOS."""
    map_los = compute_crop_fov_box_for_observer(
        box_corners_world,
        observer=map_observer,
        obstime=model_obstime,
        pad=pad,
    )
    earth_los = compute_crop_fov_box_for_observer(
        box_corners_world,
        observer=earth_observer_at(model_obstime),
        obstime=model_obstime,
        pad=pad,
    )
    return map_los, earth_los


def earth_observer_at(obstime) -> SkyCoord:
    """Convenience wrapper for Earth observer coordinates at ``obstime``."""
    return get_earth(Time(obstime))


def crop_fov_at_model_times(
    box_corners_world: SkyCoord,
    *,
    map_observer,
    model_obstime,
    pad: float = 1.1,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """FOV footprints at model time in map LOS and Earth LOS."""
    map_los = compute_crop_fov_for_observer(
        box_corners_world,
        observer=map_observer,
        obstime=model_obstime,
        pad=pad,
    )
    earth_los = compute_crop_fov_for_observer(
        box_corners_world,
        observer=earth_observer_at(model_obstime),
        obstime=model_obstime,
        pad=pad,
    )
    return map_los, earth_los


# ============================================================================
# UNIFIED SPATIAL-ONLY REFMAP CROP API
# ============================================================================


@dataclass(frozen=True)
class UnifiedCropRequest:
    """Request parameters for unified spatial-only refmap cropping."""

    smap: Map
    box_corners_world: SkyCoord
    model_obstime: str | Time | None = None
    pad: float = 1.1
    pangle_policy: str = "auto"  # "auto" (detect from CROTA2), "always", "never"


@dataclass(frozen=True)
class UnifiedCropResult:
    """Output of unified refmap crop pipeline with full provenance.

    The result is always spatial-only: no temporal transforms are applied.
    Time alignment is deferred to the model.py loader stage (gxrender newTime).
    """

    cropped_map: Map
    crop_fov: dict[str, float]
    map_observer: Any
    map_obstime: Time
    model_obstime: Time
    box_corners_world: SkyCoord
    pad_factor: float
    pangle_rotated: bool  # Whether P-angle rotation was applied
    pangle_value_deg: float | None  # P-angle value (CROTA2) if available
    crop_stage: str  # "spatial_only" - marker for upstream
    time_alignment_handled: bool  # Always False - time handled elsewhere


def crop_refmap_spatial(
    request: UnifiedCropRequest | None = None,
    *,
    smap: Map | None = None,
    box_corners_world: SkyCoord | None = None,
    model_obstime: str | Time | None = None,
    pad: float = 1.1,
    pangle_policy: str = "auto",
) -> UnifiedCropResult:
    """Unified spatial-only refmap crop: P-angle normalization + FOV projection + cyan crop.

    This is the canonical crop routine for all refmap preprocessing workflows.
    Time alignment is handled at the model.py loader stage, not here.

    Parameters
    ----------
    request : UnifiedCropRequest, optional
        Bundle request; if provided, other params are ignored.
    smap : Map
        External reference map (Earth or non-Earth observer).
    box_corners_world : SkyCoord
        Model red-box world corners (8 points, defined at model time).
    model_obstime : str | Time, optional
        Model time. If None, extracted from box_corners_world or map time.
    pad : float
        FOV padding factor (default 1.1 = 10% margin).
    pangle_policy : str
        "auto" (apply if CROTA2 > 0.1°), "always", "never".

    Returns
    -------
    UnifiedCropResult
        Cropped map, FOV, metadata, and pipeline provenance.

    Notes
    -----
    Spatial stages:
        1. P-angle roll normalization (if policy permits and CROTA2 nonzero)
        2. Red-box projection from model frame to map observer frame
        3. Blue-box FOV inscription around red box
        4. Cyan x-y aligned crop to blue-box footprint (padded)

    Time handling:
        - All geometry computations use map observation time (not model time)
        - Model time stored for metadata audit trail only
        - No temporal transformations applied
    """
    # Handle request bundle vs explicit params
    if request is not None:
        smap = request.smap
        box_corners_world = request.box_corners_world
        model_obstime = request.model_obstime
        pad = request.pad
        pangle_policy = request.pangle_policy

    # Validate inputs
    if smap is None or box_corners_world is None:
        raise ValueError("Both smap and box_corners_world are required")

    map_observer = getattr(smap, "observer_coordinate", None)
    map_obstime = getattr(smap, "date", None)
    if map_observer is None or map_obstime is None:
        raise ValueError("Reference map missing observer metadata or DATE-OBS")

    map_obstime = Time(map_obstime)

    # Infer model obstime
    if model_obstime is None:
        model_obstime = infer_model_obstime_from_box_corners(box_corners_world)
    if model_obstime is None:
        model_obstime = map_obstime
    model_obstime = Time(model_obstime)

    # Stage 1: determine whether the legacy P-angle crop path is active.
    pangle_rotated = False
    pangle_value_deg = None
    if pangle_policy in ("auto", "always"):
        try:
            crota2 = float(getattr(smap.meta, "crota2", 0.0) or 0.0)
        except (TypeError, ValueError):
            crota2 = 0.0
        pangle_value_deg = crota2
        pangle_rotated = pangle_policy == "always" or (pangle_policy == "auto" and abs(crota2) > 0.1)

    # Stage 2-4: use the legacy crop chain so the FITS artifact matches the
    # older diagnostic output exactly.
    if pangle_policy == "never":
        legacy_result = crop_refmap_to_model_box(
            smap,
            box_corners_world,
            pad=pad,
            model_obstime=model_obstime,
        )
    else:
        legacy_result = crop_refmap_to_model_box_after_pangle_rotation(
            smap,
            box_corners_world,
            pad=pad,
            model_obstime=model_obstime,
        )

    cropped = legacy_result.cropped_map
    crop_fov = legacy_result.crop_fov

    # Add provenance metadata to header
    if cropped.meta is not None:
        cropped.meta["pyampp_crop_stage"] = "spatial_only"
        cropped.meta["pyampp_crop_pangle_rotated"] = int(pangle_rotated)
        if pangle_value_deg is not None:
            cropped.meta["pyampp_crop_pangle_deg"] = float(pangle_value_deg)
        cropped.meta["pyampp_crop_version"] = "unified.v1"
        cropped.meta["pyampp_time_alignment_handled"] = False

    return UnifiedCropResult(
        cropped_map=cropped,
        crop_fov=crop_fov,
        map_observer=map_observer,
        map_obstime=map_obstime,
        model_obstime=model_obstime,
        box_corners_world=box_corners_world,
        pad_factor=float(pad),
        pangle_rotated=pangle_rotated,
        pangle_value_deg=pangle_value_deg,
        crop_stage="spatial_only",
        time_alignment_handled=False,
    )


# Backward-compat wrappers (delegate to unified function)
def crop_refmap_to_model_box_after_pangle_rotation_compat(
    smap: Map,
    box_corners_world: SkyCoord,
    *,
    pad: float = 1.1,
    model_obstime: str | Time | None = None,
) -> RefmapCropResult:
    """Deprecated: use crop_refmap_spatial(..., pangle_policy='always') instead."""
    result = crop_refmap_spatial(
        smap=smap,
        box_corners_world=box_corners_world,
        model_obstime=model_obstime,
        pad=pad,
        pangle_policy="always",
    )
    return RefmapCropResult(
        cropped_map=result.cropped_map,
        crop_fov=result.crop_fov,
        map_observer=result.map_observer,
        map_obstime=result.map_obstime,
        model_obstime=result.model_obstime,
        box_corners_world=result.box_corners_world,
        pad_factor=result.pad_factor,
    )


def crop_refmap_to_model_box_compat(
    smap: Map,
    box_corners_world: SkyCoord,
    *,
    pad: float = 1.1,
    model_obstime: str | Time | None = None,
) -> RefmapCropResult:
    """Deprecated: use crop_refmap_spatial(..., pangle_policy='never') instead."""
    result = crop_refmap_spatial(
        smap=smap,
        box_corners_world=box_corners_world,
        model_obstime=model_obstime,
        pad=pad,
        pangle_policy="never",
    )
    return RefmapCropResult(
        cropped_map=result.cropped_map,
        crop_fov=result.crop_fov,
        map_observer=result.map_observer,
        map_obstime=result.map_obstime,
        model_obstime=result.model_obstime,
        box_corners_world=result.box_corners_world,
        pad_factor=result.pad_factor,
    )
