"""Diagnostic three-panel plots for :mod:`pyampp.io.refmap_crop`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import HeliographicStonyhurst
from sunpy.map import Map

from pyampp.gxbox.gx_fov2box import _spherical_screen_context_for_observer
from pyampp.io.refmap_crop import (
    RefmapCropResult,
    composite_map_onto_canvas,
    compute_inscribing_fov_box_for_observer,
    crop_refmap_to_model_box_after_pangle_rotation,
    earth_observer_at,
    fov_rectangle_corners_hpc,
    full_disk_fov_for_map,
    make_empty_observer_fov_map,
    plot_inscribing_fov_box_on_axes,
    project_fov_box_edges_reprojected,
    project_fov_box_edges_to_observer_hpc,
    project_model_box_edges_to_observer_hpc,
    project_padded_crop_bottom_face_corners_between_observers,
    project_rectangle_corners_between_observers,
    reproject_refmap_to_observer,
    rotate_refmap_for_display,
)

_FULL_SUN_DISK_PAD = 1.05
_FULL_SUN_OVERLAY_PAD = 1.10
_MODEL_BOX_COLOR = "red"
_INSCRIBING_BOX_COLOR = "blue"
_CROP_FOV_COLOR = "cyan"


def _rsun_arcsec_from_map(smap: Map) -> float | None:
    try:
        rsun_m = float(smap.rsun_meters.to_value(u.m))
        dsun_m = float(smap.dsun.to_value(u.m))
        if np.isfinite(rsun_m) and np.isfinite(dsun_m) and dsun_m > 0:
            return float(np.degrees(np.arcsin(min(1.0, rsun_m / dsun_m))) * 3600.0)
    except Exception:
        pass
    return None


def _full_disk_fov(smap: Map, *, pad: float = _FULL_SUN_DISK_PAD) -> dict[str, float]:
    return full_disk_fov_for_map(smap, pad=pad)


def _disk_center_pixel(smap: Map) -> tuple[float, float]:
    frame = smap.coordinate_frame
    observer = getattr(frame, "observer", None)
    origin = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frame)
    try:
        with _spherical_screen_context_for_observer(observer):
            xpix, ypix = smap.world_to_pixel(origin)
        return (
            float(xpix.to_value(u.pix) if hasattr(xpix, "to_value") else xpix),
            float(ypix.to_value(u.pix) if hasattr(ypix, "to_value") else ypix),
        )
    except Exception:
        ref = smap.reference_pixel
        return float(ref.x), float(ref.y)


def _scene_overlay_pixel_coords(
    display_map: Map,
    scene: dict[str, Any],
    *,
    crop_corners: SkyCoord | None = None,
    crop_pixels: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    observer = scene["observer"]
    obstime = scene["obstime"]
    geometry_observer = scene.get("geometry_observer")
    geometry_obstime = scene.get("geometry_obstime")

    def _add_coords(coords: SkyCoord | None) -> None:
        pixels = _pixel_coords(display_map, coords)
        if pixels is None:
            return
        xpix, ypix = pixels
        finite = np.isfinite(xpix) & np.isfinite(ypix)
        xs.extend(xpix[finite].tolist())
        ys.extend(ypix[finite].tolist())

    for edge in project_model_box_edges_to_observer_hpc(
        scene["box_corners_world"],
        observer=observer,
        obstime=obstime,
    ):
        _add_coords(edge)

    inscribing_box = scene.get("inscribing_box")
    if inscribing_box is not None:
        if geometry_observer is not None and geometry_obstime is not None:
            fov_edges = project_fov_box_edges_reprojected(
                inscribing_box,
                source_observer=geometry_observer,
                source_obstime=geometry_obstime,
                target_observer=observer,
                target_obstime=obstime,
            )
            for edge in fov_edges:
                _add_coords(edge)
        else:
            for edge in project_fov_box_edges_to_observer_hpc(
                inscribing_box,
                observer=observer,
                obstime=obstime,
            ):
                _add_coords(edge)

    corners = crop_corners
    if corners is None and geometry_observer is not None and geometry_obstime is not None:
        corners = _crop_fov_corners_in_display_frame(
            scene["crop_fov"],
            source_observer=geometry_observer,
            source_obstime=geometry_obstime,
            display_map=display_map,
        )
    if corners is None:
        corners = scene.get("crop_corners")
    _add_coords(corners)

    if crop_pixels is not None:
        xpix, ypix = crop_pixels
        finite = np.isfinite(xpix) & np.isfinite(ypix)
        xs.extend(xpix[finite].tolist())
        ys.extend(ypix[finite].tolist())

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _transform_coords_to_map_frame(smap: Map, coords: SkyCoord) -> SkyCoord:
    frame = smap.coordinate_frame
    observer = getattr(frame, "observer", None)
    with _spherical_screen_context_for_observer(observer):
        return coords.transform_to(frame)


def _pixel_coords(smap: Map, coords: SkyCoord | None) -> tuple[np.ndarray, np.ndarray] | None:
    if coords is None:
        return None
    try:
        mapped = _transform_coords_to_map_frame(smap, coords)
        xpix, ypix = smap.world_to_pixel(mapped)
        x = np.asarray(xpix.to_value(u.pix) if hasattr(xpix, "to_value") else xpix, dtype=float)
        y = np.asarray(ypix.to_value(u.pix) if hasattr(ypix, "to_value") else ypix, dtype=float)
    except Exception:
        return None
    return x, y


def _plot_polyline_pixels(
    ax,
    pixels: tuple[np.ndarray, np.ndarray] | None,
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 1.4,
    zorder: int = 6,
) -> None:
    if pixels is None:
        return
    xpix, ypix = pixels
    finite = np.isfinite(xpix) & np.isfinite(ypix)
    if np.count_nonzero(finite) < 2:
        return
    ax.plot(xpix[finite], ypix[finite], color=color, linewidth=linewidth, label=label, zorder=zorder)


def _plot_coord_polyline(
    ax,
    coords: SkyCoord | None,
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 1.4,
    zorder: int = 6,
) -> bool:
    if coords is None:
        return False
    plot_coord = getattr(ax, "plot_coord", None)
    if plot_coord is None:
        return False
    try:
        plot_coord(coords, color=color, linewidth=linewidth, label=label, zorder=zorder)
        return True
    except Exception:
        return False


def _plot_edge_pixels(
    ax,
    display_map: Map,
    edge: SkyCoord,
    *,
    color: str,
    zorder: int = 6,
    use_plot_coord: bool = False,
) -> None:
    if use_plot_coord and _plot_coord_polyline(ax, edge, color=color, linewidth=1.4, zorder=zorder):
        return
    _plot_polyline_pixels(ax, _pixel_coords(display_map, edge), color=color, zorder=zorder)


def _plot_closed_sky(
    ax,
    display_map: Map,
    coords: SkyCoord | None,
    *,
    color: str,
    label: str | None = None,
    use_plot_coord: bool = False,
) -> None:
    if use_plot_coord and _plot_coord_polyline(
        ax,
        coords,
        color=color,
        label=label,
        linewidth=1.6,
        zorder=6,
    ):
        return
    _plot_polyline_pixels(ax, _pixel_coords(display_map, coords), color=color, label=label, linewidth=1.6)


def _data_footprint_pixels(smap: Map, *, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray] | None:
    data = np.asarray(smap.data, dtype=float)
    mask = np.isfinite(data) & (data > threshold)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    x0, x1 = float(xs.min()) + 0.5, float(xs.max()) + 1.5
    y0, y1 = float(ys.min()) + 0.5, float(ys.max()) + 1.5
    return (
        np.array([x0, x1, x1, x0, x0], dtype=float),
        np.array([y0, y0, y1, y1, y0], dtype=float),
    )


def _map_pixel_boundary_coords(smap: Map) -> SkyCoord:
    ny, nx = smap.data.shape
    return smap.pixel_to_world(
        np.array([0.5, nx + 0.5, nx + 0.5, 0.5, 0.5]) * u.pix,
        np.array([0.5, 0.5, ny + 0.5, ny + 0.5, 0.5]) * u.pix,
    )


def _crop_boundary_pixels_on_display(
    crop_map: Map,
    display_map: Map,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pixel outline of ``crop_map`` projected onto ``display_map`` axes."""
    corners = _map_pixel_boundary_coords(crop_map)
    mapped = _transform_coords_to_map_frame(display_map, corners)
    return _pixel_coords(display_map, mapped)


def _crop_fov_corners_in_display_frame(
    crop_fov: dict[str, float],
    *,
    source_observer,
    source_obstime,
    display_map: Map,
) -> SkyCoord | None:
    frame = display_map.coordinate_frame
    target_observer = getattr(frame, "observer", None)
    target_obstime = getattr(frame, "obstime", None) or getattr(display_map, "date", None)
    if target_observer is None or target_obstime is None:
        return None
    return project_rectangle_corners_between_observers(
        crop_fov,
        source_observer=source_observer,
        source_obstime=source_obstime,
        target_observer=target_observer,
        target_obstime=target_obstime,
    )


def _reference_frame_legend_labels(display_map: Map, obstime) -> list[str]:
    labels: list[str] = []
    time_value = obstime if obstime is not None else getattr(display_map, "date", None)
    if time_value is not None:
        try:
            labels.append(f"t={Time(time_value).isot}")
        except Exception:
            pass
    try:
        frame = display_map.coordinate_frame
        observer = getattr(frame, "observer", None)
        if observer is not None and time_value is not None:
            hgs = observer.transform_to(HeliographicStonyhurst(obstime=Time(time_value)))
            labels.append(f"L0={float(hgs.lon.to_value(u.deg)):.2f}°")
            labels.append(f"B0={float(hgs.lat.to_value(u.deg)):.2f}°")
    except Exception:
        pass
    rsun_arcsec = _rsun_arcsec_from_map(display_map)
    if rsun_arcsec is not None:
        labels.append(f"Rsun={float(rsun_arcsec):.1f}\"")
    return labels


def _viewport_sun_centered(
    display_map: Map,
    *,
    disk_pad: float = _FULL_SUN_DISK_PAD,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Viewport symmetric about the helioprojective origin (solar disk center)."""
    rsun_arcsec = _rsun_arcsec_from_map(display_map) or 960.0
    half_arcsec = float(rsun_arcsec) * max(float(disk_pad), 1.0)
    try:
        sx = abs(float(display_map.scale.axis1.to_value(u.arcsec / u.pix)))
        sy = abs(float(display_map.scale.axis2.to_value(u.arcsec / u.pix)))
        scale = max(sx, sy, 1e-6)
    except Exception:
        scale = 1.0
    half_pix = half_arcsec / scale
    origin = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=display_map.coordinate_frame)
    observer = getattr(display_map.coordinate_frame, "observer", None)
    try:
        with _spherical_screen_context_for_observer(observer):
            xpix, ypix = display_map.world_to_pixel(origin)
        cx = float(xpix.to_value(u.pix) if hasattr(xpix, "to_value") else xpix)
        cy = float(ypix.to_value(u.pix) if hasattr(ypix, "to_value") else ypix)
    except Exception:
        cx, cy = _disk_center_pixel(display_map)
    return (cx - half_pix, cx + half_pix), (cy - half_pix, cy + half_pix)


def _viewport_sun_centered_including_scene(
    display_map: Map,
    scene: dict[str, Any],
    *,
    crop_corners: SkyCoord | None = None,
    crop_pixels: tuple[np.ndarray, np.ndarray] | None = None,
    disk_pad: float = _FULL_SUN_DISK_PAD,
    overlay_pad: float = _FULL_SUN_OVERLAY_PAD,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Sun-centered viewport expanded to include overlay geometry (e.g. cyan crop box)."""
    (bx0, bx1), (by0, by1) = _viewport_sun_centered(display_map, disk_pad=disk_pad)
    cx = 0.5 * (bx0 + bx1)
    cy = 0.5 * (by0 + by1)
    half_x = 0.5 * abs(bx1 - bx0)
    half_y = 0.5 * abs(by1 - by0)
    xpix, ypix = _scene_overlay_pixel_coords(
        display_map,
        scene,
        crop_corners=crop_corners,
        crop_pixels=crop_pixels,
    )
    pad = max(float(overlay_pad), 1.0)
    if xpix.size:
        half_x = max(half_x, pad * float(np.nanmax(np.abs(xpix - cx))))
        half_y = max(half_y, pad * float(np.nanmax(np.abs(ypix - cy))))
    crop_fov = scene.get("crop_fov")
    if crop_fov is not None:
        try:
            sx = abs(float(display_map.scale.axis1.to_value(u.arcsec / u.pix)))
            sy = abs(float(display_map.scale.axis2.to_value(u.arcsec / u.pix)))
            scale = max(sx, sy, 1e-6)
            half_w = 0.5 * float(crop_fov["xsize_arcsec"]) * pad
            half_h = 0.5 * float(crop_fov["ysize_arcsec"]) * pad
            xc_arc = float(crop_fov["xc_arcsec"])
            yc_arc = float(crop_fov["yc_arcsec"])
            half_x = max(half_x, abs(xc_arc) / scale + half_w / scale)
            half_y = max(half_y, abs(yc_arc) / scale + half_h / scale)
        except Exception:
            pass
    return (cx - half_x, cx + half_x), (cy - half_y, cy + half_y)


def _scene_geometry(
    box_corners_world: SkyCoord,
    *,
    crop_fov: dict[str, float],
    inscribing_box: dict[str, float],
    observer,
    obstime,
    pad: float,
) -> dict[str, Any]:
    return {
        "observer": observer,
        "obstime": obstime,
        "crop_fov": crop_fov,
        "inscribing_box": inscribing_box,
        "crop_corners": fov_rectangle_corners_hpc(crop_fov, observer=observer, obstime=obstime),
    }


def _overlay_scene(
    ax,
    display_map: Map,
    scene: dict[str, Any],
    *,
    crop_corners: SkyCoord | None = None,
    crop_pixels: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    observer = scene["observer"]
    obstime = scene["obstime"]
    geometry_observer = scene.get("geometry_observer")
    geometry_obstime = scene.get("geometry_obstime")
    use_plot_coord = bool(scene.get("use_plot_coord"))
    for edge in project_model_box_edges_to_observer_hpc(
        scene["box_corners_world"],
        observer=observer,
        obstime=obstime,
    ):
        _plot_edge_pixels(
            ax,
            display_map,
            edge,
            color=_MODEL_BOX_COLOR,
            use_plot_coord=use_plot_coord,
        )

    inscribing_box = scene.get("inscribing_box")
    if inscribing_box is not None:
        if geometry_observer is not None and geometry_obstime is not None:
            fov_edges = project_fov_box_edges_reprojected(
                inscribing_box,
                source_observer=geometry_observer,
                source_obstime=geometry_obstime,
                target_observer=observer,
                target_obstime=obstime,
            )
            for edge in fov_edges:
                _plot_edge_pixels(
                    ax,
                    display_map,
                    edge,
                    color=_INSCRIBING_BOX_COLOR,
                    zorder=6,
                    use_plot_coord=use_plot_coord,
                )
            if fov_edges:
                ax.plot([], [], color=_INSCRIBING_BOX_COLOR, linewidth=1.4, label="inscribing FOV box")
        else:
            artists = plot_inscribing_fov_box_on_axes(
                ax,
                inscribing_box,
                observer=observer,
                obstime=obstime,
                color=_INSCRIBING_BOX_COLOR,
                linewidth=0.9,
                zorder=6,
                label="inscribing FOV box",
            )
            if not artists:
                ax.plot([], [], color=_INSCRIBING_BOX_COLOR, linewidth=1.4, label="inscribing FOV box")

    if crop_pixels is not None:
        _plot_polyline_pixels(
            ax,
            crop_pixels,
            color=_CROP_FOV_COLOR,
            label=f"crop FOV ({scene.get('pad', 1.1):.2f}x)",
            linewidth=1.6,
        )
    else:
        corners = crop_corners
        if corners is None and geometry_observer is not None and geometry_obstime is not None:
            corners = _crop_fov_corners_in_display_frame(
                scene["crop_fov"],
                source_observer=geometry_observer,
                source_obstime=geometry_obstime,
                display_map=display_map,
            )
        if corners is None:
            corners = scene.get("crop_corners")
        _plot_closed_sky(
            ax,
            display_map,
            corners,
            color=_CROP_FOV_COLOR,
            label=f"crop FOV ({scene.get('pad', 1.1):.2f}x)",
            use_plot_coord=use_plot_coord,
        )


def _overlay_scene_pixels_for_viewport(
    display_map: Map,
    scene: dict[str, Any],
    *,
    crop_corners: SkyCoord | None = None,
    crop_pixels: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    txs: list[float] = []
    tys: list[float] = []
    observer = scene["observer"]
    obstime = scene["obstime"]
    geometry_observer = scene.get("geometry_observer")
    geometry_obstime = scene.get("geometry_obstime")

    def _accum(coords: SkyCoord | None) -> None:
        if coords is None:
            return
        try:
            mapped = _transform_coords_to_map_frame(display_map, coords)
            tx = np.atleast_1d(mapped.Tx.to_value(u.arcsec))
            ty = np.atleast_1d(mapped.Ty.to_value(u.arcsec))
            finite = np.isfinite(tx) & np.isfinite(ty)
            txs.extend(tx[finite].tolist())
            tys.extend(ty[finite].tolist())
        except Exception:
            return

    for edge in project_model_box_edges_to_observer_hpc(
        scene["box_corners_world"],
        observer=observer,
        obstime=obstime,
    ):
        _accum(edge)

    inscribing_box = scene.get("inscribing_box")
    if inscribing_box is not None:
        if geometry_observer is not None and geometry_obstime is not None:
            fov_edges = project_fov_box_edges_reprojected(
                inscribing_box,
                source_observer=geometry_observer,
                source_obstime=geometry_obstime,
                target_observer=observer,
                target_obstime=obstime,
            )
            for edge in fov_edges:
                _accum(edge)
        else:
            for edge in project_fov_box_edges_to_observer_hpc(
                inscribing_box,
                observer=observer,
                obstime=obstime,
            ):
                _accum(edge)

    corners = crop_corners
    if corners is None and geometry_observer is not None and geometry_obstime is not None:
        corners = _crop_fov_corners_in_display_frame(
            scene["crop_fov"],
            source_observer=geometry_observer,
            source_obstime=geometry_obstime,
            display_map=display_map,
        )
    if corners is None:
        corners = scene.get("crop_corners")
    _accum(corners)
    if crop_pixels is not None:
        xpix, ypix = crop_pixels
        for x, y in zip(xpix, ypix, strict=False):
            if np.isfinite(x) and np.isfinite(y):
                try:
                    _accum(display_map.pixel_to_world(x * u.pix, y * u.pix))
                except Exception:
                    continue
    return np.asarray(txs, dtype=float), np.asarray(tys, dtype=float)


def _compute_viewport(
    display_map: Map,
    scene: dict[str, Any],
    *,
    crop_corners: SkyCoord | None = None,
    crop_pixels: tuple[np.ndarray, np.ndarray] | None = None,
    disk_pad: float = _FULL_SUN_DISK_PAD,
    overlay_pad: float = _FULL_SUN_OVERLAY_PAD,
) -> tuple[tuple[float, float], tuple[float, float]]:
    rsun_arcsec = _rsun_arcsec_from_map(display_map) or 960.0
    disk_half_arcsec = float(rsun_arcsec) * max(float(disk_pad), 1.0)
    tx, ty = _overlay_scene_pixels_for_viewport(
        display_map,
        scene,
        crop_corners=crop_corners,
        crop_pixels=crop_pixels,
    )
    overlay_pad = max(float(overlay_pad), 1.0)
    overlay_half_x = float(np.nanmax(np.abs(tx))) * overlay_pad if tx.size else 0.0
    overlay_half_y = float(np.nanmax(np.abs(ty))) * overlay_pad if ty.size else 0.0
    half_arcsec = max(disk_half_arcsec, overlay_half_x, overlay_half_y)

    try:
        sx = abs(float(display_map.scale.axis1.to_value(u.arcsec / u.pix)))
        sy = abs(float(display_map.scale.axis2.to_value(u.arcsec / u.pix)))
        scale = max(sx, sy, 1e-6)
    except Exception:
        scale = 1.0
    half_pix = half_arcsec / scale
    cx, cy = _disk_center_pixel(display_map)
    return (cx - half_pix, cx + half_pix), (cy - half_pix, cy + half_pix)


def _apply_viewport(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    cur_x = ax.get_xlim()
    cur_y = ax.get_ylim()
    x_dir = 1.0 if cur_x[1] >= cur_x[0] else -1.0
    y_dir = 1.0 if cur_y[1] >= cur_y[0] else -1.0
    ax.set_xlim((xlim[0], xlim[1]) if x_dir > 0 else (xlim[1], xlim[0]))
    ax.set_ylim((ylim[0], ylim[1]) if y_dir > 0 else (ylim[1], ylim[0]))


def _draw_solar_grid_mesh(ax, display_map: Map) -> None:
    try:
        display_map.draw_grid(axes=ax, color="0.45", lw=0.5, annotate=False)
    except Exception:
        pass
    try:
        display_map.draw_limb(axes=ax, color="0.45", lw=0.8)
    except Exception:
        pass


def _plot_map_layers(ax, display_map: Map, *, data_map: Map | None = None) -> None:
    if data_map is None or data_map is display_map:
        display_map.plot(axes=ax, annotate=False)
        return
    background = Map(np.zeros(np.asarray(display_map.data).shape, dtype=float), display_map.meta)
    background.plot(axes=ax, annotate=False, vmin=0.0, vmax=1.0, cmap="gray", alpha=0.0)
    data_map.plot(axes=ax, annotate=False)


def _plot_panel(
    ax,
    display_map: Map,
    scene: dict[str, Any],
    *,
    title: str,
    data_map: Map | None = None,
    crop_corners: SkyCoord | None = None,
    crop_pixels: tuple[np.ndarray, np.ndarray] | None = None,
    draw_grid_mesh: bool = False,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    square_box: bool = False,
) -> None:
    ax.set_title(title, fontsize=9)
    if square_box:
        try:
            ax.set_box_aspect(1)
        except Exception:
            try:
                ax.set_aspect("equal", adjustable="box")
            except Exception:
                pass
    _plot_map_layers(ax, display_map, data_map=data_map)
    if draw_grid_mesh:
        _draw_solar_grid_mesh(ax, display_map)
    _overlay_scene(
        ax,
        display_map,
        scene,
        crop_corners=crop_corners,
        crop_pixels=crop_pixels,
    )
    if viewport is not None:
        _apply_viewport(ax, viewport[0], viewport[1])
    for label in _reference_frame_legend_labels(display_map, scene.get("obstime")):
        ax.plot([], [], color="none", label=label)
    ax.legend(loc="upper right", fontsize=7)


def _reproject_onto_canvas(smap: Map, canvas: Map, *, algorithm: str) -> Map:
    """Reproject ``smap`` pixels onto ``canvas`` WCS (SunPy native output)."""
    for algo in (algorithm, "interpolation", "adaptive", "exact"):
        try:
            return smap.reproject_to(canvas.wcs, algorithm=algo, roundtrip_coords=False)
        except Exception:
            continue
    return canvas


def _mask_pixels_not_visible_from_source(source_map: Map, display_map: Map) -> Map:
    """Mask display pixels that map off-disk in the source observer frame.

    This keeps the Earth-view reprojection physically limited to source-visible
    solar-surface pixels instead of interpolating data across the source limb.
    """
    rsun_arcsec = _rsun_arcsec_from_map(source_map) or 960.0
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
    source_settings = getattr(source_map, "plot_settings", None) or {}
    if source_settings:
        out.plot_settings.update(source_settings)
    return out


def _save_crop_map_fits(smap: Map, path: str | Path) -> Path:
    """Persist a cropped map to FITS for round-trip verification."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    smap.save(out, overwrite=True)
    return out


def _load_crop_map_fits(path: str | Path) -> Map:
    """Reload a cropped map written by :func:`_save_crop_map_fits`."""
    from pyampp.gxbox.boxutils import load_sunpy_map_compat

    return load_sunpy_map_compat(Path(path))


def plot_refmap_crop_diagnostics(
    smap: Map,
    box_corners_world: SkyCoord,
    *,
    pad: float = 1.1,
    model_obstime=None,
    crop_result: RefmapCropResult | None = None,
    crop_fits_path: str | Path | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    viewport_disk_pad: float = _FULL_SUN_DISK_PAD,
    viewport_overlay_pad: float = _FULL_SUN_OVERLAY_PAD,
) -> tuple[Any, RefmapCropResult]:
    """Three-panel refmap crop diagnostic.

  Workflow (cropping decoupled from plotting):

  1. Left panel — full reference map @ map time with crop/overlays.
  2. Crop via :func:`crop_refmap_to_model_box_after_pangle_rotation`.
  3. Save cropped map FITS, reload through the normal SunPy loader.
  4. Middle panel — restored crop on a full-Sun viewport in the crop map frame.
  5. Overlays on the middle panel match the left panel (native map LOS).
  6. Right panel — empty Earth LOS canvas @ map time; crop map pixels and all
     overlays are expressed in that single canvas WCS (IDL-style workflow).
    """
    import tempfile

    # Step 2 — crop in the reference map native LOS (after P-angle rotation).
    result = crop_result or crop_refmap_to_model_box_after_pangle_rotation(
        smap,
        box_corners_world,
        pad=pad,
        model_obstime=model_obstime,
    )

    # Step 3 — FITS round-trip so the middle panel uses a restored Map.
    if crop_fits_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        crop_fits_path = Path(tmp.name)
        tmp.close()
    _save_crop_map_fits(result.cropped_map, crop_fits_path)
    cropped_display_map = _load_crop_map_fits(crop_fits_path)

    ref_display = rotate_refmap_for_display(smap)
    inscribing_box = compute_inscribing_fov_box_for_observer(
        box_corners_world,
        observer=result.map_observer,
        obstime=result.map_obstime,
    )
    if inscribing_box is None:
        raise ValueError("could not compute inscribing FOV box for reference map")

    ref_scene = _scene_geometry(
        box_corners_world,
        crop_fov=result.crop_fov,
        inscribing_box=inscribing_box,
        observer=result.map_observer,
        obstime=result.map_obstime,
        pad=pad,
    )
    ref_scene["box_corners_world"] = box_corners_world
    ref_scene["pad"] = pad

    # Step 4 — restored crop uses the same viewport as the left panel.
    crop_viewport = None

    # Step 6 — Earth LOS full-disk canvas; map + overlays share this WCS.
    earth_observer = earth_observer_at(result.map_obstime)
    earth_canvas_fov = _full_disk_fov(cropped_display_map, pad=viewport_disk_pad)
    crop_on_earth = reproject_refmap_to_observer(
        cropped_display_map,
        observer=earth_observer,
        obstime=result.map_obstime,
        reference_smap=cropped_display_map,
        fov=earth_canvas_fov,
        algorithm="adaptive",
        mask_off_limb=True,
    )
    earth_scene = _scene_geometry(
        box_corners_world,
        crop_fov=result.crop_fov,
        inscribing_box=inscribing_box,
        observer=earth_observer,
        obstime=result.map_obstime,
        pad=pad,
    )
    earth_scene["box_corners_world"] = box_corners_world
    earth_scene["pad"] = pad
    earth_scene["geometry_observer"] = result.map_observer
    earth_scene["geometry_obstime"] = result.map_obstime
    earth_scene["use_plot_coord"] = True
    earth_crop_corners = project_padded_crop_bottom_face_corners_between_observers(
        inscribing_box,
        pad=pad,
        source_observer=result.map_observer,
        source_obstime=result.map_obstime,
        target_observer=earth_observer,
        target_obstime=result.map_obstime,
    )
    earth_viewport = _viewport_sun_centered_including_scene(
        crop_on_earth,
        earth_scene,
        crop_corners=earth_crop_corners,
        disk_pad=viewport_disk_pad,
        overlay_pad=viewport_overlay_pad,
    )

    ref_viewport = _compute_viewport(
        ref_display,
        ref_scene,
        disk_pad=viewport_disk_pad,
        overlay_pad=viewport_overlay_pad,
    )
    crop_viewport = _compute_viewport(
        cropped_display_map,
        ref_scene,
        disk_pad=viewport_disk_pad,
        overlay_pad=viewport_overlay_pad,
    )

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    # Step 1 — illustrate crop on the full reference map.
    _plot_panel(
        fig.add_subplot(1, 3, 1, projection=ref_display),
        ref_display,
        ref_scene,
        title="1) Reference map @ map time (P-angle rotated)",
        draw_grid_mesh=True,
        viewport=ref_viewport,
    )
    # Steps 4–5 — restored crop on full-Sun native grid, identical overlays.
    _plot_panel(
        fig.add_subplot(1, 3, 2, projection=cropped_display_map),
        cropped_display_map,
        ref_scene,
        title="2) Cropped map @ map time (restored FITS, native frame)",
        draw_grid_mesh=True,
        viewport=crop_viewport,
        square_box=True,
    )
    # Step 6 — Earth LOS @ map time (single canvas WCS for data + overlays).
    _plot_panel(
        fig.add_subplot(1, 3, 3, projection=crop_on_earth),
        crop_on_earth,
        earth_scene,
        title="3) Cropped map @ Earth LOS @ map time",
        crop_corners=earth_crop_corners,
        draw_grid_mesh=True,
        viewport=earth_viewport,
        square_box=True,
    )

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, result
