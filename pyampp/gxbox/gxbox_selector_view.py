#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import Any, Optional, Sequence

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from sunpy.coordinates import Heliocentric, HeliographicStonyhurst, Helioprojective, get_earth
from astropy.time import Time
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QMessageBox

from pyampp.data.downloader import SDOImageDownloader
from pyampp.gxbox.fov_selector_gui import FovBoxSelectorDialog
from pyampp.gxbox.box import Box
from pyampp.gxbox.gx_fov2box import (
    _decode_id_text,
    _extract_execute_geometry,
    _extract_execute_paths,
    _infer_time_from_entry_loaded,
    _load_entry_box_any,
)
from sunpy.map import Map

from pyampp.io import (
    box_corners_world_from_model,
    build_fits_refmaps_for_model,
    discover_fits_refmap_map_ids,
    load_model,
    model_obstime_from_base_index,
    save_model,
)
from pyampp.gxbox.selector_api import (
    BoxGeometrySelection,
    CoordMode,
    DisplayFovBoxSelection,
    DisplayFovSelection,
    SelectorDialogResult,
    SelectorSessionInput,
)
from pyampp.gxbox.observer_restore import build_pb0r_metadata_from_ephemeris, resolve_observer_with_info

from pyampp.util.config import IDL_HMI_RSUN_M

_DEFAULT_MAP_IDS = (
    "Bz",
    "Ic",
    "B_rho",
    "B_theta",
    "B_phi",
    "disambig",
    "Vert_current",
    "chromo_mask",
    "94",
    "131",
    "1600",
    "1700",
    "171",
    "193",
    "211",
    "304",
    "335",
)


def _contains_viewer_field_payload(entry_loaded: dict[str, Any]) -> bool:
    if not isinstance(entry_loaded, dict):
        return False

    for key in ("corona", "nlfff", "pot"):
        group = entry_loaded.get(key)
        if isinstance(group, dict) and any(name in group for name in ("bx", "by", "bz", "bcube")):
            return True

    chromo = entry_loaded.get("chromo")
    if isinstance(chromo, dict):
        if any(name in chromo for name in ("bx", "by", "bz", "bcube", "chromo_bcube")):
            return True

    return False

def _normalize_observer_key(observer_key: str | None) -> str:
    raw = observer_key
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "ignore")
    key = str(raw or "earth").strip().lower()
    aliases = {
        "custom": "custom",
        "sdo": "sdo",
        "sdo/aia": "sdo",
        "sdo/hmi": "sdo",
        "earth": "earth",
        "solo": "solar orbiter",
        "solar-orbiter": "solar orbiter",
        "solarorbiter": "solar orbiter",
        "solar orbiter": "solar orbiter",
        "stereo a": "stereo-a",
        "stereo-a": "stereo-a",
        "stereoa": "stereo-a",
        "stereo b": "stereo-b",
        "stereo-b": "stereo-b",
        "stereob": "stereo-b",
    }
    return aliases.get(key, "earth")


def _observer_display_label(observer_key: str | None) -> str:
    return {
        "earth": "Earth",
        "sdo": "SDO",
        "solar orbiter": "Solar Orbiter",
        "stereo-a": "STEREO-A",
        "stereo-b": "STEREO-B",
        "custom": "Custom",
    }.get(_normalize_observer_key(observer_key), "Earth")


def _base_display_id(base_key: str) -> str:
    key = str(base_key).lower()
    aliases = {
        "bx": "Bx",
        "by": "By",
        "bz": "Bz",
        "ic": "Ic",
        "vert_current": "Vert_current",
        "chromo_mask": "chromo_mask",
    }
    return aliases.get(key, key)


def _pick_entry_file(initial: Optional[str], start_dir: Optional[str]) -> Optional[str]:
    dialog = QFileDialog(None, "Open Model Box (.h5 or .sav)")
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter("Model Box Files (*.h5 *.sav);;HDF5 Files (*.h5);;SAV Files (*.sav);;All Files (*)")
    if initial:
        candidate = Path(initial).expanduser()
        dialog.setDirectory(str(candidate.parent if candidate.parent.exists() else Path.cwd()))
        dialog.selectFile(candidate.name)
    elif start_dir:
        base = Path(start_dir).expanduser()
        dialog.setDirectory(str(base if base.exists() else Path.cwd()))
    else:
        dialog.setDirectory(str(Path.cwd()))
    if not dialog.exec_():
        return None
    selected = dialog.selectedFiles()
    return selected[0] if selected else None


def _parse_execute_box_dims_and_dx(execute_text: str) -> tuple[Optional[tuple[int, int, int]], Optional[float]]:
    dims = None
    dx_km = None
    if not execute_text:
        return dims, dx_km
    try:
        parts = shlex.split(execute_text)
    except Exception:
        parts = []
    for i, token in enumerate(parts):
        if token == "--box-dims" and i + 3 < len(parts):
            try:
                dims = (int(float(parts[i + 1])), int(float(parts[i + 2])), int(float(parts[i + 3])))
            except Exception:
                pass
        elif token.startswith("--box-dims="):
            try:
                vals = token.split("=", 1)[1].split(",")
                if len(vals) == 3:
                    dims = tuple(int(float(v)) for v in vals)
            except Exception:
                pass
        elif token == "--dx-km" and i + 1 < len(parts):
            try:
                dx_km = float(parts[i + 1])
            except Exception:
                pass
        elif token.startswith("--dx-km="):
            try:
                dx_km = float(token.split("=", 1)[1])
            except Exception:
                pass
    if dims is None:
        m = re.search(
            r"--box-dims\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)",
            execute_text,
        )
        if m:
            dims = tuple(int(round(float(m.group(i)))) for i in (1, 2, 3))
    if dx_km is None:
        m = re.search(r"--dx-km\s+([+-]?\d+(?:\.\d+)?)", execute_text)
        if m:
            dx_km = float(m.group(1))
    return dims, dx_km


def _parse_execute_refmap_paths(execute_text: str) -> tuple[str, ...]:
    if not execute_text:
        return ()
    try:
        parts = shlex.split(str(execute_text))
    except Exception:
        parts = []
    paths: list[str] = []
    flags = {"--refmaps-path", "--refmap-path"}
    for i, token in enumerate(parts):
        if token in flags and i + 1 < len(parts):
            paths.append(parts[i + 1])
        elif token.startswith("--refmaps-path=") or token.startswith("--refmap-path="):
            paths.append(token.split("=", 1)[1])
    return tuple(path for path in paths if str(path).strip())


def _merge_ref_map_paths(*path_groups: Optional[Sequence[str]]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for paths in path_groups:
        for path in paths or ():
            text = str(path).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return tuple(merged)


def _infer_dims_from_entry(entry_loaded: dict[str, Any]) -> tuple[int, int, int]:
    meta = entry_loaded.get("metadata", {}) if isinstance(entry_loaded, dict) else {}
    axis_order = _decode_id_text(meta.get("axis_order_3d", "")).strip().lower() if isinstance(meta, dict) else ""

    def _dims_from_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
        if len(shape) < 3:
            return (150, 75, 150)
        if axis_order == "zyx":
            return (int(shape[2]), int(shape[1]), int(shape[0]))
        return (int(shape[0]), int(shape[1]), int(shape[2]))

    for group_name in ("corona", "chromo"):
        grp = entry_loaded.get(group_name)
        if not isinstance(grp, dict):
            continue
        for key in ("bx", "by", "bz"):
            if key in grp:
                arr = np.asarray(grp[key])
                if arr.ndim >= 3:
                    return _dims_from_shape(tuple(int(v) for v in arr.shape[:3]))
        for key in ("bcube", "chromo_bcube"):
            if key in grp:
                arr = np.asarray(grp[key])
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    return _dims_from_shape(tuple(int(v) for v in arr.shape[:3]))
                if arr.ndim == 4 and arr.shape[0] == 3:
                    return _dims_from_shape(tuple(int(v) for v in arr.shape[1:4]))
    return (150, 75, 150)


def _infer_dx_km_from_entry(entry_loaded: dict[str, Any]) -> float:
    rsun_km = 695700.0
    for group_name in ("corona", "chromo"):
        grp = entry_loaded.get(group_name)
        if not isinstance(grp, dict):
            continue
        dr = grp.get("dr")
        if dr is None:
            continue
        try:
            arr = np.asarray(dr, dtype=float).ravel()
            if arr.size > 0:
                return float(arr[0] * rsun_km)
        except Exception:
            continue
    return 1400.0


def _observer_fov_from_entry(
    entry_loaded: dict[str, Any],
) -> tuple[Optional[DisplayFovSelection], bool, Optional[DisplayFovBoxSelection]]:
    observer = entry_loaded.get("observer")
    if not isinstance(observer, dict):
        return None, False, None
    fov = observer.get("fov")
    if not isinstance(fov, dict):
        return None, False, None
    try:
        result = DisplayFovSelection(
            center_x_arcsec=float(fov["xc_arcsec"]),
            center_y_arcsec=float(fov["yc_arcsec"]),
            width_arcsec=float(fov["xsize_arcsec"]),
            height_arcsec=float(fov["ysize_arcsec"]),
        )
        fov_box_meta = observer.get("fov_box")
        fov_box = None
        if isinstance(fov_box_meta, dict):
            try:
                fov_box = DisplayFovBoxSelection(
                    center_x_arcsec=float(fov_box_meta.get("xc_arcsec", result.center_x_arcsec)),
                    center_y_arcsec=float(fov_box_meta.get("yc_arcsec", result.center_y_arcsec)),
                    width_arcsec=float(fov_box_meta.get("xsize_arcsec", result.width_arcsec)),
                    height_arcsec=float(fov_box_meta.get("ysize_arcsec", result.height_arcsec)),
                    z_min_mm=float(fov_box_meta["zmin_mm"]),
                    z_max_mm=float(fov_box_meta["zmax_mm"]),
                    observer_key=_normalize_observer_key(
                        fov_box_meta.get("observer_key", observer.get("name", "earth"))
                    ),
                )
            except Exception:
                fov_box = None
        return result, bool(fov.get("square", False)), fov_box
    except Exception:
        return None, False, None


def _geometry_from_entry(entry_loaded: dict[str, Any], entry_path: Path) -> tuple[str, BoxGeometrySelection]:
    meta = entry_loaded.get("metadata", {}) if isinstance(entry_loaded, dict) else {}
    execute_text = _decode_id_text(meta.get("execute", "")) if isinstance(meta, dict) else ""
    time_iso = _infer_time_from_entry_loaded(entry_loaded, entry_path) or Time.now().to_datetime().strftime("%Y-%m-%dT%H:%M:%S")
    coords, frame, _projection = _extract_execute_geometry(execute_text)
    dims, dx_km = _parse_execute_box_dims_and_dx(execute_text)
    if coords is None:
        coords = (0.0, 0.0)
    if dims is None:
        dims = _infer_dims_from_entry(entry_loaded)
    if dx_km is None:
        dx_km = _infer_dx_km_from_entry(entry_loaded)
    mode = {
        "hpc": CoordMode.HPC,
        "hgc": CoordMode.HGC,
        "hgs": CoordMode.HGS,
    }.get((frame or "hpc").lower(), CoordMode.HPC)
    geometry = BoxGeometrySelection(
        coord_mode=mode,
        coord_x=float(coords[0]),
        coord_y=float(coords[1]),
        grid_x=int(dims[0]),
        grid_y=int(dims[1]),
        grid_z=int(dims[2]),
        dx_km=float(dx_km),
    )
    return time_iso, geometry


def _discover_filesystem_maps(time_iso: str, data_dir: Optional[str]) -> dict[str, str]:
    if not data_dir:
        return {}
    base = Path(data_dir).expanduser()
    if not base.exists() or not base.is_dir():
        return {}
    try:
        downloader = SDOImageDownloader(Time(time_iso), data_dir=str(base), euv=True, uv=True, hmi=True)
        files = {k: v for k, v in downloader._check_files_exist(downloader.path, returnfilelist=True).items() if v}
        files.update(
            {
                key: value
                for key, value in _discover_external_ref_map_files(
                    [downloader.path],
                    generic=False,
                ).items()
                if key not in files
            }
        )
        return files
    except Exception:
        return {}


def _discover_external_ref_map_files(ref_map_paths: Optional[Sequence[str]], *, generic: bool = True) -> dict[str, str]:
    if not ref_map_paths:
        return {}
    out: dict[str, str] = {}
    for path, refmap_id in discover_fits_refmap_map_ids(ref_map_paths, generic=generic).items():
        out[_viewer_context_id_from_refmap_id(refmap_id)] = str(path)
    return out


def _viewer_context_id_from_refmap_id(refmap_id: str) -> str:
    match = re.fullmatch(r"AIA_(\d+)", str(refmap_id))
    if match:
        return match.group(1)
    return str(refmap_id)


def _template_map_from_box_refmaps(refmaps: dict[str, Any]):
    for key in ("Bz_reference", "Ic_reference"):
        payload = refmaps.get(key)
        if not isinstance(payload, dict):
            continue
        header_text = payload.get("wcs_header")
        data = payload.get("data")
        if header_text is None or data is None:
            continue
        try:
            text = _decode_id_text(header_text).replace("\\n", "\n")
            header = fits.Header.fromstring(text, sep="\n")
            arr = np.asarray(data)
            return Map(np.zeros(arr.shape, dtype=np.float32), header)
        except Exception:
            continue
    return None


def _session_box_data_for_refmap_embed(
    session_input: SelectorSessionInput,
    entry_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build the same in-memory box payload used by save-time refmap embedding."""

    if entry_path is not None:
        try:
            box_data = load_model(Path(entry_path).expanduser().resolve())
            refmaps = dict(box_data.get("refmaps") or {})
            if isinstance(session_input.refmaps, dict):
                refmaps.update(session_input.refmaps)
            box_data["refmaps"] = refmaps
            return box_data
        except Exception:
            pass

    box_data: dict[str, Any] = {"refmaps": dict(session_input.refmaps or {})}
    if session_input.base_wcs_header:
        box_data["base"] = {"index": session_input.base_wcs_header}
    return box_data


def _box_corners_world_from_selector_geometry(
    geometry: BoxGeometrySelection,
    *,
    obstime_iso: str | None,
) -> SkyCoord:
    if geometry is None:
        raise ValueError("selector geometry is required to resolve model box corners")
    when = Time(obstime_iso) if obstime_iso else Time.now()
    observer = get_earth(when)
    rsun = (IDL_HMI_RSUN_M * u.m).to(u.km)
    frame_obs = Helioprojective(observer=observer, obstime=when, rsun=rsun)
    if geometry.coord_mode == CoordMode.HPC:
        box_origin = SkyCoord(
            Tx=geometry.coord_x * u.arcsec,
            Ty=geometry.coord_y * u.arcsec,
            obstime=when,
            observer=observer,
            rsun=rsun,
            frame=Helioprojective,
        )
    elif geometry.coord_mode == CoordMode.HGC:
        box_origin = SkyCoord(
            lon=geometry.coord_x * u.deg,
            lat=geometry.coord_y * u.deg,
            radius=rsun,
            obstime=when,
            observer=observer,
            frame=HeliographicCarrington,
        )
    else:
        box_origin = SkyCoord(
            lon=geometry.coord_x * u.deg,
            lat=geometry.coord_y * u.deg,
            radius=rsun,
            obstime=when,
            observer=observer,
            frame=HeliographicStonyhurst,
        )
    box_dims = u.Quantity([geometry.grid_x, geometry.grid_y, geometry.grid_z], u.pix)
    box_res = geometry.dx_km * u.km
    frame_hcc = Heliocentric(observer=box_origin, obstime=when)
    box_center = box_origin.transform_to(frame_hcc)
    center_z = box_center.z + (box_dims[2] / u.pix * box_res.to(u.Mm))
    box_center = SkyCoord(
        x=box_center.x,
        y=box_center.y,
        z=center_z,
        frame=frame_hcc,
    )
    box = Box(frame_obs, box_origin, box_center, box_dims, box_res.to(u.Mm))
    world = box.model_box_corners_world()
    if world is None:
        raise ValueError("could not build model box corners from selector geometry")
    return world


def _resolve_box_corners_world_for_embed(
    box_data: dict[str, Any],
    *,
    session_input: SelectorSessionInput | None = None,
    geometry: BoxGeometrySelection | None = None,
) -> SkyCoord:
    try:
        return box_corners_world_from_model(box_data)
    except ValueError:
        pass
    geom = geometry
    if geom is None and session_input is not None:
        geom = session_input.geometry
    obstime = model_obstime_from_base_index(box_data)
    if obstime is None and session_input is not None:
        obstime = session_input.time_iso
    return _box_corners_world_from_selector_geometry(geom, obstime_iso=obstime)


def _embed_external_refmaps_into_box_data(
    box_data: dict[str, Any],
    external_ref_map_paths: Optional[Sequence[str]],
    *,
    session_input: SelectorSessionInput | None = None,
    geometry: BoxGeometrySelection | None = None,
    overwrite: bool = True,
) -> tuple[list[str], list[str]]:
    """Embed FITS refmaps from explicit external paths into in-memory box_data.

    Only paths supplied via ``external_ref_map_paths`` are embedded. JSOC cache
    files discovered from ``data_dir`` are intentionally excluded.

    Returns ``(embedded_map_ids, skipped_map_ids)``.
    """

    paths = _merge_ref_map_paths(external_ref_map_paths)
    if not paths:
        return [], []

    refmaps = box_data.get("refmaps")
    if not isinstance(refmaps, dict):
        refmaps = {}
        box_data["refmaps"] = refmaps

    model_obstime = model_obstime_from_base_index(box_data)
    box_corners_world = _resolve_box_corners_world_for_embed(
        box_data,
        session_input=session_input,
        geometry=geometry,
    )
    payloads = build_fits_refmaps_for_model(
        paths,
        model_obstime=model_obstime,
        box_corners_world=box_corners_world,
        generic=True,
    )

    embedded: list[str] = []
    skipped: list[str] = []
    for map_id, payload in payloads.items():
        if map_id in refmaps and not overwrite:
            skipped.append(map_id)
            continue
        refmaps[map_id] = payload
        embedded.append(map_id)
    return embedded, skipped


def _embed_external_refmaps_into_session(
    session_input: SelectorSessionInput,
    external_ref_map_paths: Optional[Sequence[str]],
    *,
    entry_path: Path | str | None = None,
    overwrite: bool = True,
) -> tuple[list[str], list[str]]:
    """Embed FITS refmaps into the selector session's in-memory model."""

    box_data = _session_box_data_for_refmap_embed(session_input, entry_path)
    embedded, skipped = _embed_external_refmaps_into_box_data(
        box_data,
        external_ref_map_paths,
        session_input=session_input,
        overwrite=overwrite,
    )
    session_input.refmaps = dict(box_data.get("refmaps") or {})
    return embedded, skipped


def _available_map_ids_from_sources(map_files: dict[str, str], refmaps: dict[str, dict], base_maps: dict[str, Any]) -> list[str]:
    out: list[str] = []
    fs_availability = {
        "Bz": "magnetogram" in map_files,
        "Ic": "continuum" in map_files,
        "B_rho": "field" in map_files,
        "B_theta": "inclination" in map_files,
        "B_phi": "azimuth" in map_files,
        "disambig": "disambig" in map_files,
        # Backward-compatible legacy IDs; these now map to measured HPC products.
        "Br": "field" in map_files,
        "Bp": "inclination" in map_files,
        "Bt": "azimuth" in map_files,
    }
    ref_availability = {
        "Bz": "Bz_reference" in refmaps,
        "Ic": "Ic_reference" in refmaps,
        "Vert_current": "Vert_current" in refmaps,
        "94": "AIA_94" in refmaps,
        "131": "AIA_131" in refmaps,
        "1600": "AIA_1600" in refmaps,
        "1700": "AIA_1700" in refmaps,
        "171": "AIA_171" in refmaps,
        "193": "AIA_193" in refmaps,
        "211": "AIA_211" in refmaps,
        "304": "AIA_304" in refmaps,
        "335": "AIA_335" in refmaps,
    }
    base_availability = {
        "Bz": "bz" in base_maps,
        "Ic": "ic" in base_maps,
        "chromo_mask": "chromo_mask" in base_maps,
    }
    for map_id in _DEFAULT_MAP_IDS:
        if map_id in fs_availability:
            if fs_availability[map_id] or ref_availability.get(map_id, False) or base_availability.get(map_id, False):
                out.append(map_id)
        elif map_id in ref_availability:
            if ref_availability[map_id]:
                out.append(map_id)
        elif map_id in base_availability:
            if base_availability[map_id]:
                out.append(map_id)
        elif map_id in map_files:
            out.append(map_id)
    for base_key in sorted(base_maps.keys()):
        display_id = _base_display_id(base_key)
        if display_id not in out:
            out.append(display_id)
    if "Vert_current" in refmaps and "Vert_current" not in out:
        out.append("Vert_current")
    aliased_refmaps = {
        "Bz_reference",
        "Ic_reference",
        "Vert_current",
        "AIA_94",
        "AIA_131",
        "AIA_1600",
        "AIA_1700",
        "AIA_171",
        "AIA_193",
        "AIA_211",
        "AIA_304",
        "AIA_335",
    }
    for ref_key in sorted(refmaps.keys()):
        if ref_key not in aliased_refmaps and ref_key not in out:
            out.append(ref_key)
    for map_id in sorted(map_files.keys()):
        if map_id not in {
            "magnetogram",
            "continuum",
            "field",
            "inclination",
            "azimuth",
            "disambig",
        } and map_id not in out:
            out.append(map_id)
    return out or list(_DEFAULT_MAP_IDS)


def _build_session_input(entry_path: Path, ref_map_paths: Optional[Sequence[str]] = None) -> SelectorSessionInput:
    entry_loaded = _load_entry_box_any(entry_path)
    if not _contains_viewer_field_payload(entry_loaded):
        raise ValueError(
            "Incompatible model file for gxbox-view2d: missing 3D field payload "
            f"(corona/chromo). File: {entry_path}. "
            "This looks like a metadata-only thin HDF5 (e.g. export-model-metadata output), "
            "which is supported for geometry metadata workflows but not for 2D/3D viewer rendering."
        )
    time_iso, geometry = _geometry_from_entry(entry_loaded, entry_path)
    meta = entry_loaded.get("metadata", {}) if isinstance(entry_loaded, dict) else {}
    execute_text = _decode_id_text(meta.get("execute", "")) if isinstance(meta, dict) else ""
    data_dir, _gxmodel_dir = _extract_execute_paths(execute_text)
    execute_ref_map_paths = _parse_execute_refmap_paths(execute_text)
    all_ref_map_paths = _merge_ref_map_paths(execute_ref_map_paths, ref_map_paths)
    external_ref_map_paths = all_ref_map_paths
    explicit_fov, square_fov, explicit_fov_box = _observer_fov_from_entry(entry_loaded)
    observer_meta = entry_loaded.get("observer") if isinstance(entry_loaded, dict) else None
    observer_name = observer_meta.get("name", "earth") if isinstance(observer_meta, dict) else "earth"
    display_observer_key = _normalize_observer_key(observer_name)
    custom_observer_ephemeris = None
    custom_observer_label = None
    custom_observer_source = None
    if isinstance(observer_meta, dict):
        raw_ephemeris = observer_meta.get("ephemeris")
        raw_label = _decode_id_text(observer_meta.get("label", "")).strip() or None
        raw_source = _decode_id_text(observer_meta.get("source", "")).strip() or None
        custom_needed = (
            display_observer_key == "custom"
            or _normalize_observer_key(observer_name) == "custom"
            or (
                isinstance(explicit_fov_box, DisplayFovBoxSelection)
                and _normalize_observer_key(explicit_fov_box.observer_key) == "custom"
            )
        )
        if custom_needed and isinstance(raw_ephemeris, dict):
            custom_observer_ephemeris = {
                key: raw_ephemeris[key]
                for key in ("obs_date", "obs_time", "hgln_obs_deg", "hglt_obs_deg", "dsun_cm", "rsun_cm")
                if key in raw_ephemeris
            } or None
            custom_observer_label = raw_label or "Custom"
            custom_observer_source = raw_source

    map_files = _discover_filesystem_maps(time_iso, data_dir)
    map_files.update(_discover_external_ref_map_files(all_ref_map_paths))
    refmaps = {}
    raw_refmaps = entry_loaded.get("refmaps")
    if isinstance(raw_refmaps, dict):
        refmaps = raw_refmaps
    base_maps = {}
    base_wcs_header = None
    raw_base = entry_loaded.get("base")
    if isinstance(raw_base, dict):
        for key in ("index", "index_header", "wcs_header"):
            if key in raw_base:
                try:
                    base_wcs_header = _decode_id_text(raw_base.get(key))
                except Exception:
                    base_wcs_header = None
                if base_wcs_header and str(base_wcs_header).strip():
                    break
        for key, value in raw_base.items():
            try:
                arr = np.asarray(value)
            except Exception:
                continue
            if arr.ndim == 2:
                base_maps[str(key).lower()] = value
    map_ids = _available_map_ids_from_sources(map_files, refmaps, base_maps)
    initial_map = "171" if "171" in map_ids else ("Bz" if "Bz" in map_ids else (map_ids[0] if map_ids else None))
    map_source_mode = "filesystem" if map_files else ("embedded" if refmaps else "auto")

    return SelectorSessionInput(
        time_iso=time_iso,
        data_dir=data_dir or "",
        geometry=geometry,
        fov=explicit_fov,
        fov_box=explicit_fov_box,
        square_fov=square_fov,
        allow_geometry_edit=False,
        map_ids=tuple(map_ids),
        map_files=map_files or None,
        external_ref_map_paths=external_ref_map_paths,
        refmaps=refmaps or None,
        base_maps=base_maps or None,
        base_wcs_header=base_wcs_header,
        base_geometry=geometry,
        map_source_mode=map_source_mode,
        display_observer_key=display_observer_key,
        custom_observer_ephemeris=custom_observer_ephemeris,
        custom_observer_label=custom_observer_label,
        custom_observer_source=custom_observer_source,
        initial_map_id=initial_map,
        pad_frac=0.10,
    )


def _pick_save_as_h5_path(parent_widget, default_stem: str = "model") -> Path | None:
    """Show a Save-As file dialog and return the chosen .h5 path, or None if cancelled."""
    dialog = QFileDialog(parent_widget, "Save Model As")
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    dialog.setFileMode(QFileDialog.AnyFile)
    dialog.setNameFilter("HDF5 Files (*.h5);;All Files (*)")
    dialog.selectFile(f"{default_stem}.h5")
    if not dialog.exec_():
        return None
    selected = dialog.selectedFiles()
    if not selected:
        return None
    path = Path(selected[0])
    if path.suffix.lower() != ".h5":
        path = path.with_suffix(".h5")
    return path


def _persist_selector_result_to_entry(
    entry_path: Path,
    result: SelectorDialogResult,
    line_seeds=None,
    fov_box: DisplayFovBoxSelection | None = None,
    observer_state: dict[str, Any] | None = None,
    output_path: Path | None = None,
    external_ref_map_paths: Optional[Sequence[str]] = None,
    session_refmaps: Optional[dict[str, Any]] = None,
) -> bool:
    dest = output_path or entry_path
    if dest.suffix.lower() != ".h5":
        return False

    box_data = load_model(entry_path)
    observer = box_data.get("observer")
    if not isinstance(observer, dict):
        observer = {}
    else:
        observer = dict(observer)

    observer_name = observer.get("name", "earth")
    ephemeris = observer.get("ephemeris")
    display_observer_key = _normalize_observer_key(
        observer_state.get("display_observer_key") if isinstance(observer_state, dict) else observer_name
    )
    raw_custom_ephemeris = (
        observer_state.get("custom_observer_ephemeris")
        if isinstance(observer_state, dict)
        else None
    )
    custom_observer_label = (
        str(observer_state.get("custom_observer_label", "")).strip()
        if isinstance(observer_state, dict)
        else ""
    )
    custom_observer_source = (
        str(observer_state.get("custom_observer_source", "")).strip()
        if isinstance(observer_state, dict)
        else ""
    )
    if not isinstance(raw_custom_ephemeris, dict):
        raw_custom_ephemeris = None
    fov = {
        "frame": "helioprojective",
        "xc_arcsec": float(result.fov.center_x_arcsec),
        "yc_arcsec": float(result.fov.center_y_arcsec),
        "xsize_arcsec": float(result.fov.width_arcsec),
        "ysize_arcsec": float(result.fov.height_arcsec),
        "square": bool(result.square_fov),
    }
    observer["name"] = str(display_observer_key or "earth")
    observer["fov"] = fov
    if isinstance(fov_box, DisplayFovBoxSelection):
        observer["fov_box"] = fov_box.as_observer_metadata(square=bool(result.square_fov))
    else:
        observer.pop("fov_box", None)
    persisted_fov_meta = observer.get("fov_box", {}) if isinstance(observer.get("fov_box"), dict) else {}
    fov_observer_key = (
        str(fov_box.observer_key)
        if isinstance(fov_box, DisplayFovBoxSelection)
        else _normalize_observer_key(persisted_fov_meta.get("observer_key", observer["name"]))
    )
    needs_custom_ephemeris = (
        display_observer_key == "custom"
        or _normalize_observer_key(fov_observer_key) == "custom"
    )
    if needs_custom_ephemeris and custom_observer_label:
        observer["label"] = custom_observer_label
    else:
        observer["label"] = _observer_display_label(display_observer_key)
    if needs_custom_ephemeris and custom_observer_source:
        observer["source"] = custom_observer_source
    else:
        observer.pop("source", None)
    resolved_ephemeris: dict[str, float | str] = {}
    obs_time = _infer_time_from_entry_loaded(box_data, entry_path)
    if obs_time:
        try:
            when = Time(obs_time)
        except Exception:
            when = None
    else:
        when = None
    if needs_custom_ephemeris and raw_custom_ephemeris:
        resolved_ephemeris = {
            key: raw_custom_ephemeris[key]
            for key in ("obs_date", "obs_time", "hgln_obs_deg", "hglt_obs_deg", "dsun_cm", "rsun_cm")
            if key in raw_custom_ephemeris
        }
        if when is not None and "obs_date" not in resolved_ephemeris:
            resolved_ephemeris["obs_date"] = when.isot
    else:
        if when is not None:
            resolved_ephemeris["obs_date"] = when.isot
            coord, _warning, _used_key = resolve_observer_with_info(box_data, display_observer_key, when)
            if coord is not None:
                try:
                    obs_hgs = coord.transform_to(HeliographicStonyhurst(obstime=when))
                    resolved_ephemeris["hgln_obs_deg"] = float(obs_hgs.lon.to_value(u.deg))
                    resolved_ephemeris["hglt_obs_deg"] = float(obs_hgs.lat.to_value(u.deg))
                    resolved_ephemeris["dsun_cm"] = float(coord.radius.to_value(u.cm))
                except Exception:
                    pass
    if "rsun_cm" not in resolved_ephemeris:
        refmaps = box_data.get("refmaps", {}) if isinstance(box_data, dict) else {}
        for key in ("Bz_reference", "Ic_reference"):
            payload = refmaps.get(key) if isinstance(refmaps, dict) else None
            if not isinstance(payload, dict):
                continue
            header_text = payload.get("wcs_header")
            if header_text is None:
                continue
            try:
                text = _decode_id_text(header_text).replace("\\n", "\n")
                header = fits.Header.fromstring(text, sep="\n")
                if "RSUN_REF" in header:
                    resolved_ephemeris["rsun_cm"] = float(u.Quantity(header["RSUN_REF"], u.m).to_value(u.cm))
                    break
            except Exception:
                continue
    if not resolved_ephemeris and isinstance(ephemeris, dict):
        resolved_ephemeris = {
            key: ephemeris[key]
            for key in ("obs_date", "obs_time", "hgln_obs_deg", "hglt_obs_deg", "dsun_cm", "rsun_cm")
            if key in ephemeris
        }
    if needs_custom_ephemeris or display_observer_key == "custom":
        if resolved_ephemeris:
            observer["ephemeris"] = resolved_ephemeris
            pb0r = build_pb0r_metadata_from_ephemeris(
                resolved_ephemeris,
                observer_key="custom",
                obs_time=resolved_ephemeris.get("obs_date"),
            )
            if pb0r:
                observer["pb0r"] = pb0r
            else:
                observer.pop("pb0r", None)
        else:
            observer.pop("ephemeris", None)
            observer.pop("pb0r", None)
    elif resolved_ephemeris:
        observer["ephemeris"] = resolved_ephemeris
        pb0r = build_pb0r_metadata_from_ephemeris(
            resolved_ephemeris,
            observer_key=observer.get("name"),
            obs_time=resolved_ephemeris.get("obs_date"),
        )
        if pb0r:
            observer["pb0r"] = pb0r
    else:
        observer.pop("ephemeris", None)
        observer.pop("pb0r", None)
    box_data["observer"] = observer
    if isinstance(line_seeds, dict):
        box_data["line_seeds"] = line_seeds
    else:
        box_data.pop("line_seeds", None)

    if isinstance(session_refmaps, dict):
        existing_refmaps = box_data.get("refmaps")
        if not isinstance(existing_refmaps, dict):
            existing_refmaps = {}
        merged_refmaps = dict(existing_refmaps)
        merged_refmaps.update(session_refmaps)
        box_data["refmaps"] = merged_refmaps

    _embed_external_refmaps_into_box_data(
        box_data,
        external_ref_map_paths,
        geometry=result.geometry,
        overwrite=True,
    )

    # Save with contract persistence via centralized model.io loader
    save_model(box_data, dest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the FOV / Box selector from a saved .h5/.sav box file.")
    parser.add_argument("entry_box", nargs="?", help="Path to the saved .h5 or .sav box file.")
    parser.add_argument("--pick", action="store_true", help="Open a file picker even if a path is provided.")
    parser.add_argument("--dir", dest="start_dir", help="Initial directory for the file picker.")
    parser.add_argument(
        "--refmaps-path",
        "--ref-map-path",
        action="append",
        dest="refmaps_path",
        default=[],
        help=(
            "Additional FITS file or directory of FITS files to expose as "
            "filesystem context maps. May be passed more than once."
        ),
    )
    args = parser.parse_args()

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication([])
        owns_app = True

    entry_arg = args.entry_box
    if args.pick or not entry_arg:
        picked = _pick_entry_file(entry_arg, args.start_dir)
        if not picked:
            return 0
        entry_arg = picked

    entry_path = Path(entry_arg).expanduser().resolve()
    try:
        session_input = _build_session_input(entry_path, ref_map_paths=args.refmaps_path)
    except Exception as exc:
        QMessageBox.critical(
            None,
            "Incompatible Model",
            str(exc),
        )
        return 2
    dialog = FovBoxSelectorDialog(session_input=session_input, entry_box_path=entry_path)
    dialog.setWindowTitle(f"FOV / Box Selector - {entry_path.name}")
    def _on_save_as_clicked() -> None:
        out_path = _pick_save_as_h5_path(
            dialog,
            default_stem=entry_path.stem,
        )
        if out_path is None:
            return
        try:
            result = dialog.current_selection_snapshot()
            line_seeds = dialog.committed_line_seeds()
            fov_box = dialog.current_fov_box_selection()
            observer_state = dialog.current_observer_persistence_state()
            _persist_selector_result_to_entry(
                entry_path,
                result,
                line_seeds=line_seeds,
                fov_box=fov_box,
                observer_state=observer_state,
                output_path=out_path,
                external_ref_map_paths=dialog.external_ref_map_paths(),
                session_refmaps=dialog.current_session_refmaps(),
            )
            QMessageBox.information(
                dialog,
                "Model Saved",
                f"Saved updated model to:\n{out_path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                dialog,
                "Save Failed",
                f"Failed to save model to {out_path}:\n{exc}",
            )

    dialog.set_save_as_callback(_on_save_as_clicked, text="Save As")
    dialog.set_accept_button_text("Apply && Close")

    def _persist_result_if_needed() -> None:
        if dialog.result() != QDialog.Accepted:
            return
        result = dialog.accepted_selection()
        if result is None:
            return
        line_seeds = dialog.committed_line_seeds()
        fov_box = dialog.current_fov_box_selection()
        observer_state = dialog.current_observer_persistence_state()
        if entry_path.suffix.lower() != ".h5":
            btn = QMessageBox.question(
                dialog,
                "Save As Required",
                "This model cannot be updated in place.\n\n"
                "Save the updated model (with your FOV changes) to a new .h5 file?",
                QMessageBox.Save | QMessageBox.Discard,
                QMessageBox.Save,
            )
            if btn != QMessageBox.Save:
                return
            out_path = _pick_save_as_h5_path(
                dialog,
                default_stem=entry_path.stem,
            )
            if out_path is None:
                return
            try:
                _persist_selector_result_to_entry(
                    entry_path,
                    result,
                    line_seeds=line_seeds,
                    fov_box=fov_box,
                    observer_state=observer_state,
                    output_path=out_path,
                    external_ref_map_paths=dialog.external_ref_map_paths(),
                    session_refmaps=dialog.current_session_refmaps(),
                )
            except Exception as exc:
                QMessageBox.warning(
                    dialog,
                    "Save Failed",
                    f"Failed to save model to {out_path}:\n{exc}",
                )
            return
        try:
            _persist_selector_result_to_entry(
                entry_path,
                result,
                line_seeds=line_seeds,
                fov_box=fov_box,
                observer_state=observer_state,
                external_ref_map_paths=dialog.external_ref_map_paths(),
                session_refmaps=dialog.current_session_refmaps(),
            )
        except Exception as exc:
            QMessageBox.warning(
                dialog,
                "Save Failed",
                f"Failed to write updated observer FOV metadata:\n{exc}",
            )

    if owns_app:
        dialog.finished.connect(lambda _code: (_persist_result_if_needed(), app.quit()))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        app.exec_()
    else:
        accepted = dialog.exec_() == QDialog.Accepted
        if accepted:
            _persist_result_if_needed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
