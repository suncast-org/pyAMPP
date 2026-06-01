"""Utilities for adding external FITS maps to pyAMPP HDF5 refmaps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
from sunpy.map import Map, make_fitswcs_header

from pyampp.geometry.contract import infer_obstime
from pyampp.gxbox.boxutils import load_sunpy_map_compat


@dataclass(frozen=True)
class AddedRefmap:
    """Summary for one external FITS file embedded as a refmap."""

    map_id: str
    source_path: Path
    data_shape: tuple[int, ...]
    data_dtype: str


PathLike = str | Path
MapIdFactory = Callable[[Path, object], str]


_SOURCE_HEADER_KEYS = (
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
)


def add_fits_refmaps_to_h5(
    h5_path: PathLike,
    fits_paths: Iterable[PathLike],
    *,
    crop_refmap: str | None = "Bz_reference",
    map_ids: Mapping[PathLike, str] | Sequence[str] | MapIdFactory | None = None,
    overwrite: bool = False,
) -> list[AddedRefmap]:
    """Align external FITS maps and embed them in ``refmaps/`` of a model HDF5.

    Parameters
    ----------
    h5_path
        pyAMPP model HDF5 file to modify in place.
    fits_paths
        External FITS file paths to add.
    crop_refmap
        Existing refmap whose WCS footprint is used as the alignment target
        for Earth-line-of-sight maps. Use ``None`` to embed maps without a
        model-FOV target.
    map_ids
        Optional map-id source. This can be a mapping from path to id, a
        sequence aligned with ``fits_paths``, or a callable ``(path, sunpy_map)
        -> str``. When omitted, ids are inferred from FITS metadata and names.
    overwrite
        Replace existing ``refmaps/<map_id>`` groups when true.

    Returns
    -------
    list[AddedRefmap]
        One summary entry per embedded FITS file.
    """

    h5_path = Path(h5_path)
    paths = [Path(p) for p in fits_paths]
    if not paths:
        return []

    template = None
    with h5py.File(h5_path, "r+") as h5f:
        model_obstime = model_obstime_from_base_index(h5f)
        refmaps = h5f.require_group("refmaps")
        if crop_refmap is not None:
            template = _load_template_refmap(refmaps, crop_refmap)

        next_order = _next_refmap_order(refmaps)
        out: list[AddedRefmap] = []
        for idx, path in enumerate(paths):
            smap = load_sunpy_map_compat(path)
            map_id = _resolve_map_id(path, smap, map_ids, idx)
            payload = build_refmap_payload_for_model(
                smap,
                model_obstime=model_obstime,
                target_template=template,
                source_path=path,
            )

            if map_id in refmaps:
                if not overwrite:
                    raise ValueError(f"refmap already exists: {map_id}")
                del refmaps[map_id]

            group = refmaps.create_group(map_id, track_order=True)
            group.attrs["order_index"] = np.int64(next_order)
            group.attrs["source_path"] = str(path)
            next_order += 1
            group.create_dataset("data", data=np.asarray(payload["data"]))
            group.create_dataset("wcs_header", data=np.bytes_(payload["wcs_header"]))
            out.append(
                AddedRefmap(
                    map_id=map_id,
                    source_path=path,
                    data_shape=tuple(np.asarray(payload["data"]).shape),
                    data_dtype=str(np.asarray(payload["data"]).dtype),
                )
            )
    return out


def add_fits_refmaps_from_dir_to_h5(
    h5_path: PathLike,
    fits_dir: PathLike,
    *,
    pattern: str | None = None,
    recursive: bool = False,
    crop_refmap: str | None = "Bz_reference",
    map_ids: Mapping[PathLike, str] | Sequence[str] | MapIdFactory | None = None,
    overwrite: bool = False,
) -> list[AddedRefmap]:
    """Embed FITS files from a directory into ``refmaps/`` of an HDF5 model.

    When ``pattern`` is omitted, all supported FITS extensions are included:
    ``.fits``, ``.fit``, ``.fts``, their uppercase variants, and ``.fits.gz``.
    """

    fits_dir = Path(fits_dir)
    if pattern is None:
        paths = discover_fits_refmap_paths([fits_dir], recursive=recursive)
    else:
        globber = fits_dir.rglob if recursive else fits_dir.glob
        paths = sorted(p for p in globber(pattern) if p.is_file())
    return add_fits_refmaps_to_h5(
        h5_path,
        paths,
        crop_refmap=crop_refmap,
        map_ids=map_ids,
        overwrite=overwrite,
    )


def discover_fits_refmap_paths(paths: Iterable[PathLike], *, recursive: bool = False) -> list[Path]:
    """Resolve FITS files from a mix of file and directory paths."""

    out: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths or ():
        path = Path(raw_path).expanduser()
        candidates: list[Path]
        if path.is_dir():
            globber = path.rglob if recursive else path.glob
            candidates = sorted(
                p
                for pattern in (
                    "*.fits",
                    "*.fit",
                    "*.fts",
                    "*.fits.gz",
                    "*.FITS",
                    "*.FIT",
                    "*.FTS",
                    "*.FITS.GZ",
                )
                for p in globber(pattern)
                if p.is_file()
            )
        elif path.is_file():
            candidates = [path]
        else:
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def infer_fits_refmap_id(path: PathLike, smap=None, *, generic: bool = True) -> str | None:
    """Infer the canonical refmap id for a FITS map.

    Known observation products are identified from FITS metadata. When
    ``generic`` is true, unknown FITS files fall back to a sanitized file stem;
    when false, unknown files are ignored by returning ``None``.
    """

    path = Path(path)
    meta = getattr(smap, "meta", {}) or {}
    wavelength = _meta_get(meta, "WAVELNTH")
    telescope = str(_meta_get(meta, "TELESCOP") or "").upper()
    instrument = str(_meta_get(meta, "INSTRUME") or "").upper()
    if wavelength is not None and ("AIA" in telescope or "AIA" in instrument):
        try:
            return f"AIA_{int(round(float(wavelength)))}"
        except Exception:
            pass

    freq = _meta_get(meta, "CRVAL3")
    unit = str(_meta_get(meta, "CUNIT3") or "").strip().upper()
    if freq is not None and unit == "HZ" and ("EOVSA" in telescope or "EOVSA" in instrument):
        try:
            return f"EOVSA_f{float(freq) / 1e9:.3f}GHz"
        except Exception:
            pass

    if smap is None:
        try:
            with fits.open(path) as hdul:
                header = None
                for hdu in hdul:
                    hdr = hdu.header
                    if hdr.get("WAVELNTH") is not None or hdr.get("CRVAL3") is not None:
                        header = hdr
                        break
                if header is None:
                    header = hdul[0].header
                return infer_fits_refmap_id(path, _HeaderOnlyMap(header), generic=generic)
        except Exception:
            pass

    if not generic:
        return None
    return _sanitize_map_id(path.stem)


def discover_fits_refmap_map_ids(
    paths: Iterable[PathLike],
    *,
    recursive: bool = False,
    generic: bool = True,
) -> dict[Path, str]:
    """Resolve FITS refmap paths and canonical ids using the shared IO policy."""

    out: dict[Path, str] = {}
    for path in discover_fits_refmap_paths(paths, recursive=recursive):
        map_id = infer_fits_refmap_id(path, generic=generic)
        if map_id:
            out[path] = map_id
    return out


def build_fits_refmaps_for_model(
    paths: Iterable[PathLike],
    *,
    model_obstime: str | Time | None,
    target_fov: tuple[SkyCoord, SkyCoord] | None = None,
    target_template=None,
    map_ids: Mapping[PathLike, str] | Sequence[str] | MapIdFactory | None = None,
    reproject_algorithm: str = "adaptive",
    recursive: bool = False,
    generic: bool = True,
) -> dict[str, dict[str, Any]]:
    """Load FITS refmaps from paths and build model-aligned payloads."""

    if map_ids is None:
        discovered = discover_fits_refmap_map_ids(paths, recursive=recursive, generic=generic)
        fits_paths = list(discovered.keys())
        resolved_map_ids: Mapping[PathLike, str] | Sequence[str] | MapIdFactory | None = discovered
    else:
        fits_paths = discover_fits_refmap_paths(paths, recursive=recursive)
        resolved_map_ids = map_ids
    out: dict[str, dict[str, Any]] = {}
    for idx, path in enumerate(fits_paths):
        smap = load_sunpy_map_compat(path)
        map_id = _resolve_map_id(path, smap, resolved_map_ids, idx)
        out[map_id] = build_refmap_payload_for_model(
            smap,
            model_obstime=model_obstime,
            target_template=target_template,
            target_fov=target_fov,
            source_path=path,
            reproject_algorithm=reproject_algorithm,
        )
    return out


def _load_template_refmap(refmaps: h5py.Group, crop_refmap: str):
    if crop_refmap not in refmaps:
        raise KeyError(f"crop refmap not found: refmaps/{crop_refmap}")
    group = refmaps[crop_refmap]
    if "data" not in group or "wcs_header" not in group:
        raise KeyError(f"crop refmap is missing data/wcs_header: refmaps/{crop_refmap}")
    data = np.asarray(group["data"])
    header_text = _decode_h5_string(group["wcs_header"][()])
    header = fits.Header.fromstring(header_text, sep="\n")
    return Map(np.zeros(data.shape, dtype=np.float32), header)


def model_obstime_from_base_index(model_or_h5: Any) -> str | None:
    """Return the model time from canonical ``base/index`` metadata."""

    if isinstance(model_or_h5, (h5py.File, h5py.Group)):
        model = {"base": {}}
        if "base" in model_or_h5 and "index" in model_or_h5["base"]:
            model["base"]["index"] = _decode_h5_string(model_or_h5["base/index"][()])
        elif "base" in model_or_h5 and "index_header" in model_or_h5["base"]:
            model["base"]["index_header"] = _decode_h5_string(model_or_h5["base/index_header"][()])
        return infer_obstime(model)
    if isinstance(model_or_h5, Mapping):
        return infer_obstime(dict(model_or_h5))
    return None


def build_refmap_payload_for_model(
    smap,
    *,
    model_obstime: str | Time | None,
    target_template=None,
    target_fov: tuple[SkyCoord, SkyCoord] | None = None,
    source_path: Path | None = None,
    reproject_algorithm: str = "adaptive",
) -> dict[str, Any]:
    """Build a refmap payload using pyAMPP's model-time alignment policy.

    Earth-line-of-sight maps are solar-rotated to the model time and remapped
    onto a target FOV WCS. Non-Earth maps keep their native WCS and data so the
    viewer can display them in their own spacecraft LOS mode.
    """

    aligned = smap
    should_reproject = _is_earth_los_map(smap)
    if should_reproject and model_obstime is not None:
        header = _model_fov_header_for_refmap(
            smap,
            model_obstime=model_obstime,
            target_template=target_template,
            target_fov=target_fov,
        )
        if header is not None:
            with propagate_with_solar_surface():
                aligned = smap.reproject_to(
                    header,
                    algorithm=reproject_algorithm,
                    roundtrip_coords=False,
                )
    elif target_template is not None and should_reproject:
        aligned = _crop_to_template_footprint(smap, target_template)

    header_text = _refmap_wcs_header(
        aligned,
        source_path=source_path,
        model_obstime=model_obstime,
        source_obstime=_map_date_isot(smap),
        aligned_to_model=bool(aligned is not smap and should_reproject),
    )
    return {"data": np.asarray(aligned.data), "wcs_header": header_text}


def _model_fov_header_for_refmap(
    smap,
    *,
    model_obstime: str | Time,
    target_template=None,
    target_fov: tuple[SkyCoord, SkyCoord] | None = None,
):
    observer = _earth_like_observer(smap)
    obs_time = Time(model_obstime)
    if target_template is not None:
        ny, nx = np.asarray(target_template.data).shape
        bl = target_template.pixel_to_world(0 * u.pix, 0 * u.pix)
        tr = target_template.pixel_to_world((nx - 1) * u.pix, (ny - 1) * u.pix)
    elif target_fov is not None:
        bl, tr = target_fov
    else:
        return None

    try:
        bl_hpc = bl.transform_to(Helioprojective(observer=observer, obstime=obs_time))
        tr_hpc = tr.transform_to(Helioprojective(observer=observer, obstime=obs_time))
        x0, x1 = sorted([bl_hpc.Tx.to_value(u.arcsec), tr_hpc.Tx.to_value(u.arcsec)])
        y0, y1 = sorted([bl_hpc.Ty.to_value(u.arcsec), tr_hpc.Ty.to_value(u.arcsec)])
        scale_x = abs(float(smap.scale.axis1.to_value(u.arcsec / u.pix)))
        scale_y = abs(float(smap.scale.axis2.to_value(u.arcsec / u.pix)))
    except Exception:
        return None
    if not all(np.isfinite(v) and v > 0 for v in (scale_x, scale_y)):
        return None

    width = max(float(x1 - x0), scale_x)
    height = max(float(y1 - y0), scale_y)
    nx = max(2, int(np.ceil(width / scale_x)) + 1)
    ny = max(2, int(np.ceil(height / scale_y)) + 1)
    center = SkyCoord(
        Tx=(0.5 * (x0 + x1)) * u.arcsec,
        Ty=(0.5 * (y0 + y1)) * u.arcsec,
        frame=Helioprojective(observer=observer, obstime=obs_time),
    )
    header = make_fitswcs_header(
        np.empty((ny, nx), dtype=np.float32),
        center,
        scale=u.Quantity([scale_x, scale_y], u.arcsec / u.pix),
    )
    header["DATE-OBS"] = obs_time.isot
    header["DATE_OBS"] = obs_time.isot
    try:
        header["RSUN_REF"] = float(smap.rsun_meters.to_value(u.m))
    except Exception:
        pass
    return header


def _crop_to_template_footprint(smap, template):
    ny, nx = np.asarray(template.data).shape
    bottom_left = template.pixel_to_world(0 * u.pix, 0 * u.pix)
    top_right = template.pixel_to_world((nx - 1) * u.pix, (ny - 1) * u.pix)
    return smap.submap(
        bottom_left.transform_to(smap.coordinate_frame),
        top_right=top_right.transform_to(smap.coordinate_frame),
    )


def _refmap_wcs_header(
    smap,
    *,
    source_path: Path | None = None,
    model_obstime: str | Time | None = None,
    source_obstime: str | None = None,
    aligned_to_model: bool = False,
) -> str:
    try:
        header = smap.wcs.to_header()
    except Exception:
        header = fits.Header()
    meta = getattr(smap, "meta", {}) or {}
    for key in _SOURCE_HEADER_KEYS:
        value = _meta_get(meta, key)
        if value is not None:
            header[key] = value

    date = getattr(smap, "date", None)
    if date is not None:
        try:
            header["DATE-OBS"] = date.isot
            header["DATE_OBS"] = date.isot
        except Exception:
            pass

    try:
        if getattr(smap, "rsun_obs", None) is not None:
            header["RSUN_OBS"] = float(u.Quantity(smap.rsun_obs).to_value(u.arcsec))
    except Exception:
        pass
    try:
        if getattr(smap, "rsun_meters", None) is not None:
            header["RSUN_REF"] = float(u.Quantity(smap.rsun_meters).to_value(u.m))
    except Exception:
        pass
    if source_path is not None:
        header["HISTORY"] = f"Embedded by pyampp.io.refmaps from {source_path}"
    if source_obstime:
        header["SRC_DATE"] = source_obstime
    if model_obstime is not None:
        try:
            header["MODELT"] = Time(model_obstime).isot
        except Exception:
            header["MODELT"] = str(model_obstime)
    header["PYALIGN"] = bool(aligned_to_model)
    return header.tostring(sep="\n", endcard=True)


def _map_date_isot(smap) -> str | None:
    try:
        return smap.date.isot
    except Exception:
        return None


def _is_earth_los_map(smap) -> bool:
    meta = getattr(smap, "meta", {}) or {}
    telescope = str(_meta_get(meta, "TELESCOP") or "").upper()
    instrument = str(_meta_get(meta, "INSTRUME") or "").upper()
    if any(token in telescope or token in instrument for token in ("SDO", "AIA", "HMI", "EOVSA")):
        return True
    try:
        obs = smap.observer_coordinate
        lon = abs(float(obs.lon.to_value(u.deg)))
        lat = abs(float(obs.lat.to_value(u.deg)))
        return lon < 5.0 and lat < 10.0
    except Exception:
        return False


def _earth_like_observer(smap):
    try:
        return smap.observer_coordinate
    except Exception:
        return "earth"


def _resolve_map_id(
    path: Path,
    smap,
    map_ids: Mapping[PathLike, str] | Sequence[str] | MapIdFactory | None,
    index: int,
) -> str:
    if callable(map_ids):
        return _sanitize_map_id(map_ids(path, smap))
    if isinstance(map_ids, Mapping):
        if path in map_ids:
            return _sanitize_map_id(map_ids[path])
        if str(path) in map_ids:
            return _sanitize_map_id(map_ids[str(path)])
        if path.name in map_ids:
            return _sanitize_map_id(map_ids[path.name])
    elif map_ids is not None:
        return _sanitize_map_id(list(map_ids)[index])
    return _infer_map_id(path, smap)


def _infer_map_id(path: Path, smap) -> str:
    return infer_fits_refmap_id(path, smap, generic=True) or _sanitize_map_id(path.stem)


class _HeaderOnlyMap:
    def __init__(self, header: fits.Header):
        self.meta = header


def _sanitize_map_id(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    text = text.strip("_")
    if not text:
        raise ValueError("empty refmap id")
    return text


def _next_refmap_order(refmaps: h5py.Group) -> int:
    orders = []
    for name in refmaps:
        try:
            orders.append(int(refmaps[name].attrs.get("order_index", len(orders))))
        except Exception:
            orders.append(len(orders))
    return max(orders, default=-1) + 1


def _meta_get(meta, key: str):
    for candidate in (key, key.lower(), key.replace("-", "_"), key.replace("_", "-")):
        if candidate in meta:
            value = meta[candidate]
            if value is not None:
                return value
    return None


def _decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode(errors="replace")
    return str(value)
