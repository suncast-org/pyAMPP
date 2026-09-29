"""Utilities for adding external FITS maps to pyAMPP HDF5 refmaps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map

from pyampp.geometry.contract import (
    GeometryContract,
    complete_geometry_contract,
    infer_obstime,
    world_corners_from_geometry_contract,
)
from pyampp.gxbox.boxutils import load_sunpy_map_compat


@dataclass(frozen=True)
class AddedRefmap:
    """Summary for one external FITS file embedded as a refmap."""

    map_id: str
    source_path: Path
    data_shape: tuple[int, ...]
    data_dtype: str


@dataclass(frozen=True)
class RemovedRefmap:
    """Summary for one embedded refmap removed from a model HDF5 file."""

    map_id: str


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

# Source-FITS audit cards stored in an embedded refmap ``wcs_header``.
# ``SRC_RELPATH`` is longer than 8 characters, so Astropy writes it as a
# HIERARCH card. ``SRC_ROOT`` is ``data-dir`` (JSOC / ``--data-dir`` cache)
# or ``model-dir`` (the model file's directory, or ``--gxmodel-dir``).
REFMAP_SRC_PATH_KEY = "SRC_PATH"
REFMAP_SRC_RELPATH_KEY = "SRC_RELPATH"
REFMAP_SRC_ROOT_KEY = "SRC_ROOT"
REFMAP_SRC_ROOT_DATA_DIR = "data-dir"
REFMAP_SRC_ROOT_MODEL_DIR = "model-dir"
_CROP_REFMAP_IGNORED = object()


def add_fits_refmaps_to_h5(
    h5_path: PathLike,
    fits_paths: Iterable[PathLike],
    *,
    crop_refmap: str | None = _CROP_REFMAP_IGNORED,
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
        Deprecated and ignored. Embeds are cropped from model box corners.
        Passing this argument emits ``DeprecationWarning``.
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

    if crop_refmap is not _CROP_REFMAP_IGNORED:
        warnings.warn(
            "crop_refmap is ignored; refmaps are cropped from model box corners. "
            "Stop passing crop_refmap.",
            DeprecationWarning,
            stacklevel=2,
        )

    h5_path = Path(h5_path)
    paths = [Path(p) for p in fits_paths]
    if not paths:
        return []

    with h5py.File(h5_path, "r+") as h5f:
        model_ctx = _model_context_from_open_h5(h5f)
        model_obstime = model_obstime_from_base_index(model_ctx)
        box_corners_world = box_corners_world_from_model(model_ctx)
        data_dir, gxmodel_dir = _source_roots_from_model_context(model_ctx)
        refmaps = h5f.require_group("refmaps")

        next_order = _next_refmap_order(refmaps)
        out: list[AddedRefmap] = []
        for idx, path in enumerate(paths):
            smap = load_sunpy_map_compat(path)
            map_id = _resolve_map_id(path, smap, map_ids, idx)
            payload = build_refmap_payload_for_model(
                smap,
                model_obstime=model_obstime,
                box_corners_world=box_corners_world,
                source_path=path,
                data_dir=data_dir,
                model_dir=h5_path.parent,
                gxmodel_dir=gxmodel_dir,
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
    crop_refmap: str | None = _CROP_REFMAP_IGNORED,
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
    kwargs: dict[str, Any] = {"map_ids": map_ids, "overwrite": overwrite}
    if crop_refmap is not _CROP_REFMAP_IGNORED:
        kwargs["crop_refmap"] = crop_refmap
    return add_fits_refmaps_to_h5(h5_path, paths, **kwargs)


def list_embedded_refmap_ids(h5_path: PathLike) -> list[str]:
    """Return sorted ``refmaps/<map_id>`` group names stored in a model HDF5 file."""

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5f:
        refmaps = h5f.get("refmaps")
        if not isinstance(refmaps, h5py.Group):
            return []
        return sorted(str(name) for name in refmaps.keys())


def remove_refmaps_from_h5(
    h5_path: PathLike,
    map_ids: Iterable[str] | None = None,
    *,
    remove_all: bool = False,
    telescope: str | None = None,
    missing_ok: bool = True,
) -> list[RemovedRefmap]:
    """Remove embedded refmaps from ``refmaps/`` of a model HDF5 file.

    Exactly one selection mode must be provided:

    - ``map_ids``: remove the listed ``refmaps/<map_id>`` groups
    - ``telescope``: remove refmaps whose FITS ``TELESCOP`` header contains the
      token (case-insensitive), e.g. ``\"STEREO\"`` for STEREO/STEREO-A maps
    - ``remove_all=True``: remove every embedded refmap

    Parameters
    ----------
    h5_path
        pyAMPP model HDF5 file to modify in place.
    map_ids
        Explicit refmap ids to delete.
    remove_all
        Delete all groups under ``refmaps/`` when true.
    telescope
        Delete refmaps whose embedded WCS header matches this telescope token.
    missing_ok
        When false, raise ``KeyError`` if a requested refmap id is absent or if
        ``refmaps/`` does not exist.

    Returns
    -------
    list[RemovedRefmap]
        One summary entry per deleted refmap, in sorted map-id order.
    """

    if sum(bool(x) for x in (map_ids is not None, remove_all, telescope is not None)) != 1:
        raise ValueError("Provide exactly one of map_ids, telescope=..., or remove_all=True")

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r+") as h5f:
        refmaps = h5f.get("refmaps")
        if not isinstance(refmaps, h5py.Group):
            if missing_ok:
                return []
            raise KeyError("refmaps group not found")

        if remove_all:
            ids_to_remove = sorted(str(name) for name in refmaps.keys())
        elif telescope is not None:
            token = str(telescope).strip().upper()
            if not token:
                raise ValueError("telescope token must be non-empty")
            ids_to_remove = []
            for name in refmaps:
                header = _refmap_header_from_group(refmaps[name])
                tele = str(header.get("TELESCOP") or "").upper() if header is not None else ""
                if token in tele:
                    ids_to_remove.append(str(name))
            ids_to_remove.sort()
        else:
            ids_to_remove = sorted({_sanitize_map_id(item) for item in map_ids or ()})

        removed: list[RemovedRefmap] = []
        for map_id in ids_to_remove:
            if map_id not in refmaps:
                if not missing_ok:
                    raise KeyError(f"refmap not found: refmaps/{map_id}")
                continue
            del refmaps[map_id]
            removed.append(RemovedRefmap(map_id=map_id))
    return removed


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
    box_corners_world: SkyCoord | None = None,
    model: Mapping[str, Any] | None = None,
    map_ids: Mapping[PathLike, str] | Sequence[str] | MapIdFactory | None = None,
    pad: float = 1.1,
    pangle_policy: str = "auto",
    recursive: bool = False,
    generic: bool = True,
    data_dir: PathLike | None = None,
    model_dir: PathLike | None = None,
    gxmodel_dir: PathLike | None = None,
    # Deprecated legacy kwargs kept for call-site compatibility during migration.
    target_fov: tuple[SkyCoord, SkyCoord] | None = None,
    target_template=None,
    reproject_algorithm: str = "adaptive",
) -> dict[str, dict[str, Any]]:
    """Load FITS refmaps from paths and build spatially cropped embed payloads."""

    if box_corners_world is None:
        if model is None:
            raise ValueError(
                "build_fits_refmaps_for_model requires box_corners_world or model "
                "with resolvable geometry_contract."
            )
        box_corners_world = box_corners_world_from_model(model)
    if target_fov is not None or target_template is not None:
        raise ValueError(
            "target_fov/target_template are no longer supported; pass box_corners_world "
            "or model geometry instead."
        )
    _ = reproject_algorithm

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
            box_corners_world=box_corners_world,
            source_path=path,
            pad=pad,
            pangle_policy=pangle_policy,
            data_dir=data_dir,
            model_dir=model_dir,
            gxmodel_dir=gxmodel_dir,
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
        model_or_h5 = _model_context_from_open_h5(model_or_h5)
    if isinstance(model_or_h5, Mapping):
        return infer_obstime(dict(model_or_h5))
    return None


def _geometry_contract_from_model(model_dict: Mapping[str, Any]) -> GeometryContract | None:
    metadata = model_dict.get("metadata")
    if isinstance(metadata, Mapping):
        contract = metadata.get("geometry_contract")
        if isinstance(contract, GeometryContract):
            return contract
        if isinstance(contract, Mapping):
            try:
                return GeometryContract.from_dict(dict(contract))
            except Exception:
                pass
    return complete_geometry_contract(dict(model_dict), strict=False)


def box_corners_world_from_model(
    model_or_h5: Any,
    *,
    obstime: str | Time | None = None,
) -> SkyCoord:
    """Return model red-box world corners from geometry metadata."""

    if isinstance(model_or_h5, (h5py.File, h5py.Group)):
        model_dict = _model_context_from_open_h5(model_or_h5)
    elif isinstance(model_or_h5, Mapping):
        model_dict = dict(model_or_h5)
    else:
        raise TypeError("model_or_h5 must be a mapping or open HDF5 handle")

    contract = _geometry_contract_from_model(model_dict)
    if contract is None:
        raise ValueError(
            "Cannot resolve model red-box corners: geometry_contract is missing "
            "and could not be inferred from base/index and corona metadata."
        )
    when = obstime
    if when is None:
        when = model_obstime_from_base_index(model_dict) or contract.obstime
    world = world_corners_from_geometry_contract(contract, obstime=when)
    if world is None:
        raise ValueError("Could not build world corners from geometry contract.")
    return world


def _model_context_from_open_h5(h5f: h5py.Group) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    base = h5f.get("base")
    if isinstance(base, h5py.Group):
        ctx["base"] = {}
        for key in ("index", "index_header", "wcs_header"):
            if key in base:
                ctx["base"][key] = _decode_h5_string(base[key][()])
    corona = h5f.get("corona")
    if isinstance(corona, h5py.Group):
        ctx["corona"] = {}
        if "dr" in corona:
            ctx["corona"]["dr"] = np.asarray(corona["dr"])
        for key in ("bx", "by", "bz"):
            if key in corona:
                ctx["corona"][key] = np.asarray(corona[key])
                break
    metadata = h5f.get("metadata")
    if isinstance(metadata, h5py.Group):
        ctx["metadata"] = {}
        for key in metadata.keys():
            item = metadata[key]
            if isinstance(item, h5py.Group):
                group_data = {}
                for subkey in item.keys():
                    value = item[subkey][()]
                    if isinstance(value, (bytes, np.bytes_)):
                        value = value.decode("utf-8", "ignore")
                    group_data[subkey] = value
                ctx["metadata"][key] = group_data
            else:
                value = item[()]
                if isinstance(value, (bytes, np.bytes_)):
                    value = value.decode("utf-8", "ignore")
                ctx["metadata"][key] = value
    return ctx


def build_refmap_payload_for_model(
    smap,
    *,
    model_obstime: str | Time | None = None,
    box_corners_world: SkyCoord | None = None,
    source_path: Path | None = None,
    pad: float = 1.1,
    pangle_policy: str = "auto",
    data_dir: PathLike | None = None,
    model_dir: PathLike | None = None,
    gxmodel_dir: PathLike | None = None,
    # Deprecated legacy kwargs kept for call-site compatibility during migration.
    target_template=None,
    target_fov: tuple[SkyCoord, SkyCoord] | None = None,
    reproject_algorithm: str = "adaptive",
) -> dict[str, Any]:
    """Build a spatially cropped refmap payload at native map observer/time."""

    if box_corners_world is None:
        raise ValueError("box_corners_world is required for refmap embedding")
    if target_fov is not None or target_template is not None:
        raise ValueError(
            "target_fov/target_template are no longer supported; pass box_corners_world instead."
        )
    _ = reproject_algorithm

    from pyampp.io.refmap_crop import crop_refmap_spatial

    source_obstime = _map_date_isot(smap)
    result = crop_refmap_spatial(
        smap=smap,
        box_corners_world=box_corners_world,
        model_obstime=model_obstime,
        pad=pad,
        pangle_policy=pangle_policy,
    )
    cropped = result.cropped_map
    header_text = _refmap_wcs_header(
        cropped,
        source_path=source_path,
        model_obstime=model_obstime or result.model_obstime,
        source_obstime=source_obstime,
        aligned_to_model=False,
        data_dir=data_dir,
        model_dir=model_dir,
        gxmodel_dir=gxmodel_dir,
    )
    return {"data": np.asarray(cropped.data), "wcs_header": header_text}


def apply_refmap_source_cards(
    header: fits.Header,
    source_path: PathLike,
    *,
    data_dir: PathLike | None = None,
    model_dir: PathLike | None = None,
    gxmodel_dir: PathLike | None = None,
) -> None:
    """Write ``SRC_PATH`` and, when the file is under a known root, relative cards.

    ``SRC_ROOT=data-dir`` means ``SRC_RELPATH`` is relative to ``--data-dir``
    (the JSOC cache root). ``SRC_ROOT=model-dir`` means it is relative to the
    model file's directory or ``--gxmodel-dir``. When the file is inside more
    than one root, the longer root wins.
    """

    source = _as_directory(source_path)
    if source is None:
        return
    header[REFMAP_SRC_PATH_KEY] = str(source)
    roots: list[tuple[str, Path]] = []
    data_root = _as_directory(data_dir)
    if data_root is not None:
        roots.append((REFMAP_SRC_ROOT_DATA_DIR, data_root))
    for candidate in (model_dir, gxmodel_dir):
        model_root = _as_directory(candidate)
        if model_root is not None:
            roots.append((REFMAP_SRC_ROOT_MODEL_DIR, model_root))
    best: tuple[int, str, str] | None = None
    for label, root in roots:
        relative = _relative_posix_under(source, root)
        if relative is None:
            continue
        rank = len(str(root))
        if best is None or rank > best[0]:
            best = (rank, label, relative)
    if best is None:
        header.pop(REFMAP_SRC_RELPATH_KEY, None)
        header.pop(REFMAP_SRC_ROOT_KEY, None)
        return
    _label, root_name, relative = best
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        header[REFMAP_SRC_RELPATH_KEY] = relative
    header[REFMAP_SRC_ROOT_KEY] = root_name


def resolve_embedded_refmap_source(
    header: fits.Header | str,
    *,
    data_dir: PathLike | None = None,
    model_dir: PathLike | None = None,
    gxmodel_dir: PathLike | None = None,
) -> Path | None:
    """Resolve an embedded refmap's source FITS, or return None.

    Order: ``SRC_RELPATH`` under the current root named by ``SRC_ROOT``,
    then ``SRC_PATH``. A missing file is skipped.
    """

    parsed = _header_for_source_cards(header)
    if parsed is None:
        return None
    relative = _header_card_text(parsed, REFMAP_SRC_RELPATH_KEY)
    root_name = _header_card_text(parsed, REFMAP_SRC_ROOT_KEY)
    if relative and root_name == REFMAP_SRC_ROOT_DATA_DIR:
        found = _existing_file_under(data_dir, relative)
        if found is not None:
            return found
    elif relative and root_name == REFMAP_SRC_ROOT_MODEL_DIR:
        for root in (model_dir, gxmodel_dir):
            found = _existing_file_under(root, relative)
            if found is not None:
                return found
    absolute = _header_card_text(parsed, REFMAP_SRC_PATH_KEY)
    if absolute:
        try:
            candidate = Path(absolute).expanduser()
        except Exception:
            candidate = None
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    return None


def _source_roots_from_model_context(model_ctx: Mapping[str, Any]) -> tuple[str | None, str | None]:
    metadata = model_ctx.get("metadata") if isinstance(model_ctx, Mapping) else None
    execute = ""
    if isinstance(metadata, Mapping):
        execute = metadata.get("execute") or ""
        if isinstance(execute, (bytes, np.bytes_)):
            execute = _decode_h5_string(execute)
    if not str(execute).strip():
        return None, None
    from pyampp.gxbox.gx_fov2box import _extract_execute_paths

    data_dir, gxmodel_dir = _extract_execute_paths(str(execute))
    return data_dir, gxmodel_dir


def _as_directory(value: PathLike | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Path(text).expanduser().resolve()
    except Exception:
        return None


def _relative_posix_under(source: Path, root: Path) -> str | None:
    try:
        relative = source.resolve().relative_to(root.resolve())
    except Exception:
        return None
    if relative == Path("."):
        return None
    return relative.as_posix()


def _existing_file_under(root: PathLike | None, relative: str) -> Path | None:
    base = _as_directory(root)
    if base is None or not str(relative).strip():
        return None
    try:
        candidate = (base / relative).resolve()
    except Exception:
        return None
    if candidate.is_file():
        return candidate
    return None


def _header_for_source_cards(header: fits.Header | str) -> fits.Header | None:
    if isinstance(header, fits.Header):
        return header
    text = str(header or "").replace("\\n", "\n")
    if not text.strip():
        return None
    try:
        return fits.Header.fromstring(text, sep="\n")
    except Exception:
        return None


def _header_card_text(header: fits.Header, key: str) -> str | None:
    try:
        value = header.get(key)
    except Exception:
        return None
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _refmap_wcs_header(
    smap,
    *,
    source_path: Path | None = None,
    model_obstime: str | Time | None = None,
    source_obstime: str | None = None,
    aligned_to_model: bool = False,
    data_dir: PathLike | None = None,
    model_dir: PathLike | None = None,
    gxmodel_dir: PathLike | None = None,
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
    try:
        obs = getattr(smap, "observer_coordinate", None)
        obs_time = getattr(smap, "date", None)
        if obs is not None and obs_time is not None:
            obs_hgs = obs.transform_to(HeliographicStonyhurst(obstime=obs_time))
            header["HGLN_OBS"] = float(obs_hgs.lon.to_value(u.deg))
            header["HGLT_OBS"] = float(obs_hgs.lat.to_value(u.deg))
    except Exception:
        pass
    if source_path is not None:
        header["HISTORY"] = f"Embedded by pyampp.io.refmaps from {source_path}"
        apply_refmap_source_cards(
            header,
            source_path,
            data_dir=data_dir,
            model_dir=model_dir,
            gxmodel_dir=gxmodel_dir,
        )
    if source_obstime:
        header["SRC_DATE"] = source_obstime
    if model_obstime is not None:
        try:
            header["MODELT"] = Time(model_obstime).isot
        except Exception:
            header["MODELT"] = str(model_obstime)
    header["PYEMBED"] = True
    header["PYALIGN"] = bool(aligned_to_model)
    return header.tostring(sep="\n", endcard=True)


def _map_date_isot(smap) -> str | None:
    try:
        return smap.date.isot
    except Exception:
        return None


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


def _refmap_header_from_group(group: h5py.Group) -> fits.Header | None:
    if "wcs_header" not in group:
        return None
    try:
        text = _decode_h5_string(group["wcs_header"][()]).replace("\\n", "\n")
        return fits.Header.fromstring(text, sep="\n")
    except Exception:
        return None
