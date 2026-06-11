import ctypes
import subprocess
import shlex
import shutil
import sys
import os
import signal
import re
import ast
import io
import contextlib
import datetime as dt
import threading
import time as time_mod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, List

import astropy.units as u
import numpy as np
import typer
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from pyAMaFiL.mag_field_lin_fff import MagFieldLinFFF
from pyAMaFiL import mag_field_wrapper as pyamafil_mag_field_wrapper
from pyAMaFiL.mag_field_proc import (
    MagFieldProcessor,
    mfp_util_invert_index_array,
    mfp_util_transpose_index,
)
from sunpy.coordinates import (Heliocentric, HeliographicCarrington, HeliographicStonyhurst,
                               Helioprojective, get_earth)
from sunpy.map import Map

from pyampp.data.downloader import SDOImageDownloader
from pyampp.gx_chromo.combo_model import combo_model
from pyampp.gx_chromo.decompose import decompose
from pyampp.gxbox.box import Box
from pyampp.gxbox.boxutils import (
    hmi_b2ptr,
    hmi_disambig,
    read_b3d_h5,
    write_b3d_h5,
    compute_vertical_current,
    load_sunpy_map_compat,
    map_from_data_header_compat,
    remap_vertical_current_inputs,
)
from pyampp.io import load_model
from pyampp.io.refmaps import (
    build_fits_refmaps_for_model,
    build_refmap_payload_for_model,
    discover_fits_refmap_map_ids,
    model_obstime_from_base_index,
)
from pyampp.io.model import _normalize_loaded_model_dict
from pyampp.gxbox.gx_box2id import gx_box2id
from pyampp.gxbox.observer_restore import (
    build_pb0r_metadata_from_ephemeris,
    resolve_named_observer,
    normalize_observer_key,
)
from pyampp.util.config import (
    DOWNLOAD_DIR,
    GXMODEL_DIR,
    AIA_EUV_PASSBANDS,
    AIA_UV_PASSBANDS,
    IDL_HMI_RSUN_M,
)


app = typer.Typer(help="Run gx_fov2box pipeline without GUI and save model stages.")

_VALID_REPROJECT_ALGORITHMS = ("adaptive", "exact", "interpolation")


def _patch_pyamafil_mag_field_wrapper_for_legacy_libraries() -> None:
    wrapper_classes = []
    for candidate in (MagFieldProcessor.__mro__[1], pyamafil_mag_field_wrapper.MagFieldWrapper):
        if candidate not in wrapper_classes:
            wrapper_classes.append(candidate)

    for wrapper_cls in wrapper_classes:
        if getattr(wrapper_cls, "_pyampp_legacy_nlfff_patch", False):
            continue

        original_init = wrapper_cls.__init__
        original_set_setting = wrapper_cls.set_setting

        def _fallback_init(self, lib_path=""):
            if lib_path == "":
                lib_path = list(Path(pyamafil_mag_field_wrapper.__file__).resolve().parent.glob("WWNLFFFReconstruction*"))[0].as_posix()

            self._MagFieldWrapper__mptr1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
            self._MagFieldWrapper__mptr2 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C")
            self._MagFieldWrapper__mptr3 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags="C")
            self._MagFieldWrapper__mpint1 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C")
            self._MagFieldWrapper__mpint2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C")
            self._MagFieldWrapper__mpint3 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=3, flags="C")
            self._MagFieldWrapper__mp64 = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C")
            self._MagFieldWrapper__mpstr = ctypes.POINTER(ctypes.c_char)
            self._MagFieldWrapper__mvoid = ctypes.c_void_p
            self._MagFieldWrapper__mint = ctypes.c_int32
            self._MagFieldWrapper__mreal = ctypes.c_double
            self._MagFieldWrapper__mdw = ctypes.c_uint32
            self._MagFieldWrapper__m64 = ctypes.c_uint64

            lib_mfw = ctypes.CDLL(lib_path)

            create_func = lib_mfw.utilInitialize
            create_func.argtypes = []
            create_func.restype = self._MagFieldWrapper__mdw

            set_int_func = lib_mfw.utilSetInt
            set_int_func.argtypes = [self._MagFieldWrapper__mpstr, self._MagFieldWrapper__mint]
            set_int_func.restype = self._MagFieldWrapper__mint

            set_double_func = lib_mfw.utilSetDouble
            set_double_func.argtypes = [self._MagFieldWrapper__mpstr, self._MagFieldWrapper__mreal]
            set_double_func.restype = self._MagFieldWrapper__mint

            get_double_func = lib_mfw.utilGetDouble
            get_double_func.argtypes = [self._MagFieldWrapper__mpstr, self._MagFieldWrapper__mptr1]
            get_double_func.restype = self._MagFieldWrapper__mint

            set_setting_func = getattr(lib_mfw, "utilSetSetting", None)
            if set_setting_func is not None:
                set_setting_func.argtypes = [self._MagFieldWrapper__mpstr, self._MagFieldWrapper__mreal]
                set_setting_func.restype = self._MagFieldWrapper__mint

            get_version_func = lib_mfw.utilGetVersion
            get_version_func.argtypes = [self._MagFieldWrapper__mpstr, self._MagFieldWrapper__mint]
            get_version_func.restype = self._MagFieldWrapper__mint

            nlfff_func = lib_mfw.mfoNLFFFCore
            nlfff_func.argtypes = [self._MagFieldWrapper__mpint1, self._MagFieldWrapper__mptr3, self._MagFieldWrapper__mptr3, self._MagFieldWrapper__mptr3]
            nlfff_func.restype = self._MagFieldWrapper__mint

            lines_func = lib_mfw.mfoGetLines
            lines_func.restype = self._MagFieldWrapper__mdw

            self._MagFieldWrapper__lib_path = lib_path
            self._MagFieldWrapper__func_set = {
                "create_func": create_func,
                "set_int_func": set_int_func,
                "set_double_func": set_double_func,
                "get_double_func": get_double_func,
                "set_setting_func": set_setting_func,
                "get_version_func": get_version_func,
                "NLFFF_func": nlfff_func,
                "lines_func": lines_func,
            }
            self._MagFieldWrapper__pointer = create_func()

        def _compat_init(self, lib_path=""):
            try:
                original_init(self, lib_path)
            except AttributeError as exc:
                if "utilSetSetting" not in str(exc):
                    raise
                _fallback_init(self, lib_path)

        def _compat_set_setting(self, prop, value):
            func_set = self._MagFieldWrapper__func_set
            set_setting_func = func_set.get("set_setting_func")
            if set_setting_func is not None:
                return original_set_setting(self, prop, value)
            if isinstance(value, (bool, np.bool_, int, np.integer)) and not isinstance(value, (float, np.floating)):
                return func_set["set_int_func"](prop.encode("utf-8"), np.int32(value))
            return func_set["set_double_func"](prop.encode("utf-8"), np.float64(value))

        wrapper_cls.__init__ = _compat_init
        wrapper_cls.set_setting = _compat_set_setting
        wrapper_cls._pyampp_legacy_nlfff_patch = True


_patch_pyamafil_mag_field_wrapper_for_legacy_libraries()


def _spherical_screen_context_for_observer(observer):
    if hasattr(Helioprojective, "assume_spherical_screen"):
        return Helioprojective.assume_spherical_screen(observer)
    try:
        from sunpy.coordinates.screens import SphericalScreen

        return SphericalScreen(observer)
    except Exception:
        return contextlib.nullcontext()


def _is_offdisk_submap_nan_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "submap" in text
        and "nan" in text
        and (
            "off-disk" in text
            or "off disk" in text
            or "sphericalscreen" in text
            or "spherical_screen" in text
        )
    )


def _submap_with_fov_safe(smap: Map, fov_bottom_left: SkyCoord, fov_top_right: SkyCoord) -> Map:
    frame = smap.coordinate_frame
    try:
        bl = fov_bottom_left.transform_to(frame)
        tr = fov_top_right.transform_to(frame)
        return smap.submap(bl, top_right=tr)
    except ValueError as exc:
        if not _is_offdisk_submap_nan_error(exc):
            raise
        observer = getattr(frame, "observer", None)
        with _spherical_screen_context_for_observer(observer):
            bl = fov_bottom_left.transform_to(frame)
            tr = fov_top_right.transform_to(frame)
            return smap.submap(bl, top_right=tr)


def _fill_nonfinite_2d(arr: Any, *, label: str) -> np.ndarray:
    """Fill non-finite pixels in 2D maps with local 4-neighbor interpolation."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        return a

    invalid = ~np.isfinite(a)
    if not np.any(invalid):
        return a

    n_invalid = int(np.count_nonzero(invalid))
    if not np.any(np.isfinite(a)):
        print(f"Warning: {label} has no finite pixels; using zeros.")
        return np.zeros_like(a, dtype=float)

    filled = a.copy()
    finite_mean = float(np.nanmean(filled))
    filled[invalid] = finite_mean
    unresolved = invalid.copy()

    max_iter = int(filled.shape[0] + filled.shape[1])
    for _ in range(max_iter):
        if not np.any(unresolved):
            break
        up = np.full_like(filled, np.nan, dtype=float)
        down = np.full_like(filled, np.nan, dtype=float)
        left = np.full_like(filled, np.nan, dtype=float)
        right = np.full_like(filled, np.nan, dtype=float)
        up[1:, :] = filled[:-1, :]
        down[:-1, :] = filled[1:, :]
        left[:, 1:] = filled[:, :-1]
        right[:, :-1] = filled[:, 1:]
        neighbor_mean = np.nanmean(np.stack([up, down, left, right], axis=0), axis=0)
        updatable = unresolved & np.isfinite(neighbor_mean)
        if not np.any(updatable):
            break
        filled[updatable] = neighbor_mean[updatable]
        unresolved[updatable] = False

    if np.any(unresolved):
        filled[unresolved] = finite_mean

    print(f"Info: inpainted {n_invalid} non-finite pixels in {label}.")
    return filled


@dataclass
class Fov2BoxConfig:
    time: Optional[str]
    coords: Optional[Tuple[float, float]]
    hpc: bool
    hgc: bool
    hgs: bool
    cea: bool
    top: bool
    box_dims: Optional[Tuple[int, int, int]]
    dx_km: float
    pad_frac: float
    data_dir: str
    gxmodel_dir: str
    nlfff_lib: Optional[str]
    download_backend: str
    drms_sequential: bool
    force_download: bool
    hmi_time_window: float
    aia_time_window: float
    entry_box: Optional[str]
    save_empty_box: bool
    save_potential: bool
    save_bounds: bool
    save_nas: bool
    save_gen: bool
    save_chr: bool
    stop_after: Optional[str]
    empty_box_only: bool
    potential_only: bool
    nlfff_only: bool
    generic_only: bool
    use_potential: bool
    skip_lines: bool
    center_vox: bool
    reduce_passed: Optional[int]
    euv: bool
    uv: bool
    sfq: bool
    observer_name: str
    fov_xc: Optional[float]
    fov_yc: Optional[float]
    fov_xsize: Optional[float]
    fov_ysize: Optional[float]
    square_fov: bool
    jump2potential: bool
    jump2bounds: bool
    jump2nlfff: bool
    jump2lines: bool
    jump2chromo: bool
    rebuild: bool
    rebuild_from_none: bool
    info: bool
    reproject_algorithm: str
    reproject_scan: Optional[str]
    refmaps_path: Optional[Tuple[str, ...]] = None


@dataclass
class TransitionPlan:
    target_stage: str
    jump_chain: Optional[Tuple[str, ...]]
    active_jump: Optional[str]
    goto_lines: bool
    goto_chromo: bool


@dataclass
class ResumePreparedState:
    base_group: Dict[str, Any]
    refmaps: Dict[str, Any]
    base_bz_arr: np.ndarray
    base_ic_arr: np.ndarray
    bottom_bz_data: np.ndarray
    projection_tag: str
    base: str
    dr3: np.ndarray


@dataclass
class ObservationPreparedState:
    obs_time: Time
    maps: Dict[str, Any]
    base_group: Dict[str, Any]
    refmaps: Dict[str, Any]
    base_bz_arr: np.ndarray
    base_ic_arr: np.ndarray
    bottom_bz_data: np.ndarray
    vert_current_error: Optional[str]
    projection_tag: str
    base: str
    dr3: np.ndarray
    observer_metadata: Dict[str, Any]


@dataclass
class PreparedRunState:
    obs_time: Time
    maps: Optional[Dict[str, Any]]
    base_group: Dict[str, Any]
    refmaps: Dict[str, Any]
    base_bz_arr: np.ndarray
    base_ic_arr: np.ndarray
    bottom_bz_data: np.ndarray
    vert_current_error: Optional[str]
    projection_tag: str
    base: str
    dr3: np.ndarray
    observer_metadata: Optional[Dict[str, Any]]
    lineage_root: str
    lineage_marker: str
    entry_stage_for_marker: str
    entry_corona: Optional[Dict[str, Any]] = None
    entry_model: Optional[str] = None
    entry_lines: Optional[Dict[str, Any]] = None


@dataclass
class PreparedTransitionInputs:
    active_jump: Optional[str]
    goto_lines: bool
    goto_chromo: bool
    entry_lines: Optional[dict]
    pot_box: Optional[dict]
    bnd_box: Optional[dict]
    nlfff_box: Optional[dict]


@dataclass
class NasStageResult:
    nlfff_box: dict
    finalized: bool


@dataclass
class GenChrStageResult:
    finalized: bool


@dataclass
class GenStageResult:
    stage_prefix: str
    lines: Optional[dict]
    finalized: bool


@dataclass
class StageSaveContext:
    cfg: Fov2BoxConfig
    prepared_run: PreparedRunState
    execute_cmd: str
    out_dir: Path
    default_grid: dict
    empty_grid: np.ndarray
    produced: List[Path]


def _infer_time_from_path(path: Path) -> Optional[str]:
    stem = path.name
    if "hmi.M_720s." in stem:
        try:
            tag = stem.split("hmi.M_720s.", 1)[1].split(".", 1)[0]
            if len(tag) == 15:
                return f"{tag[0:4]}-{tag[4:6]}-{tag[6:8]}T{tag[9:11]}:{tag[11:13]}:{tag[13:15]}"
        except Exception:
            return None
    return None


def _extract_time_tokens(text: str) -> list[str]:
    if not text:
        return []
    s = _decode_id_text(text)
    patterns = [
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z)?",
        r"\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2}(?:_TAI)?",
        r"\d{1,2}-[A-Za-z]{3}-\d{2,4}\s+\d{2}:\d{2}:\d{2}",
    ]
    out: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, s):
            tok = m.group(0).strip()
            if tok and tok not in out:
                out.append(tok)
    return out


def _parse_time_candidate(token: str) -> Optional[Time]:
    t = _decode_id_text(token).strip()
    if not t:
        return None
    # First try astropy's native parser.
    try:
        return Time(t)
    except Exception:
        pass

    # JSOC T_REC-like values, often e.g. 2025.11.26_15:36:00_TAI
    m = re.match(r"^(\d{4})\.(\d{2})\.(\d{2})_(\d{2}):(\d{2}):(\d{2})(?:_(TAI))?$", t, flags=re.IGNORECASE)
    if m:
        yyyy, mm, dd, hh, mi, ss, scale = m.groups()
        dtx = dt.datetime(int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss))
        try:
            return Time(dtx, scale="tai" if scale else "utc").utc
        except Exception:
            return None

    # IDL-like values, e.g. 26-Nov-25 15:47:52
    for fmt in ("%d-%b-%y %H:%M:%S", "%d-%b-%Y %H:%M:%S"):
        try:
            dtx = dt.datetime.strptime(t, fmt)
            return Time(dtx, scale="utc")
        except Exception:
            pass
    return None


def _extract_execute_time(execute_text: str) -> Optional[str]:
    if not execute_text:
        return None
    for tok in _extract_time_tokens(execute_text):
        parsed = _parse_time_candidate(tok)
        if parsed is not None:
            return parsed.utc.isot
    return None


def _infer_time_from_entry_loaded(entry_loaded: Dict[str, Any], entry_path: Path) -> Optional[str]:
    """
    Infer model observation time from entry data.
    Priority:
    1) explicit metadata/chromo attrs time fields
    2) base/index metadata text (actual model index header source)
    3) execute command text (requested time)
    4) filename/path fallback
    """
    meta = entry_loaded.get("metadata", {}) if isinstance(entry_loaded, dict) else {}
    if isinstance(meta, dict):
        for key in ("obs_time", "date_obs", "date-obs", "t_rec", "time"):
            if key in meta:
                parsed = _parse_time_candidate(str(meta.get(key)))
                if parsed is not None:
                    return parsed.utc.isot

    # Group attrs can carry true observation time in converted models.
    for group_name in ("chromo", "corona", "base"):
        grp = entry_loaded.get(group_name)
        if isinstance(grp, dict):
            attrs = grp.get("attrs", {})
            if isinstance(attrs, dict):
                for key in ("obs_time", "date_obs", "date-obs", "t_rec", "time"):
                    if key in attrs:
                        parsed = _parse_time_candidate(str(attrs.get(key)))
                        if parsed is not None:
                            return parsed.utc.isot

    # Prefer index/header text over execute (execute can be requested time).
    index_candidates = []
    base = entry_loaded.get("base")
    if isinstance(base, dict) and "index" in base:
        index_candidates.append(base.get("index"))
    if isinstance(meta, dict) and "index" in meta:
        index_candidates.append(meta.get("index"))
    for raw in index_candidates:
        for tok in _extract_time_tokens(_decode_id_text(raw)):
            parsed = _parse_time_candidate(tok)
            if parsed is not None:
                return parsed.utc.isot

    execute_text = _decode_id_text(meta.get("execute", "")) if isinstance(meta, dict) else ""
    inferred_exec = _extract_execute_time(execute_text)
    if inferred_exec:
        return inferred_exec

    return _infer_time_from_path(entry_path)


def _format_coord_tag(lon_deg: float, lat_deg: float, suffix: str = "CR") -> str:
    lon = (lon_deg + 180.0) % 360.0 - 180.0
    lon_dir = "W" if lon >= 0 else "E"
    lat_dir = "N" if lat_deg >= 0 else "S"
    lon_val = f"{abs(int(round(lon))):02d}"
    lat_val = f"{abs(int(round(lat_deg))):02d}"
    return f"{lon_dir}{lon_val}{lat_dir}{lat_val}{suffix}"


def _stage_output_dir(gxmodel_dir: str, obs_time: Time) -> Path:
    date_str = obs_time.to_datetime().strftime("%Y-%m-%d")
    out_dir = Path(gxmodel_dir) / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _stage_file_base(obs_time: Time, coord_tag: str, projection_tag: str = "CEA") -> str:
    time_tag = obs_time.to_datetime().strftime("%Y%m%d_%H%M%S")
    return f"hmi.M_720s.{time_tag}.{coord_tag}.{projection_tag}"


def _stage_filename(out_dir: Path, base: str, stage_tag: str) -> Path:
    return out_dir / f"{base}.{stage_tag}.h5"


def _effective_box_dims_from_base(
    box_dims: Tuple[int, int, int],
    base_group: Dict[str, Any],
) -> Tuple[int, int, int]:
    bx = base_group.get("bx") if isinstance(base_group, dict) else None
    if bx is None:
        return box_dims
    arr = np.asarray(bx)
    if arr.ndim != 2:
        return box_dims
    ny, nx = arr.shape
    return int(nx), int(ny), int(box_dims[2])


def _last_stage_tag(stop_after: Optional[str]) -> str:
    if stop_after:
        stop = stop_after.lower()
        if stop in ("dl", "download", "downloads"):
            return "DL"
        if stop in ("none", "empty", "empty_box"):
            return "NONE"
        if stop in ("bnd", "bounds"):
            return "BND"
        if stop == "pot":
            return "POT"
        if stop == "nas":
            return "NAS"
        if stop == "gen":
            return "GEN"
        if stop == "chr":
            return "CHR"
    return "CHR"


def _canon_stage_name(tag: Optional[str]) -> str:
    if not tag:
        return "CHR"
    t = tag.lower()
    if t in ("dl", "download", "downloads"):
        return "DL"
    if t in ("none", "empty", "empty_box"):
        return "NONE"
    if t in ("pot", "potential"):
        return "POT"
    if t in ("bnd", "bounds"):
        return "BND"
    if t in ("nas", "nlfff"):
        return "NAS"
    if t in ("gen", "nas.gen", "lines"):
        return "GEN"
    if t in ("chr", "nas.chr", "chromo"):
        return "CHR"
    return t.upper()


def _entry_stage_from_loaded(loaded: Dict[str, Any], entry_path: Path) -> str:
    # Prefer explicit metadata id/model_type when present.
    model_id = _decode_id_text(loaded.get("metadata", {}).get("id", "")).upper()
    if ".POT.GEN.CHR" in model_id:
        return "CHR"
    if ".NAS.GEN.CHR" in model_id:
        return "CHR"
    if ".POT.CHR" in model_id:
        return "CHR"
    if ".POT.GEN" in model_id:
        return "GEN"
    if ".NAS.CHR" in model_id:
        return "CHR"
    if ".NAS.GEN" in model_id:
        return "GEN"
    if ".NAS." in model_id:
        return "NAS"
    if ".POT" in model_id:
        return "POT"
    if ".BND" in model_id:
        return "BND"
    if ".NONE" in model_id:
        return "NONE"

    if "chromo" in loaded:
        return "CHR"
    if "lines" in loaded:
        return "GEN"
    if "corona" in loaded:
        model_type = str(loaded["corona"].get("attrs", {}).get("model_type", "")).lower()
        if model_type in ("none",):
            return "NONE"
        if model_type in ("pot",):
            return "POT"
        if model_type in ("bnd", "bounds"):
            return "BND"
        if model_type in ("nlfff",):
            return "NAS"
        return "NAS"

    # Fallback from filename.
    up = entry_path.name.upper()
    if ".POT.GEN.CHR" in up:
        return "CHR"
    if ".NAS.GEN.CHR" in up:
        return "CHR"
    if ".POT.CHR" in up:
        return "CHR"
    if ".POT.GEN" in up:
        return "GEN"
    if ".NAS.CHR" in up:
        return "CHR"
    if ".NAS.GEN" in up:
        return "GEN"
    if ".NAS." in up:
        return "NAS"
    if ".POT" in up:
        return "POT"
    if ".BND" in up:
        return "BND"
    if ".NONE" in up:
        return "NONE"
    return "NONE"


def _detect_target_stage(cfg: Fov2BoxConfig, entry_stage: Optional[str]) -> str:
    if cfg.rebuild or cfg.rebuild_from_none:
        return "NONE"
    if cfg.jump2potential:
        return "POT"
    if cfg.jump2bounds:
        return "BND"
    if cfg.jump2nlfff:
        return "NAS"
    if cfg.jump2lines:
        return "GEN"
    if cfg.jump2chromo:
        return "CHR"
    if entry_stage:
        return entry_stage
    return "NONE"


_ALLOWED_JUMP_CHAINS: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("NONE", "NONE"): ("NONE",),
    ("NONE", "POT"): ("NONE", "POT"),
    ("NONE", "BND"): ("NONE", "POT", "BND"),
    ("POT", "NONE"): ("POT", "NONE"),
    ("POT", "POT"): ("POT",),
    ("POT", "BND"): ("POT", "BND"),
    ("POT", "NAS"): ("POT", "BND", "NAS"),
    ("POT", "GEN"): ("POT", "GEN"),
    ("POT", "CHR"): ("POT", "CHR"),
    ("BND", "NONE"): ("BND", "NONE"),
    ("BND", "POT"): ("BND", "POT"),
    ("BND", "BND"): ("BND",),
    ("BND", "NAS"): ("BND", "NAS"),
    ("NAS", "NONE"): ("NAS", "NONE"),
    ("NAS", "POT"): ("NAS", "POT"),
    ("NAS", "BND"): ("NAS", "BND"),
    ("NAS", "NAS"): ("NAS",),
    ("NAS", "GEN"): ("NAS", "GEN"),
    ("NAS", "CHR"): ("NAS", "CHR"),
    ("GEN", "NONE"): ("GEN", "NONE"),
    ("GEN", "POT"): ("GEN", "POT"),
    ("GEN", "BND"): ("GEN", "BND"),
    ("GEN", "NAS"): ("GEN", "NAS"),
    ("GEN", "GEN"): ("GEN",),
    ("GEN", "CHR"): ("GEN", "CHR"),
    ("CHR", "NONE"): ("CHR", "NONE"),
    ("CHR", "POT"): ("CHR", "POT"),
    ("CHR", "BND"): ("CHR", "BND"),
    ("CHR", "NAS"): ("CHR", "NAS"),
    ("CHR", "GEN"): ("CHR", "GEN"),
    ("CHR", "CHR"): ("CHR",),
}


def _jump_chain(entry_stage: str, target_stage: str) -> Optional[Tuple[str, ...]]:
    return _ALLOWED_JUMP_CHAINS.get((entry_stage, target_stage))


def _jump_allowed(entry_stage: str, target_stage: str) -> bool:
    return _jump_chain(entry_stage, target_stage) is not None


def _active_jump_for_target(target_stage: str, use_entry_box: bool) -> Optional[str]:
    if not use_entry_box:
        return None
    return {
        "POT": "potential",
        "BND": "bounds",
        "NAS": "nlfff",
        "GEN": "lines",
        "CHR": "chromo",
    }.get(target_stage)


def _plan_transition(cfg: Fov2BoxConfig, entry_stage: Optional[str]) -> TransitionPlan:
    target_stage = _detect_target_stage(cfg, entry_stage)
    jump_chain = None
    use_entry_resume = bool(cfg.entry_box and not cfg.rebuild and not cfg.rebuild_from_none)

    if use_entry_resume:
        assert entry_stage is not None
        jump_chain = _jump_chain(entry_stage, target_stage)
        if jump_chain is None:
            raise ValueError(
                f"Incompatible jump request: entry stage {entry_stage} cannot jump to {target_stage}. "
                "Unsupported transitions are rejected explicitly."
            )

    active_jump = _active_jump_for_target(target_stage, use_entry_resume)
    goto_lines = active_jump in ("lines", "chromo")
    goto_chromo = active_jump == "chromo" or cfg.skip_lines
    return TransitionPlan(
        target_stage=target_stage,
        jump_chain=jump_chain,
        active_jump=active_jump,
        goto_lines=goto_lines,
        goto_chromo=goto_chromo,
    )


def _prepare_resume_state(entry_loaded: Dict[str, Any], cfg: Fov2BoxConfig, obs_time: Time) -> ResumePreparedState:
    base_group = dict(entry_loaded.get("base", {}))
    if not all(k in base_group for k in ("bx", "by", "bz")):
        raise ValueError("Entry-box resume requires base/bx, base/by, and base/bz.")
    if "ic" not in base_group:
        base_group["ic"] = np.asarray(base_group["bz"], dtype=float).copy()
    if "chromo_mask" not in base_group:
        base_group["chromo_mask"] = decompose(
            np.asarray(base_group["bz"], dtype=float).T,
            np.asarray(base_group["ic"], dtype=float).T,
        ).T

    refmaps = dict(entry_loaded.get("refmaps", {}))
    base_bz_arr = np.asarray(base_group["bz"], dtype=float)
    base_ic_arr = np.asarray(base_group["ic"], dtype=float)
    projection_tag = _decode_id_text(entry_loaded.get("metadata", {}).get("projection", "CEA")).upper()

    axis_order = entry_loaded.get("metadata", {}).get("axis_order_3d")
    entry_corona_for_dr = _h5_corona_to_internal_xyz(entry_loaded.get("corona"), axis_order)
    if isinstance(entry_corona_for_dr, dict) and "dr" in entry_corona_for_dr:
        d = np.asarray(entry_corona_for_dr["dr"], dtype=float).reshape(-1)
        dr3 = np.array([d[0], d[min(1, d.size - 1)], d[min(2, d.size - 1)]], dtype=float)
    else:
        dr = float((cfg.dx_km * u.km / (IDL_HMI_RSUN_M * u.m).to(u.km)).value)
        dr3 = np.array([dr, dr, dr], dtype=float)

    src_id = _decode_id_text(entry_loaded.get("metadata", {}).get("id", "")).strip()
    base = ""
    if src_id:
        base, _ = _split_stage_id(src_id)
    if not base:
        base = Path(cfg.entry_box).stem if cfg.entry_box else _stage_file_base(obs_time, "UNKNOWN", projection_tag=projection_tag)

    return ResumePreparedState(
        base_group=base_group,
        refmaps=refmaps,
        base_bz_arr=base_bz_arr,
        base_ic_arr=base_ic_arr,
        bottom_bz_data=base_bz_arr,
        projection_tag=projection_tag,
        base=base,
        dr3=dr3,
    )


def _prepare_observation_state(
    cfg: Fov2BoxConfig,
    requested_obs_time: Time,
    box_dims_resolved: Tuple[int, int, int],
    run_logged_step: Callable[[str, Callable[[], Any]], Any],
    t_start: float,
) -> Optional[ObservationPreparedState]:
    import time as time_mod

    obs_time = requested_obs_time
    data_dir_path = Path(cfg.data_dir).expanduser().resolve()
    disambig_method = 0 if cfg.sfq else 2
    print("Checking/downloading HMI/AIA data...")
    dl_t0 = time_mod.perf_counter()
    maps, download_info = _load_hmi_maps_from_downloader(
        obs_time,
        data_dir_path,
        cfg.euv,
        cfg.uv,
        download_backend=cfg.download_backend,
        drms_sequential=cfg.drms_sequential,
        force_download=cfg.force_download,
        hmi_time_window=cfg.hmi_time_window,
        aia_time_window=cfg.aia_time_window,
        disambig_method=disambig_method,
        strict_required=(_last_stage_tag(cfg.stop_after) != "DL"),
    )
    dl_elapsed = time_mod.perf_counter() - dl_t0
    print(f"Done in {dl_elapsed:.2f}s")
    resolved_obs_time = download_info.get("resolved_obs_time")
    if resolved_obs_time:
        resolved_obs_time = Time(resolved_obs_time)
        if abs((resolved_obs_time - requested_obs_time).to_value("sec")) > 1.0:
            print(
                "Resolved source bundle time: "
                f"{resolved_obs_time.isot} "
                f"(requested {requested_obs_time.isot})"
            )
        obs_time = resolved_obs_time
    if _last_stage_tag(cfg.stop_after) == "DL":
        print("Stopped after download stage by request.")
        total = time_mod.perf_counter() - t_start
        print(f"Total elapsed: {total:.2f}s")
        return None

    def _prepare_geometry():
        rsun_local = (IDL_HMI_RSUN_M * u.m).to(u.km)
        observer_local = _resolve_cli_observer(maps.get("field"), cfg.observer_name, obs_time)

        if cfg.hpc:
            box_origin_local = SkyCoord(cfg.coords[0] * u.arcsec, cfg.coords[1] * u.arcsec,
                                        obstime=obs_time, observer=observer_local, rsun=rsun_local,
                                        frame=Helioprojective)
        elif cfg.hgc:
            box_origin_local = SkyCoord(lon=cfg.coords[0] * u.deg, lat=cfg.coords[1] * u.deg,
                                        radius=rsun_local, obstime=obs_time, observer=observer_local,
                                        frame=HeliographicCarrington)
        else:
            box_origin_local = SkyCoord(lon=cfg.coords[0] * u.deg, lat=cfg.coords[1] * u.deg,
                                        radius=rsun_local, obstime=obs_time, observer=observer_local,
                                        frame=HeliographicStonyhurst)

        box_dims_q_local = u.Quantity(list(box_dims_resolved)) * u.pix
        box_res_local = (cfg.dx_km * u.km).to(u.Mm)

        frame_obs_local = Helioprojective(observer=observer_local, obstime=obs_time, rsun=rsun_local)
        frame_hcc_local = Heliocentric(observer=box_origin_local, obstime=obs_time)
        box_center_local = box_origin_local.transform_to(frame_hcc_local)
        center_z_local = box_center_local.z + (box_dims_q_local[2] / u.pix * box_res_local) / 2
        box_center_local = SkyCoord(x=box_center_local.x, y=box_center_local.y, z=center_z_local,
                                    frame=frame_hcc_local)

        box_local = Box(frame_obs_local, box_origin_local, box_center_local, box_dims_q_local, box_res_local)
        if cfg.top:
            bottom_wcs_header_local = box_local.bottom_top_header(dsun_obs=maps["field"].dsun)
            projection_tag_local = "TOP"
        else:
            bottom_wcs_header_local = box_local.bottom_cea_header
            projection_tag_local = "CEA"
        fov_coords_local = box_local.bounds_coords_bl_tr(pad_frac=cfg.pad_frac)
        return (
            rsun_local, observer_local, box_origin_local, bottom_wcs_header_local,
            projection_tag_local, fov_coords_local,
        )

    (
        rsun, observer, box_origin, bottom_wcs_header, projection_tag, fov_coords,
    ) = run_logged_step("Preparing observer and box geometry", _prepare_geometry)

    map_bp, map_bt, map_br = run_logged_step(
        "Converting HMI vector field components",
        lambda: hmi_b2ptr(maps["field"], maps["inclination"], maps["azimuth"]),
    )

    def submap_with_fov(_map: Map) -> Map:
        return _submap_with_fov_safe(_map, fov_coords[0], fov_coords[1])

    def _extract_cutouts():
        return (
            submap_with_fov(map_bp),
            submap_with_fov(map_bt),
            submap_with_fov(map_br),
            submap_with_fov(maps["continuum"]),
            submap_with_fov(maps["magnetogram"]),
        )

    map_bp, map_bt, map_br, map_cont, map_los = run_logged_step(
        "Extracting FOV cutouts from source maps",
        _extract_cutouts,
    )

    map_bx = map_bp
    map_by = map_from_data_header_compat(-map_bt.data, map_bt.meta)
    map_bz = map_br

    def _reproject_cutouts():
        _algo = cfg.reproject_algorithm
        bottom_bx_local = map_bx.reproject_to(bottom_wcs_header, algorithm=_algo)
        bottom_by_local = map_by.reproject_to(bottom_wcs_header, algorithm=_algo)
        bottom_bz_local = map_bz.reproject_to(bottom_wcs_header, algorithm=_algo)
        base_bz_local = map_los.reproject_to(bottom_wcs_header, algorithm=_algo)
        base_ic_local = map_cont.reproject_to(bottom_wcs_header, algorithm=_algo)
        return (
            bottom_bx_local,
            bottom_by_local,
            bottom_bz_local,
            base_bz_local,
            base_ic_local,
            np.asarray(base_bz_local.data, dtype=float),
            np.asarray(base_ic_local.data, dtype=float),
            np.asarray(bottom_bz_local.data, dtype=float),
        )

    (
        bottom_bx, bottom_by, bottom_bz, base_bz, base_ic,
        base_bz_arr, base_ic_arr, bottom_bz_data,
    ) = run_logged_step("Reprojecting cutouts onto the model base grid", _reproject_cutouts)

    def _build_base_products():
        index_local = _build_index_header(
            bottom_wcs_header,
            bottom_bz,
            observer_override=observer,
            obs_time_override=obs_time,
            rsun_override=rsun,
        )
        bx_local = _fill_nonfinite_2d(bottom_bx.data, label="bottom_bx (OBS reprojection)")
        by_local = _fill_nonfinite_2d(bottom_by.data, label="bottom_by (OBS reprojection)")
        bz_local = _fill_nonfinite_2d(bottom_bz.data, label="bottom_bz (OBS reprojection)")
        ic_local = _fill_nonfinite_2d(base_ic.data, label="base_ic (OBS reprojection)")
        chromo_mask_local = decompose(base_bz_arr.T, base_ic_arr.T).T
        return {
            "bx": bx_local,
            "by": by_local,
            "bz": bz_local,
            "ic": ic_local,
            "chromo_mask": chromo_mask_local,
            "index": index_local,
        }

    base_group = run_logged_step("Building base-layer products", _build_base_products)

    refmaps: Dict[str, Any] = {}
    refmap_model_time = model_obstime_from_base_index({"base": base_group}) or obs_time.isot

    def add_refmap(ref_id: str, smap: Map) -> None:
        refmaps[ref_id] = build_refmap_payload_for_model(
            smap,
            model_obstime=refmap_model_time,
            target_fov=(fov_coords[0], fov_coords[1]),
            reproject_algorithm=cfg.reproject_algorithm,
        )

    def _collect_refmaps():
        hmi_t0 = time_mod.perf_counter()
        add_refmap("Bz_reference", maps["magnetogram"])
        add_refmap("Ic_reference", maps["continuum"])
        print(f"HMI references: 2/2 in {time_mod.perf_counter() - hmi_t0:.2f}s")

        local_vert_current_error = None

        def _compute_vert_current():
            vc_bx_local, vc_by_local, vc_bz_local = remap_vertical_current_inputs(
                map_bx, map_by, map_bz
            )
            vc_header_local = vc_bx_local.wcs.to_header().tostring(sep="\n", endcard=True)
            rsun_arcsec_local = vc_bx_local.rsun_obs.to_value(u.arcsec)
            crpix1_local, crpix2_local = vc_bx_local.wcs.wcs.crpix
            cdelt1_local = vc_bx_local.scale.axis1.to_value(u.arcsec / u.pix)
            cdelt2_local = vc_bx_local.scale.axis2.to_value(u.arcsec / u.pix)
            jz_local = compute_vertical_current(
                vc_bz_local.data,
                vc_bx_local.data,
                vc_by_local.data,
                vc_header_local,
                rsun_arcsec_local,
                crpix1=crpix1_local,
                crpix2=crpix2_local,
                cdelt1_arcsec=cdelt1_local,
                cdelt2_arcsec=cdelt2_local,
            )
            jz_map = map_from_data_header_compat(jz_local, vc_bx_local.wcs.to_header())
            refmaps["Vert_current"] = build_refmap_payload_for_model(
                jz_map,
                model_obstime=refmap_model_time,
                target_fov=(fov_coords[0], fov_coords[1]),
                reproject_algorithm=cfg.reproject_algorithm,
            )

        try:
            run_logged_step("Vertical current", _compute_vert_current)
        except Exception as exc:
            local_vert_current_error = str(exc)
            print(f"Vertical current: skipped ({local_vert_current_error})")

        aia_passbands = [pb for pb in (AIA_EUV_PASSBANDS + AIA_UV_PASSBANDS) if f"AIA_{pb}" in maps]
        total_aia = len(aia_passbands)
        aia_t0 = time_mod.perf_counter()
        for idx, pb in enumerate(aia_passbands, start=1):
            key = f"AIA_{pb}"
            add_refmap(key, maps[key])
            elapsed = time_mod.perf_counter() - aia_t0
            print(f"AIA references: {idx}/{total_aia} ({pb}) in {elapsed:.2f}s")
        cache_dir = data_dir_path / obs_time.datetime.strftime("%Y-%m-%d")
        cache_t0 = time_mod.perf_counter()
        cache_map_ids = {
            path: map_id
            for path, map_id in discover_fits_refmap_map_ids([cache_dir], generic=False).items()
            if map_id not in refmaps
        }
        if cache_map_ids:
            missing_cache_refmaps = build_fits_refmaps_for_model(
                list(cache_map_ids.keys()),
                model_obstime=refmap_model_time,
                target_fov=(fov_coords[0], fov_coords[1]),
                reproject_algorithm=cfg.reproject_algorithm,
                map_ids=cache_map_ids,
                generic=False,
            )
            for key, payload in missing_cache_refmaps.items():
                refmaps[key] = payload
            print(
                "Cached context references: "
                f"{len(missing_cache_refmaps)} from {cache_dir} "
                f"in {time_mod.perf_counter() - cache_t0:.2f}s"
            )
        if cfg.refmaps_path:
            external_t0 = time_mod.perf_counter()
            external_refmaps = build_fits_refmaps_for_model(
                cfg.refmaps_path,
                model_obstime=refmap_model_time,
                target_fov=(fov_coords[0], fov_coords[1]),
                reproject_algorithm=cfg.reproject_algorithm,
                generic=True,
            )
            for key, payload in external_refmaps.items():
                refmaps[key] = payload
            print(
                "External references: "
                f"{len(external_refmaps)} from {len(cfg.refmaps_path)} path(s) "
                f"in {time_mod.perf_counter() - external_t0:.2f}s"
            )
        return local_vert_current_error

    vert_current_error = run_logged_step("Collecting reference maps and derived products", _collect_refmaps)

    obs_dr = (cfg.dx_km * u.km) / rsun
    dr3 = np.array([obs_dr.value, obs_dr.value, obs_dr.value])
    coord_tag = _format_coord_tag(
        box_origin.transform_to(HeliographicCarrington(obstime=obs_time)).lon.to_value(u.deg),
        box_origin.transform_to(HeliographicCarrington(obstime=obs_time)).lat.to_value(u.deg),
    )
    base = _stage_file_base(obs_time, coord_tag, projection_tag=projection_tag)
    observer_metadata = _observer_metadata_from_source_map(maps["field"], cfg)

    return ObservationPreparedState(
        obs_time=obs_time,
        maps=maps,
        base_group=base_group,
        refmaps=refmaps,
        base_bz_arr=base_bz_arr,
        base_ic_arr=base_ic_arr,
        bottom_bz_data=bottom_bz_data,
        vert_current_error=vert_current_error,
        projection_tag=projection_tag,
        base=base,
        dr3=dr3,
        observer_metadata=observer_metadata,
    )


def _prepare_run_state(
    cfg: Fov2BoxConfig,
    resume_mode: bool,
    entry_loaded: Optional[Dict[str, Any]],
    entry_stage: Optional[str],
    target_stage: str,
    requested_obs_time: Time,
    box_dims_resolved: Tuple[int, int, int],
    observer_metadata: Optional[Dict[str, Any]],
    run_logged_step: Callable[[str, Callable[[], Any]], Any],
    t_start: float,
) -> Optional[PreparedRunState]:
    if resume_mode:
        assert entry_loaded is not None
        prepared_resume = _prepare_resume_state(entry_loaded, cfg, requested_obs_time)
        expected_shape_3d = _effective_box_dims_from_base(box_dims_resolved, prepared_resume.base_group)
        entry_meta = dict(entry_loaded.get("metadata", {}))
        entry_corona = _prepare_stage_corona_payload(
            entry_loaded.get("corona"),
            source="disk",
            axis_order_3d=entry_meta.get("axis_order_3d"),
            expected_shape_3d=expected_shape_3d,
        )
        entry_model = None
        if isinstance(entry_corona, dict):
            entry_model = _decode_id_text(entry_corona.get("attrs", {}).get("model_type", "")).strip().lower() or None
        entry_lines = entry_loaded.get("lines")
        if entry_lines is None:
            # Backward compatibility with legacy GEN files that stored line keys under chromo.
            entry_lines = entry_loaded.get("chromo")
        lineage_root = _decode_id_text(entry_meta.get("lineage", "")).strip()
        lineage_marker = ""
        entry_stage_for_marker = ""
        if not lineage_root:
            entry_id = _decode_id_text(entry_meta.get("id", "")).strip()
            _, entry_stage_tag = _split_stage_id(entry_id) if entry_id else ("", "")
            entry_stage_for_marker = entry_stage_tag or _stage_tag_from_stage(entry_stage or target_stage)
            lineage_marker = f"ENTRY.{entry_stage_for_marker}"
            lineage_root = lineage_marker
        return PreparedRunState(
            obs_time=requested_obs_time,
            maps=None,
            base_group=prepared_resume.base_group,
            refmaps=prepared_resume.refmaps,
            base_bz_arr=prepared_resume.base_bz_arr,
            base_ic_arr=prepared_resume.base_ic_arr,
            bottom_bz_data=prepared_resume.bottom_bz_data,
            vert_current_error=None,
            projection_tag=prepared_resume.projection_tag,
            base=prepared_resume.base,
            dr3=prepared_resume.dr3,
            observer_metadata=observer_metadata,
            lineage_root=lineage_root,
            lineage_marker=lineage_marker,
            entry_stage_for_marker=entry_stage_for_marker,
            entry_corona=entry_corona,
            entry_model=entry_model,
            entry_lines=entry_lines,
        )

    prepared_obs = _prepare_observation_state(
        cfg,
        requested_obs_time,
        box_dims_resolved,
        run_logged_step,
        t_start,
    )
    if prepared_obs is None:
        return None
    return PreparedRunState(
        obs_time=prepared_obs.obs_time,
        maps=prepared_obs.maps,
        base_group=prepared_obs.base_group,
        refmaps=prepared_obs.refmaps,
        base_bz_arr=prepared_obs.base_bz_arr,
        base_ic_arr=prepared_obs.base_ic_arr,
        bottom_bz_data=prepared_obs.bottom_bz_data,
        vert_current_error=prepared_obs.vert_current_error,
        projection_tag=prepared_obs.projection_tag,
        base=prepared_obs.base,
        dr3=prepared_obs.dr3,
        observer_metadata=prepared_obs.observer_metadata,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
        entry_corona=None,
        entry_model=None,
        entry_lines=None,
    )


_CARRY_FORWARD_LINE_KEYS: Tuple[str, ...] = (
    "start_idx", "end_idx",
    "av_field", "phys_length", "voxel_status",
)


_REQUIRED_LINE_KEYS: Tuple[str, ...] = (
    "codes", "apex_idx", "start_idx", "end_idx", "seed_idx",
    "av_field", "phys_length", "voxel_status",
)


def _prepare_stage_corona_payload(
    corona: Optional[Dict[str, Any]],
    *,
    source: str,
    axis_order_3d: Optional[str] = None,
    expected_shape_3d: Optional[Tuple[int, int, int]] = None,
    allowed_model_types: Optional[Tuple[str, ...]] = None,
) -> Optional[Dict[str, Any]]:
    if corona is None:
        return None
    if not isinstance(corona, dict):
        raise ValueError(f"Canonical corona payload must be a dict for {source} stage input.")

    prepared = _h5_corona_to_internal_xyz(corona, axis_order_3d) if source == "disk" else corona
    component_shapes = []
    for key in ("bx", "by", "bz"):
        if key not in prepared:
            raise ValueError(f"Canonical corona payload is missing corona/{key} for {source} stage input.")
        arr = np.asarray(prepared[key])
        if arr.ndim != 3:
            raise ValueError(f"Canonical corona payload must provide 3D corona/{key} for {source} stage input.")
        component_shapes.append(tuple(int(v) for v in arr.shape))

    if len(set(component_shapes)) != 1:
        raise ValueError(
            f"Canonical corona payload must use one shared 3D shape for bx/by/bz; got {component_shapes!r}."
        )

    if expected_shape_3d is not None and component_shapes[0] != tuple(int(v) for v in expected_shape_3d):
        raise ValueError(
            "Canonical corona payload shape does not match the prepared model volume. "
            f"expected={tuple(int(v) for v in expected_shape_3d)!r}, got={component_shapes[0]!r}"
        )

    dr = prepared.get("dr")
    if dr is None:
        raise ValueError(f"Canonical corona payload is missing corona/dr for {source} stage input.")
    dr_arr = np.asarray(dr, dtype=float).reshape(-1)
    if dr_arr.size == 0:
        raise ValueError(f"Canonical corona payload must provide non-empty corona/dr for {source} stage input.")

    attrs = prepared.get("attrs")
    model_type = ""
    if isinstance(attrs, dict):
        model_type = _decode_id_text(attrs.get("model_type", "")).strip().lower()
    if allowed_model_types is not None and model_type not in allowed_model_types:
        raise ValueError(
            f"Canonical corona payload model_type must be one of {allowed_model_types!r}; got {model_type!r}."
        )
    return prepared


def _prepare_stage_lines_payload(
    lines: Optional[Dict[str, Any]],
    *,
    source: str,
    require_complete: bool,
) -> Optional[Dict[str, Any]]:
    if lines is None:
        if require_complete:
            raise ValueError(f"Canonical line payload is required for {source} stage input.")
        return None
    if not isinstance(lines, dict):
        raise ValueError(f"Canonical line payload must be a dict for {source} stage input.")
    required_keys = _REQUIRED_LINE_KEYS if require_complete else _CARRY_FORWARD_LINE_KEYS
    missing = [key for key in required_keys if key not in lines]
    if missing:
        if require_complete:
            raise ValueError(
                f"Canonical line payload is missing required keys for {source} stage input: {', '.join(missing)}"
            )
        return None
    return lines


def _compute_none_stage_box(
    prepared_run: PreparedRunState,
    box_dims_resolved: Tuple[int, int, int],
) -> dict:
    # Internal in-memory cube contract is (x, y, z). HDF5 canonical (z, y, x)
    # conversion is applied only in the save path.
    nx, ny, nz = (int(box_dims_resolved[0]), int(box_dims_resolved[1]), int(box_dims_resolved[2]))
    bx_base = _fill_nonfinite_2d(prepared_run.base_group["bx"], label="base/bx (NONE boundary)")
    by_base = _fill_nonfinite_2d(prepared_run.base_group["by"], label="base/by (NONE boundary)")
    bz_base = _fill_nonfinite_2d(prepared_run.base_group["bz"], label="base/bz (NONE boundary)")
    expected_base_shape = (ny, nx)
    for label, arr in (("bx", bx_base), ("by", by_base), ("bz", bz_base)):
        if arr.shape != expected_base_shape:
            raise ValueError(
                f"NONE stage expects base/{label} shape (ny,nx)={expected_base_shape!r}; got {arr.shape!r}."
            )

    corona_bx = np.zeros((nx, ny, nz), dtype=float)
    corona_by = np.zeros((nx, ny, nz), dtype=float)
    corona_bz = np.zeros((nx, ny, nz), dtype=float)
    corona_bx[:, :, 0] = bx_base.T
    corona_by[:, :, 0] = by_base.T
    corona_bz[:, :, 0] = bz_base.T
    return {
        "corona": _prepare_stage_corona_payload(
            {
                "bx": corona_bx,
                "by": corona_by,
                "bz": corona_bz,
                "dr": prepared_run.dr3,
                "attrs": {"model_type": "none"},
            },
            source="memory",
            expected_shape_3d=box_dims_resolved,
            allowed_model_types=("none",),
        )
    }


def _normalize_runtime_stage_box_for_pipeline(
    stage_box: dict,
    *,
    prepared_run: PreparedRunState,
    stage_tag: str,
    source_axis_order_3d: str = "xyz",
) -> dict:
    working_box = dict(stage_box)

    base_group = dict(prepared_run.base_group)
    if "ic" not in base_group:
        base_group["ic"] = np.asarray(prepared_run.base_ic_arr, dtype=float).copy()
    working_box.setdefault("base", base_group)
    working_box.setdefault("refmaps", prepared_run.refmaps)

    if prepared_run.observer_metadata is not None and "observer" not in working_box:
        working_box["observer"] = dict(prepared_run.observer_metadata)

    metadata = dict(working_box.get("metadata", {})) if isinstance(working_box.get("metadata"), dict) else {}
    metadata.setdefault("id", f"{prepared_run.base}.{stage_tag}")
    metadata.setdefault("projection", _decode_id_text(prepared_run.projection_tag).upper())
    metadata.setdefault("axis_order_2d", "yx")
    metadata.setdefault("axis_order_3d", "zyx")
    metadata.setdefault("vector_layout", "split_components")
    working_box["metadata"] = metadata

    normalized_h5_box = _normalize_stage_for_h5(
        working_box,
        source_axis_order_3d=source_axis_order_3d,
    )
    normalized_model = _normalize_loaded_model_dict(
        normalized_h5_box,
        source_path=Path(f"{metadata['id']}.h5"),
        source_kind="h5",
        strict=False,
        stored_contract=None,
    )
    normalized_metadata = normalized_model.get("metadata")
    if isinstance(normalized_metadata, dict) and "geometry_contract" in normalized_metadata:
        metadata["geometry_contract"] = normalized_metadata["geometry_contract"]
        working_box["metadata"] = metadata
    return working_box


def _compute_pot_stage_box(
    prepared_run: PreparedRunState,
    box_dims_resolved: Tuple[int, int, int],
) -> dict:
    bnddata = np.asarray(prepared_run.bottom_bz_data, dtype=float).copy()
    bnddata[np.isnan(bnddata)] = 0.0
    maglib_lff = MagFieldLinFFF()
    maglib_lff.set_field(bnddata)
    pot_res = maglib_lff.LFFF_cube(nz=box_dims_resolved[2], alpha=0.0)
    return {
        "corona": _prepare_stage_corona_payload(
            {
                "bx": pot_res["by"].swapaxes(0, 1),
                "by": pot_res["bx"].swapaxes(0, 1),
                "bz": pot_res["bz"].swapaxes(0, 1),
                "dr": prepared_run.dr3,
                "attrs": {"model_type": "pot"},
            },
            source="memory",
            expected_shape_3d=box_dims_resolved,
            allowed_model_types=("pot",),
        )
    }


def _compute_bnd_stage_box(
    prepared_run: PreparedRunState,
    pot_box: dict,
    box_dims_resolved: Tuple[int, int, int],
) -> dict:
    corona = pot_box.get("corona", pot_box)
    bnd = {
        "bx": np.array(corona["bx"], copy=True),
        "by": np.array(corona["by"], copy=True),
        "bz": np.array(corona["bz"], copy=True),
        "dr": prepared_run.dr3,
        "attrs": {"model_type": "bnd"},
    }
    bnd["bx"][:, :, 0] = np.asarray(prepared_run.base_group["bx"], dtype=float).T
    bnd["by"][:, :, 0] = np.asarray(prepared_run.base_group["by"], dtype=float).T
    bnd["bz"][:, :, 0] = np.asarray(prepared_run.base_group["bz"], dtype=float).T
    return {
        "corona": _prepare_stage_corona_payload(
            bnd,
            source="memory",
            expected_shape_3d=box_dims_resolved,
            allowed_model_types=("bnd",),
        )
    }


def _prepare_transition_stage_inputs(
    prepared_run: PreparedRunState,
    transition_plan: TransitionPlan,
    *,
    entry_stage: Optional[str],
    box_dims_resolved: Tuple[int, int, int],
) -> PreparedTransitionInputs:
    active_jump = transition_plan.active_jump
    entry_corona = prepared_run.entry_corona if active_jump else None
    entry_lines = prepared_run.entry_lines if active_jump else None

    goto_lines = transition_plan.goto_lines
    goto_chromo = transition_plan.goto_chromo

    if goto_lines:
        if not entry_corona:
            raise ValueError("--jump2lines/--jump2chromo requires --entry-box with corona model_type=pot|nlfff")
        return PreparedTransitionInputs(
            active_jump=active_jump,
            goto_lines=goto_lines,
            goto_chromo=goto_chromo,
            entry_lines=entry_lines,
            pot_box=None,
            bnd_box=None,
            nlfff_box=entry_corona,
        )

    pot_box = None
    bnd_box = None
    nlfff_box = None

    if active_jump in ("potential", "bounds", "nlfff"):
        pot_box, bnd_box, nlfff_box = _prepare_resume_jump_boxes(
            active_jump,
            entry_stage,
            entry_corona,
        )
        if pot_box is not None:
            pot_box = _prepare_stage_corona_payload(
                pot_box,
                source="memory",
                expected_shape_3d=box_dims_resolved,
                allowed_model_types=("pot",),
            )
        if bnd_box is not None:
            bnd_box = _prepare_stage_corona_payload(
                bnd_box,
                source="memory",
                expected_shape_3d=box_dims_resolved,
                allowed_model_types=("bnd",),
            )
        if nlfff_box is not None:
            nlfff_box = _prepare_stage_corona_payload(
                nlfff_box,
                source="memory",
                expected_shape_3d=box_dims_resolved,
                allowed_model_types=("nlfff",),
            )
        if active_jump == "nlfff" and entry_stage == "POT" and pot_box is not None:
            bnd_box = _compute_bnd_stage_box(prepared_run, pot_box, box_dims_resolved)["corona"]

    return PreparedTransitionInputs(
        active_jump=active_jump,
        goto_lines=goto_lines,
        goto_chromo=goto_chromo,
        entry_lines=entry_lines,
        pot_box=pot_box,
        bnd_box=bnd_box,
        nlfff_box=nlfff_box,
    )


def _resolve_stage_prefix(
    nlfff_box: dict,
    *,
    resume_mode: bool,
    entry_stage: Optional[str],
    target_stage: str,
) -> str:
    is_pot_skip_path = (
        resume_mode
        and entry_stage == "POT"
        and target_stage in ("GEN", "CHR")
    )
    if str(nlfff_box.get("attrs", {}).get("model_type", "")).lower() == "pot" or is_pot_skip_path:
        return "POT"
    return "NAS"


def _run_gen_stage(
    cfg: Fov2BoxConfig,
    prepared_run: PreparedRunState,
    nlfff_box: dict,
    *,
    resume_mode: bool,
    entry_stage: Optional[str],
    target_stage: str,
    goto_chromo: bool,
    entry_lines: Optional[dict],
    save_stage: Callable[..., None],
    finalize: Callable[[], None],
    stage_times: Dict[str, float],
    stage_progress_cls: type,
) -> GenStageResult:
    stage_prefix = _resolve_stage_prefix(
        nlfff_box,
        resume_mode=resume_mode,
        entry_stage=entry_stage,
        target_stage=target_stage,
    )
    gen_stage_tag = f"{stage_prefix}.GEN"

    if goto_chromo:
        lines = _prepare_stage_lines_payload(
            entry_lines,
            source="disk",
            require_complete=False,
        )
        if cfg.skip_lines:
            print("Skipping GEN line computation by request...")
        return GenStageResult(stage_prefix=stage_prefix, lines=lines, finalized=False)

    with stage_progress_cls(f"Computing {gen_stage_tag} model") as progress:
        gen_load_t0 = time_mod.perf_counter()
        maglib = MagFieldProcessor(cfg.nlfff_lib or "")
        _load_maglib_idl_cube(maglib, nlfff_box, prepared_run.dr3)
        gen_load_elapsed = time_mod.perf_counter() - gen_load_t0
        resolved_reduce_passed = cfg.reduce_passed if cfg.reduce_passed is not None else (0 if cfg.center_vox else 1)
        resolved_chromo_level = (1000.0 / float(cfg.dx_km)) if cfg.dx_km else 1.0
        trace_kwargs = {
            "seeds": None,
            "chromo_level": resolved_chromo_level,
            "reduce_passed": resolved_reduce_passed,
        }
        gen_trace_t0 = time_mod.perf_counter()
        lines = _lines_fast(maglib, **trace_kwargs)
        gen_trace_elapsed = time_mod.perf_counter() - gen_trace_t0
        print(
            f"\n{gen_stage_tag} tracer load: {gen_load_elapsed:.2f}s | "
            f"line trace: {gen_trace_elapsed:.2f}s | "
            f"reduce_passed={resolved_reduce_passed!r} | "
            f"chromo_level={resolved_chromo_level:.12g}"
        )
        lines = _prepare_stage_lines_payload(
            lines,
            source="memory",
            require_complete=True,
        )
        compute_lines_time = progress.finish()

    gen_save_t0 = time_mod.perf_counter()
    lines_group = _make_lines_group(lines, prepared_run.dr3)
    save_stage(gen_stage_tag, {"corona": nlfff_box, "lines": lines_group})
    print(f"{gen_stage_tag} stage save: {time_mod.perf_counter() - gen_save_t0:.2f}s")
    stage_times[gen_stage_tag] = compute_lines_time
    if cfg.generic_only or _last_stage_tag(cfg.stop_after) == "GEN":
        finalize()
        return GenStageResult(stage_prefix=stage_prefix, lines=lines, finalized=True)
    return GenStageResult(stage_prefix=stage_prefix, lines=lines, finalized=False)


def _run_chr_stage(
    prepared_run: PreparedRunState,
    nlfff_box: dict,
    *,
    stage_prefix: str,
    lines: Optional[dict],
    save_stage: Callable[..., None],
    stage_times: Dict[str, float],
    stage_progress_cls: type,
) -> None:
    chr_stage_tag = f"{stage_prefix}.CHR"
    if lines is not None:
        chr_stage_tag = f"{stage_prefix}.GEN.CHR"

    header = _make_header(prepared_run.maps["field"]) if prepared_run.maps is not None else {}
    chromo_mask = None
    if "chromo_mask" in prepared_run.base_group:
        chromo_mask = np.asarray(prepared_run.base_group["chromo_mask"]).T
    chromo_box = combo_model(
        nlfff_box,
        prepared_run.dr3,
        prepared_run.base_bz_arr.T,
        prepared_run.base_ic_arr.T,
        chromo_mask=chromo_mask,
    )
    for k in ["codes", "apex_idx", "start_idx", "end_idx", "seed_idx",
              "av_field", "phys_length", "voxel_status"]:
        if lines is not None and k in lines:
            chromo_box[k] = lines[k]
    if "phys_length" in chromo_box:
        chromo_box["phys_length"] *= prepared_run.dr3[0]
    chromo_box["attrs"] = header

    with stage_progress_cls(f"Computing {chr_stage_tag} model") as progress:
        chr_stage = {"corona": nlfff_box, "chromo": _make_chromo_group(chromo_box)}
        if lines is not None:
            chr_stage["lines"] = _make_lines_group(lines, prepared_run.dr3)
        save_stage(chr_stage_tag, chr_stage, chromo_source_axis_order_2d="xy")
        stage_times[chr_stage_tag] = progress.finish()


def _save_stage_passthrough(
    stage_tag: str,
    stage_box: dict,
    *,
    context: StageSaveContext,
    source_axis_order_3d: str = "xyz",
    chromo_source_axis_order_2d: str = "yx",
) -> dict:
    if not _should_save_stage(stage_tag, context.cfg):
        return stage_box

    prepared_run = context.prepared_run
    working_box = dict(stage_box)
    if "base" not in working_box:
        working_box["base"] = prepared_run.base_group
    if "refmaps" not in working_box:
        working_box["refmaps"] = prepared_run.refmaps
    if "corona" in working_box:
        corona = working_box["corona"]
        if "dr" not in corona:
            corona["dr"] = prepared_run.dr3
        corona_for_id = corona
        if _decode_id_text(source_axis_order_3d).strip().lower() == "zyx":
            corona_for_id = _h5_corona_to_internal_xyz(dict(corona), "zyx")
        voxel_id, corona_base = gx_box2id(
            {"corona": corona_for_id, "lines": working_box.get("lines"), "chromo": working_box.get("chromo")},
            return_corona_base=True,
        )
        if corona_base is not None:
            corona["corona_base"] = int(corona_base)
        working_box["corona"] = corona
        grid = dict(context.default_grid)
        if voxel_id is not None:
            grid["voxel_id"] = voxel_id
        else:
            grid["voxel_id"] = gx_box2id({
                "corona": {"bx": context.empty_grid, "by": context.empty_grid, "bz": context.empty_grid, "dr": prepared_run.dr3}
            })
        if "chromo" in working_box and "dz" in working_box["chromo"]:
            grid["dz"] = working_box["chromo"]["dz"]
        working_box["grid"] = grid
    elif "grid" not in working_box:
        working_box["grid"] = context.default_grid

    stage_id = f"{prepared_run.base}.{stage_tag}"
    lineage_suffix = _canonical_lineage_suffix(stage_tag)
    metadata = {
        "execute": context.execute_cmd,
        "id": stage_id,
        "lineage": (
            f"{prepared_run.lineage_marker}->{_lineage_delta_from_entry(prepared_run.entry_stage_for_marker or stage_tag, stage_tag)}.h5"
            if prepared_run.lineage_marker
            else _merge_lineage(prepared_run.lineage_root, lineage_suffix)
        ),
        "disambiguation": "SFQ" if context.cfg.sfq else "HMI",
        "projection": _decode_id_text(prepared_run.projection_tag).upper(),
        "axis_order_2d": "yx",
        "axis_order_3d": "zyx",
        "vector_layout": "split_components",
    }
    if prepared_run.vert_current_error:
        metadata["vert_current_error"] = prepared_run.vert_current_error
    working_box["metadata"] = metadata
    merged_observer = _merge_observer_metadata(
        prepared_run.observer_metadata,
        working_box.get("observer") if isinstance(working_box.get("observer"), dict) else None,
    )
    if merged_observer is not None:
        working_box["observer"] = merged_observer
    working_box = _normalize_stage_for_h5(
        working_box,
        source_axis_order_3d=source_axis_order_3d,
        chromo_source_axis_order_2d=chromo_source_axis_order_2d,
    )
    out_path = _stage_filename(context.out_dir, prepared_run.base, stage_tag)
    write_b3d_h5(str(out_path), working_box)
    print(f"Saved {stage_tag}: {out_path}")
    context.produced.append(out_path)
    return stage_box


def _run_nas_stage(
    cfg: Fov2BoxConfig,
    prepared_run: PreparedRunState,
    pot_box: Optional[dict],
    bnd_box: Optional[dict],
    nlfff_box: Optional[dict],
    expected_shape_3d: Tuple[int, int, int],
    save_stage: Callable[..., None],
    finalize: Callable[[], None],
    stage_times: Dict[str, float],
    stage_progress_cls: type,
) -> NasStageResult:
    if nlfff_box is None:
        if cfg.use_potential:
            if pot_box is None:
                raise ValueError("pot_box must be available when use_potential is enabled")
            nlfff_box = {
                "bx": np.array(pot_box["bx"], copy=True),
                "by": np.array(pot_box["by"], copy=True),
                "bz": np.array(pot_box["bz"], copy=True),
                "dr": prepared_run.dr3,
                "attrs": {"model_type": "pot"},
            }
            nlfff_box = _prepare_stage_corona_payload(
                nlfff_box,
                source="memory",
                expected_shape_3d=expected_shape_3d,
                allowed_model_types=("pot",),
            )
        else:
            if bnd_box is None:
                raise ValueError("bnd_box must be available before computing the NAS stage")
            with stage_progress_cls("Computing NAS model") as progress:
                nlfff_box = _solve_nlfff_from_bnd(
                    bnd_box,
                    prepared_run.dr3,
                    nlfff_lib=cfg.nlfff_lib,
                )
                nlfff_box = _prepare_stage_corona_payload(
                    nlfff_box,
                    source="memory",
                    expected_shape_3d=expected_shape_3d,
                    allowed_model_types=("nlfff",),
                )
                save_stage("NAS", {"corona": nlfff_box})
                stage_times["NAS"] = progress.finish()
            if cfg.nlfff_only or _last_stage_tag(cfg.stop_after) == "NAS":
                finalize()
                return NasStageResult(nlfff_box=nlfff_box, finalized=True)
    elif str(nlfff_box.get("attrs", {}).get("model_type", "")).lower() == "nlfff":
        nlfff_box = _prepare_stage_corona_payload(
            nlfff_box,
            source="memory",
            expected_shape_3d=expected_shape_3d,
            allowed_model_types=("nlfff",),
        )
        # Jumped to NAS from an existing NAS/GEN/CHR entry box.
        with stage_progress_cls("Preparing NAS model from entry data") as progress:
            save_stage("NAS", {"corona": nlfff_box})
            stage_times["NAS"] = progress.finish()
        if cfg.nlfff_only or _last_stage_tag(cfg.stop_after) == "NAS":
            finalize()
            return NasStageResult(nlfff_box=nlfff_box, finalized=True)

    if nlfff_box is None:
        raise ValueError("NAS stage did not produce an nlfff_box")
    return NasStageResult(nlfff_box=nlfff_box, finalized=False)


def _run_gen_chr_stages(
    cfg: Fov2BoxConfig,
    prepared_run: PreparedRunState,
    nlfff_box: dict,
    *,
    resume_mode: bool,
    entry_stage: Optional[str],
    target_stage: str,
    goto_chromo: bool,
    entry_lines: Optional[dict],
    save_stage: Callable[..., None],
    finalize: Callable[[], None],
    stage_times: Dict[str, float],
    stage_progress_cls: type,
) -> GenChrStageResult:
    gen_result = _run_gen_stage(
        cfg,
        prepared_run,
        nlfff_box,
        resume_mode=resume_mode,
        entry_stage=entry_stage,
        target_stage=target_stage,
        goto_chromo=goto_chromo,
        entry_lines=entry_lines,
        save_stage=save_stage,
        finalize=finalize,
        stage_times=stage_times,
        stage_progress_cls=stage_progress_cls,
    )
    if gen_result.finalized:
        return GenChrStageResult(finalized=True)

    _run_chr_stage(
        prepared_run,
        nlfff_box,
        stage_prefix=gen_result.stage_prefix,
        lines=gen_result.lines,
        save_stage=save_stage,
        stage_times=stage_times,
        stage_progress_cls=stage_progress_cls,
    )

    return GenChrStageResult(finalized=False)


def _warn_impossible_save_requests(cfg: Fov2BoxConfig, start_stage: str) -> None:
    order = {"NONE": 0, "POT": 1, "BND": 2, "NAS": 3, "GEN": 4, "CHR": 5}
    start_ord = order.get(start_stage, 0)
    req = [
        ("NONE", cfg.save_empty_box, "--save-empty-box"),
        ("BND", cfg.save_bounds, "--save-bounds"),
        ("POT", cfg.save_potential, "--save-potential"),
        ("NAS", cfg.save_nas, "--save-nas"),
        ("GEN", cfg.save_gen, "--save-gen"),
        ("CHR", cfg.save_chr, "--save-chr"),
    ]
    impossible = [flag for st, enabled, flag in req if enabled and order.get(st, 0) < start_ord]
    if impossible:
        print(
            "Warning: impossible save request(s) before selected start stage "
            f"{start_stage}: {', '.join(impossible)} (ignored)."
        )


def _clone_corona_with_model_type(corona: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    cloned = dict(corona)
    attrs = cloned.get("attrs")
    attrs_dict = dict(attrs) if isinstance(attrs, dict) else {}
    attrs_dict["model_type"] = model_type
    cloned["attrs"] = attrs_dict
    return cloned


def _prepare_resume_jump_boxes(
    active_jump: Optional[str],
    entry_stage: Optional[str],
    entry_corona: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    pot_box = None
    bnd_box = None
    nlfff_box = None

    if active_jump not in ("potential", "bounds", "nlfff"):
        return pot_box, bnd_box, nlfff_box
    if not entry_corona:
        raise ValueError("--entry-box does not contain corona vectors required by selected jump/recompute action.")

    if active_jump == "potential":
        # NONE->POT must compute a fresh potential field from the base boundary.
        if entry_stage != "NONE":
            pot_box = _clone_corona_with_model_type(entry_corona, "pot")
        return pot_box, bnd_box, nlfff_box

    if active_jump == "bounds":
        # NONE->BND and POT->BND need real forward-stage computation instead of
        # relabeling the entry corona as a later stage.
        if entry_stage == "POT":
            pot_box = _clone_corona_with_model_type(entry_corona, "pot")
        elif entry_stage not in (None, "NONE"):
            bnd_box = _clone_corona_with_model_type(entry_corona, "bnd")
        return pot_box, bnd_box, nlfff_box

    if entry_stage == "POT":
        pot_box = _clone_corona_with_model_type(entry_corona, "pot")
    elif entry_stage == "BND":
        bnd_box = _clone_corona_with_model_type(entry_corona, "bnd")
    elif entry_stage in ("NAS", "GEN", "CHR"):
        nlfff_box = _clone_corona_with_model_type(entry_corona, "nlfff")
    else:
        raise ValueError(f"Unsupported entry stage for --jump2nlfff path: {entry_stage}")
    return pot_box, bnd_box, nlfff_box


def _stage_tag_from_stage(stage: str) -> str:
    return {
        "NONE": "NONE",
        "POT": "POT",
        "BND": "BND",
        "NAS": "NAS",
        "GEN": "NAS.GEN",
        "CHR": "NAS.CHR",
    }.get(stage, "NAS.CHR")


def _extract_execute_paths(execute_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract data/model directory defaults from metadata/execute.
    Supports both Python CLI form and IDL gx_fov2box form.
    """
    if not execute_text:
        return None, None

    data_dir: Optional[str] = None
    gxmodel_dir: Optional[str] = None
    text = str(execute_text)

    # Python CLI style: gx-fov2box ... --data-dir X --gxmodel-dir Y
    try:
        parts = shlex.split(text)
    except Exception:
        parts = []
    for i, token in enumerate(parts):
        if token == "--data-dir" and i + 1 < len(parts):
            data_dir = parts[i + 1]
        elif token.startswith("--data-dir="):
            data_dir = token.split("=", 1)[1]
        elif token == "--gxmodel-dir" and i + 1 < len(parts):
            gxmodel_dir = parts[i + 1]
        elif token.startswith("--gxmodel-dir="):
            gxmodel_dir = token.split("=", 1)[1]

    # IDL style fallback: TMP_DIR='...' and OUT_DIR='...'
    if data_dir is None:
        m = re.search(r"\bTMP_DIR\s*=\s*['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\bTMP_DIR\s*=\s*([^,\s]+)", text, flags=re.IGNORECASE)
        if m:
            data_dir = m.group(1).strip().strip("'\"")
    if gxmodel_dir is None:
        m = re.search(r"\bOUT_DIR\s*=\s*['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\bOUT_DIR\s*=\s*([^,\s]+)", text, flags=re.IGNORECASE)
        if m:
            gxmodel_dir = m.group(1).strip().strip("'\"")

    return data_dir, gxmodel_dir


def _extract_execute_geometry(execute_text: str) -> Tuple[Optional[Tuple[float, float]], Optional[str], Optional[str]]:
    """
    Extract center coordinates, frame and projection hints from metadata/execute text.

    Returns:
      (coords, frame, projection)
      - coords: (x, y) as floats
      - frame: one of {"hpc","hgc","hgs"} or None
      - projection: one of {"cea","top"} or None
    """
    if not execute_text:
        return None, None, None
    text = str(execute_text)
    coords: Optional[Tuple[float, float]] = None
    frame: Optional[str] = None
    projection: Optional[str] = None

    # Python CLI style: --coords X Y [--hpc|--hgc|--hgs] [--cea|--top]
    try:
        parts = shlex.split(text)
    except Exception:
        parts = []
    for i, token in enumerate(parts):
        if token == "--coords" and i + 2 < len(parts):
            try:
                coords = (float(parts[i + 1]), float(parts[i + 2]))
            except Exception:
                pass
        elif token.startswith("--coords="):
            try:
                vals = token.split("=", 1)[1].split(",")
                if len(vals) == 2:
                    coords = (float(vals[0]), float(vals[1]))
            except Exception:
                pass
        elif token == "--hpc":
            frame = "hpc"
        elif token == "--hgc":
            frame = "hgc"
        elif token == "--hgs":
            frame = "hgs"
        elif token == "--cea":
            projection = "cea"
        elif token == "--top":
            projection = "top"

    # IDL canonical pattern:
    # CENTER_ARCSEC=[ -280, -250]
    if coords is None:
        m = re.search(
            r"\bCENTER_ARCSEC\s*=\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            coords = (float(m.group(1)), float(m.group(2)))
            frame = frame or "hpc"

    # Optional degree-based patterns.
    if coords is None:
        m = re.search(
            r"\bCENTER_HGC\s*=\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            coords = (float(m.group(1)), float(m.group(2)))
            frame = "hgc"
    if coords is None:
        m = re.search(
            r"\bCENTER_HGS\s*=\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            coords = (float(m.group(1)), float(m.group(2)))
            frame = "hgs"

    # Projection hint in IDL execute.
    if projection is None:
        up = text.upper()
        if " /TOP" in up or re.search(r",\s*TOP\s*=\s*1\b", up):
            projection = "top"
        elif " /CEA" in up or re.search(r",\s*CEA\s*=\s*1\b", up):
            projection = "cea"

    return coords, frame, projection


def _flag_explicit_on_cli(flag: str) -> bool:
    args = sys.argv[1:]
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _normalize_reproject_algorithm(value: str) -> str:
    return str(value or "").strip().lower()


def _parse_reproject_scan(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw == "all":
        return _VALID_REPROJECT_ALGORITHMS
    values = []
    for item in raw.split(","):
        alg = _normalize_reproject_algorithm(item)
        if not alg:
            continue
        if alg not in _VALID_REPROJECT_ALGORITHMS:
            raise ValueError(
                f"Unsupported reprojection algorithm '{alg}' in --reproject-scan. "
                f"Choose from: {', '.join(_VALID_REPROJECT_ALGORITHMS)}."
            )
        if alg not in values:
            values.append(alg)
    if not values:
        raise ValueError("--reproject-scan did not include any valid reprojection algorithms.")
    return tuple(values)


def _warn_path_default(role: str, path_text: str) -> None:
    p = Path(path_text).expanduser()
    if p.exists():
        return
    parent = p.parent
    if parent.exists():
        print(f"Warning: {role} from metadata/execute does not exist yet: {p} (will be used as requested).")
    else:
        print(f"Warning: {role} parent directory does not exist on this system: {parent}")


def _lines_quiet(maglib: Any, **kwargs: Any) -> Dict[str, Any]:
    """Suppress noisy stdout/stderr emitted by the underlying line tracer."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return maglib.lines(**kwargs)


def _lines_fast(maglib: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Call the pyAMaFiL line tracer without the unconditional ``print(res)``
    performed by ``MagFieldProcessor.lines()``.
    """
    reduce_passed = kwargs.get("reduce_passed")
    chromo_level = kwargs.get("chromo_level", 1)
    seeds = kwargs.get("seeds")
    max_length = kwargs.get("max_length", 0)
    step = kwargs.get("step", 1.0)
    tolerance = kwargs.get("tolerance", 1e-3)
    tolerance_bound = kwargs.get("tolerance_bound", 1e-3)
    n_processes = kwargs.get("n_processes", 0)
    debug_input = kwargs.get("debug_input", False)

    bx = getattr(maglib, "_MagFieldProcessor__bx", None)
    by = getattr(maglib, "_MagFieldProcessor__by", None)
    bz = getattr(maglib, "_MagFieldProcessor__bz", None)
    if bx is None or by is None or bz is None:
        return maglib.lines(**kwargs)

    res = maglib.lines_wrapper(
        bx,
        by,
        bz,
        reduce_passed,
        chromo_level,
        seeds,
        max_length,
        step,
        tolerance,
        tolerance_bound,
        n_processes,
        debug_input,
    )

    shape_zyx = np.flip(bx.shape)
    transpose = seeds is None
    res["voxel_status"] = mfp_util_transpose_index(res["voxel_status"], shape_zyx, transpose)
    res["phys_length"] = mfp_util_transpose_index(res["phys_length"], shape_zyx, transpose)
    res["av_field"] = mfp_util_transpose_index(res["av_field"], shape_zyx, transpose)
    res["codes"] = mfp_util_transpose_index(res["codes"], shape_zyx, transpose)
    res["apex_idx"] = mfp_util_transpose_index(mfp_util_invert_index_array(res["apex_idx"], shape_zyx), shape_zyx, transpose)
    res["start_idx"] = mfp_util_transpose_index(mfp_util_invert_index_array(res["start_idx"], shape_zyx), shape_zyx, transpose)
    res["end_idx"] = mfp_util_transpose_index(mfp_util_invert_index_array(res["end_idx"], shape_zyx), shape_zyx, transpose)
    res["seed_idx"] = mfp_util_transpose_index(mfp_util_invert_index_array(res["seed_idx"], shape_zyx), shape_zyx, transpose)

    if res.get("coords") is not None:
        res["coords"] = res["coords"][:, [1, 0, 2, 3]]

    return res


def _load_maglib_idl_cube(maglib: Any, box: Dict[str, Any], dr: Any = 0.001 * u.solRad) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cube components into ``MagFieldProcessor`` in the same boundary layout
    captured from the IDL ``mfoLines`` call:

    - internal box arrays are normalized to ``(x, y, z)``
    - DLL input arrays are ``(z, x, y)``
    - ``bx``/``by`` are swapped relative to the raw box field names
    """
    bx = np.asarray(box["by"]).transpose(2, 0, 1)
    by = np.asarray(box["bx"]).transpose(2, 0, 1)
    bz = np.asarray(box["bz"]).transpose(2, 0, 1)
    load_vars = getattr(maglib, "_MagFieldProcessor__load_vars")
    load_vars(bx, by, bz, dr)
    return bx, by, bz


def _solve_nlfff_from_bnd(
    bnd_box: Dict[str, Any],
    dr: Any,
    *,
    nlfff_lib: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the NLFFF solver from an internal ``(x, y, z)`` BND cube.

    The solver-facing layout must match the IDL ``mfoLines`` path: load the
    cube through the same component swap and ``(z, x, y)`` transpose used by
    ``_load_maglib_idl_cube()``, then preserve the returned solver component
    names and cube layout. Real BND->NAS parity shows the DLL already returns
    ``bx``/``by`` under the canonical internal names expected by the stage
    contract; swapping them again corrupts the horizontal field while leaving
    ``bz`` and array shapes intact.
    """
    maglib = MagFieldProcessor(nlfff_lib or "")
    _load_maglib_idl_cube(maglib, bnd_box, dr)
    res_nlf = maglib.NLFFF()
    return {
        "bx": np.asarray(res_nlf["bx"]),
        "by": np.asarray(res_nlf["by"]),
        "bz": np.asarray(res_nlf["bz"]),
        "dr": np.asarray(bnd_box["dr"], dtype=float).copy(),
        "attrs": {"model_type": "nlfff"},
    }


def _decode_id_text(v: Any) -> str:
    if hasattr(v, "item"):
        try:
            return _decode_id_text(v.item())
        except Exception:
            pass
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).decode("utf-8", errors="ignore")
    s = str(v or "").strip()
    if s.startswith(("b'", 'b"', "B'", 'B"')):
        try:
            # Normalize uppercase byte literal prefix (e.g. B'CEA') to valid Python syntax.
            lit_src = f"b{s[1:]}" if s[:1] == "B" else s
            lit = ast.literal_eval(lit_src)
            if isinstance(lit, (bytes, bytearray)):
                return bytes(lit).decode("utf-8", errors="ignore")
        except Exception:
            pass
    return s


def _split_stage_id(model_id: str) -> Tuple[str, str]:
    sid = _decode_id_text(model_id)
    up = sid.upper()
    for suffix in (
        ".POT.GEN.CHR", ".NAS.GEN.CHR",
        ".POT.CHR", ".NAS.CHR",
        ".POT.GEN", ".NAS.GEN",
        ".NAS", ".BND", ".POT", ".NONE",
    ):
        if up.endswith(suffix):
            return sid[: -len(suffix)], suffix[1:]
    return sid, ""


def _canonical_lineage_suffix(stage_tag: str) -> str:
    mapping = {
        "NONE": "NONE",
        "POT": "NONE.POT",
        "BND": "NONE.POT.BND",
        "NAS": "NONE.POT.BND.NAS",
        "POT.GEN": "NONE.POT.GEN",
        "POT.CHR": "NONE.POT.CHR",
        "POT.GEN.CHR": "NONE.POT.GEN.CHR",
        "NAS.GEN": "NONE.POT.BND.NAS.GEN",
        "NAS.CHR": "NONE.POT.BND.NAS.CHR",
        "NAS.GEN.CHR": "NONE.POT.BND.NAS.GEN.CHR",
    }
    return mapping.get(stage_tag, stage_tag)


def _merge_lineage(root: str, suffix: str) -> str:
    r = [t for t in str(root or "").split(".") if t]
    s = [t for t in str(suffix or "").split(".") if t]
    if not r:
        return ".".join(s)
    if not s:
        return ".".join(r)
    max_k = min(len(r), len(s))
    overlap = 0
    for i in range(max_k, 0, -1):
        if r[-i:] == s[:i]:
            overlap = i
            break
    return ".".join(r + s[overlap:])


def _lineage_delta_from_entry(entry_stage_tag: str, saved_stage_tag: str) -> str:
    """Return lineage segment from entry stage to saved stage (inclusive of implied steps)."""
    e = [t for t in _canonical_lineage_suffix(entry_stage_tag).split(".") if t]
    s = [t for t in _canonical_lineage_suffix(saved_stage_tag).split(".") if t]
    # Remove shared prefix (e.g., NONE.POT), keep what was computed after entry.
    i = 0
    while i < min(len(e), len(s)) and e[i] == s[i]:
        i += 1
    tail = s[i:]
    if not tail:
        # Clone/self-save: keep explicit stage identity on RHS.
        return saved_stage_tag
    return ".".join(tail)


def _should_save_stage(stage_tag: str, cfg: Fov2BoxConfig) -> bool:
    stop_tag = _last_stage_tag(cfg.stop_after)
    if stop_tag == "GEN" and stage_tag in ("NAS.GEN", "POT.GEN"):
        return True
    if stop_tag == "CHR" and stage_tag.endswith(".CHR"):
        return True
    if stage_tag == stop_tag:
        return True
    if stage_tag == "POT":
        return cfg.save_potential
    if stage_tag == "NAS":
        return cfg.save_nas
    if stage_tag in ("NAS.GEN", "POT.GEN"):
        return cfg.save_gen
    if stage_tag.endswith(".CHR"):
        return cfg.save_chr
    if stage_tag == "NONE":
        return cfg.save_empty_box
    if stage_tag == "BND":
        return cfg.save_bounds
    return False


def _build_execute_cmd(cfg: Fov2BoxConfig) -> str:
    cmd = ["gx-fov2box"]
    if cfg.time:
        cmd += ["--time", cfg.time]
    if cfg.coords:
        cmd += ["--coords", str(cfg.coords[0]), str(cfg.coords[1])]
    if cfg.hpc:
        cmd.append("--hpc")
    elif cfg.hgc:
        cmd.append("--hgc")
    else:
        cmd.append("--hgs")
    if cfg.top:
        cmd.append("--top")
    elif cfg.cea:
        cmd.append("--cea")
    if cfg.box_dims:
        cmd += ["--box-dims", *(str(v) for v in cfg.box_dims)]
    cmd += ["--dx-km", f"{cfg.dx_km:.6f}"]
    cmd += ["--pad-frac", f"{cfg.pad_frac:.4f}"]
    cmd += ["--data-dir", cfg.data_dir]
    cmd += ["--gxmodel-dir", cfg.gxmodel_dir]
    if str(cfg.download_backend or "drms").lower() == "fido":
        cmd.append("--use-fido")
    if cfg.force_download:
        cmd.append("--force-download")
    if cfg.hmi_time_window != 720:
        cmd += ["--hmi-time-window", f"{cfg.hmi_time_window:.6g}"]
    if cfg.aia_time_window != 12:
        cmd += ["--aia-time-window", f"{cfg.aia_time_window:.6g}"]
    if cfg.euv:
        cmd.append("--euv")
    if cfg.uv:
        cmd.append("--uv")
    if cfg.save_empty_box:
        cmd.append("--save-empty-box")
    if cfg.save_potential:
        cmd.append("--save-potential")
    if cfg.save_bounds:
        cmd.append("--save-bounds")
    if cfg.save_nas:
        cmd.append("--save-nas")
    if cfg.save_gen:
        cmd.append("--save-gen")
    if cfg.save_chr:
        cmd.append("--save-chr")
    if cfg.empty_box_only:
        cmd.append("--empty-box-only")
    if cfg.potential_only:
        cmd.append("--potential-only")
    if cfg.nlfff_only:
        cmd.append("--nlfff-only")
    if cfg.generic_only:
        cmd.append("--generic-only")
    if cfg.use_potential:
        cmd.append("--use-potential")
    if cfg.skip_lines:
        cmd.append("--skip-lines")
    if cfg.center_vox:
        cmd.append("--center-vox")
    if cfg.reduce_passed is not None:
        cmd += ["--reduce-passed", str(cfg.reduce_passed)]
    if cfg.entry_box:
        cmd += ["--entry-box", cfg.entry_box]
    if cfg.sfq:
        cmd.append("--sfq")
    if cfg.observer_name:
        cmd += ["--observer-name", cfg.observer_name]
    if cfg.fov_xc is not None:
        cmd += ["--fov-xc", f"{cfg.fov_xc:.2f}"]
    if cfg.fov_yc is not None:
        cmd += ["--fov-yc", f"{cfg.fov_yc:.2f}"]
    if cfg.fov_xsize is not None:
        cmd += ["--fov-xsize", f"{cfg.fov_xsize:.2f}"]
    if cfg.fov_ysize is not None:
        cmd += ["--fov-ysize", f"{cfg.fov_ysize:.2f}"]
    if cfg.square_fov:
        cmd.append("--square-fov")
    if cfg.stop_after:
        cmd += ["--stop-after", cfg.stop_after]
    if cfg.nlfff_lib:
        cmd += ["--nlfff-lib", cfg.nlfff_lib]
    if cfg.jump2potential:
        cmd.append("--jump2potential")
    if cfg.jump2bounds:
        cmd.append("--jump2bounds")
    if cfg.jump2nlfff:
        cmd.append("--jump2nlfff")
    if cfg.jump2lines:
        cmd.append("--jump2lines")
    if cfg.jump2chromo:
        cmd.append("--jump2chromo")
    if cfg.rebuild:
        cmd.append("--rebuild")
    if cfg.rebuild_from_none:
        cmd.append("--rebuild-from-none")
    if cfg.reproject_algorithm != "adaptive":
        cmd += ["--reproject-algorithm", cfg.reproject_algorithm]
    for path in cfg.refmaps_path or ():
        cmd += ["--refmaps-path", path]
    return shlex.join(cmd)


def _explicit_observer_fov(cfg: Fov2BoxConfig) -> Optional[dict[str, Any]]:
    parts = [cfg.fov_xc, cfg.fov_yc, cfg.fov_xsize, cfg.fov_ysize]
    if all(v is None for v in parts):
        return None
    if any(v is None for v in parts):
        raise ValueError("Observer FOV requires all of --fov-xc/--fov-yc/--fov-xsize/--fov-ysize.")
    return {
        "frame": "helioprojective",
        "xc_arcsec": float(cfg.fov_xc),
        "yc_arcsec": float(cfg.fov_yc),
        "xsize_arcsec": float(cfg.fov_xsize),
        "ysize_arcsec": float(cfg.fov_ysize),
        "square": bool(cfg.square_fov),
    }


def _observer_metadata_from_source_map(source_map: Optional[Map], cfg: Fov2BoxConfig) -> dict[str, Any]:
    observer_meta: dict[str, Any] = {"name": str(cfg.observer_name or "earth")}
    observer_meta["label"] = {
        "earth": "Earth",
        "sdo": "SDO",
        "solar orbiter": "Solar Orbiter",
        "stereo-a": "STEREO-A",
        "stereo-b": "STEREO-B",
        "custom": "Custom",
    }.get(normalize_observer_key(cfg.observer_name), "Earth")
    explicit_fov = _explicit_observer_fov(cfg)
    if explicit_fov is not None:
        observer_meta["fov"] = explicit_fov
    ephemeris: dict[str, Any] = {}
    if source_map is not None:
        source_date = getattr(source_map, "date", None)
        try:
            if source_date is not None:
                ephemeris["obs_date"] = Time(source_date).isot
        except Exception:
            pass
        obs = _resolve_cli_observer(source_map, cfg.observer_name, Time(source_date) if source_date is not None else None)
        if obs is not None:
            try:
                obs_hgs = obs.transform_to(HeliographicStonyhurst(obstime=source_date))
                ephemeris["hgln_obs_deg"] = float(obs_hgs.lon.to_value(u.deg))
                ephemeris["hglt_obs_deg"] = float(obs_hgs.lat.to_value(u.deg))
            except Exception:
                pass
        meta = getattr(source_map, "meta", None)
        if meta is not None and normalize_observer_key(cfg.observer_name) == "sdo":
            try:
                if "HGLN_OBS" in meta and "hgln_obs_deg" not in ephemeris:
                    ephemeris["hgln_obs_deg"] = float(meta["HGLN_OBS"])
                if "HGLT_OBS" in meta and "hglt_obs_deg" not in ephemeris:
                    ephemeris["hglt_obs_deg"] = float(meta["HGLT_OBS"])
                if "DSUN_OBS" in meta and "dsun_cm" not in ephemeris:
                    ephemeris["dsun_cm"] = float(u.Quantity(meta["DSUN_OBS"], u.m).to_value(u.cm))
                if "RSUN_REF" in meta and "rsun_cm" not in ephemeris:
                    ephemeris["rsun_cm"] = float(u.Quantity(meta["RSUN_REF"], u.m).to_value(u.cm))
            except Exception:
                pass
        if hasattr(source_map, "rsun_meters") and source_map.rsun_meters is not None:
            ephemeris["rsun_cm"] = float(u.Quantity(source_map.rsun_meters).to_value(u.cm))
        if obs is not None:
            try:
                ephemeris["dsun_cm"] = float(u.Quantity(obs.radius).to_value(u.cm))
            except Exception:
                pass
        elif hasattr(source_map, "dsun") and source_map.dsun is not None:
            ephemeris["dsun_cm"] = float(u.Quantity(source_map.dsun).to_value(u.cm))
    if ephemeris:
        observer_meta["ephemeris"] = ephemeris
        pb0r = build_pb0r_metadata_from_ephemeris(
            ephemeris,
            observer_key=observer_meta.get("name"),
            obs_time=ephemeris.get("obs_date"),
        )
        if pb0r:
            observer_meta["pb0r"] = pb0r
    return observer_meta


def _observer_metadata_from_entry(entry_loaded: Optional[Dict[str, Any]], cfg: Fov2BoxConfig) -> Optional[dict[str, Any]]:
    if not isinstance(entry_loaded, dict):
        return None
    base_observer = entry_loaded.get("observer", {})
    observer_meta: dict[str, Any] = dict(base_observer) if isinstance(base_observer, dict) else {}
    observer_meta["name"] = str(cfg.observer_name or observer_meta.get("name") or "earth")
    observer_meta["label"] = str(observer_meta.get("label") or {
        "earth": "Earth",
        "sdo": "SDO",
        "solar orbiter": "Solar Orbiter",
        "stereo-a": "STEREO-A",
        "stereo-b": "STEREO-B",
        "custom": "Custom",
    }.get(normalize_observer_key(observer_meta.get("name")), "Earth"))
    explicit_fov = _explicit_observer_fov(cfg)
    if explicit_fov is not None:
        observer_meta["fov"] = explicit_fov
    ephemeris = dict(observer_meta.get("ephemeris", {})) if isinstance(observer_meta.get("ephemeris"), dict) else {}
    if not ephemeris:
        index_text = None
        base_group = entry_loaded.get("base")
        if isinstance(base_group, dict):
            index_text = base_group.get("index")
        if index_text:
            try:
                header = fits.Header.fromstring(_decode_id_text(index_text), sep="\n")
                if "HGLN_OBS" in header:
                    ephemeris["hgln_obs_deg"] = float(header["HGLN_OBS"])
                if "HGLT_OBS" in header:
                    ephemeris["hglt_obs_deg"] = float(header["HGLT_OBS"])
                if "RSUN_REF" in header:
                    ephemeris["rsun_cm"] = float(u.Quantity(header["RSUN_REF"], u.m).to_value(u.cm))
                if "DSUN_OBS" in header:
                    ephemeris["dsun_cm"] = float(u.Quantity(header["DSUN_OBS"], u.m).to_value(u.cm))
            except Exception:
                pass
    if ephemeris:
        ephemeris = {
            key: ephemeris[key]
            for key in ("obs_date", "hgln_obs_deg", "hglt_obs_deg", "dsun_cm", "rsun_cm")
            if key in ephemeris
        }
    if ephemeris:
        observer_meta["ephemeris"] = ephemeris
        pb0r = build_pb0r_metadata_from_ephemeris(
            ephemeris,
            observer_key=observer_meta.get("name"),
            obs_time=ephemeris.get("obs_date"),
        )
        if pb0r:
            observer_meta["pb0r"] = pb0r
    return observer_meta or None


def _merge_observer_metadata(
    base_observer: Optional[dict[str, Any]],
    stage_observer: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    merged: dict[str, Any] = {}
    if isinstance(base_observer, dict):
        merged.update(base_observer)
    if isinstance(stage_observer, dict):
        for key, value in stage_observer.items():
            if key == "ephemeris" and isinstance(value, dict):
                ephemeris = dict(merged.get("ephemeris", {})) if isinstance(merged.get("ephemeris"), dict) else {}
                ephemeris.update(value)
                merged["ephemeris"] = ephemeris
            elif key in ("fov", "fov_box", "pb0r") and isinstance(value, dict):
                merged[key] = dict(value)
            else:
                merged[key] = value
    return merged or None


def _resolve_cli_observer(source_map: Optional[Map], observer_name: str, obs_time: Optional[Time]):
    if obs_time is None:
        return getattr(source_map, "observer_coordinate", None)
    key = normalize_observer_key(observer_name)
    if key == "earth":
        return get_earth(obs_time)
    if key == "sdo":
        if source_map is not None and getattr(source_map, "observer_coordinate", None) is not None:
            try:
                return source_map.observer_coordinate.transform_to(HeliographicStonyhurst(obstime=obs_time))
            except Exception:
                return source_map.observer_coordinate
        return get_earth(obs_time)
    coord = resolve_named_observer(key, obs_time)
    return coord if coord is not None else get_earth(obs_time)


def _build_index_header(
    bottom_wcs_header,
    source_map,
    *,
    observer_override: Any = None,
    obs_time_override: Any = None,
    rsun_override: Any = None,
) -> str:
    """
    Build an IDL-GX compatible INDEX-like FITS header string.

    This mirrors the intent of IDL `wcs2fitshead(..., /structure)` for the
    base map WCS: preserve WCS cards and add key TIME/POSITION cards expected
    by legacy GX Simulator routines.
    """
    header = fits.Header(bottom_wcs_header).copy()
    date_value = source_map.date if source_map is not None and source_map.date is not None else obs_time_override

    # Normalize to IDL-GX conventions used in box.index.
    ctype1 = str(header.get("CTYPE1", ""))
    ctype2 = str(header.get("CTYPE2", ""))
    if ctype1.startswith("HGLN-"):
        try:
            if date_value is not None and "CRVAL1" in header and "CRVAL2" in header:
                rsun_value = rsun_override
                if rsun_value is None and source_map is not None and getattr(source_map, "rsun_meters", None) is not None:
                    rsun_value = source_map.rsun_meters
                if rsun_value is None:
                    rsun_value = IDL_HMI_RSUN_M * u.m
                obs_for_carr = observer_override
                if obs_for_carr is None and source_map is not None:
                    obs_for_carr = getattr(source_map, "observer_coordinate", None)
                if obs_for_carr is None:
                    obs_for_carr = "earth"
                origin_hgs = SkyCoord(
                    lon=float(header["CRVAL1"]) * u.deg,
                    lat=float(header["CRVAL2"]) * u.deg,
                    radius=u.Quantity(rsun_value).to(u.m),
                    frame=HeliographicStonyhurst(obstime=date_value),
                )
                origin_hgc = origin_hgs.transform_to(
                    HeliographicCarrington(observer=obs_for_carr, obstime=date_value)
                )
                header["CRVAL1"] = float(origin_hgc.lon.to_value(u.deg))
                header["CRVAL2"] = float(origin_hgc.lat.to_value(u.deg))
        except Exception:
            pass
        header["CTYPE1"] = "CRLN-" + ctype1.split("-", 1)[1]
    if ctype2.startswith("HGLT-"):
        header["CTYPE2"] = "CRLT-" + ctype2.split("-", 1)[1]

    header["SIMPLE"] = int(bool(header.get("SIMPLE", 1)))
    header["BITPIX"] = int(header.get("BITPIX", 8))
    header["NAXIS"] = int(header.get("NAXIS", 2))

    if source_map is not None or obs_time_override is not None:
        date_obs = date_value.isot if date_value is not None else None
        if date_obs:
            header["DATE-OBS"] = date_obs
            header["DATE_OBS"] = date_obs
            header["DATE"] = date_obs
        elif "DATE-OBS" in header and "DATE_OBS" not in header:
            header["DATE_OBS"] = header["DATE-OBS"]

        if observer_override is not None and getattr(observer_override, "radius", None) is not None:
            header["DSUN_OBS"] = float(u.Quantity(observer_override.radius).to_value(u.m))
        elif source_map is not None and hasattr(source_map, "dsun") and source_map.dsun is not None:
            header["DSUN_OBS"] = float(source_map.dsun.to_value(u.m))

        obs = observer_override if observer_override is not None else (
            source_map.observer_coordinate if source_map is not None else None
        )
        if obs is not None:
            obs_hgs = obs.transform_to(HeliographicStonyhurst(obstime=date_value))
            header["HGLN_OBS"] = float(obs_hgs.lon.to_value(u.deg))
            header["HGLT_OBS"] = float(obs_hgs.lat.to_value(u.deg))
            header["SOLAR_B0"] = float(obs_hgs.lat.to_value(u.deg))
            try:
                obs_hgc = obs.transform_to(HeliographicCarrington(observer="earth", obstime=date_value))
                header["CRLN_OBS"] = float(obs_hgc.lon.to_value(u.deg))
                header["CRLT_OBS"] = float(obs_hgc.lat.to_value(u.deg))
            except Exception:
                # If Carrington transformation fails (e.g. unsupported observer configuration),
                # proceed without CRLN_OBS/CRLT_OBS rather than aborting header creation.
                # Carrington observer transforms can fail for some observer metadata;
                # keep header generation robust and proceed with available keys.
                pass

        if rsun_override is not None:
            header["RSUN_REF"] = float(u.Quantity(rsun_override).to_value(u.m))
        elif source_map is not None and getattr(source_map, "rsun_meters", None) is not None:
            header["RSUN_REF"] = float(source_map.rsun_meters.to_value(u.m))

        if source_map is not None:
            tel = source_map.meta.get("telescop")
            if tel:
                header["TELESCOP"] = str(tel)
            instr = source_map.meta.get("instrume")
            if instr:
                header["INSTRUME"] = str(instr)
        if "WCSNAME" not in header:
            header["WCSNAME"] = "Carrington-Heliographic"

    return header.tostring(sep="\n", endcard=True)


def _print_info(cfg: Fov2BoxConfig) -> None:
    resolved_reduce_passed = cfg.reduce_passed if cfg.reduce_passed is not None else (0 if cfg.center_vox else 1)
    resolved_chromo_level = (1000.0 / float(cfg.dx_km)) if cfg.dx_km else 1.0
    import pyampp
    import sunpy

    runtime_rows = [
        ("python_executable", sys.executable, "Interpreter used for this run"),
        ("python_version", sys.version.split()[0], "Python runtime version"),
        ("gx_fov2box_module", __file__, "Loaded gx_fov2box module path"),
        ("pyampp_module", pyampp.__file__, "Loaded pyampp package path"),
        ("sunpy_version", sunpy.__version__, "Loaded sunpy version"),
        ("gx-fov2box_cli", shutil.which("gx-fov2box"), "Resolved CLI entry point on PATH"),
    ]
    rows = [
        ("time", cfg.time, "Observation time in ISO format"),
        ("coords", cfg.coords, "Center coordinates: arcsec (HPC) or degrees (HGC/HGS)"),
        ("hpc/hgc/hgs", "HPC" if cfg.hpc else "HGC" if cfg.hgc else "HGS", "Coordinate frame"),
        ("cea/top", "TOP" if cfg.top else "CEA", "Basemap projection"),
        ("box_dims", cfg.box_dims, "Box dimensions (nx, ny, nz) in pixels"),
        ("dx_km", cfg.dx_km, "Voxel size in km"),
        ("pad_frac", cfg.pad_frac, "Padding fraction used for context map FOV"),
        ("data_dir", cfg.data_dir, "SDO download/cache directory"),
        ("gxmodel_dir", cfg.gxmodel_dir, "Output gx_models directory"),
        ("nlfff_lib", cfg.nlfff_lib, "Override WWNLFFFReconstruction library path for NAS and GEN"),
        ("download_backend", cfg.download_backend, "SDO downloader backend"),
        ("drms_sequential", cfg.drms_sequential, "Force single-worker DRMS downloads (HMI and AIA)"),
        ("force_download", cfg.force_download, "Bypass local cache hits and redownload requested SDO products"),
        ("hmi_time_window", cfg.hmi_time_window, "HMI JSOC search window in seconds (IDL HMI_time_window)"),
        ("aia_time_window", cfg.aia_time_window, "AIA JSOC search window in seconds (IDL AIA_time_window)"),
        ("entry_box", cfg.entry_box, "Path to precomputed HDF5 box"),
        ("save_empty_box", cfg.save_empty_box, "Save NONE stage"),
        ("save_potential", cfg.save_potential, "Save POT stage"),
        ("save_bounds", cfg.save_bounds, "Save BND stage"),
        ("save_nas", cfg.save_nas, "Save NAS stage"),
        ("save_gen", cfg.save_gen, "Save NAS.GEN stage"),
        ("save_chr", cfg.save_chr, "Save NAS.CHR stage"),
        ("stop_after", cfg.stop_after, "Stop after stage (dl/none/bnd/pot/nas/gen/chr)"),
        ("empty_box_only", cfg.empty_box_only, "Stop after NONE stage"),
        ("potential_only", cfg.potential_only, "Stop after POT stage"),
        ("nlfff_only", cfg.nlfff_only, "Stop after NAS stage"),
        ("generic_only", cfg.generic_only, "Stop after NAS.GEN stage"),
        ("use_potential", cfg.use_potential, "Skip NLFFF; reuse potential field"),
        ("skip_lines", cfg.skip_lines, "Skip GEN line computation and proceed directly to CHR"),
        ("center_vox", cfg.center_vox, "Compute lines only through voxel centers (sets reduce_passed=0 unless overridden)"),
        ("reduce_passed", cfg.reduce_passed, "Expert override: 0|1|2|3 (takes precedence over --center-vox)"),
        ("reduce_passed_resolved", resolved_reduce_passed, "Effective line-reduction mode used by tracer"),
        ("chromo_level_resolved", resolved_chromo_level, "Effective chromo level passed to line tracer (1 Mm / dx_km)"),
        ("euv", cfg.euv, "Download AIA EUV context maps"),
        ("uv", cfg.uv, "Download AIA UV context maps"),
        ("sfq", cfg.sfq, "Use SFQ disambiguation (method=0)"),
        ("jump2potential", cfg.jump2potential, "Start from entry box and jump to POT"),
        ("jump2bounds", cfg.jump2bounds, "Start from entry box and jump to BND"),
        ("jump2nlfff", cfg.jump2nlfff, "Start from entry box and jump to NAS"),
        ("jump2lines", cfg.jump2lines, "Start from entry box and jump to GEN"),
        ("jump2chromo", cfg.jump2chromo, "Start from entry box and jump to CHR"),
        ("rebuild", cfg.rebuild, "Ignore entry stage payload and rebuild from NONE"),
        ("rebuild_from_none", cfg.rebuild_from_none, "Strip entry to NONE-equivalent and continue forward"),
    ]
    print("gx-fov2box --info")
    for name, value, desc in runtime_rows + rows:
        print(f"- {name}: {value} :: {desc}")
    missing = []
    if not cfg.time:
        missing.append("--time")
    if not cfg.coords:
        missing.append("--coords")
    if not cfg.box_dims:
        missing.append("--box-dims")
    if missing:
        print("\nWarnings:")
        for item in missing:
            print(f"- Missing required {item}")
        if cfg.entry_box:
            print("- entry_box provided; time/box-dims may be inferred if present")


def _load_hmi_maps_from_downloader(
    time: Time,
    data_dir: Path,
    euv: bool,
    uv: bool,
    download_backend: str = "drms",
    drms_sequential: bool = False,
    force_download: bool = False,
    hmi_time_window: float = 720,
    aia_time_window: float = 12,
    disambig_method: int = 2,
    strict_required: bool = True,
) -> tuple[Dict[str, Map], dict]:
    import time as time_mod

    def _missing_items(report: dict[str, Any], *categories: str) -> list[str]:
        missing: list[str] = []
        for category in categories:
            items = report.get(category, {}) if isinstance(report, dict) else {}
            if isinstance(items, dict):
                missing.extend([k for k, exists in items.items() if not exists])
        return missing

    requested_time = Time(time)
    def _new_downloader(*, obs_time: Time, want_euv: bool, want_uv: bool, want_hmi: bool):
        kwargs = {
            "data_dir": str(data_dir),
            "euv": want_euv,
            "uv": want_uv,
            "hmi": want_hmi,
            "backend": download_backend,
            "force_download": force_download,
            "hmi_time_window": hmi_time_window,
            "aia_time_window": aia_time_window,
        }
        if download_backend == "drms":
            kwargs["drms_sequential"] = drms_sequential
        try:
            return SDOImageDownloader(obs_time, **kwargs)
        except TypeError as exc:
            if "drms_sequential" in str(exc):
                kwargs.pop("drms_sequential", None)
                return SDOImageDownloader(obs_time, **kwargs)
            raise

    downloader = _new_downloader(obs_time=time, want_euv=False, want_uv=False, want_hmi=True)
    missing_before = _missing_items(downloader.existence_report, "hmi_b", "hmi_m", "hmi_ic")
    t0 = time_mod.perf_counter()
    files = dict(downloader.download_images())
    hmi_elapsed = time_mod.perf_counter() - t0
    downloaded = force_download or len(missing_before) > 0
    required = ["field", "inclination", "azimuth", "disambig", "continuum", "magnetogram"]
    optional_requested: list[str] = []
    if euv:
        optional_requested.extend(list(AIA_EUV_PASSBANDS))
    if uv:
        optional_requested.extend(list(AIA_UV_PASSBANDS))
    missing_required = [k for k in required if not files.get(k)]
    missing_optional = [k for k in optional_requested if not files.get(k)]
    if missing_required and strict_required:
        print("Download summary:")
        print(f"- required missing: {missing_required}")
        print(f"- optional missing: {missing_optional}")
        raise RuntimeError(f"Missing required HMI files: {missing_required}")
    if missing_required and not strict_required:
        print("Download summary:")
        print(f"- required missing: {missing_required}")
        print(f"- optional missing: {missing_optional}")
        info = {
            "requested_obs_time": requested_time.isot,
            "resolved_obs_time": None,
            "downloaded": downloaded,
            "elapsed": hmi_elapsed,
            "hmi_elapsed": hmi_elapsed,
            "context_elapsed": 0.0,
            "missing_before": missing_before,
            "missing_after": missing_required,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "files": files,
        }
        return {}, info

    map_field = load_sunpy_map_compat(files["field"])
    map_inclination = load_sunpy_map_compat(files["inclination"])
    map_azimuth = load_sunpy_map_compat(files["azimuth"])
    map_disambig = load_sunpy_map_compat(files["disambig"])
    map_conti = load_sunpy_map_compat(files["continuum"])
    map_losma = load_sunpy_map_compat(files["magnetogram"])

    map_azimuth = hmi_disambig(map_azimuth, map_disambig, method=disambig_method)

    maps = {
        "field": map_field,
        "inclination": map_inclination,
        "azimuth": map_azimuth,
        "disambig": map_disambig,
        "continuum": map_conti,
        "magnetogram": map_losma,
    }
    continuum_date = getattr(map_conti, "date", None)
    resolved_obs_time = Time(continuum_date) if continuum_date is not None else requested_time

    context_elapsed = 0.0
    if euv or uv:
        context_downloader = _new_downloader(
            obs_time=resolved_obs_time,
            want_euv=euv,
            want_uv=uv,
            want_hmi=False,
        )
        context_missing_before = _missing_items(context_downloader.existence_report, "euv", "uv")
        missing_before.extend(context_missing_before)
        t0 = time_mod.perf_counter()
        context_files = context_downloader.download_images()
        context_elapsed = time_mod.perf_counter() - t0
        downloaded = downloaded or force_download or len(context_missing_before) > 0
        for key, path in context_files.items():
            if key in required:
                continue
            files[key] = path
        missing_optional = [k for k in optional_requested if not files.get(k)]

    print("Download summary:")
    print(f"- required missing: {missing_required}")
    print(f"- optional missing: {missing_optional}")

    for key, path in files.items():
        if key in ("field", "inclination", "azimuth", "disambig", "continuum", "magnetogram"):
            continue
        try:
            maps[f"AIA_{key}"] = load_sunpy_map_compat(path)
        except Exception:
            # Skip optional context channels that cannot be loaded.
            continue
    info = {
        "requested_obs_time": requested_time.isot,
        "resolved_obs_time": resolved_obs_time.isot,
        "downloaded": downloaded,
        "elapsed": hmi_elapsed + context_elapsed,
        "hmi_elapsed": hmi_elapsed,
        "context_elapsed": context_elapsed,
        "missing_before": missing_before,
        "missing_after": missing_required,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "files": files,
    }
    return maps, info


def _make_header(map_field: Map) -> Dict[str, Any]:
    header_field = map_field.wcs.to_header()
    field_frame = map_field.center.heliographic_carrington.frame
    lon, lat = field_frame.lon.value, field_frame.lat.value
    obs_time = Time(map_field.date)
    dsun_obs = header_field["DSUN_OBS"]
    return {"lon": lon, "lat": lat, "dsun_obs": dsun_obs, "obs_time": str(obs_time.iso)}


def _make_lines_group(lines: dict, dr3: np.ndarray) -> dict:
    group = {
        "dr": np.asarray(dr3, dtype=float),
    }
    for key in (
        "start_idx",
        "end_idx",
        "seed_idx",
        "apex_idx",
        "codes",
        "av_field",
        "phys_length",
        "voxel_status",
    ):
        if key in lines:
            group[key] = lines[key]
    return group


def _make_chromo_group(chromo_box: dict) -> dict:
    group = {}
    for key in (
        "chromo_idx",
        "chromo_n",
        "chromo_t",
        "n_p",
        "n_hi",
        "n_htot",
        "tr",
        "tr_h",
        "chromo_layers",
        "dz",
        "chromo_mask",
    ):
        if key in chromo_box:
            group[key] = chromo_box[key]

    # Export CHR vectors explicitly in H5 (no BCUBE replacement in schema).
    chromo_bcube = chromo_box.get("chromo_bcube")
    if isinstance(chromo_bcube, np.ndarray) and chromo_bcube.ndim == 4 and chromo_bcube.shape[-1] == 3:
        group["bx"] = chromo_bcube[:, :, :, 0]
        group["by"] = chromo_bcube[:, :, :, 1]
        group["bz"] = chromo_bcube[:, :, :, 2]
    return group


def _to_h5_2d(arr: np.ndarray, ny: int, nx: int, source_axis_order_2d: str = "yx") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 2:
        return a
    axis_order = _decode_id_text(source_axis_order_2d).strip().lower()
    if axis_order == "yx":
        if a.shape == (ny, nx):
            return a
        if a.shape == (nx, ny):
            return a.T
        raise ValueError(
            f"Cannot normalize 2D map shape {a.shape} from source order {source_axis_order_2d!r} "
            f"to (ny,nx)=({ny},{nx})"
        )
    if axis_order == "xy":
        if a.shape == (nx, ny) or (nx == ny and a.shape == (ny, nx)):
            return a.T
        raise ValueError(
            f"Cannot normalize 2D map shape {a.shape} from source order {source_axis_order_2d!r} "
            f"to (ny,nx)=({ny},{nx})"
        )
    raise ValueError(f"Unsupported 2D source axis order: {source_axis_order_2d!r}")


def _to_h5_3d(arr: np.ndarray, source_axis_order_3d: str) -> np.ndarray:
    """
    Convert a 3D volume to canonical HDF5 order ``(z, y, x)``.

    ``gx_fov2box`` currently uses explicit internal ``(x, y, z)`` volumes for
    solver-facing paths, while HDF5 stores 3D volumes canonically as
    ``(z, y, x)``. Do not infer the transform from the array shape because
    cubic boxes make that ambiguous.
    """
    a = np.asarray(arr)
    if a.ndim != 3:
        return a
    axis_order = _decode_id_text(source_axis_order_3d).strip().lower()
    if axis_order == "zyx":
        return a
    if axis_order == "xyz":
        return a.transpose((2, 1, 0))
    raise ValueError(f"Unsupported 3D source axis order: {source_axis_order_3d!r}")


def _normalize_stage_for_h5(
    stage_box: dict,
    source_axis_order_3d: str = "xyz",
    base_source_axis_order_2d: str = "yx",
    chromo_source_axis_order_2d: str = "yx",
) -> dict:
    out = dict(stage_box)
    if "base" not in out:
        return out
    base = dict(out["base"])
    if "bx" in base:
        ny, nx = np.asarray(base["bx"]).shape
    elif "bz" in base:
        ny, nx = np.asarray(base["bz"]).shape
    else:
        out["base"] = base
        return out

    for k in ("bx", "by", "bz", "ic", "chromo_mask"):
        if k in base:
            base[k] = _to_h5_2d(base[k], ny, nx, source_axis_order_2d=base_source_axis_order_2d)
    out["base"] = base

    if "corona" in out and isinstance(out["corona"], dict):
        corona = dict(out["corona"])
        for k in ("bx", "by", "bz"):
            if k in corona:
                corona[k] = _to_h5_3d(corona[k], source_axis_order_3d)
        out["corona"] = corona

    if "chromo" in out and isinstance(out["chromo"], dict):
        chromo = dict(out["chromo"])
        for k in ("bx", "by", "bz", "dz"):
            if k in chromo:
                chromo[k] = _to_h5_3d(chromo[k], source_axis_order_3d)
        for k in ("tr", "tr_h", "chromo_mask"):
            if k in chromo:
                chromo[k] = _to_h5_2d(chromo[k], ny, nx, source_axis_order_2d=chromo_source_axis_order_2d)
        out["chromo"] = chromo

    if "grid" in out and isinstance(out["grid"], dict):
        grid = dict(out["grid"])
        for k in ("voxel_id", "dz"):
            if k in grid and np.asarray(grid[k]).ndim == 3:
                grid[k] = _to_h5_3d(grid[k], source_axis_order_3d)
        out["grid"] = grid
    return out


def _h5_corona_to_internal_xyz(corona: dict, axis_order_3d: Optional[str]) -> dict:
    if not isinstance(corona, dict):
        return corona
    axis_order = _decode_id_text(axis_order_3d).lower()
    if axis_order != "zyx":
        return corona
    out = dict(corona)
    for k in ("bx", "by", "bz"):
        if k in out and np.asarray(out[k]).ndim == 3:
            out[k] = np.asarray(out[k]).transpose((2, 1, 0))
    return out


def _load_entry_box_any(entry_path: Path) -> Dict[str, Any]:
    return load_model(entry_path)


def _resolve_box_params(cfg: Fov2BoxConfig) -> Tuple[Time, Tuple[int, int, int], float]:
    obs_time = Time(cfg.time) if cfg.time else None
    box_dims = cfg.box_dims
    dx_km = cfg.dx_km
    if cfg.entry_box:
        entry_path = Path(cfg.entry_box)
        if entry_path.exists():
            box_b3d = _load_entry_box_any(entry_path)
            corona = box_b3d.get("corona")
            if corona is not None:
                if box_dims is None and "bx" in corona:
                    shape = np.asarray(corona["bx"]).shape
                    axis_order = _decode_id_text(box_b3d.get("metadata", {}).get("axis_order_3d")).lower()
                    if axis_order == "zyx" and len(shape) == 3:
                        box_dims = (int(shape[2]), int(shape[1]), int(shape[0]))
                    else:
                        # internal cubes are x,y,z unless metadata says zyx
                        if len(shape) == 3:
                            box_dims = (int(shape[0]), int(shape[1]), int(shape[2]))
                        else:
                            box_dims = tuple(int(v) for v in shape)
                if "dr" in corona and dx_km == 0:
                    dr3 = corona["dr"]
                    dx_km = float(dr3[0] * (IDL_HMI_RSUN_M * u.m).to(u.km).value)
        if obs_time is None:
            inferred = _infer_time_from_entry_loaded(box_b3d, entry_path)
            if inferred:
                obs_time = Time(inferred)
    if obs_time is None:
        raise ValueError("--time is required unless it can be inferred from entry_box")
    if box_dims is None:
        raise ValueError("--box-dims is required unless it can be inferred from entry_box")
    return obs_time, tuple(int(v) for v in box_dims), float(dx_km)


@app.command()
def main(
    time: Optional[str] = typer.Option(None, "--time", help="Observation time ISO (e.g. 2024-05-12T00:00:00)"),
    coords: Optional[Tuple[float, float]] = typer.Option(None, "--coords", help="Center coords (x y)"),
    hpc: bool = typer.Option(False, "--hpc/--no-hpc", help="Use helioprojective coordinates"),
    hgc: bool = typer.Option(False, "--hgc/--no-hgc", help="Use heliographic carrington coordinates"),
    hgs: bool = typer.Option(False, "--hgs/--no-hgs", help="Use heliographic stonyhurst coordinates"),
    cea: bool = typer.Option(False, "--cea", help="Use CEA basemap projection"),
    top: bool = typer.Option(False, "--top", help="Use TOP-view basemap projection"),
    box_dims: Optional[Tuple[int, int, int]] = typer.Option(None, "--box-dims", help="Box dims in pixels"),
    dx_km: float = typer.Option(1400.0, "--dx-km", help="Voxel size in km"),
    pad_frac: float = typer.Option(0.10, "--pad-frac", help="Padding fraction for FOV"),
    data_dir: str = typer.Option(DOWNLOAD_DIR, "--data-dir", help="SDO data directory"),
    gxmodel_dir: str = typer.Option(GXMODEL_DIR, "--gxmodel-dir", help="GX model output directory"),
    nlfff_lib: Optional[str] = typer.Option(None, "--nlfff-lib", help="Override the WWNLFFFReconstruction shared library path for NAS and GEN stages"),
    download_backend: Optional[str] = typer.Option(None, "--download-backend", help="Compatibility override: explicitly set downloader backend to fido or drms"),
    use_fido: bool = typer.Option(False, "--use-fido", help="Use the legacy SunPy/Fido downloader instead of the default DRMS backend"),
    drms_sequential: bool = typer.Option(False, "--drms-sequential", help="Force DRMS downloads to single-worker mode for maximum reliability"),
    force_download: bool = typer.Option(False, "--force-download", help="Bypass local cache hits and redownload requested SDO products"),
    hmi_time_window: float = typer.Option(720.0, "--hmi-time-window", help="HMI JSOC search window in seconds (default 720, IDL HMI_time_window)"),
    aia_time_window: float = typer.Option(12.0, "--aia-time-window", help="AIA JSOC search window in seconds (default 12; UV uses max(12, 24), IDL AIA_time_window)"),
    entry_box: Optional[str] = typer.Option(None, "--entry-box", help="Existing HDF5/SAV box"),
    save_empty_box: bool = typer.Option(False, "--save-empty-box", help="Save NONE stage"),
    save_potential: bool = typer.Option(False, "--save-potential", help="Save POT stage"),
    save_bounds: bool = typer.Option(False, "--save-bounds", help="Save BND stage"),
    save_nas: bool = typer.Option(False, "--save-nas", help="Save NAS stage"),
    save_gen: bool = typer.Option(False, "--save-gen", help="Save NAS.GEN stage"),
    save_chr: bool = typer.Option(False, "--save-chr", help="Save NAS.CHR stage"),
    stop_after: Optional[str] = typer.Option(None, "--stop-after", help="Stop after stage (dl/none/pot/bnd/nas/gen/chr)"),
    empty_box_only: bool = typer.Option(False, "--empty-box-only", help="Stop after NONE"),
    potential_only: bool = typer.Option(False, "--potential-only", help="Stop after POT"),
    nlfff_only: bool = typer.Option(False, "--nlfff-only", help="Stop after NAS"),
    generic_only: bool = typer.Option(False, "--generic-only", help="Stop after NAS.GEN"),
    use_potential: bool = typer.Option(False, "--use-potential", help="Skip NLFFF"),
    skip_lines: bool = typer.Option(False, "--skip-lines", help="Skip GEN line computation and go directly to CHR"),
    center_vox: bool = typer.Option(False, "--center-vox", help="Trace lines through voxel centers"),
    reduce_passed: Optional[int] = typer.Option(
        None,
        "--reduce-passed",
        min=0,
        max=3,
        help="Expert line tracing reduction bitmask: 0|1|2|3 (overrides --center-vox)",
    ),
    euv: bool = typer.Option(False, "--euv", help="Download AIA EUV maps"),
    uv: bool = typer.Option(False, "--uv", help="Download AIA UV maps"),
    sfq: bool = typer.Option(False, "--sfq", help="Use SFQ disambiguation (method=0)"),
    observer_name: str = typer.Option("earth", "--observer-name", help="Observer identifier stored in output metadata"),
    fov_xc: Optional[float] = typer.Option(None, "--fov-xc", help="Observer/image FOV center X in arcsec"),
    fov_yc: Optional[float] = typer.Option(None, "--fov-yc", help="Observer/image FOV center Y in arcsec"),
    fov_xsize: Optional[float] = typer.Option(None, "--fov-xsize", help="Observer/image FOV width in arcsec"),
    fov_ysize: Optional[float] = typer.Option(None, "--fov-ysize", help="Observer/image FOV height in arcsec"),
    square_fov: bool = typer.Option(False, "--square-fov", help="Observer/image FOV is constrained square"),
    jump2potential: bool = typer.Option(False, "--jump2potential", help="Jump to POT"),
    jump2bounds: bool = typer.Option(False, "--jump2bounds", help="Jump to BND"),
    jump2nlfff: bool = typer.Option(False, "--jump2nlfff", help="Jump to NAS"),
    jump2lines: bool = typer.Option(False, "--jump2lines", help="Jump to GEN"),
    jump2chromo: bool = typer.Option(False, "--jump2chromo", help="Jump to CHR"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Recompute from NONE using entry-box parameters"),
    rebuild_from_none: bool = typer.Option(False, "--rebuild-from-none", help="Start from entry-box NONE-equivalent and run forward"),
    info: bool = typer.Option(False, "--info", help="Show resolved defaults and exit"),
    reproject_algorithm: str = typer.Option(
        "adaptive",
        "--reproject-algorithm",
        help="Reprojection algorithm for HMI cutouts onto model base grid: 'adaptive' (default, matches IDL wcs_remap/ssaa behaviour), 'exact' (flux-conserving, slower), or 'interpolation' (bilinear)",
    ),
    reproject_scan: Optional[str] = typer.Option(
        None,
        "--reproject-scan",
        help="Batch-run OBS->NONE once per reprojection algorithm. Use 'all' or a comma-separated list such as 'adaptive,exact,interpolation'. Each run writes to a gxmodel-dir/reproject_<algorithm> subdirectory.",
    ),
    refmaps_path: Optional[List[str]] = typer.Option(
        None,
        "--refmaps-path",
        "--refmap-path",
        help="Directory of external FITS reference maps to add to model refmaps. May be repeated.",
    ),
) -> None:
    cfg = Fov2BoxConfig(
        time=time,
        coords=coords,
        hpc=hpc,
        hgc=hgc,
        hgs=hgs,
        cea=cea,
        top=top,
        box_dims=box_dims,
        dx_km=dx_km,
        pad_frac=pad_frac,
        data_dir=data_dir,
        gxmodel_dir=gxmodel_dir,
        nlfff_lib=nlfff_lib,
        download_backend=download_backend or "drms",
        drms_sequential=drms_sequential,
        force_download=force_download,
        hmi_time_window=hmi_time_window,
        aia_time_window=aia_time_window,
        entry_box=entry_box,
        save_empty_box=save_empty_box,
        save_potential=save_potential,
        save_bounds=save_bounds,
        save_nas=save_nas,
        save_gen=save_gen,
        save_chr=save_chr,
        stop_after=stop_after,
        empty_box_only=empty_box_only,
        potential_only=potential_only,
        nlfff_only=nlfff_only,
        generic_only=generic_only,
        use_potential=use_potential,
        skip_lines=skip_lines,
        center_vox=center_vox,
        reduce_passed=reduce_passed,
        euv=euv,
        uv=uv,
        sfq=sfq,
        observer_name=observer_name,
        fov_xc=fov_xc,
        fov_yc=fov_yc,
        fov_xsize=fov_xsize,
        fov_ysize=fov_ysize,
        square_fov=square_fov,
        jump2potential=jump2potential,
        jump2bounds=jump2bounds,
        jump2nlfff=jump2nlfff,
        jump2lines=jump2lines,
        jump2chromo=jump2chromo,
        rebuild=rebuild,
        rebuild_from_none=rebuild_from_none,
        info=info,
        reproject_algorithm=reproject_algorithm,
        reproject_scan=reproject_scan,
        refmaps_path=tuple(refmaps_path or ()),
    )

    cfg.download_backend = str(cfg.download_backend or "drms").strip().lower()
    if cfg.download_backend not in {"fido", "drms"}:
        raise ValueError("Unsupported --download-backend. Expected 'fido' or 'drms'.")
    if use_fido:
        if download_backend is not None and cfg.download_backend != "fido":
            raise ValueError("Conflicting downloader controls: --use-fido and --download-backend drms.")
        cfg.download_backend = "fido"

    cfg.reproject_algorithm = _normalize_reproject_algorithm(cfg.reproject_algorithm)
    if cfg.reproject_algorithm not in _VALID_REPROJECT_ALGORITHMS:
        raise ValueError(
            f"Unsupported --reproject-algorithm '{cfg.reproject_algorithm}'. "
            f"Choose from: {', '.join(_VALID_REPROJECT_ALGORITHMS)}."
        )
    reproject_scan_algorithms = _parse_reproject_scan(cfg.reproject_scan)

    if not any([cfg.hpc, cfg.hgc, cfg.hgs]):
        cfg.hpc = True
    if cfg.cea and cfg.top:
        raise ValueError("Select only one projection: --cea or --top")
    if not cfg.cea and not cfg.top:
        cfg.cea = True
    if cfg.square_fov and cfg.fov_xsize is not None:
        cfg.fov_ysize = cfg.fov_xsize
    _explicit_observer_fov(cfg)

    legacy_stop_flags = [
        ("none", cfg.empty_box_only, "--empty-box-only"),
        ("pot", cfg.potential_only, "--potential-only"),
        ("nas", cfg.nlfff_only, "--nlfff-only"),
        ("gen", cfg.generic_only, "--generic-only"),
    ]
    active_legacy = [item for item in legacy_stop_flags if item[1]]
    if len(active_legacy) > 1:
        names = ", ".join(item[2] for item in active_legacy)
        raise ValueError(f"Conflicting legacy stop flags: {names}. Use only one or use --stop-after.")
    if active_legacy:
        legacy_stop = active_legacy[0][0]
        if cfg.stop_after is not None and _last_stage_tag(cfg.stop_after) != _last_stage_tag(legacy_stop):
            raise ValueError(
                f"Conflicting stop controls: --stop-after={cfg.stop_after} and {active_legacy[0][2]}."
            )
        cfg.stop_after = cfg.stop_after or legacy_stop

    if cfg.info:
        _print_info(cfg)
        return

    if cfg.rebuild_from_none and not cfg.entry_box:
        raise ValueError("--rebuild-from-none requires --entry-box")
    if cfg.rebuild and cfg.rebuild_from_none:
        raise ValueError("Use only one of --rebuild or --rebuild-from-none")

    jump_flags_set = any([cfg.jump2potential, cfg.jump2bounds, cfg.jump2nlfff, cfg.jump2lines, cfg.jump2chromo])
    if jump_flags_set and not cfg.entry_box:
        print("Warning: jump2* flags are ignored unless --entry-box is provided.")
    if cfg.rebuild_from_none and jump_flags_set:
        raise ValueError("--rebuild-from-none cannot be combined with jump2* flags")
    if cfg.use_potential and (cfg.jump2nlfff or cfg.jump2bounds):
        raise ValueError("--use-potential cannot be combined with --jump2nlfff or --jump2bounds")
    if cfg.skip_lines and _last_stage_tag(cfg.stop_after) == "GEN":
        raise ValueError("--skip-lines cannot be combined with stop-after GEN (or equivalent).")
    if cfg.skip_lines and cfg.center_vox:
        print("Warning: --center-vox is ignored when --skip-lines is set.")
        cfg.center_vox = False

    entry_loaded: Optional[Dict[str, Any]] = None
    entry_stage: Optional[str] = None
    target_stage = "NONE"
    data_dir_explicit = _flag_explicit_on_cli("--data-dir")
    gxmodel_dir_explicit = _flag_explicit_on_cli("--gxmodel-dir")
    if cfg.entry_box:
        entry_path = Path(cfg.entry_box)
        if not entry_path.exists():
            raise ValueError(f"--entry-box not found: {entry_path}")
        entry_loaded = _load_entry_box_any(entry_path)
        entry_stage = _entry_stage_from_loaded(entry_loaded, entry_path)
        execute_text = _decode_id_text(entry_loaded.get("metadata", {}).get("execute", ""))
        exec_data_dir, exec_gxmodel_dir = _extract_execute_paths(execute_text)
        exec_coords, exec_frame, exec_projection = _extract_execute_geometry(execute_text)
        frame_explicit = any(
            _flag_explicit_on_cli(flag) for flag in ("--hpc", "--hgc", "--hgs")
        )
        proj_explicit = any(_flag_explicit_on_cli(flag) for flag in ("--cea", "--top"))
        if exec_data_dir and not data_dir_explicit:
            cfg.data_dir = exec_data_dir
            print(f"Using --data-dir from entry metadata/execute: {cfg.data_dir}")
            _warn_path_default("data_dir", cfg.data_dir)
        if exec_gxmodel_dir and not gxmodel_dir_explicit:
            cfg.gxmodel_dir = exec_gxmodel_dir
            print(f"Using --gxmodel-dir from entry metadata/execute: {cfg.gxmodel_dir}")
            _warn_path_default("gxmodel_dir", cfg.gxmodel_dir)
        if cfg.coords is None and exec_coords is not None:
            cfg.coords = exec_coords
            print(f"Using --coords from entry metadata/execute: {cfg.coords}")
        if not frame_explicit and exec_frame in ("hpc", "hgc", "hgs"):
            cfg.hpc = exec_frame == "hpc"
            cfg.hgc = exec_frame == "hgc"
            cfg.hgs = exec_frame == "hgs"
        if not proj_explicit and exec_projection in ("cea", "top"):
            cfg.cea = exec_projection == "cea"
            cfg.top = exec_projection == "top"
    transition_plan = _plan_transition(cfg, entry_stage)
    target_stage = transition_plan.target_stage

    if reproject_scan_algorithms is not None:
        if target_stage != "NONE":
            raise ValueError(
                "--reproject-scan is limited to OBS->NONE generation. "
                "Use --stop-after none (or equivalent) for reprojection sweeps."
            )
        scan_root = Path(cfg.gxmodel_dir).expanduser().resolve()
        script_path = Path(__file__).resolve()
        print(
            "Running reprojection sweep for NONE generation: "
            + ", ".join(reproject_scan_algorithms)
        )
        for algorithm in reproject_scan_algorithms:
            child_cfg = replace(
                cfg,
                gxmodel_dir=str(scan_root / f"reproject_{algorithm}"),
                save_empty_box=True,
                save_potential=False,
                save_bounds=False,
                save_nas=False,
                save_gen=False,
                save_chr=False,
                stop_after="none",
                empty_box_only=False,
                potential_only=False,
                nlfff_only=False,
                generic_only=False,
                use_potential=False,
                skip_lines=False,
                reproject_algorithm=algorithm,
                reproject_scan=None,
            )
            child_args = shlex.split(_build_execute_cmd(child_cfg))[1:]
            child_cmd = [sys.executable, str(script_path), *child_args]
            print(f"\n[{algorithm}] writing NONE output under {child_cfg.gxmodel_dir}")
            subprocess.run(child_cmd, check=True)
        print("Reprojection sweep complete.")
        return

    observer_metadata = _observer_metadata_from_entry(entry_loaded, cfg) if entry_loaded is not None else None

    import time as time_mod
    import warnings
    import logging

    t_start = time_mod.perf_counter()
    warnings.filterwarnings("ignore", message=".*assume_spherical_screen.*")
    logging.getLogger("reproject").setLevel(logging.WARNING)
    logging.getLogger("sunpy").setLevel(logging.WARNING)

    requested_obs_time, box_dims_resolved, dx_km = _resolve_box_params(cfg)
    cfg.dx_km = dx_km
    obs_time = requested_obs_time
    resume_mode = bool(cfg.entry_box and not cfg.rebuild and entry_loaded is not None)

    if not resume_mode:
        if cfg.coords is None:
            raise ValueError("--coords is required")
        if sum([cfg.hpc, cfg.hgc, cfg.hgs]) != 1:
            raise ValueError("Select exactly one of --hpc, --hgc, --hgs")

    _warn_impossible_save_requests(cfg, target_stage)
    stage_rank = {"NONE": 0, "POT": 1, "BND": 2, "NAS": 3, "GEN": 4, "CHR": 5}
    start_rank = stage_rank.get(target_stage, 0)

    maps = None
    base_bz_arr = None
    base_ic_arr = None
    bottom_bz_data = None
    vert_current_error = None
    projection_tag = "CEA"
    base = None
    dr3 = None

    def _start_progress_emitter(label: str, start: float):
        isatty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        if hasattr(os, "fork"):
            pid = os.fork()
            if pid == 0:
                frames = "|/-\\"
                idx = 0
                last_heartbeat = 0.0
                try:
                    while True:
                        elapsed = time_mod.perf_counter() - start
                        frame = frames[idx % len(frames)]
                        if isatty:
                            print(f"\r{label}... {frame}", end="", flush=True)
                        elif elapsed - last_heartbeat >= 5.0:
                            print(f"{label}... {elapsed:.1f}s elapsed", flush=True)
                            last_heartbeat = elapsed
                        idx += 1
                        time_mod.sleep(0.15)
                finally:
                    os._exit(0)
            return pid, isatty, None
        stop_event = threading.Event()

        def _heartbeat():
            frames = "|/-\\"
            idx = 0
            last_heartbeat = 0.0
            while not stop_event.wait(0.15):
                elapsed = time_mod.perf_counter() - start
                frame = frames[idx % len(frames)]
                if isatty:
                    print(f"\r{label}... {frame}", end="", flush=True)
                elif elapsed - last_heartbeat >= 5.0:
                    print(f"{label}... {elapsed:.1f}s elapsed")
                    last_heartbeat = elapsed
                idx += 1

        thread = threading.Thread(target=_heartbeat, daemon=True)
        thread.start()
        return None, isatty, (stop_event, thread)

    def _stop_progress_emitter(pid, thread_state):
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            return
        if thread_state is not None:
            stop_event, thread = thread_state
            stop_event.set()
            thread.join(timeout=0.3)

    def _run_logged_step(label: str, func):
        start = time_mod.perf_counter()
        print(f"{label}...")
        pid, isatty, thread_state = _start_progress_emitter(label, start)
        try:
            result = func()
        except Exception:
            elapsed = time_mod.perf_counter() - start
            _stop_progress_emitter(pid, thread_state)
            if isatty:
                print(f"\r{label}... failed after {elapsed:.2f}s")
            else:
                print(f"{label} failed after {elapsed:.2f}s")
            raise

        elapsed = time_mod.perf_counter() - start
        _stop_progress_emitter(pid, thread_state)
        if isatty:
            print(f"\r{label}... done in {elapsed:.2f}s")
        else:
            print(f"{label} done in {elapsed:.2f}s")
        return result

    prepared_run = _prepare_run_state(
        cfg,
        resume_mode,
        entry_loaded,
        entry_stage,
        target_stage,
        requested_obs_time,
        box_dims_resolved,
        observer_metadata,
        _run_logged_step,
        t_start,
    )
    if prepared_run is None:
        return

    box_dims_resolved = _effective_box_dims_from_base(box_dims_resolved, prepared_run.base_group)

    empty_grid = np.zeros((box_dims_resolved[0], box_dims_resolved[1], box_dims_resolved[2]), dtype=float)
    default_grid = {
        "dx": float(prepared_run.dr3[0]),
        "dy": float(prepared_run.dr3[1]),
        "dz": np.array([float(prepared_run.dr3[2])], dtype=float),
    }

    out_dir = _stage_output_dir(cfg.gxmodel_dir, prepared_run.obs_time)
    execute_cmd = _build_execute_cmd(cfg)

    produced = []
    stage_times = {}

    class _StageProgress:
        def __init__(self, label: str):
            self.label = label
            self._start = None
            self._stop_event = threading.Event()
            self._thread = None
            self._isatty = bool(getattr(sys.stdout, "isatty", lambda: False)())
            self._finished = False

        def __enter__(self):
            self._start = time_mod.perf_counter()
            print(f"{self.label}...")
            self._pid, self._isatty, self._thread_state = _start_progress_emitter(self.label, self._start)
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is not None and not self._finished:
                elapsed = time_mod.perf_counter() - (self._start or time_mod.perf_counter())
                _stop_progress_emitter(getattr(self, "_pid", None), getattr(self, "_thread_state", None))
                if self._isatty:
                    print(f"\r{self.label}... failed after {elapsed:.2f}s")
                else:
                    print(f"{self.label} failed after {elapsed:.2f}s")
            return False

        def finish(self) -> float:
            elapsed = time_mod.perf_counter() - (self._start or time_mod.perf_counter())
            self._finished = True
            _stop_progress_emitter(getattr(self, "_pid", None), getattr(self, "_thread_state", None))
            if self._isatty:
                print(f"\r{self.label}... done in {elapsed:.2f}s")
            else:
                print(f"{self.label} done in {elapsed:.2f}s")
            return elapsed

    def finalize() -> None:
        if produced:
            print("\nCompleted gx-fov2box. Output files:")
            for path in produced:
                print(f"- {path}")
        if stage_times:
            print("Stage timings:")
            for stage, seconds in stage_times.items():
                print(f"- {stage}: {seconds:.2f}s")
        total = time_mod.perf_counter() - t_start
        print(f"Total elapsed: {total:.2f}s")

    save_context = StageSaveContext(
        cfg=cfg,
        prepared_run=prepared_run,
        execute_cmd=execute_cmd,
        out_dir=out_dir,
        default_grid=default_grid,
        empty_grid=empty_grid,
        produced=produced,
    )

    def save_stage(
        stage_tag: str,
        stage_box: dict,
        *,
        source_axis_order_3d: str = "xyz",
        chromo_source_axis_order_2d: str = "yx",
    ) -> dict:
        return _save_stage_passthrough(
            stage_tag,
            stage_box,
            context=save_context,
            source_axis_order_3d=source_axis_order_3d,
            chromo_source_axis_order_2d=chromo_source_axis_order_2d,
        )

    if start_rank <= 0 and (cfg.save_empty_box or cfg.empty_box_only or cfg.stop_after in ("none", "empty", "empty_box")):
        with _StageProgress("Computing NONE model") as progress:
            # NONE stage should preserve boundary conditions at z=0 and keep z>0 empty.
            # Canonical 3D order in H5 is (nz, ny, nx).
            stage_box = _compute_none_stage_box(prepared_run, box_dims_resolved)
            stage_box = _normalize_runtime_stage_box_for_pipeline(
                stage_box,
                prepared_run=prepared_run,
                stage_tag="NONE",
                source_axis_order_3d="xyz",
            )
            from pyampp.io.model import ensure_geometry_contract_in_metadata
            # Inject geometry_contract before writing NONE stage
            ensure_geometry_contract_in_metadata(stage_box)
            save_stage("NONE", stage_box, source_axis_order_3d="xyz")
            stage_times["NONE"] = progress.finish()
        if cfg.empty_box_only or _last_stage_tag(cfg.stop_after) == "NONE":
            finalize()
            return

    transition_inputs = _prepare_transition_stage_inputs(
        prepared_run,
        transition_plan,
        entry_stage=entry_stage,
        box_dims_resolved=box_dims_resolved,
    )
    active_jump = transition_inputs.active_jump
    entry_lines = transition_inputs.entry_lines
    goto_lines = transition_inputs.goto_lines
    goto_chromo = transition_inputs.goto_chromo
    pot_box = transition_inputs.pot_box
    bnd_box = transition_inputs.bnd_box
    nlfff_box = transition_inputs.nlfff_box

    if not goto_lines:

        if pot_box is None and bnd_box is None and nlfff_box is None:
            with _StageProgress("Computing POT model") as progress:
                pot_box = _compute_pot_stage_box(prepared_run, box_dims_resolved)["corona"]
                save_stage("POT", {"corona": pot_box})
                stage_times["POT"] = progress.finish()
            if cfg.potential_only or _last_stage_tag(cfg.stop_after) == "POT":
                finalize()
                return
        elif pot_box is not None:
            with _StageProgress("Preparing POT model from entry data") as progress:
                save_stage("POT", {"corona": pot_box})
                stage_times["POT"] = progress.finish()
            if cfg.potential_only or _last_stage_tag(cfg.stop_after) == "POT":
                finalize()
                return

        if bnd_box is None and pot_box is not None and not cfg.use_potential:
            with _StageProgress("Computing BND model") as progress:
                bnd_box = _compute_bnd_stage_box(prepared_run, pot_box, box_dims_resolved)["corona"]
                save_stage("BND", {"corona": bnd_box})
                stage_times["BND"] = progress.finish()
            if _last_stage_tag(cfg.stop_after) == "BND":
                finalize()
                return
        elif bnd_box is not None:
            with _StageProgress("Preparing BND model from entry data") as progress:
                save_stage("BND", {"corona": bnd_box})
                stage_times["BND"] = progress.finish()
            if _last_stage_tag(cfg.stop_after) == "BND":
                finalize()
                return

        if nlfff_box is None and cfg.use_potential and pot_box is None:
            pot_box = _compute_pot_stage_box(prepared_run, box_dims_resolved)["corona"]
        if nlfff_box is None and not cfg.use_potential and bnd_box is None:
            if pot_box is None:
                pot_box = _compute_pot_stage_box(prepared_run, box_dims_resolved)["corona"]
            bnd_box = _compute_bnd_stage_box(prepared_run, pot_box, box_dims_resolved)["corona"]

        nas_result = _run_nas_stage(
            cfg,
            prepared_run,
            pot_box,
            bnd_box,
            nlfff_box,
            box_dims_resolved,
            save_stage,
            finalize,
            stage_times,
            _StageProgress,
        )
        nlfff_box = _prepare_stage_corona_payload(
            nas_result.nlfff_box,
            source="memory",
            expected_shape_3d=box_dims_resolved,
            allowed_model_types=("pot", "nlfff"),
        )
        if nas_result.finalized:
            return

    gen_chr_result = _run_gen_chr_stages(
        cfg,
        prepared_run,
        nlfff_box,
        resume_mode=resume_mode,
        entry_stage=entry_stage,
        target_stage=target_stage,
        goto_chromo=goto_chromo,
        entry_lines=entry_lines,
        save_stage=save_stage,
        finalize=finalize,
        stage_times=stage_times,
        stage_progress_cls=_StageProgress,
    )
    if gen_chr_result.finalized:
        return

    finalize()


if __name__ == "__main__":
    app()
