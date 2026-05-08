"""
Geometry contract completion for model box metadata.

Contract policy:
- Tier 1 (intrinsic box): inferred from coronal cube and corona.dr only.
- Tier 2 (world embedding + time): inferred from base/index.

base/index is treated as canonical. For backward compatibility, this module also
parses legacy tuple-style base/index payloads produced by older converters.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning

# HMI solar radius in meters (fixed convention)
RSUN_HMI_METERS = 6.957e8

_ISO_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?"
_FLOAT_PATTERN = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"


@dataclass(frozen=True)
class GeometryContract:
    # Tier 1: intrinsic box
    nx: int
    ny: int
    nz: int
    dr_x: float
    dr_y: float
    dr_z: float
    rsun_m: float

    # Tier 2: world embedding + time
    anchor_lon_deg: float
    anchor_lat_deg: float
    anchor_radius_rsun: float
    frame: str
    obstime: str

    # informational provenance
    inferred_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nx": int(self.nx),
            "ny": int(self.ny),
            "nz": int(self.nz),
            "dr_x": float(self.dr_x),
            "dr_y": float(self.dr_y),
            "dr_z": float(self.dr_z),
            "rsun_m": float(self.rsun_m),
            "anchor_lon_deg": float(self.anchor_lon_deg),
            "anchor_lat_deg": float(self.anchor_lat_deg),
            "anchor_radius_rsun": float(self.anchor_radius_rsun),
            "frame": str(self.frame),
            "obstime": str(self.obstime),
            "inferred_from": str(self.inferred_from or ""),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeometryContract:
        return cls(
            nx=int(data["nx"]),
            ny=int(data["ny"]),
            nz=int(data["nz"]),
            dr_x=float(data["dr_x"]),
            dr_y=float(data["dr_y"]),
            dr_z=float(data["dr_z"]),
            rsun_m=float(data["rsun_m"]),
            anchor_lon_deg=float(data["anchor_lon_deg"]),
            anchor_lat_deg=float(data["anchor_lat_deg"]),
            anchor_radius_rsun=float(data["anchor_radius_rsun"]),
            frame=str(data["frame"]),
            obstime=str(data["obstime"]),
            inferred_from=data.get("inferred_from") or None,
        )


def infer_box_dims(model_dict: dict[str, Any]) -> tuple[int, int, int] | None:
    """Infer (nx, ny, nz) from coronal cube components only."""
    corona = model_dict.get("corona")
    if not isinstance(corona, dict):
        return None

    for key in ("bx", "by", "bz"):
        if key in corona:
            arr = np.asarray(corona[key])
            if arr.ndim >= 3:
                shape = arr.shape[:3]
                return (int(shape[0]), int(shape[1]), int(shape[2]))
    return None


def infer_voxel_resolution(model_dict: dict[str, Any]) -> tuple[float, float, float] | None:
    """Infer (dr_x, dr_y, dr_z) from corona.dr only (never from chromo)."""
    corona = model_dict.get("corona")
    if not isinstance(corona, dict):
        return None

    dr = corona.get("dr")
    if dr is None:
        return None

    try:
        arr = np.asarray(dr, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None

    if arr.size >= 3:
        return (float(arr[0]), float(arr[1]), float(arr[2]))
    if arr.size == 2:
        return (float(arr[0]), float(arr[1]), float(arr[0]))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    return None


def _decode_header_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, str):
        return value
    return str(value)


def _index_header_text_from_model(model_dict: dict[str, Any]) -> str | None:
    """
    Read index header text from canonical base/index first.

    Backward-compatible fallback order:
    1) base/index
    2) base/index_header
    3) metadata/index_header
    """
    base = model_dict.get("base")
    metadata = model_dict.get("metadata")

    header_text: str | None = None
    if isinstance(base, dict):
        for key in ("index", "index_header"):
            if key in base:
                header_text = _decode_header_text(base.get(key))
                if header_text:
                    break

    if not header_text and isinstance(metadata, dict):
        header_text = _decode_header_text(metadata.get("index_header"))

    if header_text:
        return header_text
    return None


def _fits_header_from_text(header_text: str) -> fits.Header | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyUserWarning)
            return fits.Header.fromstring(header_text, sep="\n")
    except Exception:
        return None


def _extract_obstime_from_text(header_text: str) -> str | None:
    # FITS-like key-value forms
    for key in ("DATE-OBS", "DATE_OBS", "DATE"):
        m = re.search(rf"{re.escape(key)}\s*=\s*'?({_ISO_PATTERN})'?", header_text)
        if m:
            return m.group(1)

    # Legacy tuple-style payloads contain one or more ISO timestamps; first is obs time.
    m = re.search(_ISO_PATTERN, header_text)
    if m:
        return m.group(0)
    return None


def _extract_anchor_from_text(header_text: str) -> tuple[float, float] | None:
    # FITS-like forms
    m_lon = re.search(rf"CRVAL1\s*=\s*({_FLOAT_PATTERN})", header_text)
    m_lat = re.search(rf"CRVAL2\s*=\s*({_FLOAT_PATTERN})", header_text)
    if m_lon and m_lat:
        return (float(m_lon.group(1)), float(m_lat.group(1)))

    # Legacy FITSHEAD2STRUCT tuple-like payload: after Carrington-Heliographic
    m = re.search(
        rf"Carrington-Heliographic'\s*,\s*({_FLOAT_PATTERN})\s*,\s*({_FLOAT_PATTERN})",
        header_text,
    )
    if m:
        return (float(m.group(1)), float(m.group(2)))

    return None


def _extract_anchor_radius_from_text(header_text: str) -> float:
    m = re.search(rf"RSUN_REF\s*=\s*({_FLOAT_PATTERN})", header_text)
    if not m:
        return 1.0
    try:
        rsun_ref_m = float(m.group(1))
        return rsun_ref_m / RSUN_HMI_METERS
    except Exception:
        return 1.0


def infer_obstime(model_dict: dict[str, Any]) -> str | None:
    """Infer model observation time from canonical base/index payload."""
    header_text = _index_header_text_from_model(model_dict)
    if not header_text:
        return None

    header = _fits_header_from_text(header_text)
    if header is not None:
        for key in ("DATE-OBS", "DATE_OBS", "DATE"):
            value = header.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text

    return _extract_obstime_from_text(header_text)


def infer_world_anchor_from_index(model_dict: dict[str, Any]) -> tuple[float, float, float, str] | None:
    """Infer world anchor from canonical base/index payload."""
    header_text = _index_header_text_from_model(model_dict)
    if not header_text:
        return None

    lon_lat: tuple[float, float] | None = None

    header = _fits_header_from_text(header_text)
    if header is not None:
        try:
            lon_lat = (float(header["CRVAL1"]), float(header["CRVAL2"]))
        except Exception:
            lon_lat = None

    if lon_lat is None:
        lon_lat = _extract_anchor_from_text(header_text)
    if lon_lat is None:
        return None

    lon_deg, lat_deg = lon_lat
    anchor_radius_rsun = _extract_anchor_radius_from_text(header_text)
    return (lon_deg, lat_deg, anchor_radius_rsun, "heliographic_stonyhurst")


def complete_geometry_contract(
    model_dict: dict[str, Any],
    *,
    strict: bool = False,
) -> GeometryContract | None:
    """
    Complete a geometry contract.

    Policy: base/index is canonical for time and anchor metadata.
    If observation time or anchor cannot be inferred, the contract is incomplete.
    """
    dims = infer_box_dims(model_dict)
    if dims is None:
        if strict:
            raise ValueError("Cannot infer box dimensions from coronal cube.")
        return None

    resolution = infer_voxel_resolution(model_dict)
    if resolution is None:
        if strict:
            raise ValueError("Cannot infer voxel resolution from corona.dr.")
        return None

    obstime = infer_obstime(model_dict)
    if obstime is None:
        if strict:
            raise ValueError("Cannot infer observation time from base/index.")
        return None

    anchor = infer_world_anchor_from_index(model_dict)
    if anchor is None:
        if strict:
            raise ValueError("Cannot infer anchor geometry from base/index.")
        return None

    nx, ny, nz = dims
    dr_x, dr_y, dr_z = resolution
    anchor_lon, anchor_lat, anchor_radius, frame = anchor

    return GeometryContract(
        nx=nx,
        ny=ny,
        nz=nz,
        dr_x=dr_x,
        dr_y=dr_y,
        dr_z=dr_z,
        rsun_m=RSUN_HMI_METERS,
        anchor_lon_deg=anchor_lon,
        anchor_lat_deg=anchor_lat,
        anchor_radius_rsun=anchor_radius,
        frame=frame,
        obstime=obstime,
        inferred_from="index",
    )


__all__ = [
    "RSUN_HMI_METERS",
    "GeometryContract",
    "complete_geometry_contract",
    "infer_box_dims",
    "infer_voxel_resolution",
    "infer_world_anchor_from_index",
    "infer_obstime",
]
