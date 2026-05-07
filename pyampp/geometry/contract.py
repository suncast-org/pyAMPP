"""
Geometry Contract Enforcement: Complete and validate intrinsic box geometry metadata.

This module ensures that all loaded models have complete Tier 1 (intrinsic box) and
Tier 2 (world embedding) metadata, so the geometry module never needs to recompute
or infer coordinates from multiple fallback paths.

**Tier 1: Intrinsic Box Geometry (Mandatory)**
- Nx, Ny, Nz: box dimensions in pixels (from coronal cube shape)
- dr_x, dr_y, dr_z: voxel resolution in units of solar radius Rsun
- Rsun: solar radius in meters (fixed to HMI convention, not inferred per-model)

**Tier 2: World Embedding (Mandatory)**
- anchor_lon: box anchor longitude in HeliographicStonyhurst degrees
- anchor_lat: box anchor latitude in HeliographicStonyhurst degrees
- anchor_radius: box anchor radius in solar radii
- frame: coordinate frame name (e.g., "heliographic_stonyhurst")
- obstime: ISO timestamp of observation

**Design:**
- Completion is done at model load time (SAV→H5 conversion or H5 read).
- Completed fields are stored in model["metadata"]["geometry_contract"].
- Geometry functions in gximagecomputing read this contract, eliminating fallback branching.
- Models saved to H5 cache completed metadata; unsaved models re-complete on next load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from astropy.time import Time

# HMI solar radius: fixed value, never inferred per-model
# From SDO/HMI documentation, matches heliopy/sunpy convention
RSUN_HMI_METERS = 6.957e8


@dataclass(frozen=True)
class GeometryContract:
    """
    Complete geometry metadata contract for a box model.
    
    All fields are required for a valid contract. Models lacking any field
    after completion are considered incomplete and should raise an error
    in production use.
    """
    # Tier 1: Intrinsic Box
    nx: int
    ny: int
    nz: int
    dr_x: float  # in units of Rsun
    dr_y: float  # in units of Rsun
    dr_z: float  # in units of Rsun
    rsun_m: float  # solar radius in meters
    
    # Tier 2: World Embedding
    anchor_lon_deg: float  # HeliographicStonyhurst
    anchor_lat_deg: float  # HeliographicStonyhurst
    anchor_radius_rsun: float  # in units of Rsun
    frame: str  # e.g., "heliographic_stonyhurst"
    obstime: str  # ISO timestamp
    
    # Provenance (informational only; does not affect geometry)
    inferred_from: Optional[str] = None  # e.g., "index", "execute", "defaults"
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize contract to a dictionary for H5 storage."""
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
            "inferred_from": str(self.inferred_from) if self.inferred_from else "",
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeometryContract:
        """Deserialize contract from a dictionary."""
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
    """
    Infer box dimensions (Nx, Ny, Nz) from the coronal cube shape.
    
    Uses corona group only (not chromo, which has non-uniform resolution).
    Returns (nx, ny, nz) in pixels, or None if cannot be inferred.
    """
    corona = model_dict.get("corona")
    if not isinstance(corona, dict):
        return None
    
    # Try component fields first
    for key in ("bx", "by", "bz"):
        if key in corona:
            arr = np.asarray(corona[key])
            if arr.ndim >= 3:
                # Internal cubes are stored as (x, y, z) unless metadata says otherwise
                shape = arr.shape[:3]
                return (int(shape[0]), int(shape[1]), int(shape[2]))
    
    return None


def infer_voxel_resolution(model_dict: dict[str, Any]) -> tuple[float, float, float] | None:
    """
    Infer voxel resolution (dr_x, dr_y, dr_z) in units of solar radius.
    
    Uses corona.dr only (chromo has non-uniform resolution and must not be used).
    Returns (dr_x, dr_y, dr_z) in Rsun, or None if cannot be inferred.
    """
    corona = model_dict.get("corona")
    if not isinstance(corona, dict):
        return None
    
    dr = corona.get("dr")
    if dr is None:
        return None
    
    try:
        arr = np.asarray(dr, dtype=np.float64).ravel()
        if arr.size >= 3:
            return (float(arr[0]), float(arr[1]), float(arr[2]))
        if arr.size >= 2:
            return (float(arr[0]), float(arr[1]), float(arr[0]))
        if arr.size >= 1:
            return (float(arr[0]), float(arr[0]), float(arr[0]))
    except (ValueError, TypeError):
        pass
    
    return None


def infer_world_anchor_from_index(
    model_dict: dict[str, Any],
    obstime: Time | str | None = None,
) -> tuple[float, float, float, str] | None:
    """
    Infer world anchor (lon, lat, radius, frame) from INDEX metadata.
    
    Returns (anchor_lon_deg, anchor_lat_deg, anchor_radius_rsun, frame) or None.
    Frame is always "heliographic_stonyhurst" for index-based anchors.
    """
    metadata = model_dict.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    
    index_header_text = metadata.get("index_header")
    if isinstance(index_header_text, bytes):
        index_header_text = index_header_text.decode("utf-8", "ignore")
    
    if not index_header_text:
        return None
    
    # Parse FITS-like header for WCS coordinates
    # Expects CRVAL1 (lon), CRVAL2 (lat), RSUN_REF
    lon_deg = None
    lat_deg = None
    rsun_m = None
    
    for line in str(index_header_text).split("\n"):
        line = line.strip()
        if line.startswith("CRVAL1"):
            try:
                lon_deg = float(line.split("=")[1].split("/")[0].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("CRVAL2"):
            try:
                lat_deg = float(line.split("=")[1].split("/")[0].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("RSUN_REF"):
            try:
                rsun_m = float(line.split("=")[1].split("/")[0].strip())
            except (ValueError, IndexError):
                pass
    
    if lon_deg is None or lat_deg is None:
        return None
    
    # Assume anchor is at solar surface (radius = 1 Rsun)
    anchor_radius_rsun = 1.0
    
    return (float(lon_deg), float(lat_deg), anchor_radius_rsun, "heliographic_stonyhurst")


def infer_world_anchor_from_execute(
    model_dict: dict[str, Any],
    obstime: Time | str | None = None,
) -> tuple[float, float, float, str] | None:
    """
    Infer world anchor (lon, lat, radius, frame) from EXECUTE text.
    
    Attempts to parse center coordinates and coordinate mode from EXECUTE.
    Returns (anchor_lon_deg, anchor_lat_deg, anchor_radius_rsun, frame) or None.
    """
    # This is a simplified version; full parsing would use extract_geometry_from_execute
    # For now, we return None to indicate that a real implementation would need
    # to call gx_fov2box functions to properly parse EXECUTE.
    # This is a design choice: contract.py should not import gx_fov2box (circular dependency).
    return None


def infer_world_anchor_defaults(
    obstime: Time | str | None = None,
) -> tuple[float, float, float, str]:
    """
    Provide default world anchor when no metadata is available.
    
    Defaults to disk center (lon=0, lat=0, radius=1 Rsun) in HGS.
    """
    return (0.0, 0.0, 1.0, "heliographic_stonyhurst")


def infer_obstime(model_dict: dict[str, Any]) -> str | None:
    """
    Infer observation time from model metadata.
    
    Tries: metadata.obstime, chromo.attrs.obs_time, defaults to None.
    """
    metadata = model_dict.get("metadata", {})
    if isinstance(metadata, dict):
        obs_time = metadata.get("obstime")
        if obs_time:
            if isinstance(obs_time, bytes):
                return obs_time.decode("utf-8", "ignore")
            return str(obs_time)
    
    chromo = model_dict.get("chromo", {})
    if isinstance(chromo, dict):
        attrs = getattr(chromo, "attrs", {}) if hasattr(chromo, "attrs") else chromo.get("attrs", {})
        if isinstance(attrs, dict):
            obs_time = attrs.get("obs_time")
            if obs_time:
                if isinstance(obs_time, bytes):
                    return obs_time.decode("utf-8", "ignore")
                return str(obs_time)
    
    return None


def complete_geometry_contract(
    model_dict: dict[str, Any],
    *,
    strict: bool = False,
) -> GeometryContract | None:
    """
    Complete a geometry contract from available model metadata.
    
    Enforces strict Tier 1 (intrinsic box) completeness. Tier 2 (world embedding)
    uses fallbacks to reasonable defaults if metadata is incomplete.
    
    Parameters
    ----------
    model_dict : dict
        Loaded model dictionary with corona, metadata, etc.
    strict : bool
        If True, raise ValueError if any Tier 1 or 2 field cannot be inferred.
        If False (default), use defaults for Tier 2 and return None for incomplete Tier 1.
    
    Returns
    -------
    GeometryContract or None
        Completed contract if all Tier 1 fields are present, None otherwise.
    
    Raises
    ------
    ValueError
        If strict=True and any required field cannot be inferred.
    """
    # Tier 1: Intrinsic Box (all mandatory)
    dims = infer_box_dims(model_dict)
    if dims is None:
        if strict:
            raise ValueError("Cannot infer box dimensions from model.")
        return None
    nx, ny, nz = dims
    
    resolution = infer_voxel_resolution(model_dict)
    if resolution is None:
        if strict:
            raise ValueError("Cannot infer voxel resolution from corona.dr.")
        return None
    dr_x, dr_y, dr_z = resolution
    
    # Tier 2: World Embedding (fallback to defaults if needed)
    obstime = infer_obstime(model_dict)
    if obstime is None:
        if strict:
            raise ValueError("Cannot infer observation time from model.")
        obstime = "2020-01-01T00:00:00"  # Arbitrary default
    
    # Try to infer world anchor with fallback
    inferred_from = None
    anchor_lon, anchor_lat, anchor_radius, frame = None, None, None, None
    
    # Priority 1: INDEX metadata
    result = infer_world_anchor_from_index(model_dict, obstime)
    if result:
        anchor_lon, anchor_lat, anchor_radius, frame = result
        inferred_from = "index"
    
    # Priority 2: EXECUTE metadata (not implemented here; would require gx_fov2box import)
    # result = infer_world_anchor_from_execute(model_dict, obstime)
    # if result and anchor_lon is None:
    #     anchor_lon, anchor_lat, anchor_radius, frame = result
    #     inferred_from = "execute"
    
    # Priority 3: Defaults
    if anchor_lon is None:
        anchor_lon, anchor_lat, anchor_radius, frame = infer_world_anchor_defaults(obstime)
        inferred_from = "defaults"
    
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
        inferred_from=inferred_from,
    )


__all__ = [
    "RSUN_HMI_METERS",
    "GeometryContract",
    "complete_geometry_contract",
    "infer_box_dims",
    "infer_voxel_resolution",
    "infer_world_anchor_from_index",
    "infer_world_anchor_defaults",
    "infer_obstime",
]
