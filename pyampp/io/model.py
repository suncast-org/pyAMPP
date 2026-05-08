"""
Centralized model loading and saving with geometry contract enforcement.

ARCHITECTURE:
This module is the single canonical loader for all model formats (SAV, H5).
It enforces the following contract on ALL model restores:
- Tier 1 metadata (box dims, voxel resolution) completeness
- Tier 2 metadata (world anchor, observation time) completeness
- Observer ephemeris normalization at load time

For new models: enforce strict contract completeness.
For old models (SAV, incomplete H5): compute/infer missing Tier 1+2 fields
from available fallbacks (index, execute, cube shape, dr) and add them to
the loaded model.

When an old model is saved back to H5, those computed fields get persisted.
On next load, persisted fields are used; if not saved, they are recomputed.

This eliminates the need for geometry to recompute or branch on metadata
state, and centralizes all model I/O paths through contract-enforced loaders.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pyampp.geometry.contract import (
    GeometryContract,
    complete_geometry_contract,
)
from pyampp.gxbox.boxutils import (
    normalize_observer_metadata,
    read_b3d_h5,
    write_b3d_h5,
)
from pyampp.util.build_h5_from_sav import build_h5_from_sav


def _ensure_group(f: h5py.File | h5py.Group, name: str):
    """Get or create an HDF5 group."""
    if name in f:
        return f[name]
    return f.create_group(name)


def _replace_dataset(group: h5py.Group, key: str, value: Any) -> None:
    """Replace or create a dataset in an HDF5 group."""
    if key in group:
        del group[key]
    group.create_dataset(key, data=value)


def _read_contract_from_h5(h5_path: Path | str) -> GeometryContract | None:
    """Read a pre-computed geometry contract from HDF5 if present."""
    h5_path = Path(h5_path)
    try:
        with h5py.File(h5_path, "r") as f:
            if "metadata" in f and "geometry_contract" in f["metadata"]:
                g_contract = f["metadata"]["geometry_contract"]
                data = {}
                for key in g_contract.keys():
                    val = g_contract[key][()]
                    if isinstance(val, (bytes, np.bytes_)):
                        val = val.decode("utf-8", "ignore")
                    data[key] = val
                return GeometryContract.from_dict(data)
    except Exception:
        pass
    return None


def _write_contract_to_h5(h5_path: Path | str, contract: GeometryContract) -> bool:
    """
    Persist a geometry contract to HDF5.

    Returns True if successful, False otherwise.
    """
    h5_path = Path(h5_path)
    if contract is None:
        return False

    try:
        with h5py.File(h5_path, "r+") as f:
            g_meta = _ensure_group(f, "metadata")
            g_contract = _ensure_group(g_meta, "geometry_contract")

            for key, value in contract.to_dict().items():
                if isinstance(value, str):
                    value = np.bytes_(value)
                elif isinstance(value, (int, float)):
                    value = np.array(value)
                _replace_dataset(g_contract, key, value)
        return True
    except Exception:
        return False


def load_model_from_h5(
    h5_path: Path | str,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Load a model from HDF5 with geometry contract enforcement.

    Policy:
    - Try to read pre-stored contract from HDF5 (Tier 1+2 metadata)
    - If not present, infer/compute missing Tier 1+2 metadata from available
      fallbacks (index, execute, cube shape, dr)
    - Always normalize observer ephemeris at load time
    - Return complete model dict guaranteed to have Tier 1+2 metadata

    Args:
        h5_path: Path to HDF5 file
        strict: If True, raise if contract cannot be completed.
               If False, return model with available metadata (best-effort)

    Returns:
        Model dict with Tier 1+2 metadata guaranteed to be present
        (or as complete as fallbacks allow for old models)

    Raises:
        RuntimeError: If strict=True and contract cannot be completed
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    # Read model structure from H5
    model_dict = read_b3d_h5(str(h5_path))

    # Try to use pre-stored contract first
    stored_contract = _read_contract_from_h5(h5_path)
    if stored_contract is not None:
        # Contract is already persisted; use it directly
        if "metadata" not in model_dict:
            model_dict["metadata"] = {}
        model_dict["metadata"]["geometry_contract"] = stored_contract
        # Still normalize observer metadata
        model_dict = normalize_observer_metadata(model_dict)
        return model_dict

    # Contract not persisted; try to complete/infer it
    contract = complete_geometry_contract(model_dict, strict=strict)
    if contract is not None:
        if "metadata" not in model_dict:
            model_dict["metadata"] = {}
        model_dict["metadata"]["geometry_contract"] = contract

    # Always normalize observer metadata at load time
    model_dict = normalize_observer_metadata(model_dict)

    return model_dict


def load_model_from_sav(
    sav_path: Path | str,
    *,
    strict: bool = False,
    keep_temp_h5: bool = False,
) -> dict[str, Any]:
    """
    Load a model from SAV format with geometry contract enforcement.

    This converts SAV to a temporary H5 and loads via load_model_from_h5,
    ensuring all models from SAV source go through the same contract-enforced
    loader.

    Args:
        sav_path: Path to SAV file
        strict: If True, raise if contract cannot be completed
        keep_temp_h5: If True, return tuple (model_dict, temp_h5_path).
                     If False, delete temp H5 and return just model_dict.

    Returns:
        Model dict with Tier 1+2 metadata, or (model_dict, temp_h5_path)
        if keep_temp_h5=True
    """
    sav_path = Path(sav_path)
    if not sav_path.exists():
        raise FileNotFoundError(f"SAV file not found: {sav_path}")

    # Convert SAV to temporary H5
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        temp_h5_path = Path(tmp.name)

    try:
        build_h5_from_sav(sav_path=sav_path, out_h5=temp_h5_path)
        model_dict = load_model_from_h5(temp_h5_path, strict=strict)

        if keep_temp_h5:
            return model_dict, temp_h5_path
        else:
            temp_h5_path.unlink(missing_ok=True)
            return model_dict
    except Exception:
        temp_h5_path.unlink(missing_ok=True)
        raise


def save_model_to_h5(
    model_dict: dict[str, Any],
    h5_path: Path | str,
) -> None:
    """
    Save a model to HDF5 with geometry contract persistence.

    If the model has a completed geometry_contract in metadata,
    this function persists it to HDF5 so it will be reused on next load
    without recomputation.

    Args:
        model_dict: Model dictionary to save
        h5_path: Path to write HDF5 file
    """
    h5_path = Path(h5_path)

    # Write model using standard writer
    write_b3d_h5(str(h5_path), model_dict)

    # Persist contract if present
    metadata = model_dict.get("metadata")
    if isinstance(metadata, dict):
        contract = metadata.get("geometry_contract")
        if isinstance(contract, GeometryContract):
            _write_contract_to_h5(h5_path, contract)


def complete_and_persist_contract_in_h5(
    h5_path: Path | str,
    *,
    strict: bool = False,
) -> bool:
    """
    Complete geometry contract for an existing H5 file and persist it.

    This function reads an existing H5 file, attempts to complete its
    geometry contract from available metadata, and writes the result
    back to the same file.

    Args:
        h5_path: Path to existing HDF5 file to upgrade
        strict: If True, raise if contract cannot be completed.
               If False, silently return False if incomplete.

    Returns:
        True if contract was successfully completed and persisted,
        False otherwise.

    Raises:
        RuntimeError: If strict=True and contract cannot be completed
    """
    h5_path = Path(h5_path)

    # Read model from H5
    model_dict = read_b3d_h5(str(h5_path))

    # Check if contract is already stored
    if _read_contract_from_h5(h5_path) is not None:
        return True  # Already persisted

    # Try to complete the contract
    contract = complete_geometry_contract(model_dict, strict=strict)
    if contract is None:
        if strict:
            raise RuntimeError(f"Geometry contract is incomplete for model: {h5_path}")
        return False

    # Persist the contract
    return _write_contract_to_h5(h5_path, contract)


__all__ = [
    "load_model_from_h5",
    "load_model_from_sav",
    "save_model_to_h5",
    "complete_and_persist_contract_in_h5",
]
