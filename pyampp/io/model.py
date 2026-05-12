from __future__ import annotations

"""
Canonical pyAMPP Model I/O: Provenance-Agnostic Loader/Writer Contract

This module enforces the canonical, provenance-agnostic contract for all pyAMPP model I/O:

1. All supported input formats (SAV, old H5, new H5) are normalized by the loader to a single canonical in-memory structure.
2. The loader injects or upgrades all required geometry_contract and related metadata if missing or outdated.
3. The writer serializes the in-memory structure as-is, without adding or mutating metadata.
4. The output HDF5 is canonical and idempotent: repeated load/save cycles produce identical files, regardless of provenance.
5. The CLI (e.g., clone_sav.py) is a thin wrapper around this process.

This guarantees that pyAMPP model data is provenance-agnostic and round-trip idempotent, and that all downstream consumers see a single, canonical data structure.
"""

import tempfile
from pathlib import Path
from typing import Any, Literal, overload

import h5py
import numpy as np

from pyampp.geometry.contract import (
    GeometryContract,
    complete_geometry_contract,
)

def ensure_geometry_contract_in_metadata(model_dict: dict, strict: bool = False) -> None:
    """
    Ensure model_dict["metadata"]["geometry_contract"] is present and up-to-date.
    If missing or outdated, computes and injects a new contract.
    """
    if not isinstance(model_dict, dict):
        return
    metadata = model_dict.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        model_dict["metadata"] = metadata
    contract = metadata.get("geometry_contract")
    # If already a valid GeometryContract, nothing to do
    if isinstance(contract, GeometryContract):
        return
    # If present as dict, try to upgrade
    if isinstance(contract, dict):
        try:
            metadata["geometry_contract"] = GeometryContract.from_dict(contract)
            return
        except Exception:
            pass
    # Otherwise, compute and inject
    new_contract = complete_geometry_contract(model_dict, strict=strict)
    if new_contract is not None:
        metadata["geometry_contract"] = new_contract
        model_dict["metadata"] = metadata
from pyampp.gxbox.boxutils import (
    normalize_observer_metadata,
)
from pyampp.io._sav_convert import build_h5_from_sav


_CANONICAL_AXIS_ORDER_2D = "yx"
_CANONICAL_AXIS_ORDER_3D = "zyx"
_CANONICAL_VECTOR_LAYOUT = "split_components"


def _missing_metadata_text(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _legacy_lineage_placeholder(source_kind: str) -> str:
    if source_kind == "sav":
        return "legacy-sav:unknown"
    return "legacy-h5:unknown"


def _backfill_canonical_metadata(
    model_dict: dict[str, Any],
    *,
    source_path: Path,
    source_kind: str,
) -> dict[str, Any]:
    metadata = model_dict.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        model_dict["metadata"] = metadata

    if _missing_metadata_text(metadata.get("id")):
        metadata["id"] = source_path.stem
    if _missing_metadata_text(metadata.get("axis_order_2d")):
        metadata["axis_order_2d"] = _CANONICAL_AXIS_ORDER_2D
    if _missing_metadata_text(metadata.get("axis_order_3d")):
        metadata["axis_order_3d"] = _CANONICAL_AXIS_ORDER_3D
    if _missing_metadata_text(metadata.get("vector_layout")):
        metadata["vector_layout"] = _CANONICAL_VECTOR_LAYOUT
    if _missing_metadata_text(metadata.get("lineage")):
        metadata["lineage"] = _legacy_lineage_placeholder(source_kind)

    return model_dict


@overload
def load_model(
    filename: Path | str,
    *,
    strict: bool = False,
    keep_temp_h5: Literal[False] = False,
) -> dict[str, Any]:
    ...


@overload
def load_model(
    filename: Path | str,
    *,
    strict: bool = False,
    keep_temp_h5: Literal[True],
) -> tuple[dict[str, Any], Path | None]:
    ...


def load_model(
    filename: Path | str,
    *,
    strict: bool = False,
    keep_temp_h5: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Path | None]:
    """Load a pyAMPP model through the canonical provenance-agnostic boundary."""
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix == ".h5":
        model = _load_model_h5(path, strict=strict)
        if keep_temp_h5:
            return model, None
        return model
    if suffix == ".sav":
        return _load_model_sav(path, strict=strict, keep_temp_h5=keep_temp_h5)
    raise ValueError(f"Unsupported model format for {path}; expected a canonical pyAMPP .h5 or legacy .sav file")

def _normalize_model_dict(obj):
    """
    Recursively decode bytes/arrays to strings for all dict/list fields.
    Ensures base/index and metadata fields are stringified for legacy H5/SAV.
    """
    import numpy as np
    if isinstance(obj, dict):
        return {k: _normalize_model_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_normalize_model_dict(v) for v in obj)
    if isinstance(obj, (bytes, np.bytes_)):
        return obj.decode("utf-8", "ignore")
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("S", "a") and obj.size == 1:
            return obj.astype(str).item()
        if obj.dtype.kind in ("S", "a"):
            return obj.astype(str).tolist()
        if obj.shape == ():
            return obj.item()
        return obj.tolist()
    return obj

def _prepare_model_for_h5_write(model_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-copied payload safe for generic HDF5 writers."""
    payload = dict(model_dict)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return payload

    metadata_copy = dict(metadata)
    contract = metadata_copy.get("geometry_contract")
    if isinstance(contract, GeometryContract):
        metadata_copy["geometry_contract"] = contract.to_dict()
    payload["metadata"] = metadata_copy
    return payload


def _read_b3d_h5_raw(filename: Path | str) -> dict[str, Any]:
    """Low-level HDF5 reader used internally by io.model."""

    def _read_h5_node(node):
        if isinstance(node, h5py.Group):
            out = {}
            for key in node.keys():
                out[key] = _read_h5_node(node[key])
            return out
        if node.shape == ():
            return node[()]
        return node[:]

    box_b3d: dict[str, Any] = {}
    with h5py.File(filename, "r") as hdf_file:
        for model_type in hdf_file.keys():
            group = hdf_file[model_type]
            target_type = model_type
            model_attr = None
            if model_type in ("nlfff", "pot", "potential", "bounds"):
                target_type = "corona"
                if model_type == "potential":
                    model_attr = "pot"
                elif model_type in ("bounds",):
                    model_attr = "bnd"
                else:
                    model_attr = model_type
            if target_type not in box_b3d:
                box_b3d[target_type] = {}
            component_names = list(group.keys())
            if target_type == "refmaps":

                def _refmap_sort_key(name: str):
                    obj = group[name]
                    if isinstance(obj, h5py.Group) and "order_index" in obj.attrs:
                        return (0, int(obj.attrs["order_index"]))
                    return (1, name)

                component_names = sorted(component_names, key=_refmap_sort_key)
            for component in component_names:
                ds = group[component]
                box_b3d[target_type][component] = _read_h5_node(ds)
            if len(group.attrs.keys()) > 0 or model_attr is not None:
                attrs = dict(group.attrs)
                if model_attr is not None and "model_type" not in attrs:
                    attrs["model_type"] = model_attr
                if "attrs" in box_b3d[target_type]:
                    box_b3d[target_type]["attrs"].update(attrs)
                else:
                    box_b3d[target_type]["attrs"] = attrs
    return box_b3d


def _write_b3d_h5_raw(filename: Path | str, box_b3d: dict[str, Any]) -> None:
    """Low-level HDF5 writer used internally by io.model."""

    def _encode_dataset_value(value):
        if isinstance(value, (str, bytes, np.bytes_)):
            return np.bytes_(value)
        if value is None:
            return np.bytes_("")

        arr = np.asarray(value)
        if arr.dtype.kind != "O":
            return value

        if arr.shape == ():
            scalar = arr.item()
            if isinstance(scalar, (str, bytes, np.bytes_)):
                return np.bytes_(scalar)
            if scalar is None:
                return np.bytes_("")
            try:
                return np.asarray(scalar)
            except Exception:
                return np.bytes_(str(scalar))

        as_text = np.vectorize(lambda x: "" if x is None else str(x), otypes=[str])(arr)
        return as_text.astype("S")

    def _set_group_attrs(group, attrs):
        if not isinstance(attrs, dict):
            return
        for key, value in attrs.items():
            group.attrs[key] = _encode_dataset_value(value)

    def _write_node(group, key, value, *, target_type=None):
        if isinstance(value, dict):
            sub = group.create_group(key)
            attrs = value.get("attrs", {}) if isinstance(value.get("attrs"), dict) else {}
            for sub_key, sub_val in value.items():
                if sub_key == "attrs":
                    continue
                _write_node(sub, sub_key, sub_val, target_type=target_type)
            _set_group_attrs(sub, attrs)
            return

        if target_type in ("chromo", "lines") and key == "voxel_status":
            group.create_dataset(key, data=np.asarray(value, dtype=np.uint8))
            return
        group.create_dataset(key, data=_encode_dataset_value(value))

    with h5py.File(filename, "w") as hdf_file:
        for model_type, components in box_b3d.items():
            if components is None:
                continue
            if model_type == "metadata":
                group = hdf_file.create_group("metadata")
                for key, value in components.items():
                    _write_node(group, key, value, target_type="metadata")
                continue
            target_type = model_type
            attrs = components.get("attrs", {}) if isinstance(components, dict) else {}
            if model_type in ("nlfff", "pot", "potential", "bounds"):
                target_type = "corona"
                if "model_type" not in attrs:
                    attrs = dict(attrs)
                    if model_type == "potential":
                        attrs["model_type"] = "pot"
                    elif model_type == "bounds":
                        attrs["model_type"] = "bnd"
                    else:
                        attrs["model_type"] = model_type
            if target_type in hdf_file:
                continue
            if target_type == "refmaps":
                group = hdf_file.create_group(target_type, track_order=True)
            else:
                group = hdf_file.create_group(target_type)
            refmap_idx = 0
            for component, data in components.items():
                if component == "attrs":
                    continue
                if target_type == "refmaps" and isinstance(data, dict):
                    sub = group.create_group(component, track_order=True)
                    sub.attrs["order_index"] = np.int64(refmap_idx)
                    refmap_idx += 1
                    sub_attrs = data.get("attrs", {}) if isinstance(data.get("attrs"), dict) else {}
                    for sub_key, sub_val in data.items():
                        if sub_key == "attrs":
                            continue
                        _write_node(sub, sub_key, sub_val, target_type=target_type)
                    _set_group_attrs(sub, sub_attrs)
                else:
                    _write_node(group, component, data, target_type=target_type)
            if attrs:
                _set_group_attrs(group, attrs)


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


def _decode_h5_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "ignore")
    return value


def _read_h5_node(node) -> Any:
    if isinstance(node, h5py.Group):
        out: dict[str, Any] = {}
        for key in node.keys():
            out[key] = _read_h5_node(node[key])
        if len(node.attrs.keys()) > 0:
            attrs = {}
            for key, val in node.attrs.items():
                attrs[key] = _decode_h5_scalar(val)
            out["attrs"] = attrs
        return out

    if node.shape == ():
        return _decode_h5_scalar(node[()])

    arr = node[:]
    if isinstance(arr, np.ndarray) and arr.dtype.kind in ("S", "a"):
        return arr.astype(str)
    return arr


def _load_geometry_contract_and_observer_from_h5(
    h5_path: Path | str,
) -> dict[str, Any] | None:
    """
        Read full metadata plus optional observer sections from an HDF5 model file.

    Returns None when metadata/geometry_contract is missing.
        When geometry_contract exists, returns a thin model dictionary:
      {
                "metadata": {
                        ... full metadata ...,
                        "geometry_contract": GeometryContract,
                },
        "observer": {...}  # only if present in the file
      }
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "metadata" not in f or "geometry_contract" not in f["metadata"]:
            return None

        metadata = _read_h5_node(f["metadata"])
        if not isinstance(metadata, dict):
            return None
        contract_raw = metadata.get("geometry_contract")
        if not isinstance(contract_raw, dict):
            return None
        metadata["geometry_contract"] = GeometryContract.from_dict(contract_raw)

        thin_model: dict[str, Any] = {"metadata": metadata}

        if "observer" in f:
            thin_model["observer"] = _read_h5_node(f["observer"])

    return thin_model


def save_thin_model(
    thin_model: dict[str, Any],
    h5_path: Path | str,
) -> None:
    """
    Write a thin model HDF5 containing only metadata and optional observer.

    Required input:
      - thin_model["metadata"]["geometry_contract"] present as either
        GeometryContract or a dict-like contract payload.
    Optional input:
      - thin_model["observer"]
    """
    if not isinstance(thin_model, dict):
        raise TypeError("thin_model must be a dict")

    metadata = thin_model.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("thin_model must contain a metadata dict")
    if "geometry_contract" not in metadata:
        raise ValueError("thin_model.metadata must contain geometry_contract")

    metadata_copy = dict(metadata)
    contract = metadata_copy.get("geometry_contract")
    if isinstance(contract, GeometryContract):
        metadata_copy["geometry_contract"] = contract.to_dict()
    elif isinstance(contract, dict):
        # Validate schema early so malformed contract payloads fail explicitly.
        GeometryContract.from_dict(contract)
    else:
        raise TypeError("geometry_contract must be GeometryContract or dict")

    payload: dict[str, Any] = {"metadata": metadata_copy}
    observer = thin_model.get("observer")
    if observer is not None:
        if not isinstance(observer, dict):
            raise TypeError("observer must be a dict when provided")
        payload["observer"] = observer

    write_payload = _prepare_model_for_h5_write(payload)
    _write_b3d_h5_raw(str(h5_path), write_payload)


def export_thin_model(
    source_model: Path | str,
    output_h5: Path | str | None = None,
    *,
    strict: bool = False,
) -> Path:
    """
    Generate a metadata-only thin model HDF5 from any supported full model input.

    The output file contains only:
      - metadata (full metadata section)
      - observer (if present)

    Args:
        source_model: Path to source full model (.h5 or .sav)
        output_h5: Destination path. If omitted, writes sibling
                  ``<source_stem>_metadata.h5`` next to source.
        strict: Passed to ``load_model`` for contract completion.

    Returns:
        Path to written thin HDF5 file.
    """
    source_model = Path(source_model)
    if output_h5 is None:
        output_h5 = source_model.with_name(f"{source_model.stem}_metadata.h5")
    output_h5 = Path(output_h5)

    model = load_model(source_model, strict=strict)
    metadata = model.get("metadata") if isinstance(model, dict) else None
    if not isinstance(metadata, dict) or "geometry_contract" not in metadata:
        raise RuntimeError("Source model has no geometry_contract after restore.")

    thin_model: dict[str, Any] = {"metadata": metadata}
    observer = model.get("observer") if isinstance(model, dict) else None
    if isinstance(observer, dict):
        thin_model["observer"] = observer

    save_thin_model(thin_model, output_h5)
    return output_h5


def load_model_metadata(
    model_path: Path | str,
    *,
    strict: bool = False,
) -> dict[str, Any] | None:
    """
    Load canonical metadata plus optional observer from any supported model file.

    Returns None when the restored model has no metadata dictionary or when the
    completed metadata still lacks a geometry contract.
    """
    model = load_model(model_path, strict=strict)
    if not isinstance(model, dict):
        return None

    metadata = model.get("metadata")
    if not isinstance(metadata, dict) or "geometry_contract" not in metadata:
        return None

    thin_model: dict[str, Any] = {"metadata": metadata}
    observer = model.get("observer")
    if isinstance(observer, dict):
        thin_model["observer"] = observer
    return thin_model


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


def _has_model_payload(model_dict: dict[str, Any]) -> bool:
    return any(
        key in model_dict
        for key in ("corona", "chromo", "base", "refmaps", "lines", "potential", "bounds")
    )


def _require_canonical_base_maps(model_dict: dict[str, Any]) -> None:
    if not isinstance(model_dict, dict) or not _has_model_payload(model_dict):
        return

    base = model_dict.get("base")
    if not isinstance(base, dict):
        raise RuntimeError("Canonical full models must include a base group with index and bx/by/bz/ic maps.")

    if not any(base.get(key) is not None for key in ("index", "index_header", "wcs_header")):
        raise RuntimeError("Canonical full models must include base index metadata.")

    missing = [key for key in ("bx", "by", "bz", "ic") if base.get(key) is None]
    if missing:
        raise RuntimeError(
            "Canonical full models must include base LOS maps: " + ", ".join(missing)
        )


def _load_model_h5(
    h5_path: Path | str,
    *,
    strict: bool = False,
    source_path: Path | None = None,
    source_kind: str = "h5",
) -> dict[str, Any]:
    """
    Load a model from HDF5 with geometry contract enforcement.

    Policy:
    - Try to read pre-stored contract from HDF5 (Tier 1+2 metadata)
    - If not present, infer/compute missing Tier 1+2 metadata from available
            fallbacks (base/index, cube shape, dr)
    - Always normalize observer ephemeris at load time
        - Return normalized model dict; contract presence depends on completion
            success and ``strict`` mode

    Args:
        h5_path: Path to HDF5 file
        strict: If True, raise if contract cannot be completed.
               If False, return model with available metadata (best-effort)

    Returns:
        Normalized model dict. When contract completion succeeds,
        ``metadata.geometry_contract`` is attached; when ``strict=False``,
        completion failures are tolerated and the model may be returned without
        that field.

    Raises:
        RuntimeError: If strict=True and contract cannot be completed
    """
    h5_path = Path(h5_path)
    source_path = Path(source_path) if source_path is not None else h5_path
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    # Read model structure from H5
    model_dict = _read_b3d_h5_raw(str(h5_path))

    # Recursively normalize all bytes/arrays to strings for legacy compatibility
    model_dict = _normalize_model_dict(model_dict)


    # Try to use pre-stored contract first
    stored_contract = _read_contract_from_h5(h5_path)
    if stored_contract is not None:
        if "metadata" not in model_dict:
            model_dict["metadata"] = {}
        model_dict["metadata"]["geometry_contract"] = stored_contract
        model_dict = normalize_observer_metadata(model_dict)
        model_dict = _backfill_canonical_metadata(
            model_dict,
            source_path=source_path,
            source_kind=source_kind,
        )
        _require_canonical_base_maps(model_dict)
        return model_dict


    # If corona.dr is missing but bx exists, infer dr as (1.0, 1.0, 1.0) (default)
    corona = model_dict.get("corona")
    if isinstance(corona, dict) and "dr" not in corona:
        for key in ("bx", "by", "bz"):
            if key in corona:
                arr = np.asarray(corona[key])
                if isinstance(arr, np.ndarray) and arr.ndim == 3:
                    corona["dr"] = np.array([1.0, 1.0, 1.0], dtype=np.float64)
                    break

    # Contract not persisted; always attempt to infer from available metadata and cube shapes
    contract = complete_geometry_contract(model_dict, strict=False)
    if contract is not None:
        if "metadata" not in model_dict:
            model_dict["metadata"] = {}
        model_dict["metadata"]["geometry_contract"] = contract
    elif strict:
        raise RuntimeError("Cannot infer geometry_contract from available data.")

    # Always normalize observer metadata at load time
    model_dict = normalize_observer_metadata(model_dict)
    model_dict = _backfill_canonical_metadata(
        model_dict,
        source_path=source_path,
        source_kind=source_kind,
    )
    _require_canonical_base_maps(model_dict)

    return model_dict


@overload
def _load_model_sav(
    sav_path: Path | str,
    *,
    strict: bool = False,
    keep_temp_h5: Literal[False] = False,
) -> dict[str, Any]:
    ...


@overload
def _load_model_sav(
    sav_path: Path | str,
    *,
    strict: bool = False,
    keep_temp_h5: Literal[True],
) -> tuple[dict[str, Any], Path]:
    ...


def _load_model_sav(
    sav_path: Path | str,
    *,
    strict: bool = False,
    keep_temp_h5: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Path]:
    """
    Load a model from SAV format with geometry contract enforcement.

    This converts SAV to a temporary H5 and loads via the canonical H5 reader,
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
        model_dict = _load_model_h5(
            temp_h5_path,
            strict=strict,
            source_path=sav_path,
            source_kind="sav",
        )

        if keep_temp_h5:
            return model_dict, temp_h5_path
        else:
            temp_h5_path.unlink(missing_ok=True)
            return model_dict
    except Exception:
        temp_h5_path.unlink(missing_ok=True)
        raise


def save_model(
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

    metadata = model_dict.get("metadata")
    contract = metadata.get("geometry_contract") if isinstance(metadata, dict) else None

    # Write model using standard writer
    write_payload = _prepare_model_for_h5_write(model_dict)
    _write_b3d_h5_raw(str(h5_path), write_payload)

    # Persist contract if present
    if isinstance(contract, GeometryContract):
        _write_contract_to_h5(h5_path, contract)


def _complete_and_persist_contract_in_h5(
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
    model_dict = _read_b3d_h5_raw(str(h5_path))

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
    "load_model",
    "load_model_metadata",
    "save_model",
    "save_thin_model",
    "export_thin_model",
]
