from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits
from scipy.io import readsav

from pyampp.gxbox.boxutils import extract_sav_refmaps, serialize_sav_index_header
from pyampp.gxbox.gx_box2id import gx_box2id
from pyampp.geometry.contract import complete_geometry_contract

def _apply_geometry_contract_to_h5(h5_path: Path) -> None:
    """
    Complete and store geometry contract in an existing HDF5 file.
    
    Reads model groups from H5, computes GeometryContract, and stores it
    in metadata/geometry_contract. This ensures that downstream consumers
    have complete Tier 1+2 geometry without fallback branching.
    """
    try:
        from pyampp.gxbox.boxutils import read_b3d_h5
    except ImportError:
        # If boxutils not available, skip contract completion (non-fatal)
        return
    
    try:
        model_dict = read_b3d_h5(str(h5_path))
    except Exception:
        # Skip if model cannot be read
        return
    
    try:
        contract = complete_geometry_contract(model_dict, strict=False)
    except Exception:
        # Skip if contract completion fails
        return
    
    if contract is None:
        # No complete Tier 1 metadata; skip
        return
    
    try:
        with h5py.File(h5_path, "r+") as f:
            g_meta = _ensure_group(f, "metadata")
            g_contract = _ensure_group(g_meta, "geometry_contract")
            
            contract_dict = contract.to_dict()
            for key, value in contract_dict.items():
                if isinstance(value, str):
                    value = np.bytes_(value)
                elif isinstance(value, (int, float)):
                    value = np.array(value)
                _replace_dataset(g_contract, key, value)
    except Exception:
        # Non-fatal: if contract storage fails, model is still usable
        pass


def _decode_if_bytes(v: Any) -> Any:
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", "ignore")
    return v


def _as_scalar(v: Any) -> Any:
    arr = np.asarray(v)
    if arr.shape == ():
        return arr.item()
    return arr.flat[0]


def _as_text(v: Any) -> str:
    v = _decode_if_bytes(v)
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return str(v.item())
        if v.size > 0:
            return str(_decode_if_bytes(v.flat[0]))
        return ""
    return str(v)


def _has_field(box: Any, name: str) -> bool:
    return name in (box.dtype.names or ())


def _field(box: Any, name: str, default=None):
    return box[name] if _has_field(box, name) else default


def _load_box(sav_path: Path):
    data = readsav(str(sav_path), verbose=False)
    if "box" not in data:
        raise ValueError(f"Input SAV does not contain 'box': {sav_path}")
    return data["box"].flat[0]


def _ensure_group(f: h5py.File | h5py.Group, name: str):
    if name in f:
        return f[name]
    return f.create_group(name)


def _replace_dataset(group: h5py.Group, key: str, value: Any) -> None:
    if key in group:
        del group[key]
    group.create_dataset(key, data=value)


def _coerce_h5_value(value: Any):
    arr = np.asarray(value)
    if arr.dtype.names is not None:
        raise TypeError("structured dtype")
    if arr.dtype.kind == "O":
        raise TypeError("object dtype")
    if arr.dtype.kind == "U":
        arr = arr.astype("S")
    if arr.shape == ():
        return arr.item()
    return arr


def _write_struct_fields(group: h5py.Group, record: Any, skipped: list[str], prefix: str) -> None:
    names = getattr(getattr(record, "dtype", None), "names", None) or ()
    for name in names:
        value = record[name]
        path = f"{prefix}/{name}"
        try:
            if isinstance(value, np.ndarray) and value.dtype.names is not None:
                if value.size == 0:
                    skipped.append(path)
                    continue
                sub = _ensure_group(group, name)
                _write_struct_fields(sub, value.flat[0], skipped, path)
                continue
            if hasattr(value, "dtype") and getattr(value.dtype, "names", None):
                sub = _ensure_group(group, name)
                _write_struct_fields(sub, value, skipped, path)
                continue
            _replace_dataset(group, name, _coerce_h5_value(value))
        except Exception:
            skipped.append(path)


def _write_raw_sav_dump(f: h5py.File, box: Any, base: Any, index: Any) -> None:
    """
    Preserve all convertible SAV fields (no cherry-picking) under /raw_sav.
    """
    skipped: list[str] = []
    g_raw = _ensure_group(f, "raw_sav")
    g_box = _ensure_group(g_raw, "box")

    for name in (box.dtype.names or ()):
        value = box[name]
        path = f"box/{name}"
        try:
            if isinstance(value, np.ndarray) and value.dtype.names is not None:
                if value.size == 0:
                    skipped.append(path)
                    continue
                sub = _ensure_group(g_box, name)
                _write_struct_fields(sub, value.flat[0], skipped, path)
                continue
            _replace_dataset(g_box, name, _coerce_h5_value(value))
        except Exception:
            skipped.append(path)

    if base is not None:
        g_base_raw = _ensure_group(g_raw, "base")
        _write_struct_fields(g_base_raw, base, skipped, "base")
    if index is not None:
        g_index_raw = _ensure_group(g_raw, "index")
        _write_struct_fields(g_index_raw, index, skipped, "index")

    if skipped:
        g_meta = _ensure_group(f, "metadata")
        _replace_dataset(g_meta, "raw_sav_skipped", np.bytes_(json.dumps(sorted(set(skipped)))))


def _to_line_flat(arr: Any, dtype) -> np.ndarray:
    a = np.asarray(arr, dtype=dtype)
    if a.ndim == 1:
        return a
    return a.T.reshape(-1, order="F")


def _normalize_czyx_from_components_or_bcube(box: Any) -> np.ndarray | None:
    if _has_field(box, "BCUBE"):
        b = np.asarray(box["BCUBE"], dtype=np.float32)
        if b.ndim == 4 and b.shape[0] == 3:
            return b
        if b.ndim == 4 and b.shape[-1] == 3:
            return np.moveaxis(b, -1, 0)
    if _has_field(box, "BX") and _has_field(box, "BY") and _has_field(box, "BZ"):
        bx = np.asarray(box["BX"], dtype=np.float32)
        by = np.asarray(box["BY"], dtype=np.float32)
        bz = np.asarray(box["BZ"], dtype=np.float32)
        if bx.ndim == 3 and by.shape == bx.shape and bz.shape == bx.shape:
            return np.stack([bx, by, bz], axis=0)
    return None


def _infer_stage(box: Any) -> str:
    has_chromo = _has_field(box, "CHROMO_BCUBE") and _has_field(box, "CHROMO_IDX")
    has_lines = _has_field(box, "STARTIDX") and _has_field(box, "ENDIDX") and _has_field(box, "AVFIELD")
    has_corona = _normalize_czyx_from_components_or_bcube(box) is not None
    if has_chromo:
        return "CHR"
    if has_lines:
        return "GEN"
    if has_corona:
        model_id = _as_text(_field(box, "ID", "")).upper()
        if ".POT" in model_id:
            return "POT"
        if ".BND" in model_id:
            return "BND"
        return "NAS"
    return "NONE"


def _build_refmap_wcs_header(
    data2d: np.ndarray,
    *,
    xc: float,
    yc: float,
    dx: float,
    dy: float,
    date_obs: str,
    xunits: str,
    yunits: str,
    rsun_obs: float | None = None,
    b0: float | None = None,
    l0: float | None = None,
) -> str:
    ny, nx = data2d.shape
    h = fits.Header()
    h["SIMPLE"] = True
    h["BITPIX"] = -32
    h["NAXIS"] = 2
    h["NAXIS1"] = int(nx)
    h["NAXIS2"] = int(ny)
    h["CTYPE1"] = "HPLN-TAN"
    h["CTYPE2"] = "HPLT-TAN"
    h["CUNIT1"] = xunits if xunits else "arcsec"
    h["CUNIT2"] = yunits if yunits else "arcsec"
    h["CRPIX1"] = (nx + 1.0) / 2.0
    h["CRPIX2"] = (ny + 1.0) / 2.0
    h["CRVAL1"] = float(xc)
    h["CRVAL2"] = float(yc)
    h["CDELT1"] = float(dx)
    h["CDELT2"] = float(dy)
    if date_obs:
        h["DATE-OBS"] = date_obs
    if rsun_obs is not None:
        h["RSUN_OBS"] = float(rsun_obs)
    if b0 is not None:
        h["HGLT_OBS"] = float(b0)
    if l0 is not None:
        h["HGLN_OBS"] = float(l0)
    return h.tostring(sep="\n", endcard=True)


def _extract_refmaps_from_box(box: Any) -> list[tuple[int, str, np.ndarray, str]]:
    out: list[tuple[int, str, np.ndarray, str]] = []
    if not _has_field(box, "REFMAPS"):
        return out

    try:
        refmaps_obj = box["REFMAPS"][0]
        omap = refmaps_obj["OMAP"][0]
        pointer = omap["POINTER"][0]
        ids = np.asarray(pointer["IDS"], dtype=object).ravel()
        ptrs = np.asarray(pointer["PTRS"], dtype=object).ravel()
    except Exception:
        return out

    used_names: dict[str, int] = {}
    order_index = 0
    for slot_id, entry in zip(ids, ptrs):
        if entry is None:
            continue
        slot_text = _as_text(slot_id).strip()
        if not slot_text:
            continue
        try:
            rec = np.asarray(entry).reshape(-1)[0]
            names = rec.dtype.names or ()
        except Exception:
            continue
        if "DATA" not in names:
            continue

        data = np.asarray(rec["DATA"], dtype=np.float32)
        if data.shape and data.shape[0] == 1:
            data = np.asarray(data[0], dtype=np.float32)
        if data.ndim != 2:
            continue

        map_id = _as_text(rec["ID"]) if "ID" in names else slot_text
        map_id = map_id.strip() if map_id else slot_text
        if not map_id:
            map_id = f"refmap_{slot_text}"

        base_name = map_id
        idx = used_names.get(base_name, 0)
        used_names[base_name] = idx + 1
        if idx > 0:
            map_id = f"{base_name}_{idx}"

        xc = float(_as_scalar(rec["XC"])) if "XC" in names else 0.0
        yc = float(_as_scalar(rec["YC"])) if "YC" in names else 0.0
        dx = float(_as_scalar(rec["DX"])) if "DX" in names else 1.0
        dy = float(_as_scalar(rec["DY"])) if "DY" in names else 1.0
        date_obs = _as_text(rec["TIME"]) if "TIME" in names else ""
        xunits = _as_text(rec["XUNITS"]).strip() if "XUNITS" in names else "arcsec"
        yunits = _as_text(rec["YUNITS"]).strip() if "YUNITS" in names else "arcsec"
        rsun_obs = float(_as_scalar(rec["RSUN"])) if "RSUN" in names else None
        b0 = float(_as_scalar(rec["B0"])) if "B0" in names else None
        l0 = float(_as_scalar(rec["L0"])) if "L0" in names else None

        wcs_header = _build_refmap_wcs_header(
            data,
            xc=xc,
            yc=yc,
            dx=dx,
            dy=dy,
            date_obs=date_obs,
            xunits=xunits,
            yunits=yunits,
            rsun_obs=rsun_obs,
            b0=b0,
            l0=l0,
        )
        out.append((order_index, map_id, data, wcs_header))
        order_index += 1
    return out


def build_h5_from_sav(sav_path: Path, out_h5: Path, template_h5: Path | None = None) -> Path:
    """Convert a GX SAV model (any stage) into canonical HDF5 groups."""
    sav_path = Path(sav_path).expanduser().resolve()
    out_h5 = Path(out_h5).expanduser().resolve()
    template_h5 = Path(template_h5).expanduser().resolve() if template_h5 else None

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if template_h5 is not None:
        shutil.copy2(template_h5, out_h5)

    box = _load_box(sav_path)
    stage = _infer_stage(box)
    index = box["INDEX"][0] if _has_field(box, "INDEX") else None

    base = box["BASE"][0] if _has_field(box, "BASE") else None
    refmaps = extract_sav_refmaps(box)
    dr = np.asarray(_field(box, "DR", [1.0, 1.0, 1.0]), dtype=np.float64)

    bcube_czyx = _normalize_czyx_from_components_or_bcube(box)
    bcube_xyzc = None
    if bcube_czyx is not None:
        bcube_xyzc = bcube_czyx.transpose((3, 2, 1, 0))

    has_lines = _has_field(box, "STARTIDX") and _has_field(box, "ENDIDX") and _has_field(box, "AVFIELD")
    has_chromo = _has_field(box, "CHROMO_BCUBE") and _has_field(box, "CHROMO_IDX")

    with h5py.File(out_h5, "w") as f:
        g_base = _ensure_group(f, "base")
        g_refmaps = _ensure_group(f, "refmaps")
        g_grid = _ensure_group(f, "grid")
        g_meta = _ensure_group(f, "metadata")

        if bcube_czyx is not None:
            g_corona = _ensure_group(f, "corona")
            _replace_dataset(g_corona, "bx", bcube_czyx[0, :, :, :])
            _replace_dataset(g_corona, "by", bcube_czyx[1, :, :, :])
            _replace_dataset(g_corona, "bz", bcube_czyx[2, :, :, :])
            _replace_dataset(g_corona, "dr", dr.astype(np.float64))
            corona_base = int(_field(box, "CORONA_BASE", 0))
            _replace_dataset(g_corona, "corona_base", np.array(corona_base, dtype=np.int64))
            model_type = "pot" if stage == "POT" else ("bnd" if stage == "BND" else "nlfff")
            g_corona.attrs["model_type"] = model_type

        if has_lines:
            g_lines = _ensure_group(f, "lines")
            _replace_dataset(g_lines, "av_field", _to_line_flat(box["AVFIELD"], np.float64))
            if _has_field(box, "PHYSLENGTH"):
                _replace_dataset(g_lines, "phys_length", _to_line_flat(box["PHYSLENGTH"], np.float64))
            if _has_field(box, "STATUS"):
                _replace_dataset(g_lines, "voxel_status", _to_line_flat(box["STATUS"], np.uint8))
            _replace_dataset(g_lines, "start_idx", _to_line_flat(box["STARTIDX"], np.int64))
            _replace_dataset(g_lines, "end_idx", _to_line_flat(box["ENDIDX"], np.int64))
            _replace_dataset(g_lines, "dr", dr.astype(np.float64))

        if has_chromo:
            g_chromo = _ensure_group(f, "chromo")
            chromo_bcube = np.asarray(box["CHROMO_BCUBE"], dtype=np.float32)
            if chromo_bcube.ndim == 4 and chromo_bcube.shape[0] == 3:
                _replace_dataset(g_chromo, "bx", chromo_bcube[0, :, :, :])
                _replace_dataset(g_chromo, "by", chromo_bcube[1, :, :, :])
                _replace_dataset(g_chromo, "bz", chromo_bcube[2, :, :, :])
            elif chromo_bcube.ndim == 4 and chromo_bcube.shape[-1] == 3:
                _replace_dataset(g_chromo, "bx", chromo_bcube[:, :, :, 0])
                _replace_dataset(g_chromo, "by", chromo_bcube[:, :, :, 1])
                _replace_dataset(g_chromo, "bz", chromo_bcube[:, :, :, 2])

            if _has_field(box, "DZ"):
                _replace_dataset(g_chromo, "dz", np.asarray(box["DZ"], dtype=np.float64))
            for src, dst, dt in (
                ("CHROMO_IDX", "chromo_idx", np.int64),
                ("CHROMO_N", "chromo_n", np.float32),
                ("CHROMO_T", "chromo_t", np.float32),
                ("N_P", "n_p", np.float32),
                ("N_HI", "n_hi", np.float32),
                ("N_HTOT", "n_htot", np.float32),
                ("TR", "tr", np.int64),
                ("TR_H", "tr_h", np.float64),
            ):
                if _has_field(box, src):
                    _replace_dataset(g_chromo, dst, np.asarray(box[src], dtype=dt))
            if _has_field(box, "CHROMO_LAYERS"):
                _replace_dataset(g_chromo, "chromo_layers", np.array(int(_as_scalar(box["CHROMO_LAYERS"])), dtype=np.int64))
            if base is not None and "CHROMO_MASK" in (base.dtype.names or ()):
                _replace_dataset(g_chromo, "chromo_mask", np.asarray(base["CHROMO_MASK"], dtype=np.int32))

            if index is not None and "CRVAL1" in index.dtype.names and "CRVAL2" in index.dtype.names and "DSUN_OBS" in index.dtype.names:
                g_chromo.attrs["lon"] = float(_as_scalar(index["CRVAL1"]))
                g_chromo.attrs["lat"] = float(_as_scalar(index["CRVAL2"]))
                g_chromo.attrs["dsun_obs"] = float(_as_scalar(index["DSUN_OBS"]))
                if "DATE_OBS" in index.dtype.names:
                    g_chromo.attrs["obs_time"] = str(_decode_if_bytes(_as_scalar(index["DATE_OBS"])))
                if "HGLN_OBS" in index.dtype.names:
                    g_chromo.attrs["lonC"] = float(_as_scalar(index["HGLN_OBS"]))

        if base is not None:
            if "BX" in base.dtype.names:
                _replace_dataset(g_base, "bx", np.asarray(base["BX"], dtype=np.float64))
            if "BY" in base.dtype.names:
                _replace_dataset(g_base, "by", np.asarray(base["BY"], dtype=np.float64))
            if "BZ" in base.dtype.names:
                _replace_dataset(g_base, "bz", np.asarray(base["BZ"], dtype=np.float64))
            if "IC" in base.dtype.names:
                _replace_dataset(g_base, "ic", np.asarray(base["IC"], dtype=np.float64))
            if "CHROMO_MASK" in base.dtype.names:
                _replace_dataset(g_base, "chromo_mask", np.asarray(base["CHROMO_MASK"], dtype=np.int32))
            if index is not None:
                _replace_dataset(g_base, "index", np.bytes_(serialize_sav_index_header(index)))

        if dr.size >= 2:
            _replace_dataset(g_grid, "dx", np.float64(dr[0]))
            _replace_dataset(g_grid, "dy", np.float64(dr[1]))
        if _has_field(box, "DZ"):
            _replace_dataset(g_grid, "dz", np.asarray(box["DZ"], dtype=np.float64))

        voxel_id_xyz = None
        if bcube_xyzc is not None:
            id_input = {
                "bcube": bcube_xyzc,
                "dr": dr,
                "corona_base": int(_field(box, "CORONA_BASE", 0)),
            }
            if _has_field(box, "STARTIDX"):
                id_input["start_idx"] = _to_line_flat(box["STARTIDX"], np.int64)
            if _has_field(box, "CHROMO_IDX"):
                id_input["chromo_idx"] = np.asarray(box["CHROMO_IDX"], dtype=np.int64)
            if _has_field(box, "CHROMO_T"):
                id_input["chromo_t"] = np.asarray(box["CHROMO_T"], dtype=np.float32)
            if _has_field(box, "CHROMO_N"):
                id_input["chromo_n"] = np.asarray(box["CHROMO_N"], dtype=np.float32)
            if _has_field(box, "CHROMO_LAYERS"):
                id_input["chromo_layers"] = int(_as_scalar(box["CHROMO_LAYERS"]))
            voxel_id_xyz = gx_box2id(id_input)
        if voxel_id_xyz is not None:
            _replace_dataset(g_grid, "voxel_id", np.asarray(voxel_id_xyz, dtype=np.uint32).transpose((2, 1, 0)))

        execute = _decode_if_bytes(_field(box, "EXECUTE", ""))
        model_id = _decode_if_bytes(_field(box, "ID", out_h5.stem))
        _replace_dataset(g_meta, "execute", np.bytes_(str(execute)))
        _replace_dataset(g_meta, "id", np.bytes_(str(model_id)))
        _replace_dataset(g_meta, "disambiguation", np.bytes_("IDL"))
        _replace_dataset(g_meta, "projection", np.bytes_("CEA"))
        _replace_dataset(g_meta, "axis_order_2d", np.bytes_("yx"))
        _replace_dataset(g_meta, "axis_order_3d", np.bytes_("zyx"))
        _replace_dataset(g_meta, "vector_layout", np.bytes_("split_components"))
        _replace_dataset(g_meta, "stage", np.bytes_(stage))

        for order_index, map_id, map_data, map_header in refmaps:
            map_group = _ensure_group(g_refmaps, map_id)
            _replace_dataset(map_group, "data", np.asarray(map_data))
            _replace_dataset(map_group, "wcs_header", np.bytes_(map_header))
            map_group.attrs["order_index"] = np.int64(order_index)

        # Complete and store geometry contract for downstream use
        _apply_geometry_contract_to_h5(out_h5)

    return out_h5


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert GX SAV model (any stage) to canonical HDF5.")
    p.add_argument("--sav-path", type=Path, required=False, help="Path to input SAV file.")
    p.add_argument("--out-h5", type=Path, required=False, help="Output HDF5 path.")
    p.add_argument(
        "--template-h5",
        type=Path,
        default=None,
        help="Optional template HDF5 to clone before writing model groups.",
    )
    p.add_argument("--sav", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--template", type=Path, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.sav_path is None and args.sav is not None:
        args.sav_path = args.sav
    if args.out_h5 is None and args.out is not None:
        args.out_h5 = args.out
    if args.template_h5 is None and args.template is not None:
        args.template_h5 = args.template
    if args.sav_path is None or args.out_h5 is None:
        p.error("--sav-path and --out-h5 are required")
    return args


def main() -> None:
    args = _parse_args()
    out_h5 = build_h5_from_sav(args.sav_path, args.out_h5, template_h5=args.template_h5)
    print(f"Wrote: {out_h5}")


__all__ = ["build_h5_from_sav", "main"]


if __name__ == "__main__":
    main()
