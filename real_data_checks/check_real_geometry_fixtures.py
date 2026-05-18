from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

from pyampp.geometry.contract import complete_geometry_contract
from pyampp.gxbox.boxutils import read_b3d_h5
from pyampp.io._sav_convert import build_h5_from_sav


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY_MODEL_PATH = (
    WORKSPACE_ROOT
    / "pyGXrender-test-data"
    / "raw"
    / "models"
    / "model_loader_parity_20201126T195831"
    / "hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.CHR.clone.h5"
)
DEFAULT_LEGACY_SAV_PATH = DEFAULT_LEGACY_MODEL_PATH.with_name(
    DEFAULT_LEGACY_MODEL_PATH.name.replace(".clone.h5", ".sav")
)


def _upgrade_dict_with_contract(model: dict) -> dict:
    upgraded = dict(model)
    metadata = upgraded.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    contract = complete_geometry_contract(upgraded, strict=False)
    assert contract is not None
    metadata["geometry_contract"] = contract.to_dict()
    upgraded["metadata"] = metadata
    return upgraded


def _write_geometry_contract_group(dst_h5: Path, contract_dict: dict) -> None:
    with h5py.File(dst_h5, "r+") as h5f:
        metadata = h5f.require_group("metadata")
        if "geometry_contract" in metadata:
            del metadata["geometry_contract"]
        group = metadata.create_group("geometry_contract")
        for key, value in contract_dict.items():
            if isinstance(value, str):
                group.create_dataset(key, data=np.bytes_(value))
            else:
                group.create_dataset(key, data=value)


def validate_legacy_upgrade(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Legacy HDF5 fixture not found: {model_path}")

    with tempfile.TemporaryDirectory(prefix="pyampp-upgrade-") as tmpdir:
        dst = Path(tmpdir) / "legacy_upgraded.h5"
        loaded = read_b3d_h5(str(model_path))
        upgraded = _upgrade_dict_with_contract(loaded)
        shutil.copy2(model_path, dst)
        _write_geometry_contract_group(dst, upgraded["metadata"]["geometry_contract"])

        result = read_b3d_h5(str(dst))
        metadata = result.get("metadata", {})
        contract = metadata.get("geometry_contract")
        if not isinstance(contract, dict):
            raise AssertionError("geometry_contract group was not written back correctly")
        if not all(int(contract[axis]) > 0 for axis in ("nx", "ny", "nz")):
            raise AssertionError("geometry_contract dimensions are not positive")

    print(f"Legacy upgrade geometry-contract check passed for {model_path}")


def validate_sav_mapping(sav_path: Path) -> None:
    if not sav_path.exists():
        raise FileNotFoundError(f"SAV fixture not found: {sav_path}")

    try:
        from scipy import io as scipy_io
    except ImportError as exc:
        raise RuntimeError("scipy is required for the real SAV mapping validation") from exc

    with tempfile.TemporaryDirectory(prefix="pyampp-mapping-") as tmpdir:
        out_h5 = Path(tmpdir) / "real_fixture.h5"
        build_h5_from_sav(sav_path, out_h5)

        sav = scipy_io.readsav(str(sav_path), verbose=False)
        box = sav["box"].flat[0] if "box" in sav else sav["pbox"].flat[0]

        with h5py.File(out_h5, "r") as handle:
            if not np.array_equal(np.asarray(handle["lines/av_field"]), np.asarray(box["AVFIELD"], dtype=np.float64).reshape(-1, order="C")):
                raise AssertionError("lines/av_field mapping changed")
            if not np.array_equal(np.asarray(handle["lines/phys_length"]), np.asarray(box["PHYSLENGTH"], dtype=np.float64).reshape(-1, order="C")):
                raise AssertionError("lines/phys_length mapping changed")
            if not np.array_equal(np.asarray(handle["lines/start_idx"]), np.asarray(box["STARTIDX"], dtype=np.int64).reshape(-1, order="C")):
                raise AssertionError("lines/start_idx mapping changed")
            if not np.array_equal(np.asarray(handle["lines/end_idx"]), np.asarray(box["ENDIDX"], dtype=np.int64).reshape(-1, order="C")):
                raise AssertionError("lines/end_idx mapping changed")

            chromo_idx = np.asarray(handle["chromo/chromo_idx"])
            chromo_n = np.asarray(handle["chromo/chromo_n"])
            chromo_t = np.asarray(handle["chromo/chromo_t"])
            chromo_shape = np.asarray(handle["chromo/bx"]).shape

        if not np.array_equal(chromo_idx, np.asarray(box["CHROMO_IDX"], dtype=np.int64)):
            raise AssertionError("chromo_idx mapping changed")
        if not np.array_equal(chromo_n, np.asarray(box["CHROMO_N"], dtype=np.float32)):
            raise AssertionError("chromo_n mapping changed")
        if not np.array_equal(chromo_t, np.asarray(box["CHROMO_T"], dtype=np.float32)):
            raise AssertionError("chromo_t mapping changed")

        dense_n = np.zeros(chromo_shape, dtype=np.float32)
        dense_t = np.zeros(chromo_shape, dtype=np.float32)
        dense_n.flat[chromo_idx] = chromo_n
        dense_t.flat[chromo_idx] = chromo_t

        if not np.isclose(dense_n[0, 0, 0], chromo_n[0]):
            raise AssertionError("Dense chromo_n check at [0, 0, 0] failed")
        if not np.isclose(dense_n[0, 0, 1], chromo_n[1]):
            raise AssertionError("Dense chromo_n check at [0, 0, 1] failed")
        if not np.isclose(dense_t[0, 0, 0], chromo_t[0]):
            raise AssertionError("Dense chromo_t check at [0, 0, 0] failed")

    print(f"Real SAV line/chromo mapping check passed for {sav_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-data geometry upgrade and SAV mapping sanity checks outside pytest."
    )
    parser.add_argument(
        "--legacy-model-path",
        type=Path,
        default=DEFAULT_LEGACY_MODEL_PATH,
        help="Path to the real legacy clone HDF5 fixture.",
    )
    parser.add_argument(
        "--sav-path",
        type=Path,
        default=DEFAULT_LEGACY_SAV_PATH,
        help="Path to the real SAV fixture.",
    )
    args = parser.parse_args()

    validate_legacy_upgrade(args.legacy_model_path.resolve())
    validate_sav_mapping(args.sav_path.resolve())


if __name__ == "__main__":
    main()
