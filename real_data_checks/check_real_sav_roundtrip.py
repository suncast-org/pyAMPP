from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import h5py
import numpy as np

from pyampp.io import load_model, save_model


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAV_PATH = (
    WORKSPACE_ROOT
    / "pyGXrender-test-data"
    / "raw"
    / "models"
    / "model_loader_parity_20201126T195831"
    / "hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.CHR.sav"
)


def _dataset_names(handle: h5py.File) -> list[str]:
    names: list[str] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    handle.visititems(visitor)
    names.sort()
    return names


def _chunk_slices(shape: tuple[int, ...], dtype: np.dtype, *, target_bytes: int = 8 * 1024 * 1024):
    if not shape:
        yield ()
        return

    trailing_elems = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    elems_per_chunk = max(1, target_bytes // max(dtype.itemsize, 1))
    first_axis_step = max(1, elems_per_chunk // max(trailing_elems, 1))

    for start in range(0, shape[0], first_axis_step):
        stop = min(shape[0], start + first_axis_step)
        yield (slice(start, stop),) + (slice(None),) * (len(shape) - 1)


def _datasets_equal(left: h5py.Dataset, right: h5py.Dataset) -> bool:
    if left.shape != right.shape:
        return False
    if left.dtype != right.dtype:
        return False

    if left.shape == ():
        left_value = left[()]
        right_value = right[()]
        if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
            return np.array_equal(np.asarray(left_value), np.asarray(right_value))
        return left_value == right_value

    for chunk_slice in _chunk_slices(left.shape, left.dtype):
        if not np.array_equal(left[chunk_slice], right[chunk_slice]):
            return False
    return True


def _files_equal(left_path: Path, right_path: Path) -> bool:
    with h5py.File(left_path, "r") as left_handle, h5py.File(right_path, "r") as right_handle:
        left_names = _dataset_names(left_handle)
        right_names = _dataset_names(right_handle)
        if left_names != right_names:
            return False

        for dataset_name in left_names:
            if not _datasets_equal(left_handle[dataset_name], right_handle[dataset_name]):
                return False

    return True


def validate_roundtrip(sav_path: Path) -> None:
    if not sav_path.exists():
        raise FileNotFoundError(f"SAV fixture not found: {sav_path}")

    print(f"Loading SAV model from {sav_path}", flush=True)
    with tempfile.TemporaryDirectory(prefix="pyampp-roundtrip-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        h5_path1 = tmpdir_path / "model1.h5"
        h5_path2 = tmpdir_path / "model2.h5"

        model1 = load_model(sav_path)
        print(f"Saving first canonical HDF5 to {h5_path1}", flush=True)
        save_model(model1, h5_path1)

        print(f"Reloading canonical HDF5 from {h5_path1}", flush=True)
        model2 = load_model(h5_path1)
        print(f"Saving second canonical HDF5 to {h5_path2}", flush=True)
        save_model(model2, h5_path2)

        print("Comparing canonical HDF5 files", flush=True)
        if not _files_equal(h5_path1, h5_path2):
            raise AssertionError("Round-trip HDF5 files differ; contract violated")

    print(f"Round-trip idempotence check passed for {sav_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the real-data SAV -> HDF5 -> HDF5 round-trip sanity check outside pytest."
    )
    parser.add_argument(
        "sav_path",
        nargs="?",
        type=Path,
        default=DEFAULT_SAV_PATH,
        help="Path to the real SAV fixture to validate.",
    )
    args = parser.parse_args()
    validate_roundtrip(args.sav_path.resolve())


if __name__ == "__main__":
    main()
