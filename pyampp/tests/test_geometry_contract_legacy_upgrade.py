from __future__ import annotations

from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from pyampp.geometry.contract import complete_geometry_contract
from pyampp.gxbox.boxutils import read_b3d_h5, write_b3d_h5


LEGACY_MODEL_PATH = Path(
    "/Users/gelu/code/SUNCAST-ORG/pyGXrender-test-data/raw/models/"
    "model_loader_parity_20201126T195831/"
    "hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.CHR.clone.h5"
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


def test_legacy_upgrade_roundtrip_synthetic(tmp_path: Path) -> None:
    synthetic = {
        "corona": {
            "bx": np.zeros((8, 6, 4), dtype=np.float32),
            "by": np.zeros((8, 6, 4), dtype=np.float32),
            "bz": np.zeros((8, 6, 4), dtype=np.float32),
            "dr": np.array([0.002, 0.002, 0.002], dtype=np.float64),
            "attrs": {"model_type": "nlfff"},
        },
        "metadata": {
            "obstime": "2020-11-26T19:58:31",
            "id": "synthetic.test.model",
        },
    }

    src = tmp_path / "synthetic_legacy.h5"
    dst = tmp_path / "synthetic_upgraded.h5"
    write_b3d_h5(str(src), synthetic)

    loaded = read_b3d_h5(str(src))
    upgraded = _upgrade_dict_with_contract(loaded)
    shutil.copy2(src, dst)
    _write_geometry_contract_group(dst, upgraded["metadata"]["geometry_contract"])

    result = read_b3d_h5(str(dst))
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    contract = metadata.get("geometry_contract")
    assert isinstance(contract, dict)
    assert int(contract["nx"]) == 8
    assert int(contract["ny"]) == 6
    assert int(contract["nz"]) == 4
    assert abs(float(contract["dr_x"]) - 0.002) < 1e-12


def test_legacy_upgrade_roundtrip_real_fixture(tmp_path: Path) -> None:
    if not LEGACY_MODEL_PATH.exists():
        pytest.skip("Legacy fixture model not available on this machine.")

    dst = tmp_path / "legacy_upgraded.h5"
    loaded = read_b3d_h5(str(LEGACY_MODEL_PATH))
    upgraded = _upgrade_dict_with_contract(loaded)
    shutil.copy2(LEGACY_MODEL_PATH, dst)
    _write_geometry_contract_group(dst, upgraded["metadata"]["geometry_contract"])

    result = read_b3d_h5(str(dst))
    metadata = result.get("metadata", {})
    contract = metadata.get("geometry_contract")
    assert isinstance(contract, dict)
    assert int(contract["nx"]) > 0
    assert int(contract["ny"]) > 0
    assert int(contract["nz"]) > 0
