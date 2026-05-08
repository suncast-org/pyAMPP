from __future__ import annotations

import numpy as np
import pytest


def test_thin_reader_returns_none_without_geometry_contract(tmp_path):
    h5py = pytest.importorskip("h5py")
    from pyampp.io import load_geometry_contract_and_observer_from_h5

    path = tmp_path / "no_contract.h5"
    with h5py.File(path, "w") as f:
        corona = f.create_group("corona")
        corona.create_dataset("bx", data=np.zeros((2, 2, 2), dtype=np.float32))

    thin = load_geometry_contract_and_observer_from_h5(path)
    assert thin is None


def test_thin_reader_returns_contract_and_optional_observer(tmp_path):
    h5py = pytest.importorskip("h5py")
    from pyampp.geometry.contract import GeometryContract, RSUN_HMI_METERS
    from pyampp.io import (
        load_geometry_contract_and_observer_from_h5,
        save_model_to_h5,
    )

    path = tmp_path / "with_contract.h5"
    model = {
        "corona": {
            "bx": np.zeros((8, 6, 4), dtype=np.float32),
            "by": np.zeros((8, 6, 4), dtype=np.float32),
            "bz": np.zeros((8, 6, 4), dtype=np.float32),
            "attrs": {"model_type": "nlfff"},
        },
        "metadata": {
            "geometry_contract": GeometryContract(
                nx=8,
                ny=6,
                nz=4,
                dr_x=0.002,
                dr_y=0.003,
                dr_z=0.004,
                rsun_m=RSUN_HMI_METERS,
                anchor_lon_deg=120.5,
                anchor_lat_deg=-13.2,
                anchor_radius_rsun=1.0,
                frame="heliographic_stonyhurst",
                obstime="2020-11-26T19:58:33",
                inferred_from="index",
            )
        },
        "observer": {
            "name": "earth",
            "label": "Earth",
            "ephemeris": {
                "obs_date": "2020-11-26T19:58:33",
                "hgln_obs_deg": 0.0,
                "hglt_obs_deg": 1.4,
                "dsun_cm": 1.5e13,
            },
        },
    }

    save_model_to_h5(model, path)

    thin = load_geometry_contract_and_observer_from_h5(path)

    assert thin is not None
    assert set(thin.keys()).issubset({"metadata", "observer"})
    assert "metadata" in thin
    assert "geometry_contract" in thin["metadata"]
    contract = thin["metadata"]["geometry_contract"]
    assert int(contract.nx) == 8
    assert float(contract.dr_y) == pytest.approx(0.003)
    assert float(contract.anchor_lon_deg) == pytest.approx(120.5)

    assert "observer" in thin
    assert thin["observer"]["name"] == "earth"
    assert thin["observer"]["ephemeris"]["obs_date"] == "2020-11-26T19:58:33"
