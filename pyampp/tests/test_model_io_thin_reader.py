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
            "id": "demo_model",
            "axis_order_3d": "xyz",
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
    assert thin["metadata"]["id"] == "demo_model"
    assert thin["metadata"]["axis_order_3d"] == "xyz"

    assert "observer" in thin
    assert thin["observer"]["name"] == "earth"
    assert thin["observer"]["ephemeris"]["obs_date"] == "2020-11-26T19:58:33"


def test_save_thin_model_to_h5_writes_only_metadata_and_observer(tmp_path):
    h5py = pytest.importorskip("h5py")
    from pyampp.geometry.contract import GeometryContract, RSUN_HMI_METERS
    from pyampp.io import (
        load_geometry_contract_and_observer_from_h5,
        save_thin_model_to_h5,
    )

    path = tmp_path / "thin_only.h5"
    thin_model = {
        "metadata": {
            "id": "portable_meta",
            "axis_order_3d": "xyz",
            "geometry_contract": GeometryContract(
                nx=10,
                ny=12,
                nz=14,
                dr_x=0.001,
                dr_y=0.0015,
                dr_z=0.002,
                rsun_m=RSUN_HMI_METERS,
                anchor_lon_deg=75.0,
                anchor_lat_deg=5.0,
                anchor_radius_rsun=1.0,
                frame="heliographic_stonyhurst",
                obstime="2021-01-02T03:04:05",
                inferred_from="index",
            ),
        },
        "observer": {
            "name": "earth",
            "label": "Earth",
            "ephemeris": {
                "obs_date": "2021-01-02T03:04:05",
                "hgln_obs_deg": 0.0,
                "hglt_obs_deg": 3.0,
            },
        },
    }

    save_thin_model_to_h5(thin_model, path)

    with h5py.File(path, "r") as f:
        assert "metadata" in f
        assert "observer" in f
        assert "corona" not in f
        assert "chromo" not in f
        assert "base" not in f

    roundtrip = load_geometry_contract_and_observer_from_h5(path)
    assert roundtrip is not None
    assert roundtrip["metadata"]["id"] == "portable_meta"
    assert int(roundtrip["metadata"]["geometry_contract"].nz) == 14
    assert roundtrip["observer"]["name"] == "earth"


def test_export_thin_model_from_h5_creates_sibling_metadata_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    from pyampp.geometry.contract import GeometryContract, RSUN_HMI_METERS
    from pyampp.io import export_thin_model_from_h5, save_model_to_h5

    src = tmp_path / "full_model.h5"
    model = {
        "corona": {
            "bx": np.zeros((3, 4, 5), dtype=np.float32),
            "by": np.zeros((3, 4, 5), dtype=np.float32),
            "bz": np.zeros((3, 4, 5), dtype=np.float32),
            "attrs": {"model_type": "nlfff"},
        },
        "metadata": {
            "id": "full_for_export",
            "geometry_contract": GeometryContract(
                nx=3,
                ny=4,
                nz=5,
                dr_x=0.001,
                dr_y=0.001,
                dr_z=0.001,
                rsun_m=RSUN_HMI_METERS,
                anchor_lon_deg=90.0,
                anchor_lat_deg=0.0,
                anchor_radius_rsun=1.0,
                frame="heliographic_stonyhurst",
                obstime="2020-01-01T00:00:00",
                inferred_from="index",
            ),
        },
        "observer": {"name": "earth", "label": "Earth"},
    }
    save_model_to_h5(model, src)

    out = export_thin_model_from_h5(src)
    assert out == src.with_name("full_model_metadata.h5")
    assert out.exists()

    with h5py.File(out, "r") as f:
        assert sorted(f.keys()) == ["metadata", "observer"]
