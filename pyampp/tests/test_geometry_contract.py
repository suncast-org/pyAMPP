"""Tests for geometry contract enforcement (Tier 1+2 metadata completion)."""

from __future__ import annotations

import numpy as np
import pytest

from pyampp.geometry.contract import (
    GeometryContract,
    RSUN_HMI_METERS,
    complete_geometry_contract,
    infer_box_dims,
    infer_voxel_resolution,
    infer_world_anchor_from_index,
    infer_obstime,
)


def _base_index_header(
    *,
    lon: float = 12.5,
    lat: float = -5.5,
    date_obs: str = "2024-05-12T16:00:00",
    rsun_ref: float = RSUN_HMI_METERS,
) -> str:
    return (
        "SIMPLE  = T\n"
        f"CRVAL1  = {lon}\n"
        f"CRVAL2  = {lat}\n"
        f"RSUN_REF= {rsun_ref}\n"
        f"DATE-OBS= '{date_obs}'\n"
        "END\n"
    )


def test_geometry_contract_dataclass():
    contract = GeometryContract(
        nx=100,
        ny=80,
        nz=120,
        dr_x=1.0,
        dr_y=1.0,
        dr_z=1.0,
        rsun_m=RSUN_HMI_METERS,
        anchor_lon_deg=12.5,
        anchor_lat_deg=-5.5,
        anchor_radius_rsun=1.0,
        frame="heliographic_stonyhurst",
        obstime="2024-05-12T16:00:00",
        inferred_from="index",
    )

    d = contract.to_dict()
    restored = GeometryContract.from_dict(d)
    assert restored.nx == 100
    assert restored.obstime == "2024-05-12T16:00:00"
    assert restored.inferred_from == "index"


def test_infer_box_dims():
    model_dict = {"corona": {"bx": np.zeros((100, 80, 120), dtype=np.float32)}}
    assert infer_box_dims(model_dict) == (100, 80, 120)


def test_infer_voxel_resolution_corona_only():
    model_dict = {"corona": {"dr": np.array([1.5, 1.5, 1.5], dtype=np.float64)}}
    assert infer_voxel_resolution(model_dict) == (1.5, 1.5, 1.5)

    model_dict = {"chromo": {"dr": np.array([9.0, 9.0, 9.0], dtype=np.float64)}}
    assert infer_voxel_resolution(model_dict) is None


def test_infer_obstime_from_base_index():
    model_dict = {"base": {"index": _base_index_header(date_obs="2020-11-26T19:58:31")}}
    assert infer_obstime(model_dict) == "2020-11-26T19:58:31"


def test_infer_world_anchor_from_base_index():
    model_dict = {"base": {"index": _base_index_header(lon=13.0, lat=-7.0)}}
    anchor = infer_world_anchor_from_index(model_dict)
    assert anchor is not None
    lon, lat, radius, frame = anchor
    assert lon == 13.0
    assert lat == -7.0
    assert frame == "heliographic_stonyhurst"
    assert abs(radius - 1.0) < 1e-12


def test_complete_geometry_contract_success_from_base_index():
    model_dict = {
        "corona": {
            "bx": np.zeros((100, 80, 120), dtype=np.float32),
            "dr": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        },
        "base": {
            "index": _base_index_header(lon=15.0, lat=-9.0, date_obs="2020-11-26T19:58:31"),
        },
    }

    contract = complete_geometry_contract(model_dict, strict=True)
    assert contract is not None
    assert contract.nx == 100
    assert contract.ny == 80
    assert contract.nz == 120
    assert contract.obstime == "2020-11-26T19:58:31"
    assert contract.anchor_lon_deg == 15.0
    assert contract.anchor_lat_deg == -9.0
    assert contract.inferred_from == "index"


def test_complete_geometry_contract_missing_dims():
    model_dict = {
        "corona": {},
        "base": {"index": _base_index_header()},
    }
    assert complete_geometry_contract(model_dict, strict=False) is None
    with pytest.raises(ValueError, match="Cannot infer box dimensions"):
        complete_geometry_contract(model_dict, strict=True)


def test_complete_geometry_contract_missing_dr():
    model_dict = {
        "corona": {"bx": np.zeros((10, 10, 10), dtype=np.float32)},
        "base": {"index": _base_index_header()},
    }
    assert complete_geometry_contract(model_dict, strict=False) is None
    with pytest.raises(ValueError, match="Cannot infer voxel resolution"):
        complete_geometry_contract(model_dict, strict=True)


def test_complete_geometry_contract_missing_obstime():
    model_dict = {
        "corona": {
            "bx": np.zeros((10, 10, 10), dtype=np.float32),
            "dr": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        },
        "base": {"index": "CRVAL1=1\nCRVAL2=2\nEND\n"},
    }
    assert complete_geometry_contract(model_dict, strict=False) is None
    with pytest.raises(ValueError, match="Cannot infer observation time from base/index"):
        complete_geometry_contract(model_dict, strict=True)


def test_complete_geometry_contract_missing_anchor():
    model_dict = {
        "corona": {
            "bx": np.zeros((10, 10, 10), dtype=np.float32),
            "dr": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        },
        "base": {"index": "DATE-OBS='2020-01-01T00:00:00'\nEND\n"},
    }
    assert complete_geometry_contract(model_dict, strict=False) is None
    with pytest.raises(ValueError, match="Cannot infer anchor geometry from base/index"):
        complete_geometry_contract(model_dict, strict=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
