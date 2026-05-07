"""Tests for geometry contract enforcement (Tier 1+2 metadata completion)."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from pyampp.geometry.contract import (
    GeometryContract,
    RSUN_HMI_METERS,
    complete_geometry_contract,
    infer_box_dims,
    infer_voxel_resolution,
    infer_world_anchor_defaults,
    infer_obstime,
)


def test_geometry_contract_dataclass():
    """Test GeometryContract construction and serialization."""
    contract = GeometryContract(
        nx=100,
        ny=80,
        nz=120,
        dr_x=1.0,
        dr_y=1.0,
        dr_z=1.0,
        rsun_m=RSUN_HMI_METERS,
        anchor_lon_deg=0.0,
        anchor_lat_deg=0.0,
        anchor_radius_rsun=1.0,
        frame="heliographic_stonyhurst",
        obstime="2024-05-12T16:00:00",
        inferred_from="defaults",
    )
    
    assert contract.nx == 100
    assert contract.ny == 80
    assert contract.nz == 120
    assert contract.rsun_m == RSUN_HMI_METERS
    
    # Test serialization
    d = contract.to_dict()
    assert d["nx"] == 100
    assert d["obstime"] == "2024-05-12T16:00:00"
    
    # Test deserialization
    restored = GeometryContract.from_dict(d)
    assert restored.nx == contract.nx
    assert restored.obstime == contract.obstime


def test_infer_box_dims():
    """Test box dimension inference from corona cube."""
    # Case 1: Valid corona with bx
    model_dict = {
        "corona": {
            "bx": np.zeros((100, 80, 120), dtype=np.float32),
        }
    }
    dims = infer_box_dims(model_dict)
    assert dims == (100, 80, 120)
    
    # Case 2: Corona with by instead of bx
    model_dict = {
        "corona": {
            "by": np.zeros((50, 60, 70), dtype=np.float32),
        }
    }
    dims = infer_box_dims(model_dict)
    assert dims == (50, 60, 70)
    
    # Case 3: No corona
    model_dict = {}
    dims = infer_box_dims(model_dict)
    assert dims is None
    
    # Case 4: Corona but no suitable field
    model_dict = {"corona": {"other": "data"}}
    dims = infer_box_dims(model_dict)
    assert dims is None


def test_infer_voxel_resolution():
    """Test voxel resolution inference from corona.dr."""
    # Case 1: Valid dr array with 3 elements
    model_dict = {
        "corona": {
            "dr": np.array([1.5, 1.5, 1.5], dtype=np.float64),
        }
    }
    resolution = infer_voxel_resolution(model_dict)
    assert resolution == (1.5, 1.5, 1.5)
    
    # Case 2: dr with 2 elements
    model_dict = {
        "corona": {
            "dr": np.array([2.0, 2.0], dtype=np.float64),
        }
    }
    resolution = infer_voxel_resolution(model_dict)
    assert resolution == (2.0, 2.0, 2.0)
    
    # Case 3: Single element dr
    model_dict = {
        "corona": {
            "dr": np.array([3.0], dtype=np.float64),
        }
    }
    resolution = infer_voxel_resolution(model_dict)
    assert resolution == (3.0, 3.0, 3.0)
    
    # Case 4: No corona
    model_dict = {}
    resolution = infer_voxel_resolution(model_dict)
    assert resolution is None
    
    # Case 5: Corona but no dr
    model_dict = {"corona": {"bx": np.zeros((10, 10, 10))}}
    resolution = infer_voxel_resolution(model_dict)
    assert resolution is None


def test_infer_obstime():
    """Test observation time inference."""
    # Case 1: From metadata
    model_dict = {
        "metadata": {
            "obstime": "2024-05-12T16:00:00",
        }
    }
    obstime = infer_obstime(model_dict)
    assert obstime == "2024-05-12T16:00:00"
    
    # Case 2: obstime as bytes
    model_dict = {
        "metadata": {
            "obstime": b"2024-05-12T16:00:00",
        }
    }
    obstime = infer_obstime(model_dict)
    assert obstime == "2024-05-12T16:00:00"
    
    # Case 3: No metadata
    model_dict = {}
    obstime = infer_obstime(model_dict)
    assert obstime is None


def test_infer_world_anchor_defaults():
    """Test default world anchor."""
    anchor_lon, anchor_lat, anchor_radius, frame = infer_world_anchor_defaults()
    assert anchor_lon == 0.0
    assert anchor_lat == 0.0
    assert anchor_radius == 1.0
    assert frame == "heliographic_stonyhurst"


def test_complete_geometry_contract_minimal_success():
    """Test contract completion with minimal but sufficient data."""
    model_dict = {
        "corona": {
            "bx": np.zeros((100, 80, 120), dtype=np.float32),
            "dr": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        },
        "metadata": {
            "obstime": "2024-05-12T16:00:00",
        }
    }
    
    contract = complete_geometry_contract(model_dict, strict=False)
    assert contract is not None
    assert contract.nx == 100
    assert contract.ny == 80
    assert contract.nz == 120
    assert contract.dr_x == 1.0
    assert contract.dr_y == 1.0
    assert contract.dr_z == 1.0
    assert contract.anchor_lon_deg == 0.0  # default
    assert contract.anchor_lat_deg == 0.0  # default
    assert contract.obstime == "2024-05-12T16:00:00"
    assert contract.inferred_from == "defaults"


def test_complete_geometry_contract_missing_dims():
    """Test contract completion fails gracefully without dims."""
    model_dict = {
        "corona": {},  # no bx, by, or bz
        "metadata": {
            "obstime": "2024-05-12T16:00:00",
        }
    }
    
    contract = complete_geometry_contract(model_dict, strict=False)
    assert contract is None


def test_complete_geometry_contract_missing_dr():
    """Test contract completion fails gracefully without dr."""
    model_dict = {
        "corona": {
            "bx": np.zeros((100, 80, 120), dtype=np.float32),
            # no dr
        },
        "metadata": {
            "obstime": "2024-05-12T16:00:00",
        }
    }
    
    contract = complete_geometry_contract(model_dict, strict=False)
    assert contract is None


def test_complete_geometry_contract_strict_mode():
    """Test contract completion in strict mode."""
    # Complete model: should succeed
    model_dict = {
        "corona": {
            "bx": np.zeros((100, 80, 120), dtype=np.float32),
            "dr": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        },
        "metadata": {
            "obstime": "2024-05-12T16:00:00",
        }
    }
    contract = complete_geometry_contract(model_dict, strict=True)
    assert contract is not None
    
    # Missing dims: should raise
    model_dict = {
        "corona": {},
        "metadata": {
            "obstime": "2024-05-12T16:00:00",
        }
    }
    with pytest.raises(ValueError, match="Cannot infer box dimensions"):
        complete_geometry_contract(model_dict, strict=True)
    
    # Missing obstime: should raise
    model_dict = {
        "corona": {
            "bx": np.zeros((100, 80, 120), dtype=np.float32),
            "dr": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        },
        "metadata": {},
    }
    with pytest.raises(ValueError, match="Cannot infer observation time"):
        complete_geometry_contract(model_dict, strict=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
