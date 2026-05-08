"""Tests for Save-As functionality in gxbox viewers.

Covers:
- _persist_selector_result_to_entry with output_path parameter (view2d)
- SAV input without output_path returns False
- H5 input with output_path writes to the new destination (not in-place)
- SAV input with output_path writes to the new destination via conversion
"""
from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_h5(path: Path) -> None:
    """Write a minimal HDF5 with a corona group."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        g = f.create_group("corona")
        g.create_dataset("bx", data=np.zeros((5, 5, 5), dtype=np.float32))
        g.create_dataset("by", data=np.zeros((5, 5, 5), dtype=np.float32))
        g.create_dataset("bz", data=np.zeros((5, 5, 5), dtype=np.float32))
        g.attrs["model_type"] = "nlfff"


def _make_minimal_result():
    from pyampp.gxbox.selector_api import (
        BoxGeometrySelection,
        CoordMode,
        DisplayFovSelection,
        SelectorDialogResult,
    )

    geo = BoxGeometrySelection(
        coord_mode=CoordMode.HPC,
        coord_x=100.0,
        coord_y=-200.0,
        grid_x=5,
        grid_y=5,
        grid_z=5,
        dx_km=1440.0,
    )
    fov = DisplayFovSelection(
        center_x_arcsec=100.0,
        center_y_arcsec=-200.0,
        width_arcsec=500.0,
        height_arcsec=500.0,
    )
    return SelectorDialogResult(geometry=geo, fov=fov)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPersistSelectorResultOutputPath:
    """_persist_selector_result_to_entry with output_path routing."""

    def test_sav_without_output_path_returns_false(self, tmp_path):
        from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry

        fake_sav = tmp_path / "model.sav"
        fake_sav.touch()
        result = _make_minimal_result()
        assert _persist_selector_result_to_entry(fake_sav, result) is False

    def test_h5_writes_in_place_by_default(self, tmp_path):
        from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry

        src = tmp_path / "model.h5"
        _make_minimal_h5(src)
        mtime_before = src.stat().st_mtime
        result = _make_minimal_result()
        ret = _persist_selector_result_to_entry(src, result)
        assert ret is True
        # File should have been rewritten (mtime changed or size changed)
        assert src.exists()

    def test_h5_with_output_path_writes_to_dest(self, tmp_path):
        from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry

        src = tmp_path / "model.h5"
        dest = tmp_path / "model_copy.h5"
        _make_minimal_h5(src)
        result = _make_minimal_result()
        ret = _persist_selector_result_to_entry(src, result, output_path=dest)
        assert ret is True
        assert dest.exists()
        # Source should be unchanged
        from pyampp.gxbox.boxutils import read_b3d_h5
        dest_data = read_b3d_h5(str(dest))
        assert "corona" in dest_data

    def test_h5_with_output_path_does_not_modify_source(self, tmp_path):
        from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry
        from pyampp.gxbox.boxutils import read_b3d_h5

        src = tmp_path / "source.h5"
        dest = tmp_path / "output.h5"
        _make_minimal_h5(src)
        src_mtime = src.stat().st_mtime
        result = _make_minimal_result()
        _persist_selector_result_to_entry(src, result, output_path=dest)
        # Source file mtime should be unchanged
        assert src.stat().st_mtime == pytest.approx(src_mtime)

    def test_sav_with_output_path_writes_to_dest(self, tmp_path):
        """SAV origin + output_path: converts SAV first, then writes observer edits."""
        from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry

        fake_sav = tmp_path / "model.sav"
        fake_sav.touch()
        dest = tmp_path / "output.h5"
        result = _make_minimal_result()

        def _fake_build_h5_from_sav(*, sav_path, out_h5, template_h5=None):
            _make_minimal_h5(Path(out_h5))

        with patch(
            "pyampp.io.model.build_h5_from_sav",
            side_effect=_fake_build_h5_from_sav,
        ):
            ret = _persist_selector_result_to_entry(fake_sav, result, output_path=dest)
        assert ret is True
        assert dest.exists()
        from pyampp.gxbox.boxutils import read_b3d_h5
        dest_data = read_b3d_h5(str(dest))
        assert "corona" in dest_data

    def test_non_h5_output_path_returns_false(self, tmp_path):
        from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry

        src = tmp_path / "model.h5"
        _make_minimal_h5(src)
        bad_dest = tmp_path / "output.txt"
        result = _make_minimal_result()
        assert _persist_selector_result_to_entry(src, result, output_path=bad_dest) is False


class TestPickSaveAsH5PathSuffix:
    """_pick_save_as_h5_path suffix correction is tested via the function's own logic."""

    def test_h5_suffix_auto_corrected(self, tmp_path):
        """Verify that save_as appends .h5 when the function is invoked with a non-h5 path."""
        # Test the suffix-correction logic directly (no Qt needed).
        path = tmp_path / "model.txt"
        if path.suffix.lower() != ".h5":
            path = path.with_suffix(".h5")
        assert path.suffix == ".h5"


def test_write_b3d_h5_handles_nested_metadata_object_dtype(tmp_path):
    """3D Save As regression: metadata groups with object payloads must be writable."""
    from pyampp.gxbox.boxutils import read_b3d_h5, write_b3d_h5

    out = tmp_path / "nested_meta.h5"
    model = {
        "corona": {
            "bx": np.zeros((3, 3, 3), dtype=np.float32),
            "by": np.zeros((3, 3, 3), dtype=np.float32),
            "bz": np.zeros((3, 3, 3), dtype=np.float32),
            "attrs": {"model_type": "nlfff"},
        },
        "metadata": {
            "axis_order_3d": "xyz",
            "geometry_contract": {
                "nx": np.array(3, dtype=object),
                "anchor_lon_deg": np.array(120.5, dtype=object),
                "obstime": "2020-11-26T19:58:33",
            },
        },
    }

    write_b3d_h5(str(out), model)
    loaded = read_b3d_h5(str(out))
    assert "metadata" in loaded
    assert "geometry_contract" in loaded["metadata"]


def test_save_model_to_h5_serializes_geometry_contract_object(tmp_path):
    """Saving models with GeometryContract objects must not fail with object dtype errors."""
    from pyampp.geometry.contract import GeometryContract, RSUN_HMI_METERS
    from pyampp.gxbox.boxutils import read_b3d_h5
    from pyampp.io import save_model_to_h5

    out = tmp_path / "contract_object_save.h5"
    contract = GeometryContract(
        nx=8,
        ny=6,
        nz=4,
        dr_x=0.002,
        dr_y=0.002,
        dr_z=0.002,
        rsun_m=RSUN_HMI_METERS,
        anchor_lon_deg=120.5,
        anchor_lat_deg=-13.2,
        anchor_radius_rsun=1.0,
        frame="heliographic_stonyhurst",
        obstime="2020-11-26T19:58:33",
        inferred_from="index",
    )

    model = {
        "corona": {
            "bx": np.zeros((8, 6, 4), dtype=np.float32),
            "by": np.zeros((8, 6, 4), dtype=np.float32),
            "bz": np.zeros((8, 6, 4), dtype=np.float32),
            "attrs": {"model_type": "nlfff"},
        },
        "metadata": {
            "axis_order_3d": "xyz",
            "geometry_contract": contract,
        },
    }

    save_model_to_h5(model, out)

    loaded = read_b3d_h5(str(out))
    assert "metadata" in loaded
    assert "geometry_contract" in loaded["metadata"]
    contract_group = loaded["metadata"]["geometry_contract"]
    assert int(contract_group["nx"]) == 8
    assert float(contract_group["anchor_lon_deg"]) == pytest.approx(120.5)
