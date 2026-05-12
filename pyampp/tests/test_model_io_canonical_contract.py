"""
Test canonical round-trip and provenance-agnostic contract for pyAMPP model I/O.
"""

import sys
from unittest.mock import patch
import numpy as np
import pytest
from pyampp.io import load_model

from pyampp.tests._fits_header import canonical_base_index_header


def test_load_model_dispatches_through_canonical_io_boundary(tmp_path):
    h5_path = tmp_path / "entry.h5"
    sav_path = tmp_path / "entry.sav"
    h5_path.write_bytes(b"")
    sav_path.write_bytes(b"")

    with patch("pyampp.io.model._load_model_h5", return_value={"metadata": {"id": "h5"}}) as mocked_h5:
        assert load_model(h5_path)["metadata"]["id"] == "h5"
    mocked_h5.assert_called_once_with(h5_path, strict=False)

    with patch("pyampp.io.model._load_model_sav", return_value={"metadata": {"id": "sav"}}) as mocked_sav:
        assert load_model(sav_path)["metadata"]["id"] == "sav"
    mocked_sav.assert_called_once_with(sav_path, strict=False, keep_temp_h5=False)

    with pytest.raises(ValueError, match="Unsupported model format"):
        load_model(tmp_path / "entry.txt")


def test_h5_loader_upgrades_legacy_missing_contract(tmp_path):
    # Create a minimal legacy H5 missing geometry_contract
    import h5py
    h5_path = tmp_path / "legacy_missing_contract.h5"
    with h5py.File(h5_path, "w") as f:
        corona = f.create_group("corona")
        corona.create_dataset("bx", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.create_dataset("by", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.create_dataset("bz", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.attrs["model_type"] = "nlfff"
        base = f.create_group("base")
        base.create_dataset("index", data=np.bytes_(canonical_base_index_header(date_obs="2020-11-26T19:58:31")))
        base.create_dataset("bx", data=np.zeros((2, 2), dtype=np.float32))
        base.create_dataset("by", data=np.zeros((2, 2), dtype=np.float32))
        base.create_dataset("bz", data=np.zeros((2, 2), dtype=np.float32))
        base.create_dataset("ic", data=np.ones((2, 2), dtype=np.float32))
        metadata = f.create_group("metadata")
        metadata.create_dataset("id", data=np.bytes_(b"legacy.test.model"))
    
    # Loader should inject geometry_contract
    model = load_model(h5_path)
    assert "geometry_contract" in model["metadata"], "Loader did not inject geometry_contract"
    assert model["metadata"]["id"] == "legacy.test.model"
    assert model["metadata"]["axis_order_2d"] == "yx"
    assert model["metadata"]["axis_order_3d"] == "zyx"
    assert model["metadata"]["vector_layout"] == "split_components"
    assert model["metadata"]["lineage"] == "legacy-h5:unknown"
    assert model["observer"]["name"] == "earth"
    assert set(model["observer"]["ephemeris"]).issuperset({"obs_date", "hgln_obs_deg", "hglt_obs_deg", "dsun_cm", "rsun_cm"})


def test_h5_loader_backfills_missing_canonical_metadata_without_overwriting_present_values(tmp_path):
    import h5py

    h5_path = tmp_path / "legacy_name.NONE.h5"
    with h5py.File(h5_path, "w") as f:
        corona = f.create_group("corona")
        corona.create_dataset("bx", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.create_dataset("by", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.create_dataset("bz", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.attrs["model_type"] = "nlfff"

        base = f.create_group("base")
        base.create_dataset("index", data=np.bytes_(canonical_base_index_header(date_obs="2020-11-26T19:58:31")))
        base.create_dataset("bx", data=np.zeros((2, 2), dtype=np.float32))
        base.create_dataset("by", data=np.zeros((2, 2), dtype=np.float32))
        base.create_dataset("bz", data=np.zeros((2, 2), dtype=np.float32))
        base.create_dataset("ic", data=np.ones((2, 2), dtype=np.float32))

        metadata = f.create_group("metadata")
        metadata.create_dataset("lineage", data=np.bytes_(b""))
        metadata.create_dataset("axis_order_3d", data=np.bytes_(b"xyz"))

    model = load_model(h5_path)

    assert model["metadata"]["id"] == "legacy_name.NONE"
    assert model["metadata"]["axis_order_2d"] == "yx"
    assert model["metadata"]["axis_order_3d"] == "xyz"
    assert model["metadata"]["vector_layout"] == "split_components"
    assert model["metadata"]["lineage"] == "legacy-h5:unknown"


def test_sav_loader_backfills_missing_canonical_metadata(tmp_path):
    import h5py

    from pyampp.io.model import _load_model_sav

    sav_path = tmp_path / "legacy_model.NONE.sav"
    sav_path.write_bytes(b"placeholder")

    def _write_legacy_temp_h5(*, sav_path, out_h5):
        with h5py.File(out_h5, "w") as f:
            corona = f.create_group("corona")
            corona.create_dataset("bx", data=np.zeros((2, 2, 2), dtype=np.float32))
            corona.create_dataset("by", data=np.zeros((2, 2, 2), dtype=np.float32))
            corona.create_dataset("bz", data=np.zeros((2, 2, 2), dtype=np.float32))
            corona.attrs["model_type"] = "nlfff"

            base = f.create_group("base")
            base.create_dataset("index", data=np.bytes_(canonical_base_index_header(date_obs="2020-11-26T19:58:31")))
            base.create_dataset("bx", data=np.zeros((2, 2), dtype=np.float32))
            base.create_dataset("by", data=np.zeros((2, 2), dtype=np.float32))
            base.create_dataset("bz", data=np.zeros((2, 2), dtype=np.float32))
            base.create_dataset("ic", data=np.ones((2, 2), dtype=np.float32))

    with patch("pyampp.io.model.build_h5_from_sav", side_effect=_write_legacy_temp_h5):
        model = _load_model_sav(sav_path)

    assert model["metadata"]["id"] == "legacy_model.NONE"
    assert model["metadata"]["axis_order_2d"] == "yx"
    assert model["metadata"]["axis_order_3d"] == "zyx"
    assert model["metadata"]["vector_layout"] == "split_components"
    assert model["metadata"]["lineage"] == "legacy-sav:unknown"


def test_export_model_canonicalizes_legacy_h5(tmp_path, monkeypatch, capsys):
    import h5py

    from pyampp.util import export_model

    src = tmp_path / "legacy_input.NONE.h5"
    out = tmp_path / "exported.h5"
    with h5py.File(src, "w") as f:
        corona = f.create_group("corona")
        corona.create_dataset("bx", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.create_dataset("by", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.create_dataset("bz", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.attrs["model_type"] = "nlfff"

        base = f.create_group("base")
        base.create_dataset("index", data=np.bytes_(canonical_base_index_header(date_obs="2020-11-26T19:58:31")))
        base.create_dataset("bx", data=np.zeros((2, 2), dtype=np.float32))
        base.create_dataset("by", data=np.zeros((2, 2), dtype=np.float32))
        base.create_dataset("bz", data=np.zeros((2, 2), dtype=np.float32))
        base.create_dataset("ic", data=np.ones((2, 2), dtype=np.float32))

    monkeypatch.setattr(sys, "argv", ["export_model", "--model-path", str(src), "--out-h5", str(out)])

    export_model.main()

    assert out.exists()
    loaded = load_model(out)
    assert loaded["metadata"]["id"] == "legacy_input.NONE"
    assert loaded["metadata"]["axis_order_2d"] == "yx"
    assert loaded["metadata"]["axis_order_3d"] == "zyx"
    assert loaded["metadata"]["vector_layout"] == "split_components"
    assert loaded["metadata"]["lineage"] == "legacy-h5:unknown"
    assert f"Wrote: {out}" in capsys.readouterr().out


def test_h5_loader_rejects_full_models_missing_required_base_maps(tmp_path):
    import h5py

    h5_path = tmp_path / "legacy_missing_base_maps.h5"
    with h5py.File(h5_path, "w") as f:
        corona = f.create_group("corona")
        corona.create_dataset("bx", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.create_dataset("by", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.create_dataset("bz", data=np.zeros((2, 2, 2), dtype=np.float32))
        corona.attrs["model_type"] = "nlfff"
        base = f.create_group("base")
        base.create_dataset("index", data=np.bytes_(canonical_base_index_header(date_obs="2020-11-26T19:58:31")))
        base.create_dataset("bz", data=np.zeros((2, 2), dtype=np.float32))
        base.create_dataset("ic", data=np.ones((2, 2), dtype=np.float32))

    with pytest.raises(RuntimeError, match="base LOS maps"):
        load_model(h5_path)
