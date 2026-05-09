from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from pyampp.util.h5thin_export import app


def _make_full_model(path):
    from pyampp.geometry.contract import GeometryContract, RSUN_HMI_METERS
    from pyampp.io import save_model_to_h5

    model = {
        "corona": {
            "bx": np.zeros((3, 4, 5), dtype=np.float32),
            "by": np.zeros((3, 4, 5), dtype=np.float32),
            "bz": np.zeros((3, 4, 5), dtype=np.float32),
            "attrs": {"model_type": "nlfff"},
        },
        "metadata": {
            "id": "export_cli_test",
            "execute": "gx_fov2box, ...",
            "geometry_contract": GeometryContract(
                nx=3,
                ny=4,
                nz=5,
                dr_x=0.001,
                dr_y=0.001,
                dr_z=0.001,
                rsun_m=RSUN_HMI_METERS,
                anchor_lon_deg=95.0,
                anchor_lat_deg=-10.0,
                anchor_radius_rsun=1.0,
                frame="heliographic_stonyhurst",
                obstime="2020-01-01T00:00:00",
                inferred_from="index",
            ),
        },
        "observer": {"name": "earth", "label": "Earth"},
    }
    save_model_to_h5(model, path)


def test_h5thin_export_cli_default_output(tmp_path):
    h5py = pytest.importorskip("h5py")

    src = tmp_path / "source.h5"
    _make_full_model(src)

    runner = CliRunner()
    result = runner.invoke(app, [str(src)])

    assert result.exit_code == 0
    out = tmp_path / "source_metadata.h5"
    assert out.exists()
    with h5py.File(out, "r") as f:
        assert sorted(f.keys()) == ["metadata", "observer"]


def test_h5thin_export_cli_explicit_output(tmp_path):
    h5py = pytest.importorskip("h5py")

    src = tmp_path / "source2.h5"
    out = tmp_path / "custom_thin.h5"
    _make_full_model(src)

    runner = CliRunner()
    result = runner.invoke(app, [str(src), "--output", str(out)])

    assert result.exit_code == 0
    assert out.exists()
    with h5py.File(out, "r") as f:
        assert "metadata" in f
        assert "geometry_contract" in f["metadata"]
