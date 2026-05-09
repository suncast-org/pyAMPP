from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from pyampp.util.h5thin import app


def test_h5thin_missing_contract_returns_zero_unless_required(tmp_path):
    h5py = pytest.importorskip("h5py")

    path = tmp_path / "missing_contract.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("corona")
        g.create_dataset("bx", data=np.zeros((2, 2, 2), dtype=np.float32))

    runner = CliRunner()
    result = runner.invoke(app, [str(path)])
    assert result.exit_code == 0
    assert "geometry_contract: missing" in result.stdout

    result_required = runner.invoke(app, [str(path), "--require-contract"])
    assert result_required.exit_code == 2


def test_h5thin_reports_contract_and_observer_json(tmp_path):
    from pyampp.geometry.contract import GeometryContract, RSUN_HMI_METERS
    from pyampp.io import save_model_to_h5

    path = tmp_path / "with_contract.h5"
    model = {
        "corona": {
            "bx": np.zeros((4, 3, 2), dtype=np.float32),
            "by": np.zeros((4, 3, 2), dtype=np.float32),
            "bz": np.zeros((4, 3, 2), dtype=np.float32),
            "attrs": {"model_type": "nlfff"},
        },
        "metadata": {
            "geometry_contract": GeometryContract(
                nx=4,
                ny=3,
                nz=2,
                dr_x=0.001,
                dr_y=0.002,
                dr_z=0.003,
                rsun_m=RSUN_HMI_METERS,
                anchor_lon_deg=110.0,
                anchor_lat_deg=-15.0,
                anchor_radius_rsun=1.0,
                frame="heliographic_stonyhurst",
                obstime="2020-01-01T00:00:00",
                inferred_from="index",
            )
        },
        "observer": {
            "name": "earth",
            "label": "Earth",
            "ephemeris": {
                "obs_date": "2020-01-01T00:00:00",
                "hgln_obs_deg": 0.0,
                "hglt_obs_deg": 2.0,
            },
        },
    }
    save_model_to_h5(model, path)

    runner = CliRunner()
    result = runner.invoke(app, [str(path), "--json"])
    assert result.exit_code == 0
    assert '"has_geometry_contract": true' in result.stdout.lower()
    assert '"nx": 4' in result.stdout
    assert '"observer"' in result.stdout
