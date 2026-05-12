from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from pyampp.tests._fits_header import canonical_base_index_header
from pyampp.util.show_model_metadata import app


def test_show_model_metadata_restores_missing_contract_and_observer(tmp_path):
    from pyampp.io import save_model

    path = tmp_path / "missing_contract.h5"
    model = {
        "corona": {
            "bx": np.zeros((2, 2, 2), dtype=np.float32),
            "by": np.zeros((2, 2, 2), dtype=np.float32),
            "bz": np.zeros((2, 2, 2), dtype=np.float32),
            "attrs": {"model_type": "nlfff"},
        },
        "base": {
            "index": canonical_base_index_header(date_obs="2020-01-01T00:00:00"),
            "bx": np.zeros((2, 2), dtype=np.float32),
            "by": np.zeros((2, 2), dtype=np.float32),
            "bz": np.zeros((2, 2), dtype=np.float32),
            "ic": np.ones((2, 2), dtype=np.float32),
        },
    }
    save_model(model, path)

    runner = CliRunner()
    result = runner.invoke(app, [str(path), "--json"])
    assert result.exit_code == 0
    assert '"has_geometry_contract": true' in result.stdout.lower()
    assert '"observer"' in result.stdout
    assert '"has_ephemeris": true' in result.stdout.lower()
    assert '"ephemeris_keys"' in result.stdout


def test_show_model_metadata_reports_contract_and_observer_json(tmp_path):
    from pyampp.geometry.contract import GeometryContract, RSUN_HMI_METERS
    from pyampp.io import save_model

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
        "base": {
            "index": canonical_base_index_header(date_obs="2020-01-01T00:00:00"),
            "bx": np.zeros((4, 3), dtype=np.float32),
            "by": np.zeros((4, 3), dtype=np.float32),
            "bz": np.zeros((4, 3), dtype=np.float32),
            "ic": np.ones((4, 3), dtype=np.float32),
        },
    }
    save_model(model, path)

    runner = CliRunner()
    result = runner.invoke(app, [str(path), "--json"])
    assert result.exit_code == 0
    assert '"has_geometry_contract": true' in result.stdout.lower()
    assert '"nx": 4' in result.stdout
    assert '"observer"' in result.stdout