from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from pyampp.util.h5tree import app


def _make_saved_model() -> dict:
    return {
        "corona": {
            "bx": np.zeros((2, 3, 4), dtype=np.float32),
            "by": np.zeros((2, 3, 4), dtype=np.float32),
            "bz": np.zeros((2, 3, 4), dtype=np.float32),
            "attrs": {"model_type": "nlfff"},
        },
        "metadata": {"id": "normalized_from_sav"},
        "observer": {"name": "earth"},
    }


def test_h5tree_reads_h5_directly(tmp_path):
    from pyampp.io import save_model

    path = tmp_path / "model.h5"
    save_model(_make_saved_model(), path)

    runner = CliRunner()
    result = runner.invoke(app, [str(path), "--no-metadata"])

    assert result.exit_code == 0
    assert str(path) in result.stdout
    assert "corona/" in result.stdout
    assert "bx (2, 3, 4) float32" in result.stdout


def test_h5tree_normalizes_sav_through_load_and_save(tmp_path, monkeypatch):
    from pyampp.util import h5tree

    sav_path = tmp_path / "model.sav"
    sav_path.write_bytes(b"sav")
    persisted_path = tmp_path / "normalized.h5"
    calls: list[tuple[str, Path]] = []

    def fake_load_model(path):
        calls.append(("load", Path(path)))
        return _make_saved_model()

    def fake_save_model(model, path):
        from pyampp.io import save_model as real_save_model

        calls.append(("save", Path(path)))
        real_save_model(model, path)

    monkeypatch.setattr(h5tree, "load_model", fake_load_model)
    monkeypatch.setattr(h5tree, "save_model", fake_save_model)

    runner = CliRunner()
    result = runner.invoke(app, [str(sav_path), "--save-normalized", str(persisted_path), "--no-metadata"])

    assert result.exit_code == 0
    assert calls == [("load", sav_path), ("save", persisted_path)]
    assert persisted_path.exists()
    assert str(persisted_path) in result.stdout
    assert "bx (2, 3, 4) float32" in result.stdout


def test_h5tree_sav_without_save_normalized_does_not_write_h5(tmp_path, monkeypatch):
    from pyampp.util import h5tree

    sav_path = tmp_path / "model.sav"
    sav_path.write_bytes(b"sav")
    calls: list[tuple[str, Path]] = []

    def fake_load_model(path):
        calls.append(("load", Path(path)))
        return _make_saved_model()

    def fake_save_model(model, path):
        raise AssertionError("save_model should not be called for default SAV inspection")

    monkeypatch.setattr(h5tree, "load_model", fake_load_model)
    monkeypatch.setattr(h5tree, "save_model", fake_save_model)

    runner = CliRunner()
    result = runner.invoke(app, [str(sav_path), "--no-metadata"])

    assert result.exit_code == 0
    assert calls == [("load", sav_path)]
    assert f"{sav_path} (normalized view)" in result.stdout
    assert "bx (2, 3, 4) float32" in result.stdout