from __future__ import annotations

import stat
from pathlib import Path

from pyampp.util.demo_notebooks import copy_demo_bundle, filter_protected_paths, is_protected_demo_path


def test_is_protected_demo_path_matches_notebook_assets() -> None:
    assert is_protected_demo_path("docs/notebooks/demo.ipynb")
    assert is_protected_demo_path("docs/notebooks/data/demo_metadata.h5")
    assert not is_protected_demo_path("docs/notebooks/notes.txt")
    assert not is_protected_demo_path("docs/other/demo.ipynb")


def test_filter_protected_paths_keeps_only_notebook_assets() -> None:
    paths = [
        "docs/notebooks/demo.ipynb",
        "docs/notebooks/data/demo_metadata.h5",
        "docs/notebooks/notes.txt",
        "README.rst",
    ]
    assert filter_protected_paths(paths) == [
        "docs/notebooks/demo.ipynb",
        "docs/notebooks/data/demo_metadata.h5",
    ]


def test_copy_demo_bundle_makes_files_writable(tmp_path: Path) -> None:
    source_root = tmp_path / "docs" / "notebooks"
    data_root = source_root / "data"
    data_root.mkdir(parents=True)
    notebook = source_root / "demo.ipynb"
    fixture = data_root / "demo_metadata.h5"
    notebook.write_text("notebook", encoding="utf-8")
    fixture.write_bytes(b"fixture")
    notebook.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    fixture.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    dest_root = tmp_path / "playground"
    copied = copy_demo_bundle(source_root, dest_root)

    copied_notebook = dest_root / "demo.ipynb"
    copied_fixture = dest_root / "data" / "demo_metadata.h5"
    assert copied_notebook.exists()
    assert copied_fixture.exists()
    assert copied_notebook.read_text(encoding="utf-8") == "notebook"
    assert copied_fixture.read_bytes() == b"fixture"
    assert copied_notebook.stat().st_mode & stat.S_IWUSR
    assert copied_fixture.stat().st_mode & stat.S_IWUSR
    assert copied == [copied_fixture, copied_notebook]