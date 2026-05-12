from __future__ import annotations

import ast
from pathlib import Path


RUNTIME_MODEL_ENTRYPOINTS = (
    Path("pyampp/gxbox/gx_fov2box.py"),
    Path("pyampp/gxbox/view_h5.py"),
    Path("pyampp/gxbox/gxrefmap_view.py"),
    Path("pyampp/gxbox/gxbox_selector_view.py"),
    Path("pyampp/gxbox/boxutils.py"),
    Path("pyampp/gxbox/box_view3d.py"),
    Path("pyampp/util/export_model.py"),
)


def _module_tree(path: Path) -> ast.AST:
    root = Path(__file__).resolve().parents[2]
    source = (root / path).read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path))


def test_runtime_entrypoints_use_load_model_only() -> None:
    forbidden_names = {"load_model_from_h5", "load_model_from_sav", "_load_model_h5", "_load_model_sav"}
    alternate_loader_calls = {
        Path("pyampp/gxbox/box_view3d.py"): {"prepare_model_for_viewer"},
    }

    for module_path in RUNTIME_MODEL_ENTRYPOINTS:
        tree = _module_tree(module_path)

        imported_from_io = set()
        called_names = set()
        referenced_names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pyampp.io":
                imported_from_io.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node, ast.Name):
                referenced_names.add(node.id)

        assert forbidden_names.isdisjoint(imported_from_io), f"{module_path} imports suffix-specific model loaders"
        assert forbidden_names.isdisjoint(referenced_names), f"{module_path} references suffix-specific model loaders"
        assert "load_model" in imported_from_io, f"{module_path} must import pyampp.io.load_model"
        allowed_calls = {"load_model"} | alternate_loader_calls.get(module_path, set())
        assert called_names.intersection(allowed_calls), (
            f"{module_path} must route model loading through pyampp.io.load_model"
        )