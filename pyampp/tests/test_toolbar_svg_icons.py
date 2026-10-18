"""Toolbar SVGs must ship inside the package so a wheel install can find them."""

from __future__ import annotations

from pathlib import Path

import pyampp.gxbox.box_view2d as box_view2d
from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
from pyampp.gxbox.gxampp import PyAmppGUI

# Names used by the 2D toolbar and the gxampp command buttons.
_TOOLBAR_SVGS = (
    "arrow-down.svg",
    "arrow-left.svg",
    "arrow-right.svg",
    "arrow-up.svg",
    "box.svg",
    "expand-horizontal.svg",
    "expand-vertical.svg",
    "expand.svg",
    "monitor.svg",
    "rectangle-horizontal.svg",
    "shrink-horizontal.svg",
    "shrink-vertical.svg",
    "shrink.svg",
    "zoom-in.svg",
    "zoom-out.svg",
)


def test_toolbar_svg_dir_prefers_packaged_icons():
    packaged = Path(box_view2d.__file__).resolve().parent / "icons"
    box_dir = MapBoxDisplayWidget._resolve_svg_dir()
    gui_dir = PyAmppGUI._resolve_svg_dir()

    assert box_dir == packaged
    assert gui_dir == packaged

    docs_svg = Path(__file__).resolve().parents[2] / "docs" / "svg"
    for name in _TOOLBAR_SVGS:
        icon = packaged / name
        assert icon.is_file()
        assert "<svg" in icon.read_text(encoding="utf-8")
        if docs_svg.is_dir():
            assert icon.read_bytes() == (docs_svg / name).read_bytes()
