from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import typer
from astropy.io import fits
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import colors
import sunpy.map
from sunpy.visualization import colormaps as sunpy_colormaps
from pyampp.io import load_model

from .boxutils import map_from_data_header_compat

app = typer.Typer(help="View refmaps and base maps stored in a model HDF5 file.")


@dataclass
class MapSpec:
    group: str
    name: str
    data_path: str
    wcs_path: str


_AIA_REF_CMAPS = {
    "AIA_94": "sdoaia94",
    "AIA_131": "sdoaia131",
    "AIA_1600": "sdoaia1600",
    "AIA_1700": "sdoaia1700",
    "AIA_171": "sdoaia171",
    "AIA_193": "sdoaia193",
    "AIA_211": "sdoaia211",
    "AIA_304": "sdoaia304",
    "AIA_335": "sdoaia335",
}
_BW_SIGNED_REFMAPS = {"bx", "by", "bz", "Bz_reference"}
_BW_SCALAR_REFMAPS = {"ic", "Ic_reference"}


def _decode_h5_string(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind in ("S", "O"):
        item = value[()]
        if isinstance(item, (bytes, bytearray)):
            return item.decode()
        return str(item)
    return str(value)


def _safe_header_from_string(text: str) -> fits.Header:
    header = fits.Header()
    for raw in text.splitlines():
        if not raw:
            continue
        key = raw[:8].strip()
        if not key or key in ("END", "CONTINUE", "COMMENT", "HISTORY"):
            continue
        line = raw
        if len(line) < 80:
            line = line.ljust(80)
        if len(line) > 80:
            line = line[:80]
        # Only parse FITS value cards (column 9 is '=').
        # This avoids warnings when 'base/index' contains an IDL tuple dump.
        if len(line) < 10 or line[8] != "=":
            continue
        try:
            card = fits.Card.fromstring(line)
        except Exception:
            continue
        header.append(card, end=True)
    return header


def _collect_maps(h5f: h5py.File) -> list[MapSpec]:
    specs: list[MapSpec] = []

    base_header_path = None
    if "base" in h5f:
        if "index" in h5f["base"]:
            base_header_path = "base/index"
        elif "index_header" in h5f["base"]:
            base_header_path = "base/index_header"
        elif "wcs_header" in h5f["base"]:
            base_header_path = "base/wcs_header"

    if base_header_path is not None:
        for key in ("bx", "by", "bz", "ic", "chromo_mask"):
            if key in h5f["base"]:
                specs.append(
                    MapSpec(
                        group="base",
                        name=key,
                        data_path=f"base/{key}",
                        wcs_path=base_header_path,
                    )
                )

    if "refmaps" in h5f:
        for name in sorted(h5f["refmaps"].keys()):
            group = h5f["refmaps"][name]
            if "data" in group and "wcs_header" in group:
                specs.append(
                    MapSpec(
                        group="refmaps",
                        name=name,
                        data_path=f"refmaps/{name}/data",
                        wcs_path=f"refmaps/{name}/wcs_header",
                    )
                )

    return specs


def _resolve_model_to_h5(path: Path) -> tuple[Path, Optional[Path]]:
    """Return an HDF5 path, materializing a temporary HDF5 when the canonical loader needs one."""
    path = path.expanduser().resolve()
    try:
        _model_dict, tmp_h5 = load_model(path, keep_temp_h5=True)
    except Exception as exc:
        raise RuntimeError(
            "Model input could not be materialized through pyampp.io.load_model. "
            "Run canonical conversion to H5 manually, then reopen, if legacy conversion dependencies are unavailable."
        ) from exc
    if tmp_h5 is not None:
        print(f"Materialized temporary HDF5: {tmp_h5}")
        return tmp_h5, tmp_h5
    return path, None


class RefmapViewer(QtWidgets.QMainWindow):
    def __init__(self, h5_path: Path, start: Optional[str] = None):
        super().__init__()
        self.h5_path = h5_path
        self.setWindowTitle(f"gxrefmap-view: {h5_path.name}")
        self._data_min = None
        self._data_max = None
        self._suppress_slider = False

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.selector = QtWidgets.QComboBox()
        layout.addWidget(self.selector)

        controls = QtWidgets.QHBoxLayout()
        self.vmin_label = QtWidgets.QLabel("Vmin:")
        self.vmin_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.vmin_slider.setMinimum(0)
        self.vmin_slider.setMaximum(1000)
        self.vmin_value = QtWidgets.QLabel("")
        self.vmax_label = QtWidgets.QLabel("Vmax:")
        self.vmax_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.vmax_slider.setMinimum(0)
        self.vmax_slider.setMaximum(1000)
        self.vmax_value = QtWidgets.QLabel("")
        self.log_checkbox = QtWidgets.QCheckBox("Log scale")

        controls.addWidget(self.vmin_label)
        controls.addWidget(self.vmin_slider)
        controls.addWidget(self.vmin_value)
        controls.addWidget(self.vmax_label)
        controls.addWidget(self.vmax_slider)
        controls.addWidget(self.vmax_value)
        controls.addWidget(self.log_checkbox)
        layout.addLayout(controls)

        self.figure = Figure(figsize=(7, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        with h5py.File(self.h5_path, "r") as h5f:
            self.map_specs = _collect_maps(h5f)

        for spec in self.map_specs:
            label = f"{spec.group}/{spec.name}"
            self.selector.addItem(label)

        self.selector.currentIndexChanged.connect(self._update_plot)
        self.vmin_slider.valueChanged.connect(self._update_plot)
        self.vmax_slider.valueChanged.connect(self._update_plot)
        self.log_checkbox.stateChanged.connect(self._update_plot)

        if start:
            for idx, spec in enumerate(self.map_specs):
                if spec.name == start or f"{spec.group}/{spec.name}" == start:
                    self.selector.setCurrentIndex(idx)
                    break

        self._update_plot()

    def _update_plot(self) -> None:
        idx = self.selector.currentIndex()
        if idx < 0 or idx >= len(self.map_specs):
            return
        spec = self.map_specs[idx]

        try:
            with h5py.File(self.h5_path, "r") as h5f:
                data = h5f[spec.data_path][()]
                wcs_header = _decode_h5_string(h5f[spec.wcs_path][()])
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to read {spec.group}/{spec.name}: {exc}")
            return

        header = _safe_header_from_string(wcs_header)
        self.figure.clear()
        cmap = None
        norm = None
        if spec.name in _AIA_REF_CMAPS:
            cmap = sunpy_colormaps.cm.cmlist.get(_AIA_REF_CMAPS[spec.name], None)
        elif spec.name in _BW_SIGNED_REFMAPS:
            cmap = "gray"
        elif spec.name in _BW_SCALAR_REFMAPS:
            cmap = "gray"
        elif spec.name == "chromo_mask":
            cmap = colors.ListedColormap([
                "#000000", "#1f77b4", "#ff7f0e", "#2ca02c",
                "#d62728", "#9467bd", "#8c564b", "#e377c2",
                "#7f7f7f", "#bcbd22"
            ])
            norm = colors.BoundaryNorm(np.arange(0.5, 10.5, 1), cmap.N)
        elif spec.name == "Vert_current":
            cmap = "RdBu_r"

        data_min = float(np.nanmin(data))
        data_max = float(np.nanmax(data))
        if self._data_min != data_min or self._data_max != data_max:
            self._data_min = data_min
            self._data_max = data_max
            self._suppress_slider = True
            self.vmin_slider.setValue(0)
            self.vmax_slider.setValue(1000)
            self._suppress_slider = False

        if not self._suppress_slider:
            smin = self.vmin_slider.value()
            smax = self.vmax_slider.value()
            if smin > smax:
                self._suppress_slider = True
                if self.sender() is self.vmin_slider:
                    smin = smax
                    self.vmin_slider.setValue(smin)
                else:
                    smax = smin
                    self.vmax_slider.setValue(smax)
                self._suppress_slider = False
            vmin = self._data_min + (self._data_max - self._data_min) * (smin / 1000.0)
            vmax = self._data_min + (self._data_max - self._data_min) * (smax / 1000.0)
            self.vmin_value.setText(f"{vmin:.3g}")
            self.vmax_value.setText(f"{vmax:.3g}")
        else:
            vmin = self._data_min
            vmax = self._data_max

        log_ok = vmin > 0 and vmax > 0
        if spec.name in ("chromo_mask", "bx", "by", "bz", "Bz_reference", "Vert_current"):
            log_ok = False
        self.log_checkbox.setEnabled(log_ok)
        if not log_ok:
            self.log_checkbox.setChecked(False)

        if spec.name not in ("chromo_mask",):
            if self.log_checkbox.isChecked() and vmin > 0:
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                if vmin < 0 < vmax:
                    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                else:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)

        im = None
        used_sunpy = False
        try:
            smap = map_from_data_header_compat(data, header)
            ax = self.figure.add_subplot(111, projection=smap)
            im = smap.plot(axes=ax, cmap=cmap, norm=norm)
            used_sunpy = True
        except Exception:
            try:
                from astropy.wcs import WCS
                wcs = WCS(header)
                ax = self.figure.add_subplot(111, projection=wcs)
                im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm)
            except Exception:
                ax = self.figure.add_subplot(111)
                im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm)

        if not used_sunpy:
            ax.set_xlabel("X [arcsec]")
            ax.set_ylabel("Y [arcsec]")
        ax.set_title(f"{spec.group}/{spec.name} (min={self._data_min:.3g}, max={self._data_max:.3g})")
        if im is not None:
            try:
                self.figure.colorbar(im, ax=ax, orientation="vertical", shrink=0.85, pad=0.02)
            except Exception:
                # Colorbar rendering is cosmetic; ignore any errors to avoid breaking the viewer.
                # Colorbar rendering is cosmetic; keep viewer usable if it fails.
                pass
        self.canvas.draw()


@app.command()
def main(
    ctx: typer.Context,
    h5_path: Optional[Path] = typer.Argument(None, exists=True, file_okay=True, dir_okay=False, readable=True),
    h5: Optional[Path] = typer.Option(
        None,
        "--h5",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the model HDF5 file.",
    ),
    start: Optional[str] = typer.Option(None, "--start", help="Initial map name to display."),
    list_only: bool = typer.Option(False, "--list", help="List available maps and exit."),
) -> None:
    if h5_path is None and h5 is None:
        print(ctx.get_help())
        raise typer.Exit(code=0)
    if h5_path is None:
        h5_path = h5
    temp_h5 = None
    h5_path, temp_h5 = _resolve_model_to_h5(h5_path)

    with h5py.File(h5_path, "r") as h5f:
        specs = _collect_maps(h5f)
    if list_only:
        for spec in specs:
            print(f"{spec.group}/{spec.name}")
        raise typer.Exit(code=0)

    app_qt = QtWidgets.QApplication.instance()
    if app_qt is None:
        app_qt = QtWidgets.QApplication([])

    viewer = RefmapViewer(h5_path, start=start)
    viewer.resize(900, 700)
    viewer.show()
    app_qt.exec_()
    if temp_h5 is not None:
        try:
            temp_h5.unlink(missing_ok=True)
            temp_h5.parent.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    app()
