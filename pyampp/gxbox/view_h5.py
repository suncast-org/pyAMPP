#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import warnings
import tempfile
try:
    from pyvista import PyVistaDeprecationWarning
except Exception:
    PyVistaDeprecationWarning = DeprecationWarning
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import (
    Heliocentric,
    HeliographicCarrington,
    HeliographicStonyhurst,
    Helioprojective,
    get_earth,
)
from sunpy.sun import constants as sun_consts

from pyampp.gxbox.box import Box, BoxGeometryMixin
from pyampp.gxbox.boxutils import read_b3d_h5, normalize_observer_metadata
from pyampp.gxbox.gx_fov2box import _decode_id_text, _extract_execute_geometry, _infer_time_from_entry_loaded
from pyampp.io import load_model
from pyampp.gxbox.observer_restore import resolve_observer_with_info
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog
from PyQt5.QtCore import Qt, QTimer


def _contains_viewer_field_payload(b3d: dict) -> bool:
    if not isinstance(b3d, dict):
        return False

    for key in ("corona", "nlfff", "pot"):
        group = b3d.get(key)
        if isinstance(group, dict) and any(name in group for name in ("bx", "by", "bz", "bcube")):
            return True

    chromo = b3d.get("chromo")
    if isinstance(chromo, dict):
        if any(name in chromo for name in ("bx", "by", "bz", "bcube", "chromo_bcube")):
            return True

    return False


def _ensure_viewer_compatible_model(b3d: dict, model_path: Path) -> None:
    if _contains_viewer_field_payload(b3d):
        return
    raise ValueError(
        "Incompatible model file for gxbox-view3d: missing 3D field payload "
        f"(corona/chromo). File: {model_path}. "
        "This looks like a metadata-only thin HDF5 (e.g. export-model-metadata output), "
        "which is supported for geometry metadata workflows but not for 2D/3D viewer rendering."
    )


def _decode_meta_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, (bytes, bytearray)):
            return item.decode("utf-8", "ignore")
        return str(item)
    return str(value)


def _to_xyz_if_zyx(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return np.transpose(arr, (2, 1, 0))
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return np.transpose(arr, (2, 1, 0, 3))
    if arr.ndim == 4 and arr.shape[0] == 3:
        # (c, z, y, x) -> (x, y, z, c)
        return np.transpose(arr, (3, 2, 1, 0))
    return arr


def normalize_viewer_axis_order(b3d: dict) -> dict:
    """
    Convert canonical H5 zyx cubes into viewer xyz cubes.
    MagFieldViewer expects (x, y, z).
    """
    meta = b3d.get("metadata", {}) if isinstance(b3d, dict) else {}
    axis_order = _decode_meta_text(meta.get("axis_order_3d", "")).strip().lower()
    if axis_order != "zyx":
        return b3d

    for model_key in ("corona", "chromo", "nlfff", "pot"):
        if model_key not in b3d or not isinstance(b3d[model_key], dict):
            continue
        for comp in ("bx", "by", "bz", "bcube", "chromo_bcube"):
            if comp in b3d[model_key]:
                b3d[model_key][comp] = _to_xyz_if_zyx(np.asarray(b3d[model_key][comp]))
    return b3d


@dataclass
class SimpleBox(BoxGeometryMixin):
    dims_pix: np.ndarray
    res: u.Quantity
    b3d: dict
    _frame_obs: object
    _center: SkyCoord

    @property
    def grid_coords(self):
        dims = self.dims_pix
        dx = self.res
        x = np.linspace(-dims[0] / 2, dims[0] / 2, dims[0]) * dx
        y = np.linspace(-dims[1] / 2, dims[1] / 2, dims[1]) * dx
        z = np.linspace(-dims[2] / 2, dims[2] / 2, dims[2]) * dx
        return {"x": x, "y": y, "z": z, "frame": self._frame_obs}


def infer_dims(b3d: dict) -> np.ndarray:
    for key in ("corona", "nlfff", "pot"):
        if key in b3d and "bx" in b3d[key]:
            return np.array(b3d[key]["bx"].shape, dtype=int)
    if "chromo" in b3d:
        if "bx" in b3d["chromo"]:
            return np.array(b3d["chromo"]["bx"].shape, dtype=int)
        if "bcube" in b3d["chromo"]:
            return np.array(b3d["chromo"]["bcube"].shape[:3], dtype=int)
        if "chromo_bcube" in b3d["chromo"]:
            return np.array(b3d["chromo"]["chromo_bcube"].shape[1:4], dtype=int)
    raise ValueError("Unable to infer dimensions from HDF5.")


def infer_time(b3d: dict) -> Time:
    if "chromo" in b3d and "attrs" in b3d["chromo"]:
        attrs = b3d["chromo"]["attrs"]
        if "obs_time" in attrs:
            try:
                return Time(attrs["obs_time"])
            except Exception:
                pass
    return Time.now()


def infer_res(b3d: dict) -> u.Quantity:
    if "corona" in b3d and "dr" in b3d["corona"]:
        dr = b3d["corona"]["dr"]
        if dr is not None and np.size(dr) >= 1:
            return (dr[0] * sun_consts.radius.to(u.km).value) * u.km
    if "chromo" in b3d and "dr" in b3d["chromo"]:
        dr = b3d["chromo"]["dr"]
        if dr is not None and np.size(dr) >= 1:
            return (dr[0] * sun_consts.radius.to(u.km).value) * u.km
    return 1.0 * u.Mm


def _box_from_saved_model(b3d: dict, model_path: Path):
    meta = b3d.get("metadata", {}) if isinstance(b3d, dict) else {}
    execute_text = _decode_meta_text(meta.get("execute", "")) if isinstance(meta, dict) else ""
    coords, frame_mode, _projection = _extract_execute_geometry(execute_text)
    time_iso = _infer_time_from_entry_loaded(b3d, model_path)
    if not time_iso or coords is None:
        return None, None

    try:
        obs_time = Time(time_iso)
    except Exception:
        return None, None

    dims = infer_dims(b3d)
    res = infer_res(b3d).to(u.Mm)
    # Display/LoS observer (camera, projected views) comes from saved observer metadata.
    observer, observer_warning = _resolve_model_observer(b3d, obs_time)
    if observer_warning:
        print(f"Warning: {observer_warning}")
    frame_obs = Helioprojective(observer=observer, obstime=obs_time)
    # Geometry in EXECUTE comes from the original model definition (Earth frame).
    geom_observer = get_earth(obs_time)
    frame_geom = Helioprojective(observer=geom_observer, obstime=obs_time)

    try:
        if (frame_mode or "hpc").lower() == "hgc":
            box_origin = SkyCoord(
                lon=float(coords[0]) * u.deg,
                lat=float(coords[1]) * u.deg,
                radius=sun_consts.radius.to(u.Mm),
                obstime=obs_time,
                observer=geom_observer,
                frame=HeliographicCarrington,
            ).transform_to(frame_geom)
        elif (frame_mode or "hpc").lower() == "hgs":
            box_origin = SkyCoord(
                lon=float(coords[0]) * u.deg,
                lat=float(coords[1]) * u.deg,
                radius=sun_consts.radius.to(u.Mm),
                obstime=obs_time,
                frame=HeliographicStonyhurst,
            ).transform_to(frame_geom)
        else:
            box_origin = SkyCoord(
                Tx=float(coords[0]) * u.arcsec,
                Ty=float(coords[1]) * u.arcsec,
                frame=frame_geom,
            )
    except Exception:
        return None, None

    box_dims = np.array(dims, dtype=float) * u.pix
    box_hcc = Heliocentric(observer=box_origin, obstime=obs_time)
    box_center = box_origin.transform_to(box_hcc)
    box_dimensions = box_dims / u.pix * res
    box_center = SkyCoord(
        x=box_center.x,
        y=box_center.y,
        z=box_center.z + box_dimensions[2] / 2,
        frame=box_center.frame,
    )
    box = Box(frame_obs, box_origin, box_center, box_dims, res)
    return box, obs_time


def _resolve_model_observer(b3d: dict, obs_time: Time):
    observer_meta = b3d.get("observer", {}) if isinstance(b3d, dict) else {}
    if isinstance(observer_meta, dict):
        fov_box = observer_meta.get("fov_box", {})
        if isinstance(fov_box, dict):
            coord, warning, _used_key = resolve_observer_with_info(b3d, fov_box.get("observer_key"), obs_time)
            return coord, warning
        if observer_meta.get("name") is not None:
            coord, warning, _used_key = resolve_observer_with_info(b3d, observer_meta.get("name"), obs_time)
            return coord, warning
    coord, warning, _used_key = resolve_observer_with_info(b3d, "earth", obs_time)
    return coord, warning


def _normalize_vector(vec: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size != 3:
        return None
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0:
        return None
    return arr / norm


def _viewer_camera_vectors(box, obs_time: Time) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Reproduce the legacy GxBox LoS camera basis for standalone HDF5 viewing.
    """
    box_origin = getattr(box, "_origin", None)
    frame_obs = getattr(box, "_frame_obs", None)
    box_view_up = getattr(box, "box_view_up", None)
    observer = getattr(frame_obs, "observer", None)
    if box_origin is None or frame_obs is None or observer is None or box_view_up is None:
        return None, None
    try:
        box_norm_direction = _normalize_vector(
            box_origin.transform_to(
                Heliocentric(observer=observer, obstime=obs_time)
            ).cartesian.xyz.value
        )
        if box_norm_direction is None:
            return None, None
        up_coords = np.diff(
            box_view_up.transform_to(
                Heliocentric(observer=observer, obstime=obs_time)
            ).cartesian.xyz.value
        )
        view_up = _normalize_vector(np.squeeze(up_coords))
        if view_up is None:
            return box_norm_direction, None
        if abs(view_up[1]) > 0:
            view_up = np.sign(view_up[1]) * view_up
        return box_norm_direction, view_up
    except Exception:
        return None, None


def can_prepare_model_for_viewer(model_path: str | Path) -> bool:
    """
    Return ``True`` when a file looks like a viewer-compatible saved model.

    This is a lightweight gate for enabling UI actions. It intentionally avoids
    full SAV conversion and only checks file existence, supported suffix, and the
    presence of recognizable top-level model groups for HDF5 files.
    """
    try:
        model_path = Path(model_path).expanduser().resolve()
    except Exception:
        return False
    if not model_path.exists() or not model_path.is_file():
        return False
    suffix = model_path.suffix.lower()
    if suffix == ".sav":
        return True
    if suffix != ".h5":
        return False
    try:
        b3d = read_b3d_h5(str(model_path))
    except Exception:
        return False
    return _contains_viewer_field_payload(b3d)


def prepare_model_for_viewer(model_path: str | Path) -> tuple[SimpleBox, Time, str, Path | None]:
    """
    Load a saved model file into the in-memory objects expected by ``MagFieldViewer``.

    Returns
    -------
    tuple
        ``(box, obs_time, b3dtype, temp_h5_path)`` where ``temp_h5_path`` is a
        temporary conversion artifact when the canonical loader had to materialize one.
    """
    model_path = Path(model_path).expanduser().resolve()
    temp_h5_path = None
    
    try:
        b3d, temp_h5_path = load_model(model_path, keep_temp_h5=True)
    except Exception as exc:
        raise RuntimeError(
            "Model input could not be loaded through pyampp.io.load_model. "
            "If this is a legacy SAV file, verify the conversion dependencies are available."
        ) from exc
    if temp_h5_path is not None:
        print(f"Materialized temporary HDF5: {temp_h5_path}")

    _ensure_viewer_compatible_model(b3d, model_path)
    
    b3d = normalize_viewer_axis_order(b3d)

    box, obs_time = _box_from_saved_model(b3d, model_path)
    if box is None:
        dims = infer_dims(b3d)
        obs_time = infer_time(b3d)
        res = infer_res(b3d)
        observer, observer_warning = _resolve_model_observer(b3d, obs_time)
        if observer_warning:
            print(f"Warning: {observer_warning}")
        frame = Heliocentric(observer=observer, obstime=obs_time)
        center = SkyCoord(0 * u.Mm, 0 * u.Mm, 0 * u.Mm, frame=frame)
        box = SimpleBox(dims_pix=dims, res=res.to(u.Mm), b3d=b3d, _frame_obs=frame, _center=center)

    if "corona" in b3d:
        b3dtype = "corona"
    elif "nlfff" in b3d:
        b3dtype = "corona"
        b3d["corona"] = b3d.pop("nlfff")
    elif "pot" in b3d:
        b3dtype = "corona"
        b3d["corona"] = b3d.pop("pot")
    elif "chromo" in b3d:
        b3dtype = "chromo"
        chromo = b3d.get("chromo", {})
        if "bx" not in chromo and "bcube" in chromo:
            bcube = chromo["bcube"]
            if bcube.ndim == 4 and bcube.shape[-1] == 3:
                chromo["bx"] = bcube[:, :, :, 0]
                chromo["by"] = bcube[:, :, :, 1]
                chromo["bz"] = bcube[:, :, :, 2]
                b3d["chromo"] = chromo
    else:
        raise ValueError("No known model types found in HDF5 (expected corona/chromo).")

    box.b3d = b3d
    if hasattr(box, "corona_type"):
        if "corona" in b3d and isinstance(b3d["corona"], dict):
            attrs = b3d["corona"].get("attrs", {})
            if isinstance(attrs, dict):
                box.corona_type = attrs.get("model_type")
            if hasattr(box, "corona_models") and box.corona_type:
                box.corona_models[box.corona_type] = b3d["corona"]

    return box, obs_time, b3dtype, temp_h5_path


def main() -> int:
    from pyampp.gxbox.box_view3d import MagFieldViewer

    parser = argparse.ArgumentParser(description="Open a saved HDF5 model in the 3D viewer without recomputing.")
    parser.add_argument("h5_path", nargs="?", help="Path to the HDF5 model file (positional).")
    parser.add_argument("--h5", dest="h5_opt", help="Path to the HDF5 model file.")
    parser.add_argument("--dir", dest="start_dir", help="Initial directory for file picker when no model path is given.")
    parser.add_argument("--pick", action="store_true", help="Open file picker even when model path is provided.")
    parser.add_argument("--pipeline-child", action="store_true", help="Open in restricted pyAMPP child mode tied to the current model file.")
    args = parser.parse_args()

    h5_arg = args.h5_opt or args.h5_path
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication([])
        owns_app = True

    if args.pick or not h5_arg:
        start_dir = Path(args.start_dir).expanduser() if args.start_dir else Path.cwd()
        if not start_dir.exists() or not start_dir.is_dir():
            start_dir = Path.cwd()
        dialog = QFileDialog(None, "Open Model (HDF5 or SAV)")
        # Native macOS picker may ignore selectFile(); use Qt dialog for reliable preselection.
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Model Files (*.h5 *.sav);;HDF5 Files (*.h5);;SAV Files (*.sav);;All Files (*)")
        if h5_arg:
            candidate = Path(h5_arg).expanduser()
            dialog.setDirectory(str(candidate.parent if candidate.parent.exists() else start_dir))
            dialog.selectFile(str(candidate.name))
        else:
            dialog.setDirectory(str(start_dir))
        if not dialog.exec_():
            return 0
        selected = dialog.selectedFiles()
        if not selected:
            return 0
        h5_arg = selected[0]

    model_path = Path(h5_arg).expanduser().resolve()
    loading_dialog = QProgressDialog(None, None, 0, 0, None)
    loading_dialog.setWindowTitle("Opening Model")
    loading_dialog.setLabelText(f"Loading {model_path.name}...")
    loading_dialog.setCancelButton(None)
    loading_dialog.setMinimumDuration(0)
    loading_dialog.setWindowModality(Qt.ApplicationModal)
    loading_dialog.setAutoClose(False)
    loading_dialog.setAutoReset(False)
    loading_dialog.show()
    QApplication.setOverrideCursor(Qt.WaitCursor)
    app.processEvents()
    try:
        box, obs_time, b3dtype, temp_h5_path = prepare_model_for_viewer(model_path)
    except Exception as exc:
        loading_dialog.close()
        QApplication.restoreOverrideCursor()
        try:
            QMessageBox.critical(None, "Incompatible Model", str(exc))
        except Exception:
            print(f"Error: {exc}")
        return 2
    finally:
        if loading_dialog.isVisible():
            loading_dialog.close()
        QApplication.restoreOverrideCursor()
        app.processEvents()

    warnings.filterwarnings("ignore", category=PyVistaDeprecationWarning)
    save_target = model_path if temp_h5_path is None else None
    box_norm_direction, box_view_up = _viewer_camera_vectors(box, obs_time)
    viewer = MagFieldViewer(
        box,
        time=obs_time,
        b3dtype=b3dtype,
        parent=None,
        model_path=save_target,
        source_model_path=model_path,
        box_norm_direction=box_norm_direction,
        box_view_up=box_view_up,
        session_mode="pipeline_child" if args.pipeline_child else "standalone",
    )
    if hasattr(viewer, "app_window"):
        viewer.app_window.setWindowTitle(f"GxBox 3D viewer - {model_path}")
        viewer.app_window.show()
        viewer.app_window.showNormal()
        if hasattr(viewer, "ensure_window_visible"):
            viewer.ensure_window_visible()
        viewer.app_window.raise_()
        viewer.app_window.activateWindow()
    viewer.show()
    if hasattr(viewer, "schedule_startup_los_view"):
        viewer.schedule_startup_los_view()
    else:
        QTimer.singleShot(0, viewer.set_camera_to_LOS_direction)
    if owns_app:
        app.exec_()
    # Temporary conversion artifact can be removed after viewer exits.
    if temp_h5_path is not None:
        try:
            temp_h5_path.unlink(missing_ok=True)
            temp_h5_path.parent.rmdir()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
