import copy
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QLabel, \
    QPushButton, QDoubleSpinBox, QLineEdit, QCheckBox, QMessageBox, QMenu, QHeaderView, QFileDialog, QAction, QToolButton, \
    QToolBar, QGridLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QGuiApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import Heliocentric, Helioprojective
from pyampp.gxbox.boxutils import validate_number, read_b3d_h5, write_b3d_h5, update_line_seeds_h5
from pyampp.gxbox.observer_restore import resolve_observer_with_info
import pickle
import vtk

import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTreeView, \
    QGroupBox
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import numpy as np

logging.getLogger("sunpy").setLevel(logging.WARNING)

## todo is it possible to add 3d crosshair to the plotter?
## todo integrate NLFFF extrapolation module. https://github.com/Alexey-Stupishin/pyAMaFiL
def minval(min_val):
    """
    Rounds the minimum value to the nearest hundredth.

    :param min_val: float
        The minimum value to round.
    :return: float
        The rounded minimum value.
    """
    return np.ceil(min_val * 100) / 100


def maxval(max_val):
    """
    Rounds the maximum value to the nearest hundredth.

    :param max_val: float
        The maximum value to round.
    :return: float
        The rounded maximum value.
    """
    return np.floor(max_val * 100) / 100


def _decode_seed_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_seed_value(value.item())
        if value.size == 1:
            return _decode_seed_value(value.reshape(-1)[0].item())
    return value


def generate_streamlines_from_line_seeds(box, b3dtype, line_seeds):
    if not isinstance(line_seeds, dict):
        return [], 0.0

    x = np.asarray(box.grid_coords["x"].value, dtype=float)
    y = np.asarray(box.grid_coords["y"].value, dtype=float)
    z_native = np.asarray(box.grid_coords["z"].value, dtype=float)
    z_base = float(z_native.min())
    z = z_native - z_base

    bx = np.asarray(box.b3d[b3dtype]["bx"])
    by = np.asarray(box.b3d[b3dtype]["by"])
    bz = np.asarray(box.b3d[b3dtype]["bz"])

    grid = pv.ImageData()
    grid.dimensions = (len(x), len(y), len(z))
    grid.spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    grid.origin = (x.min(), y.min(), z.min())
    grid["bx"] = bx.ravel(order="F")
    grid["by"] = by.ravel(order="F")
    grid["bz"] = bz.ravel(order="F")
    grid["vectors"] = np.c_[grid["bx"], grid["by"], grid["bz"]]

    streamlines = []
    for key, seed_data in sorted(line_seeds.items()):
        if key == "attrs" or not isinstance(seed_data, dict):
            continue
        seed_type = str(_decode_seed_value(seed_data.get("seed_type", "sphere")))
        if seed_type != "sphere":
            continue
        center = np.asarray(seed_data.get("center", ()), dtype=float).reshape(-1)
        if center.size != 3:
            continue
        radius = float(_decode_seed_value(seed_data.get("radius", 0.0)))
        n_points = int(_decode_seed_value(seed_data.get("n_points", 100)))
        sl = grid.streamlines(
            vectors="vectors",
            source_center=(float(center[0]), float(center[1]), float(center[2])),
            source_radius=radius,
            n_points=n_points,
            integration_direction="both",
            max_length=5000,
            progress_bar=False,
        )
        if sl is not None and getattr(sl, "n_lines", 0) > 0:
            streamlines.append(sl)
    return streamlines, z_base


class MagFieldViewer(BackgroundPlotter):
    """
    A class to visualize the magnetic field of a box using PyVista. It inherits from the BackgroundPlotter class.

    :param box: object
        The box containing magnetic field data.
    :param parent: object, optional
        The parent object (default is None).
    """

    def __init__(self, box, parent=None, box_norm_direction=None, box_view_up=None, time=None, b3dtype='nlfff', model_path=None, session_mode=None, *args, **kwargs):
        # Build the scene fully before first paint; callers explicitly call .show().
        kwargs.setdefault("show", False)
        super().__init__(*args, **kwargs)
        self.box = box
        self.parent = parent
        self.model_path = model_path
        self.session_mode = session_mode or ("embedded" if parent is not None else "standalone")
        self.box_norm_direction = box_norm_direction
        self.box_view_up = box_view_up
        self.updating_flag = False  # Flag to avoid recursion
        self.spheres = {}
        self.current_sphere_id = None
        self.next_sphere_id = 1
        self.current_sphere = None
        self.sphere_actor = None
        self.sphere = None
        self.axes_widget = None
        self.plane_actor = None
        self.bottom_slice_actor = None
        self.base_map_actor = None
        self.model_box_actor = None
        self.fov_box_actor = None
        self.streamlines_actor = None
        self.streamlines = None
        self.sphere_visible = True
        self.slice_visible = True
        self.base_map_visible = False
        self.model_box_visible = True
        self.fov_box_visible = True
        self.plane_visible = True
        self.use_interp = True
        self.scalar = 'bz'
        self.previous_params = {}
        self.previous_valid_values = {}
        self.scalar_selector = None
        self.slice_checkbox = None
        self.slice_axis_selector = None
        self.slice_coord_label = None
        self.scalar_selector_items = []
        self.base_map_selector = None
        self.model_box_checkbox = None
        self.fov_box_checkbox = None
        self.base_map_checkbox = None
        self.base_map_items = []
        self.center_x_input = None
        self.center_y_input = None
        self.center_z_input = None
        self.radius_input = None
        self.n_points_input = None
        self.slice_z_input = None
        self.vmin_input = None
        self.vmax_input = None
        self.slice_z_min = 0.0
        self.slice_z_max = 0.0
        self.scalar_min = 0.0
        self.scalar_max = 0.0
        self.base_scalar_min = 0.0
        self.base_scalar_max = 0.0
        self.update_button = None
        self.send_button = None
        self.save_model_button = None
        self.save_close_button = None
        self.cancel_button = None
        self.save_as_button = None
        self.parallel_proj_button = None
        self.field_lines_control_group = None
        self.sphere_control_group = None
        self._streamline_controls_enabled = True
        self._los_label_text = ""
        self._embedded_close_mode = None
        self._close_hook_installed = False
        self.base_vmin_input = None
        self.base_vmax_input = None
        self.timestr = time.to_datetime().strftime("_%Y%m%dT%H%M%S") if time is not None else ''
        self._restoring_line_seeds = False
        self._original_line_seeds = copy.deepcopy(self.box.b3d.get("line_seeds")) if isinstance(self.box.b3d.get("line_seeds"), dict) else None
        if b3dtype in ("pot", "nlfff"):
            self.b3dtype = "corona"
            self.corona_type = b3dtype
        else:
            self.b3dtype = b3dtype
            self.corona_type = None
        # self.sphere_checkbox = None
        self.grid_x = self.box.grid_coords['x'].value
        self.grid_y = self.box.grid_coords['y'].value
        self.grid_z = self.box.grid_coords['z'].value
        self.grid_xmin, self.grid_xmax = minval(self.grid_x.min()), maxval(self.grid_x.max())
        self.grid_ymin, self.grid_ymax = minval(self.grid_y.min()), maxval(self.grid_y.max())
        self.grid_zmin, self.grid_zmax = minval(self.grid_z.min()), maxval(self.grid_z.max())
        self.grid_zbase = self.grid_zmin
        self.grid_z = self.grid_z - self.grid_zbase
        self.grid_zmin, self.grid_zmax = self.grid_z.min(), self.grid_z.max()
        self.slice_axis = 'z'
        self.slice_axis_positions = {
            'x': float(np.mean(self.grid_x)),
            'y': float(np.mean(self.grid_y)),
            'z': 0.0,
        }
        self.slice_coord_min = float(self.grid_zmin)
        self.slice_coord_max = float(self.grid_zmax)
        self.default_sph_cen_x = np.mean(self.grid_x)
        self.default_sph_cen_y = np.mean(self.grid_y)
        self.default_sph_cen_z = self.grid_zmin + np.ptp(self.grid_z) * 0.1

        # self.init_ui()
        self.init_grid()
        self.add_widgets_to_window()
        self.init_plot()
        self.show_axes_all()
        # Keep startup in observer LoS (do not override with isometric).
        self.plane_checkbox.setChecked(False)
        self.app_window.setWindowTitle("GxBox 3D viewer")
        self.add_menu_options()  # Add this line to include menu options
        self.add_parallel_projection_button() # Add parallel projection button
        if self.box_norm_direction is not None and self.box_view_up is not None:
            self.add_observer_cam_button()  # Add this line to include the observer cam button
        self._apply_streamline_control_state()
        self._restore_line_seeds_from_box()
        self._install_embedded_close_hook()

        ## Connect the camera modified event to the callback function
        # self.interactor.AddObserver('ModifiedEvent', self.print_camera_position)

    def print_camera_position(self, caller, event):
        """
        Prints the camera position whenever the camera is moved.
        """
        camera = self.camera
        position = camera.position
        focal_point = camera.focal_point
        view_up = camera.up

        print(f"Camera position: {position}")
        print(f"Focal point: {focal_point}")
        print(f"View up: {view_up}")

    def set_camera_to_LOS_direction(self):
        """
        Set the camera to the observer line-of-sight.

        The authoritative orientation comes only from the observer WCS and the
        box frame. The optional FOV box is used only to choose the framing
        target (center/zoom), never to define the camera basis itself.
        """

        def normalize(v):
            arr = np.asarray(v, dtype=float).reshape(-1)
            if arr.size != 3:
                return None
            norm = np.linalg.norm(arr)
            if not np.isfinite(norm) or norm <= 0:
                return None
            return arr / norm

        box_frame = getattr(getattr(self.box, "_center", None), "frame", None)
        frame_obs = getattr(self.box, "_frame_obs", None)
        observer = getattr(frame_obs, "observer", None)
        obstime = getattr(frame_obs, "obstime", None)
        if observer is None and box_frame is not None:
            observer = getattr(box_frame, "observer", None)
        if obstime is None and box_frame is not None:
            obstime = getattr(box_frame, "obstime", None)
        observer_meta = self.box.b3d.get("observer", {}) if isinstance(getattr(self.box, "b3d", None), dict) else {}
        if isinstance(observer_meta, dict):
            observer_key = observer_meta.get("name")
            if observer_key:
                try:
                    resolved, warning, used_key = resolve_observer_with_info(
                        getattr(self.box, "b3d", None) if isinstance(getattr(self.box, "b3d", None), dict) else {},
                        observer_key,
                        obstime,
                    )
                    if warning:
                        print(f"Warning: {warning}")
                    if resolved is not None:
                        observer = resolved
                except Exception:
                    pass
        if box_frame is None or observer is None:
            return

        center = getattr(self.box, "_center", None)
        if center is None:
            return
        if frame_obs is None:
            return
        step_arcsec = 10.0 * u.arcsec
        step_mm = 1.0 * u.Mm
        try:
            # Use the observer WCS itself (same sky plane used by the 2D view)
            # as the authoritative LOS basis.
            fov_corners = self._fov_box_corners_local()
            if isinstance(fov_corners, np.ndarray) and fov_corners.shape == (8, 3):
                focal_point_arr = np.mean(fov_corners, axis=0)
                ref_local = SkyCoord(
                    x=focal_point_arr[0] * u.Mm,
                    y=focal_point_arr[1] * u.Mm,
                    z=(focal_point_arr[2] + float(self.grid_zbase)) * u.Mm,
                    frame=box_frame,
                )
            else:
                focal_point_arr = np.asarray(
                    [
                        0.5 * (self.grid_xmin + self.grid_xmax),
                        0.5 * (self.grid_ymin + self.grid_ymax),
                        0.5 * (self.grid_zmin + self.grid_zmax),
                    ],
                    dtype=float,
                )
                ref_local = center

            ref_obs = ref_local.transform_to(frame_obs)
            ref_dist = getattr(ref_obs, "distance", None)
            if ref_dist is None:
                return
            right_ref_obs = SkyCoord(
                Tx=ref_obs.Tx + step_arcsec,
                Ty=ref_obs.Ty,
                distance=ref_dist,
                frame=frame_obs,
            )
            up_ref_obs = SkyCoord(
                Tx=ref_obs.Tx,
                Ty=ref_obs.Ty + step_arcsec,
                distance=ref_dist,
                frame=frame_obs,
            )
            toward_ref_obs = SkyCoord(
                Tx=ref_obs.Tx,
                Ty=ref_obs.Ty,
                distance=ref_dist - step_mm,
                frame=frame_obs,
            )

            right_ref_local = right_ref_obs.transform_to(box_frame)
            up_ref_local = up_ref_obs.transform_to(box_frame)
            toward_ref_local = toward_ref_obs.transform_to(box_frame)
        except Exception:
            # Fallback to a best-effort HCC-based approximation if the WCS path fails.
            frame_hcc = Heliocentric(observer=observer, obstime=obstime)
            step = 1.0 * u.Mm
            try:
                center_hcc = center.transform_to(frame_hcc)
                right_ref_local = SkyCoord(
                    x=center_hcc.x + step,
                    y=center_hcc.y,
                    z=center_hcc.z,
                    frame=frame_hcc,
                ).transform_to(box_frame)
                up_ref_local = SkyCoord(
                    x=center_hcc.x,
                    y=center_hcc.y + step,
                    z=center_hcc.z,
                    frame=frame_hcc,
                ).transform_to(box_frame)
                toward_ref_local = SkyCoord(
                    x=center_hcc.x,
                    y=center_hcc.y,
                    z=center_hcc.z + step,
                    frame=frame_hcc,
                ).transform_to(box_frame)
                focal_point_arr = np.asarray(
                    [
                        0.5 * (self.grid_xmin + self.grid_xmax),
                        0.5 * (self.grid_ymin + self.grid_ymax),
                        0.5 * (self.grid_zmin + self.grid_zmax),
                    ],
                    dtype=float,
                )
            except Exception:
                return

        def delta_from_focal(ref):
            try:
                return np.array(
                    [
                        float(ref.x.to_value(u.Mm) - focal_point_arr[0]),
                        float(ref.y.to_value(u.Mm) - focal_point_arr[1]),
                        float(ref.z.to_value(u.Mm) - (focal_point_arr[2] + float(self.grid_zbase))),
                    ],
                    dtype=float,
                )
            except Exception:
                try:
                    return np.array(
                        [
                            float(ref.x.to_value(u.Mm) - focal_point_arr[0]),
                            float(ref.y.to_value(u.Mm) - focal_point_arr[1]),
                            float(ref.z.to_value(u.Mm) - focal_point_arr[2]),
                        ],
                        dtype=float,
                    )
                except Exception:
                    return None

        right_local = normalize(delta_from_focal(right_ref_local))
        up_local = normalize(delta_from_focal(up_ref_local))
        toward_observer_local = normalize(delta_from_focal(toward_ref_local))
        if right_local is None or up_local is None or toward_observer_local is None:
            return
        # Re-orthogonalize to remove transform noise while keeping the observer
        # WCS as the truth source.
        right_local = normalize(right_local)
        up_local = normalize(up_local - np.dot(up_local, right_local) * right_local)
        if right_local is None or up_local is None:
            return
        toward_observer_local = normalize(np.cross(right_local, up_local))
        if toward_observer_local is None:
            return
        up_local = normalize(np.cross(toward_observer_local, right_local))
        if up_local is None:
            return
        view_local = -toward_observer_local

        fov_corners = self._fov_box_corners_local()
        if isinstance(fov_corners, np.ndarray) and fov_corners.shape == (8, 3):
            focal_point_arr = np.mean(fov_corners, axis=0)

        scene_span = max(
            float(self.grid_xmax - self.grid_xmin),
            float(self.grid_ymax - self.grid_ymin),
            float(self.grid_zmax - self.grid_zmin),
            1.0,
        )
        camera_distance = 4.0 * scene_span
        focal_point = [
            float(focal_point_arr[0]),
            float(focal_point_arr[1]),
            float(focal_point_arr[2]),
        ]
        self.camera.up = [float(up_local[0]), float(up_local[1]), float(up_local[2])]
        self.camera.focal_point = focal_point
        self.camera.position = [
            float(focal_point_arr[0] - view_local[0] * camera_distance),
            float(focal_point_arr[1] - view_local[1] * camera_distance),
            float(focal_point_arr[2] - view_local[2] * camera_distance),
        ]
        self.camera.ParallelProjectionOn()

        # Use the FOV box only to control framing in the already-defined LOS
        # basis. Otherwise fall back to the full model extents.
        if isinstance(fov_corners, np.ndarray) and fov_corners.shape == (8, 3):
            centered = np.asarray(fov_corners, dtype=float) - focal_point_arr.reshape((1, 3))
            half_h = float(np.max(np.abs(centered @ up_local)))
            half_w = float(np.max(np.abs(centered @ right_local)))
            try:
                render_size = self.render_window.GetSize()
                win_w = max(1, int(render_size[0]))
                win_h = max(1, int(render_size[1]))
                aspect = max(1e-6, float(win_w) / float(win_h))
            except Exception:
                aspect = 1.0
            # Use a more generous framing margin so the full projected FOV
            # rectangle remains visible in LoS parallel view.
            parallel_scale = max(half_h, half_w / max(aspect, 1e-6), 1e-3) * 1.18
            self.camera.parallel_scale = parallel_scale
        else:
            self.camera.parallel_scale = max(0.5 * scene_span, 1e-3)

        try:
            self.camera.SetClippingRange(0.1, max(10.0, 10.0 * camera_distance))
        except Exception:
            pass

        if self.parallel_proj_button is not None:
            self.parallel_proj_button.setChecked(True)
        self._update_los_scene_label()
        self.render()

    @staticmethod
    def _normalize_observer_key(observer_key):
        raw = observer_key
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "ignore")
        if isinstance(raw, np.ndarray) and raw.shape == ():
            raw = raw.item()
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", "ignore")
        key = str(raw or "earth").strip().lower()
        aliases = {
            "sdo": "SDO",
            "earth": "Earth",
            "solo": "Solar Orbiter",
            "solar orbiter": "Solar Orbiter",
            "solar-orbiter": "Solar Orbiter",
            "solarorbiter": "Solar Orbiter",
            "stereo-a": "STEREO-A",
            "stereo a": "STEREO-A",
            "stereoa": "STEREO-A",
            "stereo-b": "STEREO-B",
            "stereo b": "STEREO-B",
            "stereob": "STEREO-B",
        }
        return aliases.get(key, str(raw))

    def _current_los_label(self) -> str:
        observer_meta = self.box.b3d.get("observer", {}) if isinstance(self.box.b3d, dict) else {}
        if not isinstance(observer_meta, dict):
            observer_meta = {}
        if "name" in observer_meta:
            return self._normalize_observer_key(observer_meta.get("name"))
        fov_box = observer_meta.get("fov_box", {})
        if isinstance(fov_box, dict) and "observer_key" in fov_box:
            return self._normalize_observer_key(fov_box.get("observer_key"))
        return "Earth"

    def _update_los_scene_label(self) -> None:
        text = f"Observer LOS: {self._current_los_label()}"
        self._los_label_text = text
        try:
            self.add_text(
                text,
                position="upper_left",
                font_size=10,
                color="black",
                name="observer_los_label",
                shadow=False,
            )
        except Exception:
            pass

    def _apply_startup_los_view(self) -> None:
        self.set_camera_to_LOS_direction()
        self.reset_camera_clipping_range()
        self.render()
        # Some platforms defer first paint; trigger a repaint to avoid waiting
        # for manual mouse interaction before the scene stabilizes.
        window = getattr(self, "app_window", None)
        if window is not None:
            try:
                window.repaint()
            except Exception:
                pass

    def schedule_startup_los_view(self) -> None:
        # Run after Qt has realized the window size so LoS framing matches the
        # manual "LoS" toolbar action behavior.
        QTimer.singleShot(0, self._apply_startup_los_view)
        # A second pass catches late size/layout updates and removes startup jitter.
        QTimer.singleShot(120, self._apply_startup_los_view)

    def ensure_window_visible(self) -> None:
        window = getattr(self, "app_window", None)
        if window is None:
            return
        try:
            handle = window.windowHandle()
            screen = handle.screen() if handle is not None else None
        except Exception:
            screen = None
        if screen is None:
            try:
                screen = QGuiApplication.screenAt(window.frameGeometry().center())
            except Exception:
                screen = None
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        try:
            avail = screen.availableGeometry()
            if not avail.isValid():
                return
            frame = window.frameGeometry()
            width = min(max(frame.width(), 900), max(900, int(avail.width() * 0.92)))
            height = min(max(frame.height(), 650), max(650, int(avail.height() * 0.92)))
            left = frame.left()
            top = frame.top()
            if top < avail.top() or top > avail.bottom() - 80 or left > avail.right() - 80 or left < avail.left():
                left = avail.left() + max(0, (avail.width() - width) // 2)
                top = avail.top() + max(0, (avail.height() - height) // 2)
            else:
                left = min(max(left, avail.left()), max(avail.left(), avail.right() - width + 1))
                top = min(max(top, avail.top()), max(avail.top(), avail.bottom() - height + 1))
            window.resize(width, height)
            window.move(left, top)
        except Exception:
            return

    def _install_embedded_close_hook(self) -> None:
        if self.session_mode != "embedded":
            return
        if self._close_hook_installed:
            return
        window = getattr(self, "app_window", None)
        if window is None:
            return
        original_close_event = window.closeEvent

        def _wrapped_close_event(event):
            # Treat system-window close as "Cancel" for embedded mode.
            if self._embedded_close_mode is None:
                if self.parent is not None and hasattr(self.parent, "cancel_live_3d_edits"):
                    try:
                        self.parent.cancel_live_3d_edits()
                    except Exception:
                        pass
            try:
                original_close_event(event)
            finally:
                self._embedded_close_mode = None

        window.closeEvent = _wrapped_close_event
        self._close_hook_installed = True


    def add_parallel_projection_button(self):
        """
        Adds a toggle button for parallel projection to the toolbar.
        """
        toolbar = self.app_window.findChild(QToolBar)
        if toolbar:
            # Create Parallel Projection button
            self.parallel_proj_button = QToolButton()
            self.parallel_proj_button.setText("Parallel Proj.")
            self.parallel_proj_button.setToolTip("Toggle parallel projection")
            self.parallel_proj_button.setCheckable(True)
            self.parallel_proj_button.setChecked(self.camera.GetParallelProjection())
            self.parallel_proj_button.toggled.connect(self.toggle_parallel_projection)

            # Find the "Reset" button and insert the separator and parallel projection button after it
            for action in toolbar.actions():
                if action.text() == "Reset":
                    # toolbar.insertSeparator(action)
                    toolbar.insertWidget(action, self.parallel_proj_button)
                    # toolbar.insertSeparator(action)
                    break

    def toggle_parallel_projection(self, state):
        """
        Toggles the parallel projection mode of the camera.
        """
        if state:
            self.camera.ParallelProjectionOn()
        else:
            self.camera.ParallelProjectionOff()


    def add_observer_cam_button(self):
        """
        Adds a button to the toolbar to set the camera to the normal direction.
        """
        toolbar = self.app_window.findChild(QToolBar)
        if toolbar:
            observer_cam_button = QToolButton()
            observer_cam_button.setText("LoS")
            observer_cam_button.setToolTip("Set the camera to the observer's normal direction")
            observer_cam_button.clicked.connect(self.set_camera_to_LOS_direction)

            for action in toolbar.actions():
                if action.text() == "Isometric":
                    toolbar.insertWidget(action, observer_cam_button)
                    break

    def add_menu_options(self):
        menubar = self.app_window.menuBar()
        file_menu = None
        for action in menubar.actions():
            if action.text() == "File":
                file_menu = action.menu()
                break

        if file_menu is None:
            file_menu = menubar.addMenu("File")

        load_action = QAction("Load State", self.app_window)
        load_action.triggered.connect(self.load_state)

        save_action = QAction("Save State", self.app_window)
        save_action.triggered.connect(
            lambda: self.save_state(f'magfield_viewer_state{self.timestr}.pkl'))

        # Find the position of the separator and insert the new actions above it
        separator_action = None
        for action in file_menu.actions():
            if action.isSeparator():
                separator_action = action
                break

        if separator_action:
            file_menu.insertAction(separator_action, load_action)
            file_menu.insertAction(separator_action, save_action)
        else:
            file_menu.addAction(load_action)
            file_menu.addAction(save_action)

    def save_state(self,default_filename='magfield_viewer_state.pkl'):
        """
        Saves the current state of spheres to a file. Prompts the user to select a directory and input a filename.

        :param default_filename: str
            The default name of the file to save the state data.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self.app_window, "Save State", default_filename,
                                                  "Pickle Files (*.pkl)", options=options)

        if filename:
            # Create a serializable version of the spheres
            serializable_spheres = {
                sphere_id: {
                    'center': sphere['center'],
                    'radius': sphere['radius'],
                    'n_points': sphere['n_points'],
                    'sphere_visible': sphere['sphere_visible']
                }
                for sphere_id, sphere in self.spheres.items()
            }
            with open(filename, 'wb') as f:
                pickle.dump(serializable_spheres, f)
            print(f"State saved to {filename}")

    def load_state(self, filename = None):
        """
        Loads the state of spheres from a file. Prompts the user to select a file.
        """
        if not isinstance(filename, str):
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            filename, _ = QFileDialog.getOpenFileName(self.app_window, "Load State", f'magfield_viewer_state{self.timestr}.pkl', "Pickle Files (*.pkl)",
                                                      options=options)

        if filename:
            with open(filename, 'rb') as f:
                serializable_spheres = pickle.load(f)

            self._on_clear_spheres()

            # Recreate the spheres from the serializable data
            for sphere_id, sphere_data in serializable_spheres.items():
                # Update the sphere control widgets
                print(sphere_id, sphere_data)
                self.center_x_input.setText(f"{sphere_data['center'][0]:.2f}")
                self.center_y_input.setText(f"{sphere_data['center'][1]:.2f}")
                self.center_z_input.setText(f"{sphere_data['center'][2]:.2f}")
                self.radius_input.setText(f"{sphere_data['radius']:.2f}")
                self.n_points_input.setText(f"{sphere_data['n_points']}")

                # Add the sphere using the _on_add_sphere method
                self._on_add_sphere()

                # Update the sphere visibility
                # self.update_sphere_visibility(sphere_data['sphere_visible'])

            print(f"State loaded from {filename}")

    @staticmethod
    def _decode_seed_type(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.ndarray) and value.shape == ():
            return MagFieldViewer._decode_seed_type(value.item())
        return str(value)

    @staticmethod
    def _as_scalar(value, default):
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
        return value

    def _serialize_line_seeds(self):
        seeds = {}
        for sphere_id in sorted(self.spheres):
            sphere = self.spheres[sphere_id]
            seeds[f"seed_{int(sphere_id)}"] = {
                "seed_type": np.bytes_("sphere"),
                "center": np.asarray(sphere["center"], dtype=float),
                "radius": float(sphere["radius"]),
                "n_points": np.int64(sphere["n_points"]),
                "sphere_visible": np.uint8(1 if sphere.get("sphere_visible", True) else 0),
            }
        seeds["attrs"] = {
            "schema_version": np.int64(1),
            "current_seed_id": np.int64(self.current_sphere_id if self.current_sphere_id is not None else -1),
            "next_seed_id": np.int64(self.next_sphere_id),
        }
        return seeds

    def _persist_line_seeds(self):
        if self._restoring_line_seeds:
            return
        if not hasattr(self.box, "b3d") or self.box.b3d is None:
            self.box.b3d = {}
        if self.spheres:
            self.box.b3d["line_seeds"] = self._serialize_line_seeds()
        else:
            self.box.b3d.pop("line_seeds", None)

    def _restore_line_seeds(self, line_seeds):
        attrs = line_seeds.get("attrs", {}) if isinstance(line_seeds.get("attrs"), dict) else {}
        current_seed_id = int(self._as_scalar(attrs.get("current_seed_id"), -1))
        next_seed_id = int(self._as_scalar(attrs.get("next_seed_id"), 1))

        seed_entries = []
        for key, seed_data in line_seeds.items():
            if key == "attrs" or not isinstance(seed_data, dict):
                continue
            try:
                sphere_id = int(str(key).split("_")[-1])
            except Exception:
                continue
            seed_type = self._decode_seed_type(seed_data.get("seed_type", b"sphere"))
            if seed_type != "sphere":
                continue
            center = np.asarray(seed_data.get("center", ()), dtype=float).reshape(-1)
            if center.size != 3:
                continue
            radius = float(self._as_scalar(seed_data.get("radius"), 0.0))
            n_points = int(self._as_scalar(seed_data.get("n_points"), 100))
            sphere_visible = bool(int(self._as_scalar(seed_data.get("sphere_visible"), 1)))
            seed_entries.append((sphere_id, center, radius, n_points, sphere_visible))

        self._restoring_line_seeds = True
        try:
            self._on_clear_spheres()
            if not seed_entries:
                return
            seed_entries.sort(key=lambda item: item[0])
            for sphere_id, center, radius, n_points, sphere_visible in seed_entries:
                self.center_x_input.setText(f"{center[0]:.2f}")
                self.center_y_input.setText(f"{center[1]:.2f}")
                self.center_z_input.setText(f"{center[2]:.2f}")
                self.radius_input.setText(f"{radius:.2f}")
                self.n_points_input.setText(f"{n_points}")
                self.next_sphere_id = sphere_id
                self._on_add_sphere()
                if not sphere_visible and self.current_sphere_id in self.spheres:
                    self.update_sphere_visibility(False)
            if current_seed_id in self.spheres:
                self.select_sphere(current_seed_id)
            self.next_sphere_id = max(next_seed_id, self.next_sphere_id)
        finally:
            self._restoring_line_seeds = False
        self._persist_line_seeds()

    def _restore_line_seeds_from_box(self):
        line_seeds = getattr(self.box, "b3d", {}).get("line_seeds")
        if not isinstance(line_seeds, dict):
            return
        self._restore_line_seeds(line_seeds)

    def _model_stage_tag(self):
        meta = self.box.b3d.get("metadata", {}) if isinstance(getattr(self.box, "b3d", None), dict) else {}
        model_id = _decode_seed_value(meta.get("id", "")).upper() if isinstance(meta, dict) else ""
        for suffix in (
            ".POT.GEN.CHR", ".NAS.GEN.CHR",
            ".POT.CHR", ".NAS.CHR",
            ".POT.GEN", ".NAS.GEN",
            ".NAS", ".BND", ".POT", ".NONE",
        ):
            if model_id.endswith(suffix):
                return suffix[1:]
        return ""

    def _has_usable_streamline_field(self):
        b3d = getattr(self.box, "b3d", None)
        if not isinstance(b3d, dict):
            return False
        field_group = b3d.get(self.b3dtype)
        if not isinstance(field_group, dict):
            return False
        attrs = field_group.get("attrs", {}) if isinstance(field_group.get("attrs"), dict) else {}
        model_type = _decode_seed_value(attrs.get("model_type", "")).strip().lower()
        if model_type == "none":
            return False
        if self._model_stage_tag() == "NONE":
            return False
        try:
            bx = np.asarray(field_group["bx"])
            by = np.asarray(field_group["by"])
            bz = np.asarray(field_group["bz"])
        except Exception:
            return False
        if bx.size == 0 or by.size == 0 or bz.size == 0:
            return False
        return bool(np.any(bx) or np.any(by) or np.any(bz))

    def _apply_streamline_control_state(self):
        enabled = self._has_usable_streamline_field()
        self._streamline_controls_enabled = enabled
        reason = None if enabled else "Field-line seeding is unavailable for NONE/no-field boxes."
        for widget in (
            self.field_lines_control_group,
            self.sphere_control_group,
            self.add_sphere_button,
            self.delete_sphere_button,
            self.clear_sphere_button,
            self.viz_sphere_button,
            self.tree_view,
            self.center_x_input,
            self.center_y_input,
            self.center_z_input,
            self.radius_input,
            self.n_points_input,
            self.lock_z_checkbox,
        ):
            if widget is None:
                continue
            widget.setEnabled(enabled)
            if reason:
                widget.setToolTip(reason)
        if not enabled and self.spheres:
            self._on_clear_spheres()

    def add_widgets_to_window(self):
        """
        Adds the input widgets to the window.
        """
        # Get the central widget's layout
        central_widget = self.app_window.centralWidget()
        main_layout = central_widget.layout()
        render_widget = None
        while main_layout.count():
            item = main_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and render_widget is None:
                render_widget = widget
            elif widget is not None:
                widget.setParent(None)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(12)

        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        if render_widget is not None:
            render_widget.setMinimumSize(520, 520)
            left_layout.addWidget(render_widget, 1)
        left_panel.setLayout(left_layout)
        body_layout.addWidget(left_panel, 3)

        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_panel.setLayout(right_layout)
        right_panel.setMinimumWidth(520)
        right_panel.setMaximumWidth(680)
        body_layout.addWidget(right_panel, 2)

        field_lines_control_group = QGroupBox("Field Line Browser")
        self.field_lines_control_group = field_lines_control_group
        field_lines_control_layout = QHBoxLayout()
        field_lines_control_layout.setSpacing(12)
        field_lines_control_group.setLayout(field_lines_control_layout)
        field_lines_control_group.setMinimumHeight(220)
        field_lines_control_group.setMaximumHeight(300)
        right_layout.addWidget(field_lines_control_group)

        browser_panel = QWidget()
        browser_layout = QVBoxLayout()
        browser_layout.setContentsMargins(0, 0, 0, 0)
        browser_panel.setLayout(browser_layout)
        field_lines_control_layout.addWidget(browser_panel, 1)

        # Create and add the tree view
        self.tree_view = QTreeView()
        self.sphere_items = QStandardItemModel()
        self.sphere_items.setHorizontalHeaderLabels(["Sphere"])
        self.tree_view.setModel(self.sphere_items)

        # Align the text to the left
        self.tree_view.setStyleSheet("QTreeView::item { text-align: left; }")

        # Adjust the maximum width to ensure it fits the content properly
        self.tree_view.setMinimumWidth(110)
        self.tree_view.setMaximumWidth(130)

        # Resize the columns to fit the contents
        self.tree_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tree_view.header().setStretchLastSection(True)  # Stretch the last section to fill the width
        self.tree_view.header().setSectionResizeMode(QHeaderView.Stretch)  # Resize the section to fill the width

        self.tree_view.selectionModel().selectionChanged.connect(self._on_tb_selection_changed)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._on_tb_right_click)
        browser_layout.addWidget(self.tree_view)

        spheres_manage_layout = QHBoxLayout()
        spheres_manage_layout.setSpacing(2)
        button_width = 35  # Set a fixed width for each button

        # In add_widgets_to_window method, add the "+" button near the QTreeView
        self.add_sphere_button = QPushButton("+")
        self.add_sphere_button.setToolTip("Add a sphere")
        self.add_sphere_button.setFixedWidth(button_width)
        self.add_sphere_button.clicked.connect(self._on_add_sphere)
        spheres_manage_layout.addWidget(self.add_sphere_button)

        self.delete_sphere_button = QPushButton("-")
        self.delete_sphere_button.setToolTip("Delete the elected sphere")
        self.delete_sphere_button.setFixedWidth(button_width)
        self.delete_sphere_button.clicked.connect(self._on_delete_sphere)
        spheres_manage_layout.addWidget(self.delete_sphere_button)

        self.clear_sphere_button = QPushButton("⊗")
        self.clear_sphere_button.setToolTip("Clear all spheres")
        self.clear_sphere_button.setFixedWidth(button_width)
        self.clear_sphere_button.clicked.connect(self._on_clear_spheres)
        spheres_manage_layout.addWidget(self.clear_sphere_button)

        self.viz_sphere_button = QPushButton("⦿")  # Use a suitable symbol for status
        self.viz_sphere_button.setToolTip("Hide the sphere")
        self.viz_sphere_button.setFixedWidth(button_width)
        self.viz_sphere_button.setCheckable(True)
        self.viz_sphere_button.setChecked(True)
        self.viz_sphere_button.toggled.connect(self.toggle_sphere_visibility)
        spheres_manage_layout.addWidget(self.viz_sphere_button)

        # Ensure the layout is aligned
        spheres_manage_layout.addStretch()
        browser_layout.addLayout(spheres_manage_layout)

        # Create and add the properties panel
        properties_panel = QWidget()
        properties_layout = QVBoxLayout()
        properties_panel.setLayout(properties_layout)
        properties_panel.setMinimumHeight(320)
        right_layout.addWidget(properties_panel, 1)

        # Add widgets to the layout
        # Slice Control Group
        sphere_control_group = QGroupBox("Sphere")
        self.sphere_control_group = sphere_control_group
        sphere_control_layout = QGridLayout()
        sphere_control_layout.setHorizontalSpacing(8)
        sphere_control_layout.setVerticalSpacing(12)

        center_label = QLabel("Location [Mm]:")
        center_label.setToolTip(
            f"Enter the X, Y, and Z coordinates for the center of the sphere.")
        self.center_x_input = QLineEdit(f"{self.default_sph_cen_x:.2f}")
        self.center_y_input = QLineEdit(f"{self.default_sph_cen_y:.2f}")
        self.center_z_input = QLineEdit(f"{self.default_sph_cen_z:.2f}")
        self.center_x_input.setToolTip(
            f"Enter the X coordinate for the center of the sphere in the range of {self.grid_xmin:.2f} to {self.grid_xmax:.2f} Mm.")
        self.center_y_input.setToolTip(
            f"Enter the Y coordinate for the center of the sphere in the range of {self.grid_ymin:.2f} to {self.grid_ymax:.2f} Mm.")
        self.center_z_input.setToolTip(
            f"Enter the Z coordinate for the center of the sphere in the range of {0:.2f} to {self.grid_zmax:.2f} Mm.")
        self.center_x_input.returnPressed.connect(lambda: self._on_center_x_input_returnPressed(self.center_x_input))
        self.center_y_input.returnPressed.connect(lambda: self._on_center_y_input_returnPressed(self.center_y_input))
        self.center_z_input.returnPressed.connect(lambda: self._on_center_z_input_returnPressed(self.center_z_input))
        self.center_x_input.setMinimumWidth(72)
        self.center_y_input.setMinimumWidth(72)
        self.center_z_input.setMinimumWidth(72)

        location_inputs_layout = QHBoxLayout()
        location_inputs_layout.setContentsMargins(0, 0, 0, 0)
        location_inputs_layout.setSpacing(6)
        location_inputs_layout.addWidget(self.center_x_input)
        location_inputs_layout.addWidget(self.center_y_input)
        location_inputs_layout.addWidget(self.center_z_input)
        location_inputs_widget = QWidget()
        location_inputs_widget.setLayout(location_inputs_layout)

        self.lock_z_checkbox = QCheckBox("Lock Z")
        self.lock_z_checkbox.setChecked(False)
        self.lock_z_checkbox.stateChanged.connect(self.on_lock_z_changed)

        radius_label = QLabel("Radius [Mm]:")
        radius_label.setToolTip(
            f"Enter the radius of the sphere.")
        self.radius_input = QLineEdit(
            f"{min(np.ptp(self.grid_x), np.ptp(self.grid_y), np.ptp(self.grid_z)) * 0.05:.2f}")
        self.radius_input.setToolTip(
            f"Enter the radius of the sphere in Mm.")
        self.radius_input.returnPressed.connect(lambda: self._on_radius_input_returnPressed(self.radius_input))
        self.radius_input.setMinimumWidth(90)

        n_points_label = QLabel("# of Field Lines:")
        n_points_label.setToolTip(
            "Enter the number of seed points for the field lines.")
        self.n_points_input = QLineEdit("100")
        self.n_points_input.setToolTip(
            "Enter the number of seed points for the field lines.")
        self.n_points_input.returnPressed.connect(lambda: self._on_n_points_input_returnPressed(self.n_points_input))
        self.n_points_input.setMinimumWidth(90)

        sphere_control_layout.addWidget(center_label, 0, 0, 1, 2)
        sphere_control_layout.addWidget(location_inputs_widget, 1, 0, 1, 2)
        sphere_control_layout.addWidget(self.lock_z_checkbox, 2, 0, 1, 2)
        sphere_control_layout.addWidget(radius_label, 3, 0)
        sphere_control_layout.addWidget(self.radius_input, 3, 1)
        sphere_control_layout.addWidget(n_points_label, 4, 0)
        sphere_control_layout.addWidget(self.n_points_input, 4, 1)
        sphere_control_layout.setColumnStretch(1, 1)
        sphere_control_layout.setRowStretch(5, 1)

        sphere_control_group.setLayout(sphere_control_layout)
        field_lines_control_layout.addWidget(sphere_control_group, 1)

        subcontrols_layout = QHBoxLayout()
        subcontrols_layout.setSpacing(10)
        properties_layout.addLayout(subcontrols_layout)

        slice_control_group = QGroupBox("Slice")
        slice_control_layout = QGridLayout()
        slice_control_layout.setHorizontalSpacing(8)
        slice_control_layout.setVerticalSpacing(14)

        slice_axis_label = QLabel("Axis:")
        slice_axis_label.setToolTip("Choose which box-local axis to slice.")
        self.slice_axis_selector = QComboBox()
        self.slice_axis_selector.addItems(["Z", "Y", "X"])
        self.slice_axis_selector.setCurrentText("Z")
        self.slice_axis_selector.currentTextChanged.connect(self._on_slice_axis_changed)
        slice_control_layout.addWidget(slice_axis_label, 0, 0)
        slice_control_layout.addWidget(self.slice_axis_selector, 0, 1)

        self.slice_coord_label = QLabel("Z [Mm]:")
        self.slice_coord_label.setToolTip(f"Enter the Z coordinate for the slice in the range of 0 to {self.grid_zmax:.2f} Mm.")
        self.slice_z_input = QDoubleSpinBox()
        self.slice_z_input.setDecimals(2)
        self.slice_z_input.setRange(0, self.grid_zmax)
        self.slice_z_input.setSingleStep(max(self.grid_zmax / 200, 0.1))
        self.slice_z_input.setAccelerated(True)
        self.slice_z_input.setValue(0.0)
        self.slice_z_input.valueChanged.connect(lambda: self._on_slice_z_input_returnPressed(self.slice_z_input))
        self.slice_z_input.setToolTip(
            f"Use arrows or mouse wheel. Range: 0 to {self.grid_zmax:.2f} Mm.")
        slice_control_layout.addWidget(self.slice_coord_label, 1, 0)
        slice_control_layout.addWidget(self.slice_z_input, 1, 1)

        scalar_label = QLabel("Select Scalar:")
        scalar_label.setToolTip("Select the scalar field to display on the slice.")
        self.scalar_selector = QComboBox()
        self.scalar_selector.addItems(self.scalar_selector_items)
        self.scalar_selector.setCurrentText(self.scalar)
        self.scalar_selector.currentTextChanged.connect(self.update_plot)
        slice_control_layout.addWidget(scalar_label, 2, 0)
        slice_control_layout.addWidget(self.scalar_selector, 2, 1)

        vmin_label = QLabel("Vmin [G]:")
        vmin_label.setToolTip("Enter the minimum value for the color scale.")
        self.vmin_input = QDoubleSpinBox()
        self.vmin_input.setDecimals(2)
        self.vmin_input.setRange(-5e4, 5e4)
        self.vmin_input.setSingleStep(10.0)
        self.vmin_input.setAccelerated(True)
        self.vmin_input.setValue(-1000.0)
        self.vmin_input.valueChanged.connect(lambda: self._on_vmin_input_returnPressed(self.vmin_input))
        self.vmin_input.setToolTip("Use arrows or mouse wheel to change Vmin.")
        self.vmax_input = QDoubleSpinBox()
        self.vmax_input.setDecimals(2)
        self.vmax_input.setRange(-5e4, 5e4)
        self.vmax_input.setSingleStep(10.0)
        self.vmax_input.setAccelerated(True)
        self.vmax_input.setValue(1000.0)
        self.vmax_input.valueChanged.connect(lambda: self._on_vmax_input_returnPressed(self.vmax_input))
        self.vmax_input.setToolTip("Use arrows or mouse wheel to change Vmax.")
        vmax_label = QLabel("Vmax [G]:")
        vmax_label.setToolTip("Enter the maximum value for the color scale.")
        slice_control_layout.addWidget(vmin_label, 3, 0)
        slice_control_layout.addWidget(self.vmin_input, 3, 1)
        slice_control_layout.addWidget(vmax_label, 4, 0)
        slice_control_layout.addWidget(self.vmax_input, 4, 1)

        self.slice_checkbox = QCheckBox("Show Slice")
        self.slice_checkbox.setChecked(True)
        self.slice_checkbox.setToolTip("Hide or show the z-slice image while keeping the current scalar selection.")
        self.slice_checkbox.stateChanged.connect(
            lambda state: (setattr(self, "slice_visible", state == Qt.Checked), self.update_plot())
        )

        self.plane_checkbox = QCheckBox("Show Plane")
        self.plane_checkbox.setChecked(True)
        self.plane_checkbox.stateChanged.connect(self.toggle_plane_visibility)

        self.interp_checkbox = QCheckBox("Interpolate")
        self.interp_checkbox.setChecked(True)
        self.interp_checkbox.setToolTip("Toggle interpolation for slice display.")
        self.interp_checkbox.stateChanged.connect(self.update_plot)
        slice_control_layout.addWidget(self.slice_checkbox, 5, 0, 1, 2)
        slice_control_layout.addWidget(self.plane_checkbox, 6, 0, 1, 2)
        slice_control_layout.addWidget(self.interp_checkbox, 7, 0, 1, 2)
        slice_control_layout.setRowStretch(8, 1)

        slice_control_group.setLayout(slice_control_layout)
        subcontrols_layout.addWidget(slice_control_group, 1)

        base_control_group = QGroupBox("Bottom Map")
        base_control_layout = QGridLayout()
        base_control_layout.setHorizontalSpacing(8)
        base_control_layout.setVerticalSpacing(10)

        base_map_label = QLabel("Map:")
        base_map_label.setToolTip("Display a fixed base/ref map at the box bottom (z-min plane).")
        self.base_map_selector = QComboBox()
        self.base_map_selector.addItems(self.base_map_items)
        default_base_map = "bz" if "bz" in self.base_map_items else (self.base_map_items[0] if self.base_map_items else "")
        if default_base_map:
            self.base_map_selector.setCurrentText(default_base_map)
        self.base_map_selector.currentTextChanged.connect(self._on_base_map_changed)
        base_control_layout.addWidget(base_map_label, 0, 0)
        base_control_layout.addWidget(self.base_map_selector, 0, 1)

        base_vmin_label = QLabel("Min:")
        base_vmin_label.setToolTip("Minimum intensity for the fixed bottom map.")
        self.base_vmin_input = QDoubleSpinBox()
        self.base_vmin_input.setDecimals(2)
        self.base_vmin_input.setRange(-5e6, 5e6)
        self.base_vmin_input.setSingleStep(10.0)
        self.base_vmin_input.setAccelerated(True)
        self.base_vmin_input.setValue(-1000.0)
        self.base_vmin_input.valueChanged.connect(lambda: self._on_base_vmin_input_returnPressed(self.base_vmin_input))
        self.base_vmax_input = QDoubleSpinBox()
        self.base_vmax_input.setDecimals(2)
        self.base_vmax_input.setRange(-5e6, 5e6)
        self.base_vmax_input.setSingleStep(10.0)
        self.base_vmax_input.setAccelerated(True)
        self.base_vmax_input.setValue(1000.0)
        self.base_vmax_input.valueChanged.connect(lambda: self._on_base_vmax_input_returnPressed(self.base_vmax_input))
        base_vmax_label = QLabel("Max:")
        base_vmax_label.setToolTip("Maximum intensity for the fixed bottom map.")
        base_control_layout.addWidget(base_vmin_label, 1, 0)
        base_control_layout.addWidget(self.base_vmin_input, 1, 1)
        base_control_layout.addWidget(base_vmax_label, 2, 0)
        base_control_layout.addWidget(self.base_vmax_input, 2, 1)

        self.base_map_checkbox = QCheckBox("Show Map")
        self.base_map_checkbox.setChecked(False)
        self.base_map_checkbox.setToolTip("Hide or show the selected bottom map while keeping the current map selection.")
        self.base_map_checkbox.stateChanged.connect(self.toggle_base_map_visibility)
        base_control_layout.addWidget(self.base_map_checkbox, 3, 0, 1, 2)

        self.model_box_checkbox = QCheckBox("Show Model Box")
        self.model_box_checkbox.setChecked(True)
        self.model_box_checkbox.setToolTip("Hide or show the red wireframe 3D model box.")
        self.model_box_checkbox.stateChanged.connect(self.toggle_model_box_visibility)
        base_control_layout.addWidget(self.model_box_checkbox, 4, 0, 1, 2)

        self.fov_box_checkbox = QCheckBox("Show FOV Box")
        self.fov_box_checkbox.setChecked(True)
        self.fov_box_checkbox.setToolTip("Hide or show the blue observer-aligned 3D FOV box.")
        self.fov_box_checkbox.stateChanged.connect(self.toggle_fov_box_visibility)
        base_control_layout.addWidget(self.fov_box_checkbox, 5, 0, 1, 2)
        base_control_layout.setRowStretch(6, 1)

        base_control_group.setLayout(base_control_layout)
        subcontrols_layout.addWidget(base_control_group, 1)

        action_layout = QHBoxLayout()

        self.save_model_button = QPushButton("Apply && Close")
        if self.session_mode == "standalone":
            self.save_model_button.setText("Save")
            if self.model_path is None:
                self.save_model_button.setEnabled(False)
                self.save_model_button.setToolTip(
                    "SAV models are read-only — use Save As to save to a new .h5 file."
                )
            else:
                self.save_model_button.setToolTip("Save the current seed state back into the opened model file.")
                self.save_model_button.clicked.connect(self.save_current_model)
        else:
            self.save_model_button.setToolTip("Accept the current seed edits and return to the 2D viewer.")
            self.save_model_button.setStyleSheet("font-weight: 600;")
            self.save_model_button.clicked.connect(self.accept_and_close)

        self.save_close_button = QPushButton("Undo && Restore")
        if self.session_mode == "standalone":
            self.save_close_button.setToolTip("Restore the original seed state from when this 3D viewer was opened.")
        else:
            self.save_close_button.setToolTip("Restore the original seed state received from the 2D viewer.")
        self.save_close_button.clicked.connect(self.undo_and_restore)

        self.cancel_button = QPushButton("Undo && Close")
        if self.session_mode == "standalone":
            self.cancel_button.setText("Close")
            self.cancel_button.setToolTip("Close this 3D viewer.")
            self.cancel_button.clicked.connect(self._close_window)
        else:
            self.cancel_button.setToolTip("Discard seed edits made in this 3D session and return to the 2D viewer.")
            self.cancel_button.clicked.connect(self.cancel_and_close)

        self.load_box_button = QPushButton("Load Box")
        self.load_box_button.setToolTip("Load the box data from a .hd5 file.")
        self.load_box_button.clicked.connect(self.load_box)


        self.save_box_button = QPushButton("Save As")
        self.save_box_button.setToolTip("Save the full current box data to a new .h5 file.")
        self.save_box_button.clicked.connect(self.save_box)


        action_layout.addWidget(self.save_model_button)
        action_layout.addWidget(self.save_close_button)
        if self.session_mode != "standalone":
            action_layout.addWidget(self.cancel_button)
        else:
            action_layout.addWidget(self.cancel_button)
            action_layout.addWidget(self.load_box_button)
            action_layout.addWidget(self.save_box_button)

        # self.update_button = QPushButton("Update")
        # self.update_button.clicked.connect(self.update_plot)
        # action_layout.addWidget(self.update_button)

        # self.sphere_checkbox = QCheckBox("Show Sphere")
        # self.sphere_checkbox.setChecked(True)
        # self.sphere_checkbox.stateChanged.connect(self.toggle_sphere_visibility)
        # action_layout.addWidget(self.sphere_checkbox)

        right_layout.addStretch()

        main_layout.addLayout(body_layout, 1)
        main_layout.addLayout(action_layout)

    def _on_add_sphere(self):
        """
        Adds a new sphere to the viewer and tree view, hiding the current sphere.
        """
        if not self._streamline_controls_enabled:
            print("Sphere controls are disabled for this box because no volumetric field is available.")
            return
        # Create a new sphere and its streamlines
        if self.current_sphere_id in self.spheres:
            self.spheres[self.current_sphere_id]['sphere_actor'].Off()

        sphere_id = self.next_sphere_id
        center_x = float(self.center_x_input.text())
        center_y = float(self.center_y_input.text())
        # if keep_current_parms:
        #     pass
        # else:
        #     center_x = np.mean(self.grid_x)
        #     center_y = np.mean(self.grid_y)
        # center_z = self.grid_zmin + np.ptp(self.grid_z) * 0.1
        center_z = float(self.center_z_input.text())
        radius = float(self.radius_input.text())
        n_points = int(self.n_points_input.text())

        self.center_x_input.setText(f"{center_x:.2f}")
        self.center_y_input.setText(f"{center_y:.2f}")
        self.center_z_input.setText(f"{center_z:.2f}")

        self.create_streamlines(center_x, center_y, center_z, radius, n_points)
        self.current_sphere_id = sphere_id

        self.spheres[sphere_id] = {
            'center': (center_x, center_y, center_z),
            'radius': radius,
            'n_points': n_points,
            'sphere_actor': self.sphere_actor,
            'streamlines': self.streamlines,
            'streamlines_actor': self.streamlines_actor,
            'sphere_visible': True
        }

        self.streamlines_actor = None
        self.streamlines = None

        self.update_sphere_visibility(True)

        # Add the new sphere to the tree view
        sphere_item = QStandardItem(f"{self.next_sphere_id}")
        self.sphere_items.appendRow(sphere_item)
        self.tree_view.setCurrentIndex(self.sphere_items.indexFromItem(sphere_item))
        self.next_sphere_id += 1
        self._persist_line_seeds()

    def select_sphere(self, sphere_id):
        sphere = self.spheres[sphere_id]
        self.center_x_input.setText(f"{sphere['center'][0]:.2f}")
        self.center_y_input.setText(f"{sphere['center'][1]:.2f}")
        self.center_z_input.setText(f"{sphere['center'][2]:.2f}")
        self.radius_input.setText(f"{sphere['radius']:.2f}")
        self.n_points_input.setText(f"{sphere['n_points']}")

        # self.spheres[self.current_sphere_id]['streamlines_actor'].SetVisibility(False)
        if self.current_sphere_id in self.spheres:
            self.spheres[self.current_sphere_id]['sphere_actor'].Off()

        # Restore the streamlines actor for the selected sphere
        streamlines_actor = sphere['streamlines_actor']
        sphere_actor = sphere['sphere_actor']

        # if streamlines_actor is not None:
        #     streamlines_actor.SetVisibility(True)
        if sphere_actor is not None:
            sphere_actor.On()

        self.current_sphere_id = sphere_id

    def deselect_sphere(self):
        """
        Handles the deselection of a sphere.
        Clears the inputs and hides the current sphere and its streamlines.
        """
        # self.center_x_input.clear()
        # self.center_y_input.clear()
        # self.center_z_input.clear()
        # self.radius_input.clear()
        # self.n_points_input.clear()

        if self.current_sphere_id in self.spheres:
            sphere_actor = self.spheres[self.current_sphere_id]['sphere_actor']
            streamlines_actor = self.spheres[self.current_sphere_id]['streamlines_actor']
            if sphere_actor is not None:
                sphere_actor.Off()
            if streamlines_actor is not None:
                streamlines_actor.SetVisibility(False)

        self.current_sphere_id = None

    def _on_tb_selection_changed(self, selected, deselected):
        indexes = selected.indexes()
        if indexes:
            item = self.sphere_items.itemFromIndex(indexes[0])
            sphere_id = int(item.text())
            self.select_sphere(sphere_id)
        # else:
        #     self.deselect_sphere()

    def _on_delete_sphere(self):
        """
        Deletes the currently selected sphere in the tree view.
        """
        if self.sphere_items.rowCount() > 0:
            indexes = self.tree_view.selectionModel().selectedIndexes()
            if indexes:
                item = self.sphere_items.itemFromIndex(indexes[0])
                sphere_id = int(item.text())
                self.delete_sphere_from_tb(sphere_id)
            if len(self.spheres) > 0:
                self.update_sphere_visibility(True)

    def _on_clear_spheres(self):
        """
        Removes all spheres from the tree view and clears the corresponding data.
        """
        while self.sphere_items.rowCount() > 0:
            item = self.sphere_items.item(0)
            sphere_id = int(item.text())
            self.delete_sphere_from_tb(sphere_id)

        self.spheres.clear()
        self.current_sphere_id = None
        self.next_sphere_id = 1
        self._persist_line_seeds()

    def delete_sphere_from_tb(self, sphere_id):
        sphere = self.spheres.pop(sphere_id, None)
        if sphere and sphere['streamlines_actor'] is not None:
            self.remove_actor(sphere['streamlines_actor'])
        if sphere and sphere['streamlines'] is not None:
            sphere['streamlines'] = None
        if sphere and sphere['sphere_actor'] is not None:
            sphere['sphere_actor'].Off()
            sphere['sphere_actor'].RemoveAllObservers()
        # Remove from tree view
        nrows = self.sphere_items.rowCount()
        for row in range(nrows):
            item = self.sphere_items.item(row)
            if item.text() == f"{sphere_id}":
                self.sphere_items.removeRow(row)
                break

        # Update next_sphere_id to be 1 plus the largest sphere index
        if nrows > 1:
            max_sphere_id = max(int(self.sphere_items.item(row).text()) for row in range(nrows - 1))
        else:
            max_sphere_id = 0
        self.next_sphere_id = max_sphere_id + 1
        self._persist_line_seeds()

    def _on_tb_right_click(self, pos):
        index = self.tree_view.indexAt(pos)
        if index.isValid():
            item = self.sphere_items.itemFromIndex(index)
            sphere_id = int(item.text())
            menu = QMenu()
            delete_action = menu.addAction("Delete")
            action = menu.exec_(self.tree_view.viewport().mapToGlobal(pos))
            if action == delete_action:
                self.delete_sphere_from_tb(sphere_id)

    @validate_number
    def _on_center_x_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the center X input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_sphere()

    @validate_number
    def _on_center_y_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the center Y input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_sphere()

    @validate_number
    def _on_center_z_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the center Z input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_sphere()

    @validate_number
    def _on_radius_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the radius input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_sphere()

    @validate_number
    def _on_n_points_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the number of seeds input.

        :param widget: QLineEdit
            The input widget.
        """

        self.update_sphere()

    @validate_number
    def _on_slice_z_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the slice Z input.

        :param widget: QLineEdit
            The input widget.
        """
        self.slice_axis_positions[self.slice_axis] = float(widget.value()) if isinstance(widget, QDoubleSpinBox) else float(widget.text())
        self.update_plot()

    def _slice_axis_bounds(self, axis=None):
        axis = (axis or self.slice_axis).lower()
        if axis == 'x':
            return float(self.grid_xmin), float(self.grid_xmax)
        if axis == 'y':
            return float(self.grid_ymin), float(self.grid_ymax)
        return float(self.grid_zmin), float(self.grid_zmax)

    def _slice_normal_vector(self, axis=None):
        axis = (axis or self.slice_axis).lower()
        if axis == 'x':
            return (1.0, 0.0, 0.0)
        if axis == 'y':
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)

    def _slice_origin(self, coord_value=None, axis=None):
        axis = (axis or self.slice_axis).lower()
        coord_value = self.slice_axis_positions.get(axis, 0.0) if coord_value is None else float(coord_value)
        origin = [
            0.5 * (self.grid_xmin + self.grid_xmax),
            0.5 * (self.grid_ymin + self.grid_ymax),
            0.5 * (self.grid_zmin + self.grid_zmax),
        ]
        idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        origin[idx] = coord_value
        return tuple(origin)

    def _on_slice_axis_changed(self, axis_text):
        new_axis = (axis_text or "Z").lower()
        if self.slice_z_input is not None:
            self.slice_axis_positions[self.slice_axis] = float(self.slice_z_input.value())
        self.slice_axis = new_axis
        self._set_slice_slider_range()
        if self.slice_coord_label is not None:
            self.slice_coord_label.setText(f"{new_axis.upper()} [Mm]:")
            min_val, max_val = self._slice_axis_bounds(new_axis)
            self.slice_coord_label.setToolTip(
                f"Enter the {new_axis.upper()} coordinate for the slice in the range of {min_val:.2f} to {max_val:.2f} Mm."
            )
        if self.slice_z_input is not None:
            self.slice_z_input.setToolTip(
                f"Use arrows or mouse wheel. Range: {self.slice_coord_min:.2f} to {self.slice_coord_max:.2f} Mm."
            )
        self.update_plane()
        self.update_plot()

    @validate_number
    def _on_vmin_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the Vmin input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_plot()

    @validate_number
    def _on_vmax_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the Vmax input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_plot()

    @validate_number
    def _on_base_vmin_input_returnPressed(self, widget):
        self._update_base_map_from_controls()

    @validate_number
    def _on_base_vmax_input_returnPressed(self, widget):
        self._update_base_map_from_controls()

    def _on_base_map_changed(self, _map_name):
        self._set_base_scalar_range(self.base_map_selector.currentText(), reset_values=True)
        self._update_base_map_from_controls()

    def _update_base_map_from_controls(self):
        if self.base_map_selector is None:
            return
        base_map = self.base_map_selector.currentText()
        if base_map not in self.grid_bottom.array_names:
            self.update_base_map(None, 0.0, 1.0, False)
            return
        bmin = self.validate_input(
            self.base_vmin_input,
            self.base_scalar_min,
            self.base_scalar_max,
            self.previous_valid_values.get(self.base_vmin_input, self.base_scalar_min),
            paired_widget=self.base_vmax_input,
            paired_type='vmin',
        )
        bmax = self.validate_input(
            self.base_vmax_input,
            self.base_scalar_min,
            self.base_scalar_max,
            self.previous_valid_values.get(self.base_vmax_input, self.base_scalar_max),
            paired_widget=self.base_vmin_input,
            paired_type='vmax',
        )
        self.update_base_map(base_map, bmin, bmax, self.base_map_visible)

    def validate_input(self, widget, min_val, max_val, original_value, to_int=False, paired_widget=None,
                       paired_type=None):
        '''
        Validates the input of a QLineEdit widget and returns the value if it is valid. If the input is invalid, a warning message is displayed and the original value is restored.

        :param widget: QLineEdit
            The widget to validate.
        :param min_val: float
            The minimum valid value.
        :param max_val: float
            The maximum valid value.
        :param original_value: float
            The original value of the widget.
        :param to_int: bool
            Whether to convert the value to an integer.
        :param paired_widget: QLineEdit, optional
            The paired widget to compare the value with.
        :param paired_type: str, optional
            The type of comparison to perform with the paired widget.
        :return: float
            The valid value.
        '''
        try:
            if isinstance(widget, QDoubleSpinBox):
                value = float(widget.value())
            else:
                value = float(widget.text())
            if not min_val <= value <= max_val:
                original_value = min_val if value < min_val else max_val
                raise ValueError

            if paired_widget:
                if isinstance(paired_widget, QDoubleSpinBox):
                    paired_value = float(paired_widget.value())
                else:
                    paired_value = float(paired_widget.text())
                if paired_type == 'vmin' and value >= paired_value:
                    raise ValueError
                if paired_type == 'vmax' and value <= paired_value:
                    raise ValueError

            if to_int:
                value = int(value)

            self.previous_valid_values[widget] = value
            return value
        except ValueError:
            # if paired_type == 'vmin':
            #     QMessageBox.warning(self, "Invalid Input",
            #                         f"Please enter a number between {min_val:.3f} and {max_val:.3f} that is less than the corresponding max value.")
            # elif paired_type == 'vmax':
            #     QMessageBox.warning(self, "Invalid Input",
            #                         f"Please enter a number between {min_val:.3f} and {max_val:.3f} that is greater than the corresponding min value.")
            # else:
            #     QMessageBox.warning(self, "Invalid Input",
            #                         f"Please enter a number between {min_val:.3f} and {max_val:.3f}. Revert to the original value.")

            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(original_value))
            else:
                widget.setText(str(original_value))
            return original_value

    def _set_slice_slider_range(self):
        self.slice_coord_min, self.slice_coord_max = self._slice_axis_bounds()
        current_value = self.slice_axis_positions.get(self.slice_axis, self.slice_coord_min)
        current_value = min(max(current_value, self.slice_coord_min), self.slice_coord_max)
        self.slice_axis_positions[self.slice_axis] = current_value
        self.slice_z_min = self.slice_coord_min
        self.slice_z_max = self.slice_coord_max
        if self.slice_z_input is None:
            return
        self.slice_z_input.blockSignals(True)
        self.slice_z_input.setRange(self.slice_coord_min, self.slice_coord_max)
        step = max((self.slice_coord_max - self.slice_coord_min) / 200.0, 0.1)
        self.slice_z_input.setSingleStep(step)
        self.slice_z_input.setValue(current_value)
        self.slice_z_input.blockSignals(False)

    def _set_scalar_range(self, scalar_name):
        data = None
        if scalar_name in self.grid.array_names:
            data = self.grid[scalar_name]
        elif self.bottom_name is not None and scalar_name == self.bottom_name:
            data = self.grid_bottom[scalar_name]
        if data is None:
            return
        self.scalar_min = float(np.nanmin(data))
        self.scalar_max = float(np.nanmax(data))
        if self.scalar_min == self.scalar_max:
            self.scalar_min -= 1.0
            self.scalar_max += 1.0

        if self.vmin_input is not None and self.vmax_input is not None:
            self.vmin_input.blockSignals(True)
            self.vmax_input.blockSignals(True)
            self.vmin_input.setRange(self.scalar_min, self.scalar_max)
            self.vmax_input.setRange(self.scalar_min, self.scalar_max)
            step = max((self.scalar_max - self.scalar_min) / 200.0, 1.0)
            self.vmin_input.setSingleStep(step)
            self.vmax_input.setSingleStep(step)
            self.vmin_input.blockSignals(False)
            self.vmax_input.blockSignals(False)

    def _set_base_scalar_range(self, base_map_name, reset_values=False):
        if self.base_vmin_input is None or self.base_vmax_input is None:
            return
        if base_map_name is None or base_map_name not in self.grid_bottom.array_names:
            self.base_vmin_input.setEnabled(False)
            self.base_vmax_input.setEnabled(False)
            return

        data = self.grid_bottom[base_map_name]
        self.base_scalar_min = float(np.nanmin(data))
        self.base_scalar_max = float(np.nanmax(data))
        if self.base_scalar_min == self.base_scalar_max:
            self.base_scalar_min -= 1.0
            self.base_scalar_max += 1.0

        self.base_vmin_input.blockSignals(True)
        self.base_vmax_input.blockSignals(True)
        self.base_vmin_input.setEnabled(True)
        self.base_vmax_input.setEnabled(True)
        self.base_vmin_input.setRange(self.base_scalar_min, self.base_scalar_max)
        self.base_vmax_input.setRange(self.base_scalar_min, self.base_scalar_max)
        step = max((self.base_scalar_max - self.base_scalar_min) / 200.0, 1.0e-3)
        self.base_vmin_input.setSingleStep(step)
        self.base_vmax_input.setSingleStep(step)
        if reset_values:
            self.base_vmin_input.setValue(self.base_scalar_min)
            self.base_vmax_input.setValue(self.base_scalar_max)
            self.previous_valid_values[self.base_vmin_input] = self.base_scalar_min
            self.previous_valid_values[self.base_vmax_input] = self.base_scalar_max
        self.base_vmin_input.blockSignals(False)
        self.base_vmax_input.blockSignals(False)

    def init_grid(self):
        x = self.grid_x
        y = self.grid_y
        z = self.grid_z

        self.bottom_name = None
        self.base_map_items = []

        bx = self.box.b3d[self.b3dtype]['bx']
        by = self.box.b3d[self.b3dtype]['by']
        bz = self.box.b3d[self.b3dtype]['bz']


        self.grid = pv.ImageData()
        self.grid.dimensions = (len(x), len(y), len(z))
        self.grid.spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
        self.grid.origin = (x.min(), y.min(), z.min())
        self.grid_dims = (len(x), len(y), len(z))
        self.grid_spacing = self.grid.spacing

        self.grid['bx'] = bx.ravel(order='F')
        self.grid['by'] = by.ravel(order='F')
        self.grid['bz'] = bz.ravel(order='F')
        self.grid['vectors'] = np.c_[self.grid['bx'] , self.grid['by'], self.grid['bz']]
        self.scalar_selector_items = ['bx', 'by', 'bz']

        self.grid_bottom = pv.ImageData()
        self.grid_bottom.dimensions = (len(x), len(y), 1)
        self.grid_bottom.spacing = (x[1] - x[0], y[1] - y[0], 0)
        self.grid_bottom.origin = (x.min(), y.min(), z.min())

        base_group = self.box.b3d.get("base", {}) if isinstance(self.box.b3d, dict) else {}
        if isinstance(base_group, dict):
            for key in ("bx", "by", "bz", "ic", "chromo_mask"):
                if key not in base_group:
                    continue
                arr = np.asarray(base_group[key])
                if arr.ndim != 2:
                    continue
                # Base maps are stored as (y, x); grid_bottom expects flattened (x, y).
                if arr.shape != (len(y), len(x)):
                    continue
                self.grid_bottom[key] = arr.T.ravel(order='F')
                self.base_map_items.append(key)

        # Include compatible refmaps (e.g., Vert_current) when they match bottom dimensions.
        refmaps_group = self.box.b3d.get("refmaps", {}) if isinstance(self.box.b3d, dict) else {}
        if isinstance(refmaps_group, dict):
            for ref_name, ref_obj in refmaps_group.items():
                if not isinstance(ref_obj, dict) or "data" not in ref_obj:
                    continue
                arr = np.asarray(ref_obj["data"])
                if arr.ndim != 2:
                    continue
                if arr.shape != (len(y), len(x)):
                    continue
                key = str(ref_name)
                if key in self.grid_bottom.array_names:
                    continue
                self.grid_bottom[key] = arr.T.ravel(order='F')
                if key not in self.base_map_items:
                    self.base_map_items.append(key)

        if self.parent is not None and hasattr(self.parent, "mapBottomSelector") and hasattr(self.parent, "map_bottom"):
            self.bottom_name = self.parent.mapBottomSelector.currentText()
            self.grid_bottom[self.bottom_name] = self.parent.map_bottom.data.T.ravel(order='F')
            if self.bottom_name not in self.base_map_items:
                self.base_map_items.append(self.bottom_name)

        self._set_slice_slider_range()
        self._set_scalar_range(self.scalar)


    def init_plot(self):
        """
        Initializes and displays the plot with the magnetic field data.
        """
        self._set_slice_slider_range()
        self._set_scalar_range(self.scalar)
        self._set_base_scalar_range(self.base_map_selector.currentText() if self.base_map_selector is not None else "none",
                                    reset_values=True)

        def _val(widget):
            if isinstance(widget, QDoubleSpinBox):
                return float(widget.value())
            return float(widget.text())

        self.previous_valid_values = {
            self.center_x_input: _val(self.center_x_input),
            self.center_y_input: _val(self.center_y_input),
            self.center_z_input: _val(self.center_z_input),
            self.radius_input: _val(self.radius_input),
            self.slice_z_input: _val(self.slice_z_input),
            self.n_points_input: int(self.n_points_input.text()),
            self.vmin_input: _val(self.vmin_input),
            self.vmax_input: _val(self.vmax_input),
            self.base_vmin_input: _val(self.base_vmin_input) if self.base_vmin_input is not None else -1000.0,
            self.base_vmax_input: _val(self.base_vmax_input) if self.base_vmax_input is not None else 1000.0,
        }

        self.update_plot(init=True)

    def update_plot(self, init=False):
        """
        Updates the plot based on the current input parameters.
        """

        if self.updating_flag:  # Check if already updating
            return

        self.updating_flag = True  # Set the flag

        # Get current parameters
        center_x = self.validate_input(self.center_x_input, self.grid_xmin, self.grid_xmax,
                                       self.previous_valid_values[self.center_x_input])
        center_y = self.validate_input(self.center_y_input, self.grid_ymin, self.grid_ymax,
                                       self.previous_valid_values[self.center_y_input])
        center_z = self.validate_input(self.center_z_input, 0, self.grid_zmax,
                                       self.previous_valid_values[self.center_z_input])
        radius = self.validate_input(self.radius_input, 0, min(np.ptp(self.grid_x), np.ptp(self.grid_y), np.ptp(self.grid_z)),
                                     self.previous_valid_values[self.radius_input])
        n_points = self.validate_input(self.n_points_input, 1, 1000, self.previous_valid_values[self.n_points_input],
                                       to_int=True)

        if not init:
            self.update_sphere()

        self.update_plane()
        scalar = self.scalar_selector.currentText()
        self._set_scalar_range(scalar)
        base_map = self.base_map_selector.currentText() if self.base_map_selector is not None else None
        self._set_base_scalar_range(base_map, reset_values=False)
        slice_z = self.validate_input(self.slice_z_input, self.slice_coord_min, self.slice_coord_max,
                                      self.previous_valid_values[self.slice_z_input])
        self.slice_axis_positions[self.slice_axis] = slice_z
        vmin = self.validate_input(self.vmin_input, -5e4, 5e4, self.previous_valid_values[self.vmin_input],
                                   paired_widget=self.vmax_input, paired_type='vmin')
        vmax = self.validate_input(self.vmax_input, -5e4, 5e4, self.previous_valid_values[self.vmax_input],
                                   paired_widget=self.vmin_input, paired_type='vmax')
        if base_map in self.grid_bottom.array_names:
            bmin = self.validate_input(
                self.base_vmin_input,
                self.base_scalar_min,
                self.base_scalar_max,
                self.previous_valid_values[self.base_vmin_input],
                paired_widget=self.base_vmax_input,
                paired_type='vmin',
            )
            bmax = self.validate_input(
                self.base_vmax_input,
                self.base_scalar_min,
                self.base_scalar_max,
                self.previous_valid_values[self.base_vmax_input],
                paired_widget=self.base_vmin_input,
                paired_type='vmax',
            )
        else:
            bmin = vmin
            bmax = vmax
        sphere_visible = self.viz_sphere_button.isChecked()
        plane_visible = self.plane_visible
        use_interp = self.interp_checkbox.isChecked() if self.interp_checkbox is not None else True
        slice_visible = self.slice_visible
        base_map_visible = self.base_map_visible
        model_box_visible = self.model_box_visible
        fov_box_visible = self.fov_box_visible

        # Create a dictionary of current parameters
        current_params = {
            "center_x": center_x,
            "center_y": center_y,
            "center_z": center_z,
            "radius": radius,
            "slice_z": slice_z,
            "slice_axis": self.slice_axis,
            "n_points": n_points,
            "vmin": vmin,
            "vmax": vmax,
            "scalar": scalar,
            "base_map": base_map,
            "base_vmin": bmin,
            "base_vmax": bmax,
            "base_map_visible": base_map_visible,
            "slice_visible": slice_visible,
            "use_interp": use_interp,
            "sphere_visible": sphere_visible,
            "plane_visible": plane_visible,
            "model_box_visible": model_box_visible,
            "fov_box_visible": fov_box_visible,
        }

        # Check if parameters have changed
        if current_params == self.previous_params:
            self.updating_flag = False  # Reset the flag
            return

        # Update only relevant objects based on parameter changes
        if current_params['slice_z'] != self.previous_params.get('slice_z') or \
                current_params['slice_axis'] != self.previous_params.get('slice_axis') or \
                current_params['scalar'] != self.previous_params.get('scalar') or \
                current_params['vmin'] != self.previous_params.get('vmin') or \
                current_params['vmax'] != self.previous_params.get('vmax') or \
                current_params['slice_visible'] != self.previous_params.get('slice_visible') or \
                current_params['use_interp'] != self.previous_params.get('use_interp'):
            self.update_slice(current_params['slice_axis'], current_params['slice_z'], current_params['scalar'], current_params['vmin'],
                              current_params['vmax'], current_params['use_interp'], current_params['slice_visible'])

        if current_params['base_map'] != self.previous_params.get('base_map') or \
                current_params['base_map_visible'] != self.previous_params.get('base_map_visible') or \
                current_params['base_vmin'] != self.previous_params.get('base_vmin') or \
                current_params['base_vmax'] != self.previous_params.get('base_vmax'):
            self.update_base_map(
                current_params['base_map'],
                current_params['base_vmin'],
                current_params['base_vmax'],
                current_params['base_map_visible'],
            )

        if current_params['plane_visible'] != self.previous_params.get('plane_visible'):
            self.update_plane_visibility(current_params['plane_visible'])

        if current_params['model_box_visible'] != self.previous_params.get('model_box_visible') or init:
            self.update_model_box(current_params['model_box_visible'], do_render=False)

        if current_params['fov_box_visible'] != self.previous_params.get('fov_box_visible') or init:
            self.update_fov_box(current_params['fov_box_visible'], do_render=False)

        if not init:
            if current_params['center_x'] != self.previous_params.get('center_x') or \
                    current_params['center_y'] != self.previous_params.get('center_y') or \
                    current_params['center_z'] != self.previous_params.get('center_z') or \
                    current_params['radius'] != self.previous_params.get('radius') or \
                    current_params['n_points'] != self.previous_params.get('n_points'):
                self.update_streamlines(current_params['center_x'], current_params['center_y'],
                                        current_params['center_z'],
                                        current_params['radius'], current_params['n_points'])

            if current_params['sphere_visible'] != self.previous_params.get('sphere_visible'):
                self.update_sphere_visibility(current_params['sphere_visible'])

        # Update previous parameters
        self.previous_params = current_params

        # self.plotter.show()
        self.updating_flag = False  # Reset the flag
        self.reset_camera_clipping_range()
        self.render()

    def update_slice(self, slice_axis, slice_z, scalar, vmin, vmax, use_interp=True, slice_visible=True):
        """
        Updates the slice plot based on the given parameters.

        :param slice_axis: str
            The axis normal to the slice plane.
        :param slice_z: float
            The slice coordinate along the selected axis.
        :param scalar: str
            The scalar field to use for the slice.
        :param vmin: float
            The minimum value for the color scale.
        :param vmax: float
            The maximum value for the color scale.
        """
        if not slice_visible:
            if self.bottom_slice_actor is not None:
                self.remove_actor(self.bottom_slice_actor)
                self.bottom_slice_actor = None
            return

        axis = slice_axis.lower()
        if slice_z==0:
            slice_z = 1.0e-6
        slice_origin = self._slice_origin(slice_z, axis)
        if use_interp:
            new_slice = self.grid.slice(normal=axis, origin=slice_origin)
            pref = 'point'
            scalar_name = scalar
            scalars = scalar_name
        else:
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            spacing_axis = self.grid_spacing[axis_idx]
            idx = int(round((slice_z - self.grid.origin[axis_idx]) / spacing_axis))
            idx = max(0, min(idx, self.grid_dims[axis_idx] - 1))

            nx, ny, nz = self.grid_dims
            if scalar in ('bx', 'by', 'bz'):
                cube = self.box.b3d[self.b3dtype][scalar]
            else:
                cube = self.box.b3d[self.b3dtype]['bz']

            cube = np.asarray(cube)
            if cube.ndim == 4 and cube.shape[-1] == 3 and scalar in ('bx', 'by', 'bz'):
                comp_idx = {'bx': 0, 'by': 1, 'bz': 2}[scalar]
                cube = cube[..., comp_idx]

            if cube.ndim != 3 and cube.size == nx * ny * nz:
                cube = cube.reshape((nx, ny, nz), order='F')

            if cube.ndim == 3:
                if axis == 'x':
                    slice_data = cube[idx, :, :]
                elif axis == 'y':
                    slice_data = cube[:, idx, :]
                else:
                    slice_data = cube[:, :, idx]
            elif cube.ndim == 2 and cube.size == nx * ny:
                slice_data = cube
            else:
                # Fallback to interpolated slice if cube shape is unexpected
                new_slice = self.grid.slice(normal=axis, origin=slice_origin)
                pref = 'point'
                scalar_name = scalar
                scalars = scalar_name
                if self.bottom_slice_actor is None:
                    self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalars, clim=(vmin, vmax), show_edges=False,
                                                            cmap='gray', pickable=False, show_scalar_bar=False,
                                                            preference=pref)
                else:
                    self.remove_actor(self.bottom_slice_actor)
                    self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalars, clim=(vmin, vmax), show_edges=False,
                                                            cmap='gray', pickable=False, reset_camera=False,
                                                            show_scalar_bar=False, preference=pref)
                return

            expected_size = {
                'x': ny * nz,
                'y': nx * nz,
                'z': nx * ny,
            }[axis]
            if slice_data.ndim != 2 or slice_data.size != expected_size:
                if slice_data.size == expected_size:
                    if axis == 'x':
                        slice_data = slice_data.reshape((ny, nz), order='F')
                    elif axis == 'y':
                        slice_data = slice_data.reshape((nx, nz), order='F')
                    else:
                        slice_data = slice_data.reshape((nx, ny), order='F')
                else:
                    # Fallback to interpolated slice if reshaping is impossible
                    new_slice = self.grid.slice(normal=axis, origin=slice_origin)
                    pref = 'point'
                    scalar_name = scalar
                    scalars = scalar_name
                    if self.bottom_slice_actor is None:
                        self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalars, clim=(vmin, vmax), show_edges=False,
                                                                cmap='gray', pickable=False, show_scalar_bar=False,
                                                                preference=pref)
                    else:
                        self.remove_actor(self.bottom_slice_actor)
                        self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalars, clim=(vmin, vmax), show_edges=False,
                                                                cmap='gray', pickable=False, reset_camera=False,
                                                                show_scalar_bar=False, preference=pref)
                    return

            flat_slice = slice_data.ravel(order='F')
            spacing_x = (self.grid_xmax - self.grid_xmin) / float(nx)
            spacing_y = (self.grid_ymax - self.grid_ymin) / float(ny)
            spacing_z = (self.grid_zmax - self.grid_zmin) / float(max(nz, 1))
            scalar_name = "slice_scalar"
            if axis == 'x':
                new_slice = pv.ImageData(
                    dimensions=(1, ny + 1, nz + 1),
                    spacing=(1, spacing_y, spacing_z),
                    origin=(slice_z, self.grid_ymin, self.grid_zmin),
                )
            elif axis == 'y':
                new_slice = pv.ImageData(
                    dimensions=(nx + 1, 1, nz + 1),
                    spacing=(spacing_x, 1, spacing_z),
                    origin=(self.grid_xmin, slice_z, self.grid_zmin),
                )
            else:
                new_slice = pv.ImageData(
                    dimensions=(nx + 1, ny + 1, 1),
                    spacing=(spacing_x, spacing_y, 1),
                    origin=(self.grid_xmin, self.grid_ymin, slice_z),
                )
            new_slice.cell_data[scalar_name] = flat_slice
            new_slice.set_active_scalars(scalar_name, preference='cell')
            pref = 'cell'
            scalars = scalar_name
        if self.bottom_slice_actor is None:
            self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalars, clim=(vmin, vmax), show_edges=False,
                                                    cmap='gray', pickable=False, show_scalar_bar=False,
                                                    preference=pref)
        else:
            self.remove_actor(self.bottom_slice_actor)
            self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalars, clim=(vmin, vmax), show_edges=False,
                                                    cmap='gray', pickable=False, reset_camera=False,
                                                    show_scalar_bar=False, preference=pref)

    def update_base_map(self, base_map, vmin, vmax, base_map_visible=True):
        """
        Render a fixed bottom-plane base map independently of the moving z-slice.
        """
        if (not base_map_visible) or base_map is None or base_map not in self.grid_bottom.array_names:
            if self.base_map_actor is not None:
                self.remove_actor(self.base_map_actor)
                self.base_map_actor = None
            return

        if self.base_map_actor is None:
            self.base_map_actor = self.add_mesh(
                self.grid_bottom,
                scalars=base_map,
                clim=(vmin, vmax),
                show_edges=False,
                cmap='gray',
                pickable=False,
                show_scalar_bar=False,
            )
        else:
            self.remove_actor(self.base_map_actor)
            self.base_map_actor = self.add_mesh(
                self.grid_bottom,
                scalars=base_map,
                clim=(vmin, vmax),
                show_edges=False,
                cmap='gray',
                pickable=False,
                reset_camera=False,
                show_scalar_bar=False,
            )

    @staticmethod
    def _wireframe_box_from_points(points: np.ndarray):
        pts = np.asarray(points, dtype=float).reshape((-1, 3))
        if pts.shape != (8, 3):
            return None
        edges = (
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        line_cells = []
        for start, end in edges:
            line_cells.extend((2, int(start), int(end)))
        mesh = pv.PolyData()
        mesh.points = pts
        mesh.lines = np.asarray(line_cells, dtype=np.int32)
        return mesh

    def _model_box_mesh(self):
        corners = self.box.model_box_corners_local_mm()
        return self._wireframe_box_from_points(corners)

    def _fov_box_corners_local(self):
        corners = self.box.fov_box_corners_local_mm()
        if corners is None:
            observer_meta = self.box.b3d.get("observer", {}) if isinstance(self.box.b3d, dict) else {}
            fov_box = observer_meta.get("fov_box") if isinstance(observer_meta, dict) else None
            if not isinstance(fov_box, dict):
                print("FOV box overlay: missing observer['fov_box'] metadata.")
            else:
                print("FOV box overlay: incomplete or invalid observer['fov_box'] metadata.")
            return None
        return np.asarray(corners, dtype=float)

    def _fov_box_mesh(self):
        corners = self._fov_box_corners_local()
        if corners is None:
            return None
        return self._wireframe_box_from_points(corners)

    def update_model_box(self, visible=True, do_render=True):
        mesh = self._model_box_mesh()
        if mesh is None:
            if self.model_box_actor is not None:
                self.remove_actor(self.model_box_actor)
                self.model_box_actor = None
                if do_render:
                    self.render()
            return
        if self.model_box_actor is None:
            self.model_box_actor = self.add_mesh(
                mesh.tube(radius=0.35),
                color="red",
                pickable=False,
                reset_camera=False,
                lighting=False,
            )
        self.model_box_actor.SetVisibility(bool(visible))
        if do_render:
            self.render()

    def update_fov_box(self, visible=True, do_render=True):
        mesh = self._fov_box_mesh()
        if mesh is None:
            if self.fov_box_actor is not None:
                self.remove_actor(self.fov_box_actor)
                self.fov_box_actor = None
                if do_render:
                    self.render()
            return
        if self.fov_box_actor is None:
            self.fov_box_actor = self.add_mesh(
                mesh.tube(radius=0.35),
                color="deepskyblue",
                pickable=False,
                reset_camera=False,
                scalars=None,
                lighting=False,
            )
        self.fov_box_actor.SetVisibility(bool(visible))
        if do_render:
            self.render()

    def create_streamlines(self, center_x, center_y, center_z, radius, n_points):
        self.streamlines = self.grid.streamlines(vectors='vectors', source_center=(center_x, center_y, center_z),
                                                 source_radius=radius, n_points=n_points, integration_direction='both',
                                                 max_length=5000, progress_bar=False)
        if self.streamlines.n_points > 0:
            tube = self.streamlines.tube(radius=0.1)
            if tube.n_points <= 0:
                self.streamlines_actor = None
                print("No streamlines generated.")
                return
            if self.streamlines_actor is None:
                self.streamlines_actor = self.add_mesh(tube, pickable=False,
                                                       reset_camera=False, show_scalar_bar=False)
            else:
                self.remove_actor(self.streamlines_actor)
                self.streamlines_actor = self.add_mesh(tube, pickable=False,
                                                       reset_camera=False, show_scalar_bar=False)
        else:
            self.streamlines_actor = None
            print("No streamlines generated.")

    def update_streamlines(self, center_x, center_y, center_z, radius, n_points):
        """
        Updates the streamline plot based on the given parameters.

        :param center_x: float
            The X coordinate of the center of the sphere.
        :param center_y: float
            The Y coordinate of the center of the sphere.
        :param center_z: float
            The Z coordinate of the center of the sphere.
        :param radius: float
            The radius of the sphere.
        :param n_points: int
            The number of seed points for the streamlines.
        """
        sphere = self.spheres[self.current_sphere_id]
        streamlines_actor = sphere['streamlines_actor']
        streamlines = self.grid.streamlines(vectors='vectors', source_center=(center_x, center_y, center_z),
                                            source_radius=radius, n_points=n_points, integration_direction='both',
                                            max_length=5000, progress_bar=False)
        self.spheres[self.current_sphere_id]['streamlines'] = streamlines
        if streamlines.n_points > 0:
            tube = streamlines.tube(radius=0.1)
            if tube.n_points <= 0:
                if streamlines_actor is not None:
                    self.remove_actor(streamlines_actor)
                self.spheres[self.current_sphere_id]['streamlines_actor'] = None
                print("No streamlines generated.")
                return
            if streamlines_actor is None:
                streamlines_actor = self.add_mesh(tube, pickable=False,
                                                  reset_camera=False, show_scalar_bar=False)
            else:
                self.remove_actor(streamlines_actor)
                streamlines_actor = self.add_mesh(tube, pickable=False,
                                                  reset_camera=False, show_scalar_bar=False)
            self.spheres[self.current_sphere_id]['streamlines_actor'] = streamlines_actor
        else:
            if streamlines_actor is not None:
                self.remove_actor(streamlines_actor)
            self.spheres[self.current_sphere_id]['streamlines_actor'] = None
            print("No streamlines generated.")

    def update_sphere(self):
        """
        Updates the sphere widget based on the current input parameters.
        """
        if self.current_sphere_id in self.spheres:
            if 'sphere_actor' in self.spheres[self.current_sphere_id]:
                sphere_actor = self.spheres[self.current_sphere_id]['sphere_actor']
            else:
                sphere_actor = None
        else:
            sphere_actor = None
        if sphere_actor is not None:
            center_x = float(self.center_x_input.text())
            center_y = float(self.center_y_input.text())
            center_z = float(self.center_z_input.text())
            radius = float(self.radius_input.text())

            self.spheres[self.current_sphere_id]['center'] = (center_x, center_y, center_z)
            self.spheres[self.current_sphere_id]['radius'] = radius
            sphere_actor.SetCenter(self.spheres[self.current_sphere_id]['center'])
            sphere_actor.SetRadius(self.spheres[self.current_sphere_id]['radius'])
            self.update_plot()
            self._persist_line_seeds()


    def on_lock_z_changed(self, state):
        if state == Qt.Checked:
            if self.current_sphere_id in self.spheres:
                center_x = float(self.center_x_input.text())
                center_y = float(self.center_y_input.text())
                center_z = float(self.center_z_input.text())
                radius = float(self.radius_input.text())
                self.spheres[self.current_sphere_id]['sphere_actor'].Off()
                self.spheres[self.current_sphere_id]['sphere_actor'].RemoveAllObservers()
                self.spheres[self.current_sphere_id]['sphere_actor'] = self.add_sphere_widget(
                    self._on_sphere_constrained_move,
                    center=(center_x, center_y, center_z),
                    radius=radius,
                    theta_resolution=18,
                    phi_resolution=18,
                    style='wireframe'
                )
        else:
            if self.current_sphere_id in self.spheres:
                center = self.spheres[self.current_sphere_id]['center']
                radius = self.spheres[self.current_sphere_id]['radius']
                self.spheres[self.current_sphere_id]['sphere_actor'].Off()
                self.spheres[self.current_sphere_id]['sphere_actor'].RemoveAllObservers()
                self.spheres[self.current_sphere_id]['sphere_actor'] = self.add_sphere_widget(
                    self._on_sphere_moved,
                    center=center,
                    radius=radius,
                    theta_resolution=18,
                    phi_resolution=18,
                    style='wireframe'
                )

    def update_sphere_visibility(self, sphere_visible):
        """
        Updates the visibility of the sphere widget.

        :param sphere_visible: bool
            Whether the sphere widget is visible.
        """

        if self.current_sphere_id in self.spheres:
            if 'sphere_actor' in self.spheres[self.current_sphere_id]:
                sphere_actor = self.spheres[self.current_sphere_id]['sphere_actor']
            else:
                sphere_actor = None
        else:
            sphere_actor = None

        self.spheres[self.current_sphere_id]['sphere_visible'] = sphere_visible
        if sphere_visible:
            if sphere_actor is None:
                center_x = float(self.center_x_input.text())
                center_y = float(self.center_y_input.text())
                center_z = float(self.center_z_input.text())
                radius = float(self.radius_input.text())
                move_callback = self._on_sphere_constrained_move if self.lock_z_checkbox.isChecked() else self._on_sphere_moved
                # move_callback = self._on_sphere_moved
                sphere_actor = self.add_sphere_widget(move_callback,
                                                      center=(center_x, center_y, center_z),
                                                      radius=radius, theta_resolution=18, phi_resolution=18,
                                                      style='wireframe')
                self.spheres[self.current_sphere_id]['sphere_actor'] = sphere_actor
                # self.spheres[self.current_sphere_id]['initial_position'] = (center_x, center_y, center_z)
                # sphere_actor.AddObserver("StartInteractionEvent", self.start_sphere_interaction)
            else:
                sphere_actor.On()
        else:
            if sphere_actor is not None:
                sphere_actor.Off()

        if self.viz_sphere_button.isChecked() != sphere_visible:
            self.viz_sphere_button.disconnect()
            self.viz_sphere_button.setChecked(sphere_visible)
            self.viz_sphere_button.toggled.connect(self.toggle_sphere_visibility)
        self._persist_line_seeds()

    def _on_sphere_moved(self, center):
        """
        Handles the event when the sphere widget is moved.

        :param center: list of float
            The new center coordinates of the sphere.
        """
        self.center_x_input.setText(f"{center[0]:.2f}")
        self.center_y_input.setText(f"{center[1]:.2f}")
        self.center_z_input.setText(f"{center[2]:.2f}")
        self.update_sphere()

    def _on_sphere_constrained_move(self, center):
        """
        Moves the sphere in the plane z = center_z_input when 'Lock Z' is checked.

        :param center: list of float
            The new center coordinates of the sphere.
        """
        fixed_z = float(self.center_z_input.text())

        # Update the sphere's position but constrain the Z coordinate to fixed_z
        new_sphere_pos = [center[0], center[1], fixed_z]

        # Update the sphere actor position
        if  self.spheres[self.current_sphere_id]['sphere_actor'] is not None:
            self.spheres[self.current_sphere_id]['sphere_actor'].SetCenter(new_sphere_pos)


        # Update the input fields
        self.center_x_input.setText(f"{center[0]:.2f}")
        self.center_y_input.setText(f"{center[1]:.2f}")
        self.update_sphere()

    def toggle_sphere_visibility(self, state):
        """
        Toggles the visibility of the sphere widget.

        :param state: int
            The state of the checkbox (checked or unchecked).
        """
        if self.viz_sphere_button.isChecked():
            self.viz_sphere_button.setToolTip("Hide the sphere")
        else:
            self.viz_sphere_button.setToolTip("Show the sphere")

        self.sphere_visible = state == Qt.Checked
        if len(self.spheres) > 0:
            self.update_plot()

    def update_plane(self):
        """
        Updates the plane widget based on the current input parameters.
        """
        if self.plane_actor is not None:
            slice_pos = float(self.slice_z_input.value()) if isinstance(self.slice_z_input, QDoubleSpinBox) else float(self.slice_z_input.text())
            self.slice_axis_positions[self.slice_axis] = slice_pos
            origin = self._slice_origin(slice_pos)
            if hasattr(self.plane_actor, "SetNormal"):
                self.plane_actor.SetNormal(self._slice_normal_vector())
            self.plane_actor.SetOrigin(origin)
            self.update_plot()

    def update_plane_visibility(self, plane_visible):
        """
        Updates the visibility of the plane widget.

        :param plane_visible: bool
            Whether the plane widget is visible.
        """
        if plane_visible:
            if self.plane_actor is None:
                slice_pos = float(self.slice_z_input.value()) if isinstance(self.slice_z_input, QDoubleSpinBox) else float(self.slice_z_input.text())
                self.slice_axis_positions[self.slice_axis] = slice_pos
                self.plane_actor = self.add_plane_widget(self._on_plane_moved, normal=self._slice_normal_vector(),
                                                         origin=self._slice_origin(slice_pos), bounds=(
                        self.grid_xmin, self.grid_xmax, self.grid_ymin, self.grid_ymax, self.grid_zmin, self.grid_zmax),
                                                         normal_rotation=False)
            else:
                if hasattr(self.plane_actor, "SetNormal"):
                    self.plane_actor.SetNormal(self._slice_normal_vector())
                self.plane_actor.SetOrigin(self._slice_origin())
                self.plane_actor.On()
        else:
            if self.plane_actor is not None:
                self.plane_actor.Off()

    def _on_plane_moved(self, normal, origin):
        """
        Handles the event when the plane widget is moved.

        :param normal: list of float
            The normal vector of the plane.
        :param origin: list of float
            The new origin coordinates of the plane.
        """
        coord = float(origin[{'x': 0, 'y': 1, 'z': 2}[self.slice_axis]])
        self.slice_axis_positions[self.slice_axis] = coord
        if isinstance(self.slice_z_input, QDoubleSpinBox):
            self.slice_z_input.setValue(coord)
        else:
            self.slice_z_input.setText(f"{coord:.2f}")
        self.update_plane()

    def toggle_plane_visibility(self, state):
        """
        Toggles the visibility of the plane widget.

        :param state: int
            The state of the checkbox (checked or unchecked).
        """
        self.plane_visible = state == Qt.Checked
        self.update_plot()

    def toggle_slice_visibility(self, state):
        """
        Toggles the visibility of the z-slice actor while preserving the selected scalar.

        :param state: int
            The state of the checkbox (checked or unchecked).
        """
        self.slice_visible = state == Qt.Checked
        self.update_plot()

    def toggle_base_map_visibility(self, state):
        self.base_map_visible = state == Qt.Checked
        base_map = self.base_map_selector.currentText() if self.base_map_selector is not None else "none"
        self.update_base_map(
            base_map,
            float(self.base_vmin_input.value()) if self.base_vmin_input is not None else -1000.0,
            float(self.base_vmax_input.value()) if self.base_vmax_input is not None else 1000.0,
            self.base_map_visible,
        )
        self.previous_params["base_map_visible"] = self.base_map_visible
        self.previous_params["base_map"] = base_map
        self.reset_camera_clipping_range()
        self.render()

    def toggle_model_box_visibility(self, state):
        self.model_box_visible = state == Qt.Checked
        self.update_model_box(self.model_box_visible)
        self.previous_params["model_box_visible"] = self.model_box_visible
        self.reset_camera_clipping_range()
        self.render()

    def toggle_fov_box_visibility(self, state):
        self.fov_box_visible = state == Qt.Checked
        self.update_fov_box(self.fov_box_visible)
        self.previous_params["fov_box_visible"] = self.fov_box_visible
        self.reset_camera_clipping_range()
        self.render()

    def send_streamlines(self):
        """
        Sends the streamline data of all spheres to the parent object (if any).
        """
        print(f"Sending streamlines to {self.parent}")
        if self.parent is not None:
            streamlines = []
            for sphere in self.spheres.values():
                if sphere['streamlines_actor'] is not None:
                    if sphere['streamlines'].n_lines > 0:
                        streamlines.append(sphere['streamlines'])
            if streamlines != []:
                self.parent.plot_fieldlines(streamlines, z_base=self.grid_zbase)

    def _collect_streamlines(self):
        streamlines = []
        for sphere in self.spheres.values():
            if sphere.get('streamlines_actor') is not None and sphere.get('streamlines') is not None:
                if sphere['streamlines'].n_lines > 0:
                    streamlines.append(sphere['streamlines'])
        return streamlines

    def _collect_line_seeds_snapshot(self):
        line_seeds = self.box.b3d.get("line_seeds")
        if isinstance(line_seeds, dict):
            return copy.deepcopy(line_seeds)
        return None

    def save_current_model(self):
        if self.session_mode == "embedded":
            return False
        if not self.model_path:
            QMessageBox.warning(self.app_window, "Save Failed", "No writable .h5 model path is attached to this 3D viewer.")
            return False
        try:
            update_line_seeds_h5(str(self.model_path), self._collect_line_seeds_snapshot())
            self._original_line_seeds = self._collect_line_seeds_snapshot()
            print(f"Saved line seeds to {self.model_path}")
            return True
        except Exception as exc:
            QMessageBox.warning(self.app_window, "Save Failed", f"Could not save line seeds to the current model:\n{exc}")
            return False

    def _close_window(self):
        if hasattr(self, "app_window"):
            self.app_window.close()
        else:
            self.close()

    def accept_and_close(self):
        if self.session_mode == "embedded" and self.parent is not None and hasattr(self.parent, "commit_live_3d_edits"):
            self.parent.commit_live_3d_edits(
                self._collect_line_seeds_snapshot(),
                self._collect_streamlines(),
                z_base=self.grid_zbase,
            )
            self._embedded_close_mode = "accept"
            self._close_window()
            return
        if self.session_mode == "pipeline_child":
            if self.save_current_model():
                self._close_window()
            return
        self._close_window()

    def cancel_and_close(self):
        if self.session_mode == "embedded" and self.parent is not None and hasattr(self.parent, "cancel_live_3d_edits"):
            self.parent.cancel_live_3d_edits()
            self._embedded_close_mode = "cancel"
        self._close_window()

    def undo_and_restore(self):
        self._restore_line_seeds(self._original_line_seeds if isinstance(self._original_line_seeds, dict) else {})


    def save_box(self):
        box_dims_str = 'x'.join(map(str, self.box.dims_pix))
        default_filename = f'b3d_data_{self.box._frame_obs.obstime.to_datetime().strftime("%Y%m%dT%H%M%S")}_dim{box_dims_str}.h5'
        parent_widget = self.app_window if hasattr(self, "app_window") else None
        filename = QFileDialog.getSaveFileName(parent_widget, "Save Box As", default_filename, "HDF5 Files (*.h5)")[0]
        if not filename:
            return
        write_b3d_h5(filename, self.box.b3d)
        try:
            from pyampp.util.build_h5_from_sav import _apply_geometry_contract_to_h5
            _apply_geometry_contract_to_h5(Path(filename))
        except Exception:
            pass  # geometry contract application is best-effort
        self.model_path = Path(filename)
        if self.save_model_button is not None and not self.save_model_button.isEnabled():
            self.save_model_button.setEnabled(True)
            self.save_model_button.setToolTip("Save the current seed state back into the opened model file.")
            self.save_model_button.clicked.connect(self.save_current_model)

    def load_box(self):
        default_filename = "b3d_data.h5"
        filename = QFileDialog.getOpenFileName(self, "Load Box", default_filename, "HDF5 Files (*.h5)")[0]
        if not filename:
            return
        self.box.b3d = read_b3d_h5(filename)

        if "corona" in self.box.b3d:
            self.b3dtype = "corona"
        elif "nlfff" in self.box.b3d:
            self.b3dtype = "corona"
            self.box.b3d["corona"] = self.box.b3d.pop("nlfff")
        elif "pot" in self.box.b3d:
            self.b3dtype = "corona"
            self.box.b3d["corona"] = self.box.b3d.pop("pot")
        elif "chromo" in self.box.b3d:
            self.b3dtype = "chromo"
            chromo = self.box.b3d.get("chromo", {})
            if "bx" not in chromo and "bcube" in chromo:
                bcube = chromo["bcube"]
                if bcube.ndim == 4 and bcube.shape[-1] == 3:
                    chromo["bx"] = bcube[:, :, :, 0]
                    chromo["by"] = bcube[:, :, :, 1]
                    chromo["bz"] = bcube[:, :, :, 2]
                    self.box.b3d["chromo"] = chromo
        self.init_grid()
        self._apply_streamline_control_state()
        self.previous_params = {}
        self.update_plot()
        self._restore_line_seeds_from_box()
