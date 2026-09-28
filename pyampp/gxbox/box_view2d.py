from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import astropy.units as u
import matplotlib.colors as mcolors
from astropy.io import fits
from astropy.constants import R_sun
from astropy.coordinates import SkyCoord
from astropy.time import Time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import Qt, QTimer, QThread, QSize
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QStyle,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from sunpy.map import Map, make_fitswcs_header
from sunpy.coordinates import (
    Heliocentric,
    HeliographicCarrington,
    HeliographicStonyhurst,
    Helioprojective,
    get_earth,
    sun,
)
from sunpy.visualization import colormaps as sunpy_colormaps

from pyampp.gxbox.gx_fov2box import _submap_with_fov_safe
from pyampp.geometry import (
    build_fov_box_from_red_box_world,
    build_fov_box_from_user_hpc_and_red_box_world,
    local_cartesian_to_world,
    observer_fov_box_to_world_corners,
    observer_rectangle_to_hpc_corners,
    project_box_front_face_to_observer_hpc,
    project_coordinate_edges_to_observer_hpc,
    project_world_to_observer_hpc,
    project_world_to_pixel,
)
from .box import Box
from .boxutils import load_sunpy_map_compat, map_from_data_header_compat
from .observer_restore import (
    build_ephemeris_from_pb0r,
    build_pb0r_metadata_from_ephemeris,
    resolve_observer_parameters_from_ephemeris,
    resolve_observer_with_info,
)
from .selector_api import BoxGeometrySelection, CoordMode, DisplayFovBoxSelection, DisplayFovSelection, SelectorSessionInput

logging.getLogger("sunpy").setLevel(logging.WARNING)

_CONTEXT_DISPLAY_MAP_ALIASES = {
    "Bz": "magnetogram",
    "Ic": "continuum",
    "B_rho": "field",
    "B_theta": "inclination",
    "B_phi": "azimuth",
    "disambig": "disambig",
    # Backward-compatible legacy IDs now mapped to measured HPC products.
    "Br": "field",
    "Bp": "inclination",
    "Bt": "azimuth",
}
_BOTTOM_DISPLAY_MAP_ALIASES = {
    "Bx": "bx",
    "By": "by",
    "Bz": "bz",
    "Ic": "ic",
    "chromo_mask": "chromo_mask",
    "Chromo_mask": "chromo_mask",
    "Vert_current": "vert_current",
    "vert_current": "vert_current",
}

_HMI_VECTOR_SEGMENTS = ("field", "inclination", "azimuth", "disambig")
_HMI_DISPLAY_KEYS = {"magnetogram", "continuum", "field", "inclination", "azimuth", "disambig"}
_SIGNED_MAGNETIC_KEYS = {"magnetogram", "bx", "by", "bz"}
_TRANSVERSE_MAGNETIC_KEYS = set()
_VERT_CURRENT_KEYS = {"Vert_current", "vert_current"}
_CHROMO_MASK_KEYS = {"chromo_mask"}
_HMI_VECTOR_DISPLAY_KEYS = {"field", "inclination", "azimuth", "disambig"}
_AIA_COLOR_KEYS = {"94", "131", "1600", "1700", "171", "193", "211", "304", "335"}
_EOVSA_REFMAP_PREFIX = "EOVSA_"
_BOTTOM_OVERLAY_CONTEXT_KEYS = _AIA_COLOR_KEYS | _HMI_VECTOR_DISPLAY_KEYS
_EMBEDDED_REFMAP_FLAG = "PYEMBED"
_CONTEXT_PREPARE_VARIANT_FULL_DISK = "full_disk"
_CONTEXT_PREPARE_VARIANT_FOV_CROP = "fov_crop"
_EMBEDDED_REFMAP_FOV_PAD_FACTOR = 1.10
_MIN_DISPLAY_MAP_SIDE = 32
# Unified map display policy (regression-locked):
# Tier 1 (_native_crop_cache): on upload, each reference map is cropped in its native
# LOS by projecting the model box from fov_definition_observer_key, inscribing a FOV,
# and cropping at _EMBEDDED_REFMAP_FOV_PAD_FACTOR. Background prewarm must not touch
# the visible viewport.
# Tier 2 (_display_prepared_cache): on display, maps are reprojected to display_observer
# when native LOS differs; results are cached per session and reused on later switches.
# _prepare_map_for_display() is the single entry point for context, base overlays,
# foreground display, and prewarm.
# PYALIGN embedded maps skip tier-1 crop (pre-cropped at embed). Cross-observer PYALIGN
# display uses a projected 1.1x FOV ROI for reprojection ([roi], not [full]).
# Locked by: pyampp/tests/test_pyalign_cross_observer_reprojection.py,
# pyampp/tests/test_embedded_display_crop.py
# Full Sun View viewport policy (regression-locked):
# Zoom to Full Sun must center on display-observer HPC disk center (0, 0), not the FOV
# center, and size the square viewport from RSUN_ARCSEC / map plate scale (primary path).
# Overlay pixel rects (red/blue boxes) may expand the square but must never replace disk
# sizing. Do not fall back to _projected_box_bbox_rect for disk extent conversion.
# Embedded PYALIGN maps in full_sun mode reproject via disk HPC ROI (see _prepare_context_map).
# Locked by: pyampp/tests/test_box_view_full_sun.py
_FULL_SUN_DISK_EXTENT_PAD = 1.05
_FULL_SUN_VIEWPORT_PAD = 1.02
_BOX_EDGE_INDEX_PAIRS = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_DISPLAY_OBSERVER_OPTIONS = (
    ("earth", "Earth"),
    ("solar orbiter", "Solar Orbiter"),
    ("stereo-a", "STEREO-A"),
    ("stereo-b", "STEREO-B"),
)
_DISPLAY_OBSERVER_LABELS = {
    **dict(_DISPLAY_OBSERVER_OPTIONS),
    "custom": "Custom",
}
_DISPLAY_OBSERVER_HORIZONS = {
    "solar orbiter": "Solar Orbiter",
    "stereo-a": "STEREO-A",
    "stereo-b": "STEREO-B",
}


def _prepare_model_for_viewer(*args, **kwargs):
    from .view_h5 import prepare_model_for_viewer

    return prepare_model_for_viewer(*args, **kwargs)


def _viewer_camera_basis(*args, **kwargs):
    from .view_h5 import _viewer_camera_vectors

    return _viewer_camera_vectors(*args, **kwargs)


def _generate_streamlines_from_seeds(*args, **kwargs):
    from .box_view3d import generate_streamlines_from_line_seeds

    return generate_streamlines_from_line_seeds(*args, **kwargs)


def _magfield_viewer_cls():
    from .box_view3d import MagFieldViewer

    return MagFieldViewer


@dataclass
class MapBoxViewState:
    """
    Reusable state container for focused map+box visualization tools.

    This is intentionally lightweight and plotting-backend agnostic so it can be
    reused by multiple future GUIs (FOV selector, box inspector, model preview, ...).
    """

    session_input: SelectorSessionInput
    selected_context_id: Optional[str] = None
    selected_bottom_id: Optional[str] = None
    geometry: Optional[BoxGeometrySelection] = None
    fov: Optional[DisplayFovSelection] = None
    fov_box: Optional[DisplayFovBoxSelection] = None
    map_files: dict[str, str] | None = None
    refmaps: dict | None = None
    base_maps: dict | None = None
    base_wcs_header: str | None = None
    base_geometry: Optional[BoxGeometrySelection] = None
    map_source_mode: str = "auto"
    square_fov: bool = False
    display_observer_key: str = "earth"
    geometry_definition_observer_key: str = "earth"
    fov_definition_observer_key: str = "earth"
    custom_observer_ephemeris: dict | None = None
    custom_observer_label: str | None = None
    custom_observer_source: str | None = None


class _SquareCanvasHost(QWidget):
    """Keep the embedded plot canvas left-aligned with a fixed-width rectangular host."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._canvas = None
        self.setMinimumWidth(600)
        self.setMaximumWidth(600)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    def set_canvas(self, canvas: QWidget) -> None:
        self._canvas = canvas
        self._canvas.setParent(self)
        self._canvas.show()
        self._reposition_canvas()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_canvas()

    def _reposition_canvas(self) -> None:
        if self._canvas is None:
            return
        w, h = self.width(), self.height()
        x = 0
        self._canvas.setGeometry(x, 0, max(1, w), max(1, h))


class MapBoxDisplayWidget(QWidget):
    """
    Reusable widget shell for map display + interactive box overlays.

    Current implementation provides:
    - SunPy map plotting using the map's native WCS projection
    - a static box-outline overlay derived from the current geometry state
    - a stable API for future drag/resize interaction layers
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._svg_dir = self._resolve_svg_dir()
        self._state: Optional[MapBoxViewState] = None
        self._geometry_change_callback = None
        self._map_summary_cache: dict[str, str] = {}
        self._loaded_map_cache = {}
        self._native_crop_cache = {}
        self._display_prepared_cache = {}
        self._raw_map_cache = {}
        self._cache_lock = threading.RLock()
        self._background_cache_enabled = False
        self._background_cache_generation = 0
        self._background_cache_thread = None
        self._context_prewarm_generation = 0
        self._context_prewarm_thread = None
        self._context_prewarm_active = False
        self._current_map = None
        self._current_axes = None
        self._overlay_rect = None
        self._overlay_bbox_rect = None
        self._projected_box_bbox_rect = None
        self._projected_box_fov = None
        self._overlay_center_artist = None
        self._overlay_corner_artists = []
        self._overlay_line_artists = []
        self._drag_preview_box_artist = None
        self._drag_preview_fov_artist = None
        self._drag_preview_center_artist = None
        self._drag_preview_background = None
        self._drag_preview_active = False
        self._drag_preview_geometry = None
        self._zoom_anchor_px: tuple[float, float] | None = None
        self._drag_state = None
        self._entry_box_path: Optional[Path] = None
        self._viewer3d = None
        self._viewer3d_temp_h5_path: Optional[Path] = None
        self._viewer3d_watchdog = QTimer(self)
        self._viewer3d_watchdog.setInterval(400)
        self._viewer3d_watchdog.timeout.connect(self._check_viewer3d_state)
        self._hidden_for_live_3d = False
        self._viewer3d_close_handled = False
        self._committed_line_seeds = None
        self._session_box_template = None
        self._session_obs_time = None
        self._session_b3dtype = None
        self._session_temp_h5_path: Optional[Path] = None
        self._session_model_loaded = False
        self._fieldline_frame_hcc = None
        self._fieldline_frame_obs = None
        self._fieldline_streamlines = []
        self._fieldline_z_base = 0.0
        self._fieldline_artists = []
        self._map_info_callback = None
        self._status_callback = None
        self._fov_change_callback = None
        self._observer_info_callback = None
        self._observer_coord_cache: dict[str, SkyCoord] = {}
        self._observer_metadata_cache: dict[tuple[str, ...], dict] = {}
        self._observer_warning_cache: set[str] = set()
        self._observer_refresh_serial = 0
        self._available_observer_keys_override: set[str] | None = None
        self._observer_availability_notice: str | None = None
        self._refmap_display_notices: list[str] = []
        self._pending_launch_margin_fix = False
        self._last_map_info_text = "Map info: <uninitialized>"
        self._last_status_base_text = "Map/box display initialized"
        self._last_status_text = "Map/box display initialized"
        self._last_context_summary_text = "Context map: <uninitialized>"
        self._last_bottom_summary_text = "Base map: <uninitialized>"
        self._prep_trace_counts: dict[str, int] = {}
        self._prep_trace_order: list[str] = []
        self._view_mode = "box_fov"
        self._full_view_limits = None
        self._interaction_mode = "auto"
        self._mouse_actions_enabled = False
        self._geometry_edit_enabled = True
        self._action_state_callback = None
        self._fig = Figure(figsize=(7.5, 5.0))
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cid_press = self._canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._cid_move = self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._cid_release = self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._cid_scroll = self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas_host = _SquareCanvasHost()
        self._canvas_host.set_canvas(self._canvas)
        self._nav_toolbar = NavigationToolbar(self._canvas, self)
        self._nav_toolbar.setMinimumWidth(600)
        self._nav_toolbar.setMaximumWidth(600)
        self._nav_toolbar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.show_loading_placeholder("Preparing viewer data...\nPlease wait.")

        self._full_view_btn = self._make_svg_button("expand.svg", "Full Sun View", self.show_full_sun_view)
        self._box_view_btn = self._make_svg_button("shrink.svg", "Zoom canvas to image FOV", self.show_box_fov_view)
        self._recompute_fov_btn = self._make_svg_button("rectangle-horizontal.svg", "Recompute image FOV from projected 3D box", self.recompute_fov_from_box)
        self._control_mode_label = QLabel("BOX Controls")
        self._left_btn = self._make_svg_button("arrow-left.svg", "Move box center left", lambda: self._nudge_primary_center("x", -1))
        self._right_btn = self._make_svg_button("arrow-right.svg", "Move box center right", lambda: self._nudge_primary_center("x", +1))
        self._down_btn = self._make_svg_button("arrow-down.svg", "Move box center down", lambda: self._nudge_primary_center("y", -1))
        self._up_btn = self._make_svg_button("arrow-up.svg", "Move box center up", lambda: self._nudge_primary_center("y", +1))
        self._x_minus_btn = self._make_svg_button("shrink-horizontal.svg", "Decrease X box size", lambda: self._nudge_primary_size("x", -1))
        self._x_plus_btn = self._make_svg_button("expand-horizontal.svg", "Increase X box size", lambda: self._nudge_primary_size("x", +1))
        self._y_minus_btn = self._make_svg_button("shrink-vertical.svg", "Decrease Y box size", lambda: self._nudge_primary_size("y", -1))
        self._y_plus_btn = self._make_svg_button("expand-vertical.svg", "Increase Y box size", lambda: self._nudge_primary_size("y", +1))
        self._xy_minus_btn = self._make_svg_button("shrink.svg", "Decrease X and Y box size together", lambda: self._nudge_primary_size_xy(-1))
        self._xy_plus_btn = self._make_svg_button("expand.svg", "Increase X and Y box size together", lambda: self._nudge_primary_size_xy(+1))
        self._zoom_in_btn = self._make_svg_button("zoom-in.svg", "Zoom In (centered on image FOV)", lambda: self._scale_view(1 / 1.25))
        self._zoom_out_btn = self._make_svg_button("zoom-out.svg", "Zoom Out (centered on image FOV)", lambda: self._scale_view(1.25))
        self._can_open_3d = False
        self._can_clear_lines = False

        zoom_label = QLabel("Zoom Controls")
        zoom_toolbar = QHBoxLayout()
        zoom_toolbar.setContentsMargins(0, 0, 0, 0)
        zoom_toolbar.setSpacing(4)
        zoom_toolbar.addWidget(zoom_label)
        zoom_toolbar.addSpacing(8)
        zoom_toolbar.addWidget(self._full_view_btn)
        zoom_toolbar.addWidget(self._box_view_btn)
        zoom_toolbar.addWidget(self._recompute_fov_btn)
        zoom_toolbar.addSpacing(8)
        zoom_toolbar.addWidget(self._zoom_in_btn)
        zoom_toolbar.addWidget(self._zoom_out_btn)
        zoom_toolbar.addStretch()

        control_toolbar = QHBoxLayout()
        control_toolbar.setContentsMargins(0, 0, 0, 0)
        control_toolbar.setSpacing(4)
        control_toolbar.addWidget(self._control_mode_label)
        control_toolbar.addSpacing(8)
        control_toolbar.addWidget(self._left_btn)
        control_toolbar.addWidget(self._right_btn)
        control_toolbar.addWidget(self._down_btn)
        control_toolbar.addWidget(self._up_btn)
        control_toolbar.addSpacing(8)
        control_toolbar.addWidget(self._x_minus_btn)
        control_toolbar.addWidget(self._x_plus_btn)
        control_toolbar.addWidget(self._y_minus_btn)
        control_toolbar.addWidget(self._y_plus_btn)
        control_toolbar.addWidget(self._xy_minus_btn)
        control_toolbar.addWidget(self._xy_plus_btn)
        control_toolbar.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(zoom_toolbar)
        layout.addLayout(control_toolbar)
        layout.addWidget(self._canvas_host, 0, Qt.AlignLeft)
        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(0)
        nav_row.addWidget(self._nav_toolbar, 0, Qt.AlignLeft)
        nav_row.addStretch()
        layout.addLayout(nav_row)
        self._refresh_control_mode_ui()

    def show_loading_placeholder(self, message: str = "Preparing viewer data...") -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.54, message, ha="center", va="center", fontsize=13)
        ax.text(
            0.5,
            0.42,
            "The viewer will populate when maps are ready.",
            ha="center",
            va="center",
            fontsize=10,
            alpha=0.7,
        )
        ax.axis("off")
        self._fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.94)
        self._canvas.draw_idle()

    def showEvent(self, event):
        super().showEvent(event)
        if self._pending_launch_margin_fix:
            QTimer.singleShot(0, self._post_show_margin_refresh)

    def _post_show_margin_refresh(self) -> None:
        if not self._pending_launch_margin_fix:
            return
        ax = self._current_axes
        if ax is None:
            return
        self._pending_launch_margin_fix = False
        adjusted = self._auto_adjust_axes_margins(ax, top=0.93, pad_px=10.0)
        self._render_fieldlines()
        if adjusted:
            self._canvas.draw()
        else:
            self._canvas.draw_idle()

    def _make_mode_button(self, text: str, mode: str, checked: bool = False) -> QToolButton:
        btn = QToolButton(self)
        btn.setCheckable(True)
        btn.setAutoRaise(False)
        btn.setIconSize(QSize(32, 32))
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        icon, fallback = self._mode_button_icon(mode)
        btn.setIcon(icon)
        if icon.isNull():
            btn.setText(fallback)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setToolTip(text)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.clicked.connect(lambda _=False, m=mode: self._set_interaction_mode(m))
        self._mode_group.addButton(btn)
        return btn

    def _make_glyph_button(self, glyph: str, tooltip: str, callback) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(glyph)
        btn.setToolTip(tooltip)
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setAutoRaise(False)
        btn.setFixedSize(24, 24)
        f = QFont(btn.font())
        f.setPointSize(10)
        btn.setFont(f)
        btn.clicked.connect(callback)
        return btn

    def _make_text_button(self, text: str, tooltip: str, callback) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setAutoRaise(False)
        btn.setMinimumHeight(24)
        btn.clicked.connect(callback)
        return btn

    def _make_svg_button(self, svg_name: str, tooltip: str, callback) -> QToolButton:
        btn = QToolButton(self)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(False)
        btn.setFixedSize(32, 32)
        btn.setIconSize(QSize(20, 20))
        icon_path = self._svg_dir / svg_name
        if icon_path.exists():
            btn.setIcon(QIcon(str(icon_path)))
        else:
            btn.setText("?")
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.clicked.connect(callback)
        return btn

    @staticmethod
    def _resolve_svg_dir() -> Path:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[2] / "docs" / "svg",
            here.parents[1] / "docs" / "svg",
            Path.cwd() / "docs" / "svg",
        ]
        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate
            except Exception:
                continue
        return candidates[0]

    def _make_icon_button(self, icon_names, tooltip, callback, fallback_text="") -> QToolButton:
        btn = QToolButton(self)
        btn.setAutoRaise(False)
        btn.setIconSize(QSize(32, 32))
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        icon = self._theme_icon(icon_names)
        if icon.isNull():
            icon = self._fallback_std_icon()
        if not icon.isNull():
            btn.setIcon(icon)
        else:
            btn.setText(fallback_text)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setToolTip(tooltip)
        btn.clicked.connect(callback)
        return btn

    def _mode_button_icon(self, mode: str):
        mapping = {
            "auto": (["transform-move"], "◎"),
            "move": (["transform-move"], "✛"),
            "resize_xy": (["transform-scale"], "⤡"),
            "resize_x": (["object-flip-horizontal"], "↔"),
            "resize_y": (["object-flip-vertical"], "↕"),
        }
        names, fallback = mapping.get(mode, ([], mode))
        icon = self._theme_icon(names)
        if icon.isNull():
            # Use standard cursor-like fallback icons where possible.
            if mode == "move":
                icon = self.style().standardIcon(QStyle.SP_ArrowUp)
            elif mode in {"resize_x", "resize_y", "resize_xy"}:
                icon = self.style().standardIcon(QStyle.SP_TitleBarShadeButton)
        return icon, fallback

    @staticmethod
    def _theme_icon(names) -> QIcon:
        for name in names:
            icon = QIcon.fromTheme(name)
            if not icon.isNull():
                return icon
        return QIcon()

    def _fallback_std_icon(self) -> QIcon:
        try:
            return self.style().standardIcon(QStyle.SP_ArrowRight)
        except Exception:
            return QIcon()

    def _set_interaction_mode(self, mode: str) -> None:
        self._interaction_mode = mode
        self._refresh_status_text()
        self._update_cursor_for_mode()

    @staticmethod
    def _normalize_observer_key(observer_key: str | None) -> str:
        raw = observer_key
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "ignore")
        if isinstance(raw, np.ndarray) and raw.shape == ():
            raw = raw.item()
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", "ignore")
        key = str(raw or "earth").strip().lower()
        aliases = {
            "custom": "custom",
            "sdo": "earth",
            "sdo/aia": "earth",
            "sdo/hmi": "earth",
            "earth": "earth",
            "solo": "solar orbiter",
            "solar-orbiter": "solar orbiter",
            "solarorbiter": "solar orbiter",
            "solar orbiter": "solar orbiter",
            "stereo a": "stereo-a",
            "stereo-a": "stereo-a",
            "stereoa": "stereo-a",
            "stereo b": "stereo-b",
            "stereo-b": "stereo-b",
            "stereob": "stereo-b",
        }
        return aliases.get(key, "earth")

    @staticmethod
    def _observer_label_for_key(observer_key: str | None) -> str:
        return _DISPLAY_OBSERVER_LABELS.get(
            MapBoxDisplayWidget._normalize_observer_key(observer_key),
            "Earth",
        )

    def _enabled_observer_keys(self) -> set[str]:
        allow_only_earth = self._entry_box_path is None
        enabled = {
            key for key, _label in _DISPLAY_OBSERVER_OPTIONS
            if (not allow_only_earth) or key == "earth"
        }
        if self._available_observer_keys_override is not None:
            enabled &= set(self._available_observer_keys_override)
            enabled.add("earth")
        return enabled

    def _observer_source_b3d(self) -> dict:
        source_b3d = getattr(self._session_box_template, "b3d", None)
        if isinstance(source_b3d, dict):
            return source_b3d
        if self._state is None:
            return {}
        payload: dict = {}
        if isinstance(self._state.refmaps, dict) and self._state.refmaps:
            payload["refmaps"] = self._state.refmaps
        return payload

    def _normalize_display_observer_state(self) -> None:
        if self._state is None:
            return
        enabled_keys = self._enabled_observer_keys()
        if (
            "earth" in enabled_keys
            and self._state.display_observer_key not in enabled_keys
            and self._normalize_observer_key(self._state.display_observer_key) != "custom"
        ):
            self._state.display_observer_key = "earth"

    def set_display_observer_key(self, observer_key: str | None) -> None:
        """Switch display LOS for map reprojection/overlays only.

        Does not mutate ``_state.fov``, ``fov_box``, or ``fov_definition_observer_key``.
        Users may inspect another LOS and switch back without losing the prior FOV.
        Save-time alignment is handled explicitly via the FOV mismatch dialog.
        """
        if self._state is None:
            return
        key = self._normalize_observer_key(observer_key)
        enabled_keys = self._enabled_observer_keys()
        if "earth" in enabled_keys and key not in enabled_keys and key != "custom":
            key = "earth"
        if key == self._state.display_observer_key:
            self._normalize_display_observer_state()
            self._emit_observer_info()
            return
        self._state.display_observer_key = key
        self._normalize_display_observer_state()
        self._invalidate_display_prepared_cache()
        self._refresh_status_text()
        self._emit_observer_info()
        preserve_current_view = self._should_preserve_pixel_view()
        if self._view_mode == "box_fov":
            preserve_current_view = False
        self._schedule_observer_refresh(
            preserve_current_view=preserve_current_view,
            align_projected_fov=(self._view_mode == "box_fov"),
        )

    def set_custom_display_observer_pb0r(
        self,
        *,
        b0_deg,
        l0_deg,
        rsun_arcsec,
        obs_date=None,
        rsun_cm=None,
        label: str | None = None,
        source: str | None = None,
    ) -> bool:
        if self._state is None:
            return False
        ephemeris = build_ephemeris_from_pb0r(
            b0_deg=b0_deg,
            l0_deg=l0_deg,
            rsun_arcsec=rsun_arcsec,
            obs_date=obs_date,
            rsun_cm=rsun_cm,
        )
        if ephemeris is None:
            return False
        self._state.custom_observer_ephemeris = ephemeris
        if label is not None:
            self._state.custom_observer_label = str(label).strip() or "Custom"
        elif not self._state.custom_observer_label:
            self._state.custom_observer_label = "Custom"
        if source is not None:
            self._state.custom_observer_source = str(source).strip() or None
        self._state.display_observer_key = "custom"
        self._normalize_display_observer_state()
        self._refresh_status_text()
        self._emit_observer_info()
        preserve_current_view = self._should_preserve_pixel_view()
        if self._view_mode == "box_fov":
            preserve_current_view = False
        self._schedule_observer_refresh(
            preserve_current_view=preserve_current_view,
            align_projected_fov=(self._view_mode == "box_fov"),
        )
        return True

    def set_custom_observer_identity(self, *, label: str | None = None, source: str | None = None) -> None:
        if self._state is None:
            return
        changed = False
        if label is not None:
            normalized_label = str(label).strip() or "Custom"
            if normalized_label != (self._state.custom_observer_label or ""):
                self._state.custom_observer_label = normalized_label
                changed = True
        if source is not None:
            normalized_source = str(source).strip() or None
            if normalized_source != self._state.custom_observer_source:
                self._state.custom_observer_source = normalized_source
                changed = True
        if changed:
            self._refresh_status_text()
            self._emit_observer_info()

    def _schedule_observer_refresh(self, *, preserve_current_view: bool, align_projected_fov: bool) -> None:
        self._observer_refresh_serial += 1
        serial = self._observer_refresh_serial

        def _run() -> None:
            if serial != self._observer_refresh_serial:
                return
            self._refresh_plot(preserve_current_view=preserve_current_view)
            if align_projected_fov and self._view_mode == "box_fov":
                try:
                    self._set_view_to_projected_fov(pad_factor=1.10)
                    self._canvas.draw_idle()
                except Exception:
                    pass

        QTimer.singleShot(0, _run)

    _DATE_OBS_META_KEYS = ("DATE-OBS", "DATE_OBS", "date-obs", "date_obs")
    _OBSTIME_FALLBACK_META_KEYS = ("SRC_DATE", "MODELT")

    @staticmethod
    def _parse_obstime(value) -> Time | None:
        if value is None or value == "":
            return None
        try:
            return value if isinstance(value, Time) else Time(value)
        except Exception:
            return None

    @staticmethod
    def _meta_obstime_text(meta, keys: tuple[str, ...]) -> str | None:
        if not meta:
            return None
        for key in keys:
            for candidate in (key, key.lower(), key.upper()):
                value = meta.get(candidate)
                if value not in (None, ""):
                    return str(value)
        return None

    @staticmethod
    def _header_obstime_text(header) -> str | None:
        if header is None:
            return None
        for key in MapBoxDisplayWidget._DATE_OBS_META_KEYS + MapBoxDisplayWidget._OBSTIME_FALLBACK_META_KEYS:
            for candidate in (key, key.upper(), key.lower()):
                try:
                    value = header.get(candidate)
                except Exception:
                    value = None
                if value not in (None, ""):
                    return str(value)
        return None

    @staticmethod
    def _header_has_explicit_date_obs(header) -> bool:
        if header is None:
            return False
        for key in MapBoxDisplayWidget._DATE_OBS_META_KEYS:
            for candidate in (key, key.upper(), key.lower()):
                try:
                    value = header.get(candidate)
                except Exception:
                    value = None
                if value not in (None, ""):
                    return True
        return False

    def _ensure_embedded_header_obstime(self, header: fits.Header) -> None:
        if self._header_has_explicit_date_obs(header):
            return
        for key in self._OBSTIME_FALLBACK_META_KEYS:
            value = header.get(key)
            if value not in (None, ""):
                header["DATE-OBS"] = str(value)
                header["DATE_OBS"] = str(value)
                return
        fallback_iso = None
        if self._state is not None:
            fallback_iso = self._state.session_input.time_iso
        if fallback_iso:
            header["DATE-OBS"] = str(fallback_iso)
            header["DATE_OBS"] = str(fallback_iso)

    @staticmethod
    def _obstime_for_map(smap, fallback_iso: str | None = None):
        if smap is None:
            return MapBoxDisplayWidget._parse_obstime(fallback_iso)
        meta = getattr(smap, "meta", None) or {}
        for keys in (
            MapBoxDisplayWidget._DATE_OBS_META_KEYS,
            MapBoxDisplayWidget._OBSTIME_FALLBACK_META_KEYS,
        ):
            text = MapBoxDisplayWidget._meta_obstime_text(meta, keys)
            if text:
                parsed = MapBoxDisplayWidget._parse_obstime(text)
                if parsed is not None:
                    return parsed
        parsed = MapBoxDisplayWidget._parse_obstime(fallback_iso)
        if parsed is not None:
            return parsed
        return getattr(smap, "date", None)

    @staticmethod
    def _format_time_delta_short(delta_seconds: float) -> str:
        if not np.isfinite(delta_seconds):
            return "Δt=?"
        if abs(float(delta_seconds)) < 0.05:
            return "Δt=0"
        sign = "+" if delta_seconds >= 0 else "-"
        abs_s = abs(float(delta_seconds))
        if abs_s < 90.0:
            return f"Δt={sign}{abs_s:.0f}s"
        if abs_s < 5400.0:
            minutes = abs_s / 60.0
            if minutes < 10.0:
                return f"Δt={sign}{minutes:.1f}min"
            return f"Δt={sign}{minutes:.0f}min"
        hours = abs_s / 3600.0
        return f"Δt={sign}{hours:.1f}h"

    @staticmethod
    def _format_display_time_banner(obstime, model_obstime) -> str:
        if obstime is None:
            return ""
        try:
            when = obstime if isinstance(obstime, Time) else Time(obstime)
        except Exception:
            return ""
        obs_text = f"OBS {when.isot}"
        if model_obstime is None:
            return obs_text
        try:
            model_when = model_obstime if isinstance(model_obstime, Time) else Time(model_obstime)
            delta_s = float((when - model_when).to_value(u.s))
            return f"{obs_text} ({MapBoxDisplayWidget._format_time_delta_short(delta_s)})"
        except Exception:
            return obs_text

    def _display_obstime_anchor(self) -> Time | None:
        """Return the display time anchor (selected context refmap DATE-OBS)."""
        if self._state is None:
            return None
        fallback_iso = self._state.session_input.time_iso
        map_id = getattr(self._state, "selected_context_id", None)
        raw = None
        if map_id:
            try:
                canonical = self._canonical_map_key(map_id, purpose="context")
                raw = self._load_raw_map(canonical, purpose="context")
            except Exception:
                raw = None
        obstime = self._obstime_for_map(raw, fallback_iso)
        if obstime is not None:
            return obstime
        if fallback_iso:
            try:
                return Time(fallback_iso)
            except Exception:
                return None
        return None

    def _display_obstime_cache_token(self) -> str:
        anchor = self._display_obstime_anchor()
        if anchor is None:
            return "unknown"
        try:
            return anchor.isot
        except Exception:
            return "unknown"

    @staticmethod
    def _observer_cache_number(value, digits: int = 6) -> str:
        try:
            number = float(value)
        except Exception:
            return ""
        if not np.isfinite(number):
            return ""
        return f"{number:.{digits}f}"

    def _custom_observer_metadata_token(self) -> tuple[str, ...]:
        ephemeris = self._state.custom_observer_ephemeris if self._state is not None else None
        if not isinstance(ephemeris, dict):
            return ("", "", "", "", "")
        return (
            str(ephemeris.get("obs_date", ephemeris.get("obs_time", "")) or ""),
            self._observer_cache_number(ephemeris.get("hgln_obs_deg")),
            self._observer_cache_number(ephemeris.get("hglt_obs_deg")),
            self._observer_cache_number(ephemeris.get("dsun_cm"), digits=1),
            self._observer_cache_number(ephemeris.get("rsun_cm"), digits=1),
        )

    def _observer_metadata_cache_key(self, observer_key: str | None, obstime) -> tuple[str, ...] | None:
        key = self._normalize_observer_key(observer_key)
        if obstime is None:
            return None
        when = obstime if isinstance(obstime, Time) else Time(obstime)
        cache_key: tuple[str, ...] = (key, when.isot)
        if key == "custom":
            cache_key += self._custom_observer_metadata_token()
        return cache_key

    def _resolve_display_observer_metadata(self, observer_key: str | None, obstime) -> dict | None:
        key = self._normalize_observer_key(observer_key)
        if obstime is None:
            return None
        when = obstime if isinstance(obstime, Time) else Time(obstime)
        cache_key = self._observer_metadata_cache_key(key, when)
        if cache_key is not None and cache_key in self._observer_metadata_cache:
            return self._observer_metadata_cache[cache_key]

        metadata = None
        if key == "custom":
            ephemeris = self._state.custom_observer_ephemeris if self._state is not None else None
            metadata = resolve_observer_parameters_from_ephemeris(
                ephemeris,
                observer_key="custom",
                obs_time=when,
            )
            if metadata is None:
                return None
            coord = metadata.get("observer_coordinate")
            if coord is not None:
                self._observer_coord_cache[key] = coord
        else:
            coord = self._observer_coord_cache.get(key)
            warning = None
            used_key = key
            if coord is None:
                source_b3d = self._observer_source_b3d()
                coord, warning, used_key = resolve_observer_with_info(
                    source_b3d if isinstance(source_b3d, dict) else {},
                    key,
                    when,
                )
                if warning and key not in self._observer_warning_cache:
                    self._observer_warning_cache.add(key)
                    self._last_status_text = warning
                    if self._status_callback is not None:
                        self._status_callback(warning)
                if coord is not None:
                    self._observer_coord_cache[key] = coord
            if coord is None:
                return None
            try:
                hgs = coord.transform_to(HeliographicStonyhurst(obstime=when))
            except Exception:
                hgs = coord
            ephemeris_card = {
                "hgln_obs_deg": float(hgs.lon.to_value(u.deg)),
                "hglt_obs_deg": float(hgs.lat.to_value(u.deg)),
                "dsun_cm": float(coord.radius.to_value(u.cm)),
                "rsun_cm": float(R_sun.to_value(u.cm)),
                "obs_date": when.isot,
            }
            metadata = resolve_observer_parameters_from_ephemeris(
                ephemeris_card,
                observer_key=used_key,
                obs_time=when,
            )
            if metadata is None:
                metadata = {
                    "observer_coordinate": coord,
                    "observer_key": used_key,
                    "obs_time": when,
                    "b0_deg": float(hgs.lat.to_value(u.deg)),
                    "l0_deg": float(hgs.lon.to_value(u.deg)),
                    "p_deg": float(sun.P(when).to_value(u.deg)) if used_key == "earth" else None,
                    "dsun_cm": float(coord.radius.to_value(u.cm)),
                    "rsun_cm": float(R_sun.to_value(u.cm)),
                    "rsun_arcsec": None,
                    "source": "session",
                }
                rsun_value = self._rsun_arcsec_from_observer_metadata(metadata)
                if rsun_value is not None:
                    metadata["rsun_arcsec"] = rsun_value

        observer = metadata.get("observer_coordinate") if isinstance(metadata, dict) else None
        if observer is None:
            return None
        try:
            hgs = observer.transform_to(HeliographicStonyhurst(obstime=when))
            metadata["los_signature"] = (
                f"hgs:{when.isot}:"
                f"{float(hgs.lon.to_value(u.deg)):.6f}:"
                f"{float(hgs.lat.to_value(u.deg)):.6f}"
            )
        except Exception:
            metadata["los_signature"] = key
        if cache_key is not None:
            self._observer_metadata_cache[cache_key] = metadata
        return metadata

    def _resolve_display_observer_coord(self, observer_key: str | None, obstime) -> SkyCoord | None:
        metadata = self._resolve_display_observer_metadata(observer_key, obstime)
        if metadata is None:
            return None
        return metadata.get("observer_coordinate")

    def _observer_context(self, observer_key: str | None, obstime):
        coord = self._resolve_display_observer_coord(observer_key, obstime)
        if coord is None:
            return None
        return SimpleNamespace(observer_coordinate=coord, date=obstime)

    def _resolved_observer_for_map(self, smap, observer_key: str | None = None):
        obstime = getattr(smap, "date", None) if smap is not None else None
        key = observer_key
        if key is None and self._state is not None:
            key = self._state.display_observer_key
        context = self._observer_context(key, obstime)
        if context is not None and getattr(context, "observer_coordinate", None) is not None:
            return getattr(context, "observer_coordinate")
        return getattr(smap, "observer_coordinate", None) if smap is not None else None

    def _display_observer_cache_token(self, smap, observer_key: str | None = None) -> str:
        key = self._normalize_observer_key(
            observer_key if observer_key is not None else (
                self._state.display_observer_key if self._state is not None else "earth"
            )
        )
        if smap is None:
            return key
        obstime = self._display_obstime_anchor()
        if obstime is None:
            obstime = self._obstime_for_map(
                smap,
                self._state.session_input.time_iso if self._state is not None else None,
            )
        metadata = self._resolve_display_observer_metadata(key, obstime)
        if metadata is None:
            return key
        return str(metadata.get("los_signature") or key)

    def _observers_share_los(self, observer_key_a: str | None, observer_key_b: str | None, obstime) -> bool:
        meta_a = self._resolve_display_observer_metadata(observer_key_a, obstime)
        meta_b = self._resolve_display_observer_metadata(observer_key_b, obstime)
        if meta_a is None or meta_b is None:
            return self._normalize_observer_key(observer_key_a) == self._normalize_observer_key(observer_key_b)
        return str(meta_a.get("los_signature") or "") == str(meta_b.get("los_signature") or "")

    def _infer_native_display_observer_key_from_map(self, smap) -> str | None:
        """Map spacecraft products to a display observer key when metadata is unambiguous."""

        meta = getattr(smap, "meta", {}) or {}
        tele = str(meta.get("telescop") or meta.get("TELESCOP") or "").upper()
        instr = str(meta.get("instrume") or meta.get("INSTRUME") or "").upper()
        detector = str(getattr(smap, "detector", "") or meta.get("detector") or meta.get("DETECTOR") or "").upper()
        if "SOLO" in tele or "SOLAR ORBITER" in tele or "SOLO" in instr:
            return "solar orbiter"
        if "STEREO" in tele or "SECCHI" in instr or "EUVI" in detector:
            candidates = ("stereo-a", "stereo-b")
        elif not MapBoxDisplayWidget._is_earth_native_los_map(smap):
            candidates = ("stereo-a", "stereo-b", "solar orbiter")
        else:
            return None
        try:
            map_observer = smap.observer_coordinate
            obstime = getattr(smap, "date", None)
        except Exception:
            return None
        if map_observer is None or obstime is None:
            return None
        best_key = None
        best_sep_deg = None
        for key in candidates:
            candidate = self._resolve_display_observer_coord(key, obstime)
            if candidate is None:
                continue
            sep_deg = float(map_observer.separation(candidate).to_value(u.deg))
            if best_sep_deg is None or sep_deg < best_sep_deg:
                best_sep_deg = sep_deg
                best_key = key
        if best_key is not None and best_sep_deg is not None and best_sep_deg < 1.0:
            return best_key
        return None

    def _map_matches_display_observer_native_los(self, smap, display_key: str) -> bool:
        native_key = self._infer_native_display_observer_key_from_map(smap)
        return native_key is not None and native_key == self._normalize_observer_key(display_key)

    @staticmethod
    def _is_earth_native_los_map(smap) -> bool:
        """Return whether a map is natively expressed in an Earth-like LOS."""
        meta = getattr(smap, "meta", {}) or {}
        if bool(meta.get("PYALIGN", False)):
            return True
        telescope = str(meta.get("telescop") or meta.get("TELESCOP") or "").upper()
        instrument = str(meta.get("instrume") or meta.get("INSTRUME") or "").upper()
        if any(token in telescope or token in instrument for token in ("SDO", "AIA", "HMI", "EOVSA")):
            return True
        try:
            obs = smap.observer_coordinate
            lon = abs(float(obs.lon.to_value(u.deg)))
            lat = abs(float(obs.lat.to_value(u.deg)))
            return lon < 5.0 and lat < 10.0
        except Exception:
            return True

    @staticmethod
    def _is_native_spacecraft_payload(smap) -> bool:
        meta = getattr(smap, "meta", {}) or {}
        if bool(meta.get("PYALIGN", False)):
            return False
        tele = str(meta.get("telescop") or meta.get("TELESCOP") or "").upper()
        instr = str(meta.get("instrume") or meta.get("INSTRUME") or "").upper()
        detector = str(getattr(smap, "detector", "") or meta.get("detector") or meta.get("DETECTOR") or "").upper()
        if (
            "STEREO" in tele
            or "SECCHI" in instr
            or "EUVI" in detector
            or "SOLO" in tele
            or "SOLAR ORBITER" in tele
            or "SOLO" in instr
        ):
            return True
        return not MapBoxDisplayWidget._is_earth_native_los_map(smap)

    @staticmethod
    def _rotate_map_for_display(smap):
        try:
            data = np.asarray(smap.data)
            if np.issubdtype(data.dtype, np.integer):
                fill = 0
                if data.dtype == np.bool_:
                    fill = False
                return smap.rotate(order=3, missing=fill, clip=False)
            return smap.rotate(order=3)
        except Exception:
            return smap

    @staticmethod
    def _is_embedded_pyalign_map(smap) -> bool:
        meta = getattr(smap, "meta", {}) or {}
        if not bool(meta.get(_EMBEDDED_REFMAP_FLAG, False)):
            return False
        return bool(meta.get("PYALIGN", False))

    def _embedded_context_needs_display_crop(self, smap) -> bool:
        """Return whether embedded context maps need a display-time FOV crop.

        Earth-view maps persisted in the H5 workflow are already cropped to the
        model FOV at embed time (PYALIGN=True). Reprojecting them to a spacecraft
        view uses a projected 1.1x FOV ROI instead of a second pixel crop. Native
        spacecraft embedded maps (PYALIGN=False) still need a 1.1x FOV crop after
        P-angle rotation, using the FOV frame defined by ``fov_definition_observer_key``.
        """
        if not self._is_embedded_pyalign_map(smap):
            return True
        return False

    @staticmethod
    def _valid_map_array(smap, *, min_side: int = _MIN_DISPLAY_MAP_SIDE) -> bool:
        try:
            data = np.asarray(smap.data)
        except Exception:
            return False
        if data.ndim != 2:
            return False
        ny = int(data.shape[0])
        nx = int(data.shape[1])
        side = int(min_side)
        return ny >= side and nx >= side

    def _fov_observer_coord_for_submap(self, smap, *, prefer_display_observer: bool = False):
        observer_key = "earth"
        if self._state is not None:
            if prefer_display_observer:
                observer_key = self._state.display_observer_key
            else:
                observer_key = self._state.fov_definition_observer_key
        source_context = self._observer_context(observer_key, getattr(smap, "date", None))
        observer = getattr(source_context, "observer_coordinate", None) or "earth"
        obstime = getattr(source_context, "date", None) or getattr(smap, "date", None)
        return observer, obstime

    def _map_source_cache_token(self) -> str:
        if self._state is None:
            return "auto"
        return str(getattr(self._state, "map_source_mode", None) or "auto")

    @staticmethod
    def _is_embedded_native_spacecraft_map(smap) -> bool:
        meta = getattr(smap, "meta", {}) or {}
        if not bool(meta.get(_EMBEDDED_REFMAP_FLAG, False)):
            return False
        return MapBoxDisplayWidget._is_native_spacecraft_payload(smap)

    def _geometry_cache_token(self) -> str:
        if self._state is None:
            return "none"
        parts = [
            self._normalize_observer_key(getattr(self._state, "fov_definition_observer_key", "earth")),
            self._normalize_observer_key(getattr(self._state, "geometry_definition_observer_key", "earth")),
        ]
        if getattr(self._state, "fov", None) is not None:
            fov = self._state.fov
            parts.append(
                f"{fov.center_x_arcsec:.2f},{fov.center_y_arcsec:.2f},"
                f"{fov.width_arcsec:.2f},{fov.height_arcsec:.2f}"
            )
        geom = getattr(self._state, "geometry", None)
        if geom is not None:
            parts.append(
                f"{geom.coord_x},{geom.coord_y},{geom.grid_x},{geom.grid_y},{geom.grid_z}"
            )
        return "|".join(parts)

    def _native_crop_cache_key(self, map_key: str) -> str:
        return (
            f"__native_crop__:{self._map_source_cache_token()}:"
            f"{self._geometry_cache_token()}:{map_key}"
        )

    def _display_prepared_cache_key(self, map_key: str, purpose: str) -> str:
        observer_key = self._normalize_observer_key(
            self._state.display_observer_key if self._state is not None else "earth"
        )
        context_token = "__none__"
        if self._state is not None:
            context_token = str(getattr(self._state, "selected_context_id", None) or "__none__")
        return (
            f"__display__:{self._map_source_cache_token()}:"
            f"{self._display_obstime_cache_token()}:"
            f"{observer_key}:{context_token}:"
            f"{self.__dict__.get('_view_mode', 'box_fov') or 'box_fov'}:{purpose}:{map_key}"
        )

    def _infer_map_native_observer_key(self, smap) -> str:
        native_key = self._infer_native_display_observer_key_from_map(smap)
        if native_key is not None:
            return native_key
        return "earth"

    def _model_fov_in_definition_frame(self, smap) -> DisplayFovSelection | None:
        if self._state is None:
            return None
        if self._state.fov is not None:
            fov = self._state.fov
            return DisplayFovSelection(
                center_x_arcsec=float(fov.center_x_arcsec),
                center_y_arcsec=float(fov.center_y_arcsec),
                width_arcsec=float(max(fov.width_arcsec, 1e-3)),
                height_arcsec=float(max(fov.height_arcsec, 1e-3)),
            )
        geometry_observer_key = self._state.geometry_definition_observer_key
        box = self._build_legacy_box(smap, geometry_observer_key=geometry_observer_key)
        if box is None:
            return None
        return self._box_bounds_to_fov_selection(box, smap)

    def _native_box_crop_fov(self, smap) -> DisplayFovSelection | None:
        if self._state is None:
            return None
        base_fov = self._model_fov_in_definition_frame(smap)
        if base_fov is None:
            return None
        fov_key = self._normalize_observer_key(self._state.fov_definition_observer_key)
        native_key = self._infer_map_native_observer_key(smap)
        obstime = getattr(smap, "date", None) or self._state.session_input.time_iso
        projected = self._project_fov_between_observers(base_fov, fov_key, native_key, obstime)
        if projected is None:
            return None
        return self._padded_fov_selection(projected, _EMBEDDED_REFMAP_FOV_PAD_FACTOR)

    def _crop_map_at_observer_fov(
        self,
        smap,
        fov: DisplayFovSelection,
        observer_key: str,
        *,
        pad_factor: float = 1.0,
    ):
        half_w = 0.5 * max(float(fov.width_arcsec), 1e-3) * float(pad_factor)
        half_h = 0.5 * max(float(fov.height_arcsec), 1e-3) * float(pad_factor)
        obstime = getattr(smap, "date", None)
        source_context = self._observer_context(self._normalize_observer_key(observer_key), obstime)
        observer = getattr(source_context, "observer_coordinate", None) or "earth"
        obstime = getattr(source_context, "date", None) or obstime
        bottom_left = SkyCoord(
            Tx=(float(fov.center_x_arcsec) - half_w) * u.arcsec,
            Ty=(float(fov.center_y_arcsec) - half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        top_right = SkyCoord(
            Tx=(float(fov.center_x_arcsec) + half_w) * u.arcsec,
            Ty=(float(fov.center_y_arcsec) + half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        try:
            cropped = _submap_with_fov_safe(smap, bottom_left, top_right)
            if not self._valid_map_array(cropped):
                return smap
            return cropped
        except Exception:
            return smap

    def _build_native_crop(self, map_key: str, smap):
        if self._is_embedded_pyalign_map(smap):
            self._record_prepare_event(f"native crop: {map_key} skipped (pre-cropped at embed)")
            return smap, None
        crop_fov = self._native_box_crop_fov(smap)
        if crop_fov is None:
            return smap, None
        native_key = self._infer_map_native_observer_key(smap)
        self._record_prepare_event(
            f"native crop: {map_key} @ {_EMBEDDED_REFMAP_FOV_PAD_FACTOR:.2f}x FOV in {native_key}"
        )
        cropped = self._crop_map_at_observer_fov(smap, crop_fov, native_key)
        if cropped is smap:
            return smap, None
        return cropped, crop_fov

    def _ensure_cache_initialized(self) -> None:
        self.__dict__.setdefault("_cache_lock", threading.RLock())
        self.__dict__.setdefault("_native_crop_cache", {})
        self.__dict__.setdefault("_display_prepared_cache", {})
        self.__dict__.setdefault("_loaded_map_cache", {})

    def _ensure_native_crop_cache(self, map_key: str, smap=None):
        self._ensure_cache_initialized()
        cache_key = self._native_crop_cache_key(map_key)
        native_cache = self.__dict__.setdefault("_native_crop_cache", {})
        with self._cache_lock:
            entry = native_cache.get(cache_key)
            if entry is not None:
                return entry.get("map")
        if smap is None:
            smap = self._load_raw_map(map_key, purpose="context")
        if smap is None:
            return None
        cropped, crop_fov = self._build_native_crop(map_key, smap)
        with self._cache_lock:
            native_cache[cache_key] = {"map": cropped, "crop_fov": crop_fov}
        return cropped

    def _get_native_cropped_map(self, map_key: str, raw_smap):
        self._ensure_cache_initialized()
        cache_key = self._native_crop_cache_key(map_key)
        native_cache = self.__dict__.setdefault("_native_crop_cache", {})
        with self._cache_lock:
            entry = native_cache.get(cache_key)
            if entry is not None:
                return entry.get("map")
        cropped, crop_fov = self._build_native_crop(map_key, raw_smap)
        with self._cache_lock:
            native_cache[cache_key] = {"map": cropped, "crop_fov": crop_fov}
        return cropped

    def _apply_hmi_context_adjustments(self, map_key: str, smap):
        """Rotate HMI products for display. Do not resample them onto the model grid.

        The Carrington ``base/index`` WCS is the model CEA frame. Reprojecting
        the context magnetogram onto it is what drew Carrington longitude and
        latitude in the selector. Observer LOS reprojection, when the map is
        not already helioprojective, happens in
        ``_reproject_map_for_display_observer``.
        """
        if map_key not in _HMI_DISPLAY_KEYS:
            return smap
        try:
            return smap.rotate(order=3)
        except Exception:
            return smap

    def _map_display_los_matches(self, smap, display_key: str) -> bool:
        if not self._is_helioprojective_map(smap):
            return False
        display_key = self._normalize_observer_key(display_key)
        if self._map_matches_display_observer_native_los(smap, display_key):
            return True
        if not self._is_native_spacecraft_payload(smap):
            return display_key == "earth"
        return False

    def _cross_observer_roi_fov_for_display(self, smap, *, pad_factor: float | None = None) -> DisplayFovSelection | None:
        if self._state is None:
            return None
        pad = _EMBEDDED_REFMAP_FOV_PAD_FACTOR if pad_factor is None else float(pad_factor)
        pad_fov = self._embedded_context_crop_fov(smap)
        if pad_fov is None and self._state.fov is not None:
            pad_fov = self._padded_fov_selection(self._state.fov, pad)
        if pad_fov is None:
            return None
        return self._fov_selection_projected_to_display_observer(
            pad_fov,
            self._display_obstime_anchor(),
        )

    def _reproject_fov_override_for_display(self, map_key: str, smap, *, purpose: str):
        view_mode = self.__dict__.get("_view_mode", "box_fov")
        display_key = self._normalize_observer_key(self._state.display_observer_key)
        if view_mode == "full_sun" and purpose == "context" and self._is_embedded_pyalign_map(smap):
            disk_fov = self._full_sun_disk_hpc_fov(smap, pad_factor=_FULL_SUN_DISK_EXTENT_PAD)
            if disk_fov is None:
                return None
            projected = self._fov_selection_projected_to_display_observer(
                disk_fov,
                self._display_obstime_anchor(),
            )
            if projected is not None:
                self._record_prepare_event(f"context reproj full sun: {map_key} [roi]")
            return projected
        if self._map_display_los_matches(smap, display_key):
            return None
        if purpose != "context":
            return None
        if view_mode != "box_fov":
            return None
        projected = self._cross_observer_roi_fov_for_display(smap)
        if projected is not None:
            self._record_prepare_event(
                f"context reproj roi: {map_key} @ {_EMBEDDED_REFMAP_FOV_PAD_FACTOR:.2f}x FOV -> {display_key}"
            )
        return projected

    def _crop_bottom_to_display_window(self, map_key: str, smap):
        if self._view_mode != "box_fov":
            return smap
        display_bounds = self._display_window_pixel_bounds(smap)
        if display_bounds is not None:
            return self._submap_to_pixel_bounds(smap, display_bounds)
        display_fov = self._display_window_fov_selection(smap)
        if display_fov is not None:
            return self._submap_to_explicit_fov(
                smap,
                fov_override=display_fov,
                pad_factor=1.0,
                prefer_display_observer=True,
            )
        box = self._build_legacy_box(smap)
        if box is None:
            return smap
        return self._submap_to_box_bounds(smap, box)

    def _should_skip_native_crop_for_cross_observer_display(self, smap) -> bool:
        """Skip tier-1 native crop only for Earth-native maps shown at spacecraft LOS.

        Native spacecraft maps are always cropped in their own LOS first, even when
        the display observer differs (crop native → ROI reproject to display).
        """
        if self._state is None:
            return False
        display_key = self._normalize_observer_key(self._state.display_observer_key)
        if self._map_display_los_matches(smap, display_key):
            return False
        if self._is_embedded_pyalign_map(smap):
            return True
        return not self._is_native_spacecraft_payload(smap)

    def _should_use_native_crop(
        self,
        *,
        purpose: str,
        use_native_crop: bool | None,
        smap=None,
    ) -> bool:
        if purpose != "context":
            return False
        if self.__dict__.get("_view_mode", "box_fov") != "box_fov":
            return False
        if use_native_crop is False:
            return False
        if smap is not None and self._should_skip_native_crop_for_cross_observer_display(smap):
            return False
        return use_native_crop is not False

    def _prepare_bottom_for_display(self, map_key: str, smap):
        """Model-grid base maps stay on their native WCS and are autoaligned.

        Those pixels live on the Carrington CEA grid. Resampling them with that
        plate scale treated as helioprojective arcsec builds a 32-pixel canvas
        and tears the overlay into quadrants. ``plot(..., autoalign=True)``
        warps the native grid onto the observer-LOS context axes.
        """
        if self._is_known_non_los_map(smap):
            self._apply_display_scaling(smap, map_key)
            return smap, None
        display_key = self._normalize_observer_key(
            self._state.display_observer_key if self._state is not None else "earth"
        )
        los_matches = self._map_display_los_matches(smap, display_key)
        display_map = smap
        if self.__dict__.get("_view_mode", "box_fov") == "box_fov":
            if los_matches:
                display_map = self._crop_bottom_to_display_window(map_key, display_map)
            else:
                anchor = self._display_obstime_anchor()
                base_fov = self._state.fov if self._state is not None and self._state.fov is not None else self.projected_box_fov()
                fov_override = None
                if base_fov is not None and anchor is not None:
                    fov_override = self._fov_selection_projected_to_display_observer(base_fov, anchor)
                display_map, _ = self._reproject_map_for_display_observer(
                    display_map,
                    fov_override=fov_override,
                )
        self._apply_display_scaling(display_map, map_key)
        return display_map, None

    def _prepare_map_for_display(
        self,
        map_key: str,
        smap,
        *,
        purpose: str = "context",
        use_native_crop: bool | None = None,
    ):
        """Single display-preparation pipeline for context, base overlays, and prewarm."""
        if purpose == "bottom":
            return self._prepare_bottom_for_display(map_key, smap)

        working = smap
        if self._should_use_native_crop(
            purpose=purpose,
            use_native_crop=use_native_crop,
            smap=smap,
        ):
            working = self._get_native_cropped_map(map_key, smap)

        working = self._apply_hmi_context_adjustments(map_key, working)

        reproject_fov_override = self._reproject_fov_override_for_display(
            map_key,
            smap,
            purpose=purpose,
        )
        display_map, coverage_fov = self._reproject_map_for_display_observer(
            working,
            fov_override=reproject_fov_override,
        )

        self._apply_display_scaling(display_map, map_key)
        return display_map, coverage_fov

    def _context_prepared_cache_key(
        self,
        source_token: str,
        prepare_variant: str,
        observer_token: str,
        canonical_key: str,
    ) -> str:
        key = (
            f"__context_prepared__:{source_token}:{prepare_variant}:"
            f"{self._display_obstime_cache_token()}:{observer_token}:{canonical_key}"
        )
        if self._state is not None and self._state.fov is not None:
            fov = self._state.fov
            key = (
                f"{key}@{fov.center_x_arcsec:.2f},{fov.center_y_arcsec:.2f},"
                f"{fov.width_arcsec:.2f},{fov.height_arcsec:.2f}"
            )
        return key

    def _context_prepare_variant_for_display(self, canonical_key: str) -> str:
        mode = self._map_source_cache_token()
        if mode == "embedded":
            return _CONTEXT_PREPARE_VARIANT_FOV_CROP
        if mode == "filesystem":
            return _CONTEXT_PREPARE_VARIANT_FULL_DISK
        if self._filesystem_path_for_key(canonical_key):
            return _CONTEXT_PREPARE_VARIANT_FULL_DISK
        return _CONTEXT_PREPARE_VARIANT_FOV_CROP

    def _uses_cropped_context_display(self, canonical_key: str | None = None) -> bool:
        if self._state is None:
            return False
        if canonical_key is None and self._state.selected_context_id:
            canonical_key = self._canonical_map_key(self._state.selected_context_id, purpose="context")
        if canonical_key is None:
            return self._map_source_cache_token() == "embedded"
        return self._context_prepare_variant_for_display(canonical_key) == _CONTEXT_PREPARE_VARIANT_FOV_CROP

    def _embedded_context_crop_fov(self, smap) -> DisplayFovSelection | None:
        if self._state is None or self._state.fov is None:
            geometry_observer_key = self._state.geometry_definition_observer_key if self._state is not None else None
            box = self._build_legacy_box(smap, geometry_observer_key=geometry_observer_key)
            if box is None:
                return None
            fov = self._box_bounds_to_fov_selection(box, smap)
            return DisplayFovSelection(
                center_x_arcsec=float(fov.center_x_arcsec),
                center_y_arcsec=float(fov.center_y_arcsec),
                width_arcsec=float(max(fov.width_arcsec, 1e-3) * _EMBEDDED_REFMAP_FOV_PAD_FACTOR),
                height_arcsec=float(max(fov.height_arcsec, 1e-3) * _EMBEDDED_REFMAP_FOV_PAD_FACTOR),
            )
        return self._padded_fov_selection(self._state.fov, _EMBEDDED_REFMAP_FOV_PAD_FACTOR)

    @staticmethod
    def _context_map_key_from_ref_key(ref_key: str) -> str:
        key = str(ref_key)
        if key == "Bz_reference":
            return "magnetogram"
        if key == "Ic_reference":
            return "continuum"
        if key.lower() == "vert_current":
            return "vert_current"
        if key.startswith("AIA_") and key[4:].isdigit():
            return key[4:]
        return key

    def _iter_warmable_context_map_keys(self) -> list[str]:
        if self._state is None:
            return []
        keys: set[str] = set()
        for ref_key in (self._state.refmaps or {}):
            keys.add(self._context_map_key_from_ref_key(ref_key))
        return sorted(keys)

    def _reproject_map_for_display_observer(
        self,
        smap,
        *,
        fov_override: DisplayFovSelection | None = None,
    ):
        if self._state is None:
            return smap, None
        display_key = self._normalize_observer_key(self._state.display_observer_key)
        los_frame = self._is_helioprojective_map(smap)
        if display_key == "earth" and fov_override is None and los_frame:
            cross_observer_native = (
                self._is_native_spacecraft_payload(smap)
                and not self._map_matches_display_observer_native_los(smap, display_key)
            )
            if not cross_observer_native:
                return smap, None
        native_spacecraft = self._is_native_spacecraft_payload(smap)
        if native_spacecraft and display_key in {"stereo-a", "stereo-b", "solar orbiter"}:
            try:
                self._record_prepare_event("display rotate: native spacecraft solar north")
                return self._rotate_map_for_display(smap), None
            except Exception:
                return smap, None
        if self._map_matches_display_observer_native_los(smap, display_key):
            try:
                self._record_prepare_event("display rotate: native spacecraft solar north")
                return self._rotate_map_for_display(smap), None
            except Exception:
                return smap, None
        obstime = self._display_obstime_anchor()
        if obstime is None:
            obstime = self._obstime_for_map(smap, self._state.session_input.time_iso)
        observer = self._resolve_display_observer_coord(display_key, obstime)
        if observer is None:
            return smap, None
        try:
            current_observer = getattr(smap, "observer_coordinate", None)
            if current_observer is not None:
                same_lon = np.isclose(
                    float(current_observer.lon.to_value(u.deg)),
                    float(observer.lon.to_value(u.deg)),
                    rtol=0.0,
                    atol=0.01,
                )
                same_lat = np.isclose(
                    float(current_observer.lat.to_value(u.deg)),
                    float(observer.lat.to_value(u.deg)),
                    rtol=0.0,
                    atol=0.01,
                )
                if same_lon and same_lat and los_frame:
                    return smap, None
        except Exception:
            pass
        source_label = str(
            getattr(smap, "detector", None)
            or getattr(smap, "observatory", None)
            or getattr(smap, "nickname", None)
            or "map"
        )
        from pyampp.io.refmap_crop import (
            reproject_map_to_target_observer_fov,
            reproject_refmap_to_observer,
        )

        try:
            if fov_override is not None:
                self._record_prepare_event(f"observer reproj: {source_label} -> {display_key} [roi]")
                reprojected = reproject_map_to_target_observer_fov(
                    smap,
                    target_fov=self._fov_selection_to_dict(fov_override),
                    target_observer=observer,
                    target_obstime=obstime,
                    mask_off_limb=True,
                )
                return reprojected, fov_override
            self._record_prepare_event(f"observer reproj: {source_label} -> {display_key} [full]")
            reprojected = reproject_refmap_to_observer(
                smap,
                observer=observer,
                obstime=obstime,
                reference_smap=smap,
                mask_off_limb=True,
            )
            return reprojected, None
        except Exception:
            return smap, None

    def _solar_disk_center_for_observer(self, smap, observer, obstime):
        try:
            source_observer = getattr(smap, "observer_coordinate", None) or "earth"
            solar_center = SkyCoord(
                0 * u.arcsec,
                0 * u.arcsec,
                frame=Helioprojective(observer=source_observer, obstime=obstime),
            )
            return solar_center.transform_to(Helioprojective(observer=observer, obstime=obstime))
        except Exception:
            return None

    @staticmethod
    def _rsun_arcsec_from_observer_metadata(metadata: dict | None) -> float | None:
        if not isinstance(metadata, dict):
            return None
        rsun_arcsec = metadata.get("rsun_arcsec")
        if rsun_arcsec is not None:
            value = float(rsun_arcsec)
            if np.isfinite(value) and value > 0:
                return value
        rsun_cm = metadata.get("rsun_cm")
        dsun_cm = metadata.get("dsun_cm")
        if dsun_cm is None:
            observer = metadata.get("observer_coordinate")
            if observer is not None:
                try:
                    dsun_cm = float(observer.radius.to_value(u.cm))
                except Exception:
                    dsun_cm = None
        if rsun_cm is None:
            try:
                rsun_cm = float(R_sun.to_value(u.cm))
            except Exception:
                rsun_cm = None
        if dsun_cm is None or rsun_cm is None:
            return None
        try:
            dsun_value = float(dsun_cm)
            rsun_value = float(rsun_cm)
        except Exception:
            return None
        if not (np.isfinite(dsun_value) and np.isfinite(rsun_value) and dsun_value > 0):
            return None
        ratio = float(np.clip(rsun_value / dsun_value, -1.0, 1.0))
        return float(np.arcsin(ratio) * u.rad.to(u.arcsec))

    def _full_sun_disk_rsun_arcsec(self, smap, obstime) -> float | None:
        compare_time = obstime
        if compare_time is None and self._state is not None:
            compare_time = self._state.session_input.time_iso
        if self._state is not None:
            metadata = self._resolve_display_observer_metadata(
                self._state.display_observer_key,
                compare_time,
            )
            value = self._rsun_arcsec_from_observer_metadata(metadata)
            if value is not None:
                return value
            ephemeris = self._state.custom_observer_ephemeris
            if isinstance(ephemeris, dict):
                params = resolve_observer_parameters_from_ephemeris(
                    ephemeris,
                    observer_key=self._state.display_observer_key,
                    obs_time=compare_time,
                )
                value = self._rsun_arcsec_from_observer_metadata(params)
                if value is not None:
                    return value
        if smap is not None:
            try:
                rsun_m = float(smap.rsun_meters.to_value(u.m))
                dsun_m = float(smap.dsun.to_value(u.m))
                if np.isfinite(rsun_m) and np.isfinite(dsun_m) and dsun_m > 0:
                    return float(np.degrees(np.arcsin(min(1.0, rsun_m / dsun_m))) * 3600.0)
            except Exception:
                pass
        return None

    def _full_sun_disk_hpc_fov(self, smap, *, pad_factor: float = 1.05) -> DisplayFovSelection | None:
        rsun_arcsec = self._full_sun_disk_rsun_arcsec(smap, getattr(smap, "date", None))
        if rsun_arcsec is None:
            return None
        side = 2.0 * float(rsun_arcsec) * float(pad_factor)
        return DisplayFovSelection(0.0, 0.0, side, side)

    @staticmethod
    def _union_pixel_rect_bounds(*rects) -> tuple[float, float, float, float] | None:
        xmins: list[float] = []
        xmaxs: list[float] = []
        ymins: list[float] = []
        ymaxs: list[float] = []
        for rect in rects:
            if rect is None:
                continue
            try:
                x0 = float(rect.get_x())
                y0 = float(rect.get_y())
                x1 = x0 + float(rect.get_width())
                y1 = y0 + float(rect.get_height())
                if not all(np.isfinite(v) for v in (x0, x1, y0, y1)):
                    continue
                xmins.append(min(x0, x1))
                xmaxs.append(max(x0, x1))
                ymins.append(min(y0, y1))
                ymaxs.append(max(y0, y1))
            except Exception:
                continue
        if not xmins:
            return None
        return (min(xmins), max(xmaxs), min(ymins), max(ymaxs))

    @staticmethod
    def _union_pixel_view_windows(
        *windows: tuple[tuple[float, float], tuple[float, float]] | None,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        xmins: list[float] = []
        xmaxs: list[float] = []
        ymins: list[float] = []
        ymaxs: list[float] = []
        for window in windows:
            if window is None:
                continue
            try:
                (x0, x1), (y0, y1) = window
                vals = (float(x0), float(x1), float(y0), float(y1))
                if not all(np.isfinite(v) for v in vals):
                    continue
                xmins.append(min(vals[0], vals[1]))
                xmaxs.append(max(vals[0], vals[1]))
                ymins.append(min(vals[2], vals[3]))
                ymaxs.append(max(vals[2], vals[3]))
            except Exception:
                continue
        if not xmins:
            return None
        return ((min(xmins), max(xmaxs)), (min(ymins), max(ymaxs)))

    def _full_sun_disk_half_extent_pixels(
        self,
        smap,
        *,
        pad_factor: float = 1.05,
    ) -> float | None:
        """Convert display-observer solar radius to a pixel half-extent."""
        rsun_arcsec = self._full_sun_disk_rsun_arcsec(smap, getattr(smap, "date", None))
        if rsun_arcsec is None:
            return None
        try:
            sx = abs(float(smap.scale.axis1.to_value(u.arcsec / u.pix)))
            sy = abs(float(smap.scale.axis2.to_value(u.arcsec / u.pix)))
            if np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0:
                scale = max(sx, sy)
                return float(rsun_arcsec) * float(pad_factor) / scale
        except Exception:
            pass
        disk_fov = self._full_sun_disk_hpc_fov(smap, pad_factor=pad_factor)
        if disk_fov is None:
            return None
        projected = self._fov_selection_projected_to_display_observer(
            disk_fov,
            getattr(smap, "date", None),
        )
        if projected is not None:
            disk_fov = projected
        window = self._fov_selection_to_pixel_window(
            smap,
            disk_fov,
            use_display_observer=True,
        )
        if window is None:
            return None
        (x0, x1), (y0, y1) = window
        return 0.5 * max(abs(x1 - x0), abs(y1 - y0), 4.0)

    def _display_disk_center_pixel(self, smap) -> tuple[float, float] | None:
        if self._state is None or smap is None:
            return None
        observer = self._resolved_observer_for_map(smap, self._state.display_observer_key) or "earth"
        obstime = getattr(smap, "date", None)
        center = SkyCoord(
            Tx=0 * u.arcsec,
            Ty=0 * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        for projector in (
            lambda coord: smap.wcs.world_to_pixel(coord),
            lambda coord: smap.world_to_pixel(coord.transform_to(smap.coordinate_frame)),
            lambda coord: smap.world_to_pixel(coord),
        ):
            try:
                px, py = projector(center)
                px = float(np.asarray(px, dtype=float).ravel()[0])
                py = float(np.asarray(py, dtype=float).ravel()[0])
                if np.isfinite(px) and np.isfinite(py):
                    return px, py
            except Exception:
                continue
        try:
            ny, nx = np.asarray(smap.data).shape[:2]
            return 0.5 * max(0, nx - 1), 0.5 * max(0, ny - 1)
        except Exception:
            return None

    def _fov_selection_to_pixel_window(
        self,
        smap,
        fov: DisplayFovSelection | None,
        *,
        use_display_observer: bool = False,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if smap is None or fov is None or self._state is None:
            return None
        if use_display_observer:
            observer_key = self._state.display_observer_key
        else:
            observer_key = self._state.fov_definition_observer_key
        source_context = self._observer_context(observer_key, getattr(smap, "date", None))
        observer = getattr(source_context, "observer_coordinate", None) or "earth"
        obstime = getattr(source_context, "date", None) or getattr(smap, "date", None)
        half_w = 0.5 * max(float(fov.width_arcsec), 1e-3)
        half_h = 0.5 * max(float(fov.height_arcsec), 1e-3)
        cx = float(fov.center_x_arcsec)
        cy = float(fov.center_y_arcsec)
        frame = Helioprojective(observer=observer, obstime=obstime)
        corners = SkyCoord(
            Tx=np.asarray([cx - half_w, cx + half_w, cx - half_w, cx + half_w], dtype=float) * u.arcsec,
            Ty=np.asarray([cy - half_h, cy - half_h, cy + half_h, cy + half_h], dtype=float) * u.arcsec,
            frame=frame,
        )
        try:
            px, py = smap.wcs.world_to_pixel(corners)
            px = np.asarray(px, dtype=float).ravel()
            py = np.asarray(py, dtype=float).ravel()
            finite = np.isfinite(px) & np.isfinite(py)
            if not np.any(finite):
                return None
            px = px[finite]
            py = py[finite]
            return (
                (float(np.nanmin(px)), float(np.nanmax(px))),
                (float(np.nanmin(py)), float(np.nanmax(py))),
            )
        except Exception:
            return None

    def _set_view_to_full_sun_disk(self, pad_factor: float | None = None) -> None:
        """After plot: center on the solar disk and include projected box/FOV overlays.

        See ``_FULL_SUN_DISK_EXTENT_PAD`` / ``_FULL_SUN_VIEWPORT_PAD`` policy comment.
        """
        if pad_factor is None:
            pad_factor = _FULL_SUN_VIEWPORT_PAD
        if self._current_axes is None or self._current_map is None or self._state is None:
            return
        smap = self._current_map
        center = self._display_disk_center_pixel(smap)
        if center is None:
            return
        cx, cy = center
        half_extent = (
            self._full_sun_disk_half_extent_pixels(smap, pad_factor=_FULL_SUN_DISK_EXTENT_PAD) or 0.0
        )
        for rect in (self._projected_box_bbox_rect, self._overlay_bbox_rect):
            if rect is None:
                continue
            try:
                x0 = float(rect.get_x())
                y0 = float(rect.get_y())
                x1 = x0 + float(rect.get_width())
                y1 = y0 + float(rect.get_height())
                if all(np.isfinite(v) for v in (x0, x1, y0, y1)):
                    half_extent = max(
                        half_extent,
                        abs(x0 - cx),
                        abs(x1 - cx),
                        abs(y0 - cy),
                        abs(y1 - cy),
                    )
            except Exception:
                continue
        if half_extent <= 0.0:
            return
        side = max(2.0 * half_extent, 4.0) * float(pad_factor)
        self._set_view_window(cx, cy, side, side)

    def _is_non_earth_display_observer(self) -> bool:
        if self._state is None:
            return False
        return self._normalize_observer_key(self._state.display_observer_key) != "earth"

    def initialize(self, session_input: SelectorSessionInput) -> None:
        selected_context_id = self._default_context_id(session_input)
        selected_bottom_id = self._default_bottom_id(session_input)
        self._clear_prepare_trace()
        self._map_summary_cache.clear()
        self._observer_coord_cache.clear()
        self._observer_metadata_cache.clear()
        self._observer_warning_cache.clear()
        self._invalidate_map_caches()
        available_keys = getattr(session_input, "available_observer_keys", None)
        self._available_observer_keys_override = (
            {self._normalize_observer_key(key) for key in available_keys}
            if available_keys
            else None
        )
        self._observer_availability_notice = (
            str(getattr(session_input, "observer_availability_notice", "")).strip() or None
        )
        self._state = MapBoxViewState(
            session_input=session_input,
            selected_context_id=selected_context_id,
            selected_bottom_id=selected_bottom_id,
            geometry=session_input.geometry,
            fov=session_input.fov,
            fov_box=session_input.fov_box,
            map_files=dict(session_input.map_files or {}),
            refmaps=dict(session_input.refmaps or {}),
            base_maps=dict(session_input.base_maps or {}),
            base_wcs_header=str(session_input.base_wcs_header) if session_input.base_wcs_header else None,
            base_geometry=session_input.base_geometry,
            map_source_mode=str(session_input.map_source_mode or "auto"),
            square_fov=bool(session_input.square_fov),
            display_observer_key=self._normalize_observer_key(
                getattr(session_input, "display_observer_key", "earth")
            ),
            geometry_definition_observer_key="earth",
            fov_definition_observer_key=self._normalize_observer_key(
                getattr(session_input.fov_box, "observer_key", "earth")
            ),
            custom_observer_ephemeris=copy.deepcopy(
                getattr(session_input, "custom_observer_ephemeris", None)
            ) if isinstance(getattr(session_input, "custom_observer_ephemeris", None), dict) else None,
            custom_observer_label=str(getattr(session_input, "custom_observer_label", "") or "").strip() or None,
            custom_observer_source=str(getattr(session_input, "custom_observer_source", "") or "").strip() or None,
        )
        self._normalize_display_observer_state()
        self._refresh_status_text()
        self._refresh_map_info()
        self._emit_observer_info()
        self._update_fov_control_enabled_state()

    def refresh_session_view(self) -> None:
        self._refresh_plot()
        self._ensure_session_model_loaded()
        self._refresh_fieldlines_from_committed_seeds()
        self._update_fov_control_enabled_state()

    def set_available_maps(self, map_ids: Iterable[str]) -> None:
        if self._state is None:
            return
        map_ids = list(map_ids)
        self._state.session_input.map_ids = tuple(map_ids)
        context_ids = self._available_context_map_ids(self._state.session_input)
        if not context_ids:
            self._state.selected_context_id = None
        elif (
            self._state.selected_context_id is not None
            and self._state.selected_context_id not in map_ids
        ):
            self._state.selected_context_id = self._default_context_id(self._state.session_input)
        if self._state.selected_bottom_id not in map_ids:
            self._state.selected_bottom_id = self._default_bottom_id(self._state.session_input)
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot()

    def set_context_map_id(self, map_id: Optional[str]) -> None:
        if self._state is None:
            return
        if self._state.selected_context_id == map_id:
            return
        self._state.selected_context_id = map_id
        self._invalidate_display_prepared_cache()
        self._invalidate_geometry_dependent_display_maps()
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot(preserve_current_view=False)

    def set_bottom_map_id(self, map_id: Optional[str]) -> None:
        if self._state is None:
            return
        if self._state.selected_bottom_id == map_id:
            return
        self._state.selected_bottom_id = map_id
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot(preserve_current_view=self._should_preserve_pixel_view())

    def set_map_file_paths(self, map_files: dict[str, str]) -> None:
        if self._state is None:
            return
        normalized = dict(map_files or {})
        if dict(self._state.map_files or {}) == normalized:
            return
        self.update_refmap_sources(map_files=normalized)

    def set_session_refmaps(self, refmaps: dict[str, dict] | None) -> None:
        if self._state is None:
            return
        normalized = dict(refmaps or {})
        if dict(self._state.refmaps or {}) == normalized:
            return
        self.update_refmap_sources(refmaps=normalized)

    def update_refmap_sources(
        self,
        *,
        map_files: dict[str, str] | None = None,
        refmaps: dict[str, dict] | None = None,
    ) -> None:
        if self._state is None:
            return
        files_changed = False
        refmaps_changed = False
        if map_files is not None:
            normalized_files = dict(map_files or {})
            files_changed = dict(self._state.map_files or {}) != normalized_files
            if files_changed:
                self._state.map_files = normalized_files
        if refmaps is not None:
            normalized_refmaps = dict(refmaps or {})
            refmaps_changed = dict(self._state.refmaps or {}) != normalized_refmaps
            if refmaps_changed:
                self._state.refmaps = normalized_refmaps
                self._state.session_input.refmaps = normalized_refmaps
        if not files_changed and not refmaps_changed:
            return
        self._map_summary_cache.clear()
        self._invalidate_map_caches()
        self._schedule_context_prewarm()
        self._refresh_map_info()
        preserve = (
            self.__dict__.get("_current_axes") is not None
            and self.__dict__.get("_current_map") is not None
        )
        self._refresh_plot(preserve_current_view=preserve)

    def set_map_source_mode(self, mode: str) -> None:
        if self._state is None:
            return
        mode = str(mode or "auto").lower()
        if mode not in {"auto", "filesystem", "embedded"}:
            mode = "auto"
        if self._state.map_source_mode == mode:
            return
        self._state.map_source_mode = mode
        self._map_summary_cache.clear()
        self._invalidate_display_on_source_mode_change()
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot(preserve_current_view=False)

    def set_geometry_edit_enabled(self, enabled: bool) -> None:
        self._geometry_edit_enabled = bool(enabled)
        self._refresh_control_mode_ui()
        self._refresh_status_text()

    def set_entry_box_path(self, entry_box_path: Optional[str | Path], *, load_session_model: bool = True) -> None:
        self._entry_box_path = Path(entry_box_path).expanduser().resolve() if entry_box_path else None
        self._session_box_template = None
        self._session_obs_time = None
        self._session_b3dtype = None
        self._session_temp_h5_path = None
        self._committed_line_seeds = None
        self._session_model_loaded = False
        self._observer_coord_cache.clear()
        self._observer_metadata_cache.clear()
        self._observer_warning_cache.clear()
        if load_session_model:
            self._load_session_model_from_entry()
        self._normalize_display_observer_state()
        self._emit_observer_info()
        self._refresh_open_3d_state()
        self._emit_action_state()
        if load_session_model:
            self._refresh_fieldlines_from_committed_seeds()

    def set_geometry_selection(self, selection: BoxGeometrySelection) -> None:
        if self._state is None:
            return
        self._state.geometry = selection
        self._invalidate_geometry_dependent_display_maps()
        self._refresh_status_text()
        self._refresh_plot(preserve_current_view=self._should_preserve_pixel_view())
        if self._geometry_change_callback is not None:
            self._geometry_change_callback(selection)

    def set_fov_selection(self, selection: DisplayFovSelection) -> None:
        if self._state is None:
            return
        if self._state.square_fov:
            selection = DisplayFovSelection(
                center_x_arcsec=selection.center_x_arcsec,
                center_y_arcsec=selection.center_y_arcsec,
                width_arcsec=selection.width_arcsec,
                height_arcsec=selection.width_arcsec,
            )
        self._state.fov = selection
        self._sync_fov_box_to_selection()
        self._invalidate_geometry_dependent_display_maps()
        self._refresh_status_text()
        self._refresh_plot(preserve_current_view=self._should_preserve_pixel_view())
        if self._fov_change_callback is not None:
            self._fov_change_callback(selection)

    def set_square_fov(self, enabled: bool, *, refresh: bool = True) -> None:
        if self._state is None:
            return
        self._state.square_fov = bool(enabled)
        self._update_fov_control_enabled_state()
        if enabled and self._state.fov is not None:
            selection = DisplayFovSelection(
                center_x_arcsec=self._state.fov.center_x_arcsec,
                center_y_arcsec=self._state.fov.center_y_arcsec,
                width_arcsec=self._state.fov.width_arcsec,
                height_arcsec=self._state.fov.width_arcsec,
            )
            if refresh:
                self.set_fov_selection(selection)
            else:
                self._state.fov = selection
                self._sync_fov_box_to_selection()

    def _update_fov_control_enabled_state(self) -> None:
        self._refresh_control_mode_ui()

    def set_action_state_callback(self, callback) -> None:
        self._action_state_callback = callback
        self._emit_action_state()

    def _emit_action_state(self) -> None:
        if self._action_state_callback is not None:
            self._action_state_callback(self._can_open_3d, self._can_clear_lines)

    def _refresh_open_3d_state(self) -> None:
        self._can_open_3d = (
            self._viewer3d is None
            and self._entry_box_path is not None
        )

    def _load_session_model_from_entry(self) -> None:
        if self._entry_box_path is None:
            return
        try:
            box, obs_time, b3dtype, temp_h5_path = _prepare_model_for_viewer(self._entry_box_path)
        except Exception:
            self._session_box_template = None
            self._session_obs_time = None
            self._session_b3dtype = None
            self._session_temp_h5_path = None
            self._committed_line_seeds = None
            self._session_model_loaded = True
            return
        self._session_box_template = box
        self._session_obs_time = obs_time
        self._session_b3dtype = b3dtype
        self._session_temp_h5_path = temp_h5_path
        line_seeds = box.b3d.get("line_seeds")
        self._committed_line_seeds = copy.deepcopy(line_seeds) if isinstance(line_seeds, dict) else None
        self._session_model_loaded = True

    def _ensure_session_model_loaded(self) -> None:
        if self._session_model_loaded:
            return
        self._load_session_model_from_entry()
        self._refresh_open_3d_state()
        self._emit_action_state()

    def _clone_session_model(self):
        if self._session_box_template is None:
            return None, None, None
        box = copy.deepcopy(self._session_box_template)
        return box, self._session_obs_time, self._session_b3dtype

    def _apply_live_session_state(self, box, *, update_frame_obs: bool = True) -> None:
        if isinstance(self._committed_line_seeds, dict):
            box.b3d["line_seeds"] = copy.deepcopy(self._committed_line_seeds)
        else:
            box.b3d.pop("line_seeds", None)
        if not isinstance(box.b3d, dict):
            return
        observer_meta = box.b3d.get("observer", {})
        if not isinstance(observer_meta, dict):
            observer_meta = {}
        if self._state is not None:
            if self._state.fov is not None:
                observer_meta["fov"] = {
                    "frame": "helioprojective",
                    "xc_arcsec": float(self._state.fov.center_x_arcsec),
                    "yc_arcsec": float(self._state.fov.center_y_arcsec),
                    "xsize_arcsec": float(self._state.fov.width_arcsec),
                    "ysize_arcsec": float(self._state.fov.height_arcsec),
                    "square": bool(self._state.square_fov),
                }
            if self._state.fov_box is not None:
                observer_meta["fov_box"] = self._state.fov_box.as_observer_metadata(
                    square=bool(self._state.square_fov)
                )
            observer_meta["name"] = str(
                self._normalize_observer_key(
                    self._state.display_observer_key if self._state is not None else "earth"
                )
            )
            if self._normalize_observer_key(observer_meta.get("name")) == "custom":
                observer_meta["label"] = str(self._state.custom_observer_label or "Custom")
                if self._state.custom_observer_source:
                    observer_meta["source"] = str(self._state.custom_observer_source)
                else:
                    observer_meta.pop("source", None)
            else:
                observer_meta["label"] = self._observer_label_for_key(observer_meta.get("name"))
                observer_meta.pop("source", None)
            ephemeris = copy.deepcopy(self._state.custom_observer_ephemeris or {})
            needs_custom_ephemeris = (
                self._normalize_observer_key(self._state.display_observer_key) == "custom"
                or self._normalize_observer_key(self._state.fov_definition_observer_key) == "custom"
                or (
                    self._state.fov_box is not None
                    and self._normalize_observer_key(getattr(self._state.fov_box, "observer_key", None)) == "custom"
                )
            )
            if needs_custom_ephemeris and ephemeris:
                observer_meta["ephemeris"] = ephemeris
                pb0r = build_pb0r_metadata_from_ephemeris(
                    ephemeris,
                    observer_key="custom",
                    obs_time=ephemeris.get("obs_date", self._session_obs_time),
                )
                if pb0r:
                    observer_meta["pb0r"] = pb0r
            else:
                observer_meta.pop("ephemeris", None)
                observer_meta.pop("pb0r", None)
        box.b3d["observer"] = observer_meta
        # Keep the live 3D frame observer aligned with the active 2D observer
        # context so LOS camera and FOV overlay are evaluated in one frame.
        if not update_frame_obs:
            return
        try:
            frame_obs = getattr(box, "_frame_obs", None)
            obs_time = getattr(frame_obs, "obstime", None)
            if obs_time is None:
                return
            desired_key = self._normalize_observer_key(
                self._state.display_observer_key if self._state is not None else "earth"
            )
            desired_observer = self._resolve_display_observer_coord(desired_key, obs_time)
            if desired_observer is not None:
                box._frame_obs = Helioprojective(observer=desired_observer, obstime=obs_time)
        except Exception:
            pass

    def _refresh_live_3d_viewer_state(self) -> None:
        viewer = self._viewer3d
        if viewer is None:
            return
        try:
            self._apply_live_session_state(viewer.box)
            if hasattr(viewer, "_update_los_scene_label"):
                viewer._update_los_scene_label()
            if hasattr(viewer, "previous_params"):
                viewer.previous_params = {}
            if hasattr(viewer, "update_plot"):
                viewer.update_plot(init=True)
            elif hasattr(viewer, "update_fov_box"):
                viewer.update_fov_box(getattr(viewer, "fov_box_visible", True), do_render=False)
                if hasattr(viewer, "render"):
                    viewer.render()
        except Exception:
            pass

    def _on_viewer3d_closed(self, *_args) -> None:
        close_was_handled = self._viewer3d_close_handled
        if not close_was_handled and self._viewer3d is not None:
            try:
                self.commit_live_3d_edits(
                    self._viewer3d._collect_line_seeds_snapshot(),
                    self._viewer3d._collect_streamlines(),
                    z_base=self._viewer3d.grid_zbase,
                )
                close_was_handled = True
            except Exception:
                pass
        self._viewer3d = None
        self._viewer3d_temp_h5_path = None
        self._viewer3d_watchdog.stop()
        if self._hidden_for_live_3d and not self._viewer3d_close_handled:
            host = self.window()
            host.show()
            host.raise_()
            host.activateWindow()
            self._hidden_for_live_3d = False
        self._viewer3d_close_handled = False
        self._refresh_open_3d_state()
        self._emit_action_state()
        if not close_was_handled:
            self._set_runtime_status("Live 3D viewer closed.")

    def committed_line_seeds(self):
        return copy.deepcopy(self._committed_line_seeds) if isinstance(self._committed_line_seeds, dict) else None

    def _refresh_fieldlines_from_committed_seeds(self) -> None:
        if self._current_map is None or self._current_axes is None:
            return
        # Session models are loaded lazily in dialog startup; ensure seeds are
        # available before attempting to rebuild and project field lines.
        self._ensure_session_model_loaded()
        if self._session_box_template is None:
            self.clear_fieldlines()
            return
        if not isinstance(self._committed_line_seeds, dict):
            self.clear_fieldlines()
            return
        try:
            box, _obs_time, b3dtype = self._clone_session_model()
            if box is None:
                self.clear_fieldlines()
                return
            self._apply_live_session_state(box)
            self._fieldline_frame_hcc = getattr(getattr(box, "_center", None), "frame", None)
            self._fieldline_frame_obs = getattr(box, "_frame_obs", None)
            streamlines, z_base = _generate_streamlines_from_seeds(box, b3dtype, self._committed_line_seeds)
            if streamlines:
                self.plot_fieldlines(streamlines, z_base=z_base)
            else:
                self.clear_fieldlines()
        except Exception as exc:
            self._set_runtime_status(f"Failed to restore saved field lines: {exc}")

    def commit_live_3d_edits(self, line_seeds, streamlines, z_base=0.0) -> None:
        self._committed_line_seeds = copy.deepcopy(line_seeds) if isinstance(line_seeds, dict) else None
        self._viewer3d_close_handled = True
        if self._hidden_for_live_3d:
            host = self.window()
            host.show()
            host.raise_()
            host.activateWindow()
            self._hidden_for_live_3d = False
        self.plot_fieldlines(streamlines, z_base=z_base)
        self._set_runtime_status("Accepted 3D seed edits into the 2D session model.")

    def cancel_live_3d_edits(self) -> None:
        self._viewer3d_close_handled = True
        if self._hidden_for_live_3d:
            host = self.window()
            host.show()
            host.raise_()
            host.activateWindow()
            self._hidden_for_live_3d = False
        self._set_runtime_status("Canceled 3D seed edits; kept the 2D session model unchanged.")

    def _check_viewer3d_state(self) -> None:
        if self._viewer3d is None:
            self._viewer3d_watchdog.stop()
            return
        try:
            window = self._viewer3d.app_window if hasattr(self._viewer3d, "app_window") else self._viewer3d
            if not window.isVisible():
                self._on_viewer3d_closed()
        except Exception:
            self._on_viewer3d_closed()

    def _control_target_mode(self) -> str:
        return "box" if self._geometry_edit_enabled else "fov"

    def _refresh_control_mode_ui(self) -> None:
        mode = self._control_target_mode()
        can_recompute_fov = self._projected_box_fov is not None
        if mode == "box":
            self._control_mode_label.setText("BOX Controls")
            self._left_btn.setToolTip("Move box center left")
            self._right_btn.setToolTip("Move box center right")
            self._down_btn.setToolTip("Move box center down")
            self._up_btn.setToolTip("Move box center up")
            self._x_minus_btn.setToolTip("Decrease X box size")
            self._x_plus_btn.setToolTip("Increase X box size")
            self._y_minus_btn.setToolTip("Decrease Y box size")
            self._y_plus_btn.setToolTip("Increase Y box size")
            self._xy_minus_btn.setToolTip("Decrease X and Y box size together")
            self._xy_plus_btn.setToolTip("Increase X and Y box size together")
            self._recompute_fov_btn.setEnabled(bool(self._entry_box_path is None and can_recompute_fov))
            self._y_minus_btn.setEnabled(True)
            self._y_plus_btn.setEnabled(True)
            self._xy_minus_btn.setEnabled(True)
            self._xy_plus_btn.setEnabled(True)
        else:
            square = bool(self._state.square_fov) if self._state is not None else False
            self._control_mode_label.setText("FOV Controls")
            self._left_btn.setToolTip("Move FOV center left")
            self._right_btn.setToolTip("Move FOV center right")
            self._down_btn.setToolTip("Move FOV center down")
            self._up_btn.setToolTip("Move FOV center up")
            self._x_minus_btn.setToolTip("Decrease FOV X size")
            self._x_plus_btn.setToolTip("Increase FOV X size")
            self._y_minus_btn.setToolTip("Decrease FOV Y size")
            self._y_plus_btn.setToolTip("Increase FOV Y size")
            self._xy_minus_btn.setToolTip("Decrease FOV X and Y size together")
            self._xy_plus_btn.setToolTip("Increase FOV X and Y size together")
            self._recompute_fov_btn.setEnabled(can_recompute_fov)
            self._y_minus_btn.setEnabled(not square)
            self._y_plus_btn.setEnabled(not square)
            self._xy_minus_btn.setEnabled(True)
            self._xy_plus_btn.setEnabled(True)

    def _nudge_primary_center(self, axis: str, sign: int) -> None:
        if self._control_target_mode() == "box":
            self._nudge_box_center(axis, sign)
        else:
            self._nudge_fov_center(axis, sign)

    def _nudge_primary_size(self, axis: str, sign: int) -> None:
        if self._control_target_mode() == "box":
            self._nudge_box_size(axis, sign)
        else:
            self._nudge_fov_size(axis, sign)

    def _nudge_primary_size_xy(self, sign: int) -> None:
        if self._control_target_mode() == "box":
            self._nudge_box_size_xy(sign)
        else:
            self._nudge_fov_size_xy(sign)

    def current_geometry_selection(self) -> Optional[BoxGeometrySelection]:
        if self._state is None:
            return None
        return self._state.geometry

    def current_fov_selection(self) -> Optional[DisplayFovSelection]:
        if self._state is None:
            return None
        return self._state.fov

    def exportable_fov_selection(self) -> Optional[DisplayFovSelection]:
        """Return the blue FOV rectangle in display-observer HPC at the display time anchor."""
        if self._state is None:
            return None
        base = self._state.fov or self.projected_box_fov()
        if base is None:
            return None
        anchor = self._display_obstime_anchor()
        if anchor is None:
            return None
        def_key = self._normalize_observer_key(self._state.fov_definition_observer_key)
        display_key = self._normalize_observer_key(self._state.display_observer_key)
        if self._observers_share_los(def_key, display_key, anchor):
            return base
        return self._project_fov_between_observers(base, def_key, display_key, anchor)

    def exportable_fov_box_selection(self) -> Optional[DisplayFovBoxSelection]:
        """Return the 3D FOV box footprint at the display time anchor for synthesis export."""
        if self._state is None:
            return None
        export_fov = self.exportable_fov_selection()
        if export_fov is None:
            return None
        z_min_mm = z_max_mm = None
        if self._state.fov_box is not None:
            z_min_mm = float(self._state.fov_box.z_min_mm)
            z_max_mm = float(self._state.fov_box.z_max_mm)
        else:
            recomputed = self._compute_fov_box_from_current_selection()
            if recomputed is not None:
                z_min_mm = float(recomputed.z_min_mm)
                z_max_mm = float(recomputed.z_max_mm)
        if z_min_mm is None or z_max_mm is None:
            return None
        return DisplayFovBoxSelection(
            center_x_arcsec=float(export_fov.center_x_arcsec),
            center_y_arcsec=float(export_fov.center_y_arcsec),
            width_arcsec=float(export_fov.width_arcsec),
            height_arcsec=float(export_fov.height_arcsec),
            z_min_mm=z_min_mm,
            z_max_mm=z_max_mm,
            observer_key=self._normalize_observer_key(self._state.display_observer_key),
        )

    def fov_persistence_issue(self) -> str | None:
        """Return a user-facing reason when FOV cannot be saved for the current display observer."""
        if self._state is None:
            return None
        if self._state.fov is None and self.projected_box_fov() is None:
            return None
        export_fov = self.exportable_fov_selection()
        if export_fov is not None:
            export_box = self.exportable_fov_box_selection()
            display_key = self._normalize_observer_key(self._state.display_observer_key)
            if export_box is not None:
                box_key = self._normalize_observer_key(export_box.observer_key)
                if box_key != display_key:
                    return (
                        f"The FOV box observer ({self._observer_label_for_key(box_key)}) "
                        f"does not match the display observer "
                        f"({self._observer_label_for_key(display_key)})."
                    )
            return None
        def_label = self._observer_label_for_key(self._state.fov_definition_observer_key)
        display_label = self._observer_label_for_key(self._state.display_observer_key)
        anchor = self._display_obstime_cache_token()
        anchor_text = anchor if anchor != "unknown" else "undefined"
        return (
            f"The FOV was defined for {def_label}, but the display observer is "
            f"{display_label} (time anchor: {anchor_text})."
        )

    def recompute_fov_for_display_observer(self) -> bool:
        """Recompute the inscribing FOV rectangle and box for the current display observer."""
        if self._state is None:
            return False
        if self._projected_box_fov is not None:
            self.recompute_fov_from_box()
        else:
            base = self._state.fov or self.projected_box_fov()
            if base is None:
                return False
            anchor = self._display_obstime_anchor()
            if anchor is None:
                return False
            def_key = self._normalize_observer_key(self._state.fov_definition_observer_key)
            display_key = self._normalize_observer_key(self._state.display_observer_key)
            if self._observers_share_los(def_key, display_key, anchor):
                self._state.fov_definition_observer_key = display_key
                self._sync_fov_box_to_selection()
            else:
                projected = self._project_fov_between_observers(base, def_key, display_key, anchor)
                if projected is None:
                    return False
                self._state.fov_definition_observer_key = display_key
                self.set_fov_selection(projected)
        return self.exportable_fov_selection() is not None

    def current_fov_box_selection(self) -> Optional[DisplayFovBoxSelection]:
        if self._state is None:
            return None
        return self._state.fov_box

    def projected_box_fov(self) -> Optional[DisplayFovSelection]:
        return self._projected_box_fov

    def set_geometry_change_callback(self, callback) -> None:
        self._geometry_change_callback = callback

    def set_map_info_callback(self, callback) -> None:
        self._map_info_callback = callback
        if callback is not None:
            callback(self._last_map_info_text)

    def set_status_callback(self, callback) -> None:
        self._status_callback = callback
        if callback is not None and self._state is not None:
            callback(self._last_status_text)

    def current_status_text(self) -> str:
        return str(self._last_status_text or "")

    def set_observer_info_callback(self, callback) -> None:
        self._observer_info_callback = callback
        if callback is not None and self._state is not None:
            callback(self.current_observer_info())

    def set_fov_change_callback(self, callback) -> None:
        self._fov_change_callback = callback
        if callback is not None and self._state is not None and self._state.fov is not None:
            callback(self._state.fov)

    def state(self) -> Optional[MapBoxViewState]:
        return self._state

    def observer_options(self) -> tuple[tuple[str, str], ...]:
        return tuple(_DISPLAY_OBSERVER_OPTIONS)

    def observer_enabled_keys(self) -> set[str]:
        return set(self._enabled_observer_keys())

    def set_available_observer_keys(
        self,
        observer_keys: Iterable[str] | None,
        *,
        notice: str | None = None,
    ) -> None:
        self._available_observer_keys_override = (
            {self._normalize_observer_key(key) for key in observer_keys}
            if observer_keys
            else None
        )
        self._observer_availability_notice = str(notice or "").strip() or None
        if self._state is None:
            return
        self._normalize_display_observer_state()
        self._refresh_status_text()
        self._emit_observer_info()
        self._update_fov_control_enabled_state()

    def current_display_observer_key(self) -> str:
        if self._state is None:
            return "earth"
        return self._normalize_observer_key(self._state.display_observer_key)

    def current_observer_persistence_state(self) -> dict[str, Any]:
        if self._state is None:
            return {
                "display_observer_key": "earth",
                "custom_observer_ephemeris": None,
                "custom_observer_label": "",
                "custom_observer_source": "",
                "fov_definition_observer_key": "earth",
            }
        return {
            "display_observer_key": self._normalize_observer_key(self._state.display_observer_key),
            "custom_observer_ephemeris": copy.deepcopy(self._state.custom_observer_ephemeris),
            "custom_observer_label": str(self._state.custom_observer_label or ""),
            "custom_observer_source": str(self._state.custom_observer_source or ""),
            "fov_definition_observer_key": self._normalize_observer_key(self._state.fov_definition_observer_key),
            "display_fov_obstime": self._display_obstime_cache_token(),
        }

    def current_observer_info(self) -> dict[str, str]:
        info = {
            "name": "",
            "label": "",
            "source": "",
            "model_time": "",
            "obs_date": "",
            "b0_deg": "",
            "l0_deg": "",
            "rsun_arcsec": "",
            "p_deg": "",
            "hgln_obs_deg": "",
            "hglt_obs_deg": "",
            "dsun_cm": "",
            "rsun_cm": "",
        }
        if self._state is None:
            return info
        info["model_time"] = str(self._state.session_input.time_iso or "")
        info["name"] = self._observer_label_for_key(self._state.display_observer_key)
        info["label"] = info["name"]
        if self._normalize_observer_key(self._state.display_observer_key) == "custom":
            ephemeris = self._state.custom_observer_ephemeris or {}
            try:
                when = ephemeris.get("obs_date", self._state.session_input.time_iso)
                when = when if isinstance(when, Time) else Time(when)
            except Exception:
                return info
            params = self._resolve_display_observer_metadata("custom", when)
            if params is None:
                return info
            info["name"] = "Custom"
            info["label"] = str(self._state.custom_observer_label or "Custom")
            info["source"] = str(self._state.custom_observer_source or "")
            info["obs_date"] = when.isot
            observer = params.get("observer_coordinate")
            if observer is not None:
                try:
                    hgs = observer.transform_to(HeliographicStonyhurst(obstime=when))
                    info["hgln_obs_deg"] = f"{float(hgs.lon.to_value(u.deg)):.6f}"
                    info["hglt_obs_deg"] = f"{float(hgs.lat.to_value(u.deg)):.6f}"
                except Exception:
                    pass
            for key, digits in (("b0_deg", 6), ("l0_deg", 6), ("p_deg", 6), ("rsun_arcsec", 2)):
                value = params.get(key)
                if value is None:
                    continue
                try:
                    info[key] = f"{float(value):.{digits}f}"
                except Exception:
                    pass
            for key in ("dsun_cm", "rsun_cm"):
                value = params.get(key)
                if value is None:
                    continue
                try:
                    info[key] = f"{float(value):.6e}"
                except Exception:
                    pass
            return info
        source_b3d = self._observer_source_b3d()
        observer_meta = source_b3d.get("observer", {}) if isinstance(source_b3d, dict) else {}
        ephemeris = observer_meta.get("ephemeris", {}) if isinstance(observer_meta, dict) else {}
        obs_time = self._state.session_input.time_iso
        if isinstance(ephemeris, dict):
            obs_time = ephemeris.get("obs_date", ephemeris.get("obs_time", obs_time))
        smap = self._current_map
        if smap is not None:
            obs_time = self._obstime_for_map(smap, obs_time)
        try:
            when = obs_time if isinstance(obs_time, Time) else Time(obs_time)
        except Exception:
            return info
        if isinstance(source_b3d, dict):
            observer, _warning, used_key = resolve_observer_with_info(
                source_b3d,
                self._state.display_observer_key,
                when,
            )
        else:
            observer = self._resolve_display_observer_coord(self._state.display_observer_key, when)
            used_key = self._normalize_observer_key(self._state.display_observer_key)
        if observer is None:
            return info
        info["name"] = self._observer_label_for_key(used_key)
        info["label"] = info["name"]
        try:
            hgs = observer.transform_to(HeliographicStonyhurst(obstime=when))
        except Exception:
            hgs = observer
        rsun_cm = None
        if isinstance(ephemeris, dict) and ephemeris.get("rsun_cm") is not None:
            try:
                rsun_cm = float(ephemeris.get("rsun_cm"))
            except Exception:
                rsun_cm = None
        elif self._current_map is not None and getattr(self._current_map, "rsun_meters", None) is not None:
            try:
                rsun_cm = float(u.Quantity(self._current_map.rsun_meters).to_value(u.cm))
            except Exception:
                rsun_cm = None
        params = resolve_observer_parameters_from_ephemeris(
            {
                "hgln_obs_deg": float(hgs.lon.to_value(u.deg)),
                "hglt_obs_deg": float(hgs.lat.to_value(u.deg)),
                "dsun_cm": float(hgs.radius.to_value(u.cm)),
                "rsun_cm": rsun_cm,
                "obs_date": when.isot,
            },
            observer_key=used_key,
            obs_time=when,
        )
        if params is None:
            return info
        info["obs_date"] = when.isot
        info["hgln_obs_deg"] = f"{float(hgs.lon.to_value(u.deg)):.6f}"
        info["hglt_obs_deg"] = f"{float(hgs.lat.to_value(u.deg)):.6f}"
        for key, digits in (("b0_deg", 6), ("l0_deg", 6), ("p_deg", 6), ("rsun_arcsec", 2)):
            value = params.get(key)
            if value is None:
                continue
            try:
                info[key] = f"{float(value):.{digits}f}"
            except Exception:
                info[key] = str(value)
        for key in ("dsun_cm", "rsun_cm"):
            value = params.get(key)
            if value is None:
                continue
            try:
                info[key] = f"{float(value):.3e}"
            except Exception:
                info[key] = str(value)
        return info

    def _sync_fov_box_to_selection(self) -> None:
        if self._state is None or self._state.fov is None:
            return
        recomputed = self._compute_fov_box_from_current_selection()
        if recomputed is not None:
            self._state.fov_box = recomputed
            return
        if self._state.fov_box is None:
            return
        # Fallback path: preserve existing z extent if geometry recomputation fails.
        self._state.fov_box = DisplayFovBoxSelection(
            center_x_arcsec=float(self._state.fov.center_x_arcsec),
            center_y_arcsec=float(self._state.fov.center_y_arcsec),
            width_arcsec=float(self._state.fov.width_arcsec),
            height_arcsec=float(self._state.fov.height_arcsec),
            z_min_mm=float(self._state.fov_box.z_min_mm),
            z_max_mm=float(self._state.fov_box.z_max_mm),
            observer_key=str(self._state.fov_box.observer_key or self._state.fov_definition_observer_key),
        )

    def _compute_fov_box_from_current_selection(self) -> Optional[DisplayFovBoxSelection]:
        if self._state is None or self._state.fov is None or self._current_map is None:
            return None
        obstime = getattr(self._current_map, "date", None)
        geometry_observer_key = self._state.geometry_definition_observer_key
        source_map = self._observer_context(geometry_observer_key, obstime) or self._current_map
        box = self._build_legacy_box(
            source_map,
            geometry_observer_key=geometry_observer_key,
        )
        if box is None:
            return None
        world = box.model_box_corners_world()
        if world is None:
            return None
        fov_observer_key = self._normalize_observer_key(self._state.fov_definition_observer_key)
        observer = self._resolved_observer_for_map(self._current_map, fov_observer_key) or "earth"
        try:
            fov_box = build_fov_box_from_user_hpc_and_red_box_world(
                world,
                xc_arcsec=float(self._state.fov.center_x_arcsec),
                yc_arcsec=float(self._state.fov.center_y_arcsec),
                xsize_arcsec=float(self._state.fov.width_arcsec),
                ysize_arcsec=float(self._state.fov.height_arcsec),
                observer=observer,
                obstime=obstime,
            )
            if fov_box is None:
                return None
            return DisplayFovBoxSelection(
                center_x_arcsec=float(fov_box["xc_arcsec"]),
                center_y_arcsec=float(fov_box["yc_arcsec"]),
                width_arcsec=float(fov_box["xsize_arcsec"]),
                height_arcsec=float(fov_box["ysize_arcsec"]),
                z_min_mm=float(fov_box["zmin_mm"]),
                z_max_mm=float(fov_box["zmax_mm"]),
                observer_key=fov_observer_key,
            )
        except Exception:
            return None

    def _compute_fov_box_local_corners(
        self,
        fov_box: DisplayFovBoxSelection | None = None,
    ) -> tuple[tuple[float, float, float], ...] | None:
        if self._state is None or self._current_map is None:
            return None
        fov_box = fov_box or self._state.fov_box
        if fov_box is None:
            return None
        source_context = self._observer_context(
            getattr(fov_box, "observer_key", None),
            getattr(self._current_map, "date", None),
        )
        source_map = source_context or self._current_map
        box = self._build_legacy_box(
            source_map,
            geometry_observer_key=self._state.geometry_definition_observer_key,
        )
        if box is None:
            return None
        corners = box.fov_box_corners_local_mm(
            fov_box.as_observer_metadata(square=bool(self._state.square_fov))
        )
        if corners is None:
            return None
        return tuple(tuple(float(v) for v in row) for row in np.asarray(corners, dtype=float))

    def _should_preserve_pixel_view(self) -> bool:
        return self._current_axes is not None

    def _on_gui_thread(self) -> bool:
        try:
            gui_thread = self.thread()
        except Exception:
            return True
        if gui_thread is None:
            return True
        return QThread.currentThread() is gui_thread

    def _status_text_with_prepare_trace(self, base_text: str) -> str:
        text = str(base_text or "")
        if not self._prep_trace_order:
            return text
        lines = []
        for label in self._prep_trace_order[-12:]:
            count = self._prep_trace_counts.get(label, 0)
            if count <= 0:
                continue
            lines.append(f"{count}x {label}")
        if not lines:
            return text
        return f"{text}\n\nprep_trace:\n" + "\n".join(lines)

    def _emit_status_text(self) -> None:
        self._last_status_text = self._status_text_with_prepare_trace(self._last_status_base_text)
        if not self._on_gui_thread():
            return
        if self._status_callback is not None:
            self._status_callback(self._last_status_text)

    def _clear_prepare_trace(self) -> None:
        self._prep_trace_counts.clear()
        self._prep_trace_order.clear()

    def _record_prepare_event(self, label: str) -> None:
        if not self._on_gui_thread():
            return
        key = str(label or "").strip()
        if not key:
            return
        if getattr(self, "_context_prewarm_active", False) and not key.startswith("[prewarm]"):
            key = f"[prewarm] {key}"
        if not key:
            return
        if key not in self._prep_trace_counts:
            self._prep_trace_order.append(key)
            if len(self._prep_trace_order) > 20:
                old = self._prep_trace_order.pop(0)
                self._prep_trace_counts.pop(old, None)
            self._prep_trace_counts[key] = 0
        self._prep_trace_counts[key] += 1
        self._emit_status_text()

    def _refresh_status_text(self) -> None:
        if not self._on_gui_thread():
            return
        if self._state is None:
            self._last_status_base_text = "Map/box display placeholder (uninitialized)"
            self._emit_status_text()
            return
        geom = self._state.geometry
        if geom is None:
            geom_text = "geometry: <none>"
        else:
            geom_text = (
                f"geometry: {geom.coord_mode.value} "
                f"({geom.coord_x:.3f}, {geom.coord_y:.3f}), "
                f"dims={geom.grid_x}x{geom.grid_y}x{geom.grid_z}, dx={geom.dx_km:.3f} km"
            )
        if self._state.fov is None:
            fov_text = "fov: <auto>"
        else:
            fov = self._state.fov
            fov_text = (
                f"fov: center=({fov.center_x_arcsec:.2f}, {fov.center_y_arcsec:.2f}) arcsec, "
                f"size={fov.width_arcsec:.2f}x{fov.height_arcsec:.2f} arcsec"
            )
        base_text = (
            "Map/box selector interaction\n"
            f"mouse_actions={'on' if self._mouse_actions_enabled else 'off'}\n"
            f"geometry_edit={'on' if self._geometry_edit_enabled else 'off'}\n"
            f"display_observer={self._observer_label_for_key(self._state.display_observer_key)}\n"
            f"geometry_frame={self._observer_label_for_key(self._state.geometry_definition_observer_key)}\n"
            f"fov_frame={self._observer_label_for_key(self._state.fov_definition_observer_key)}\n"
            f"context={self._display_map_label(self._state.selected_context_id, bottom=False)!r}, "
            f"base={self._display_map_label(self._state.selected_bottom_id, bottom=True)!r}\n"
            f"map_source={self._state.map_source_mode}\n"
            f"square_fov={'on' if self._state.square_fov else 'off'}\n"
            f"{geom_text}\n{fov_text}"
        )
        if self._observer_availability_notice:
            base_text = f"{base_text}\n\n{self._observer_availability_notice}"
        if self._refmap_display_notices:
            base_text = f"{base_text}\n\n" + "\n".join(self._refmap_display_notices)
        model_time = str(self._state.session_input.time_iso or "")
        if model_time:
            base_text = f"{base_text}\nmodel_time={model_time}"
        anchor = self._display_obstime_anchor()
        if anchor is not None:
            anchor_banner = self._format_display_time_banner(anchor, self._state.session_input.time_iso)
            if anchor_banner:
                base_text = f"{base_text}\ndisplay_time={anchor_banner}"
        observer_time = ""
        if self._normalize_observer_key(self._state.display_observer_key) == "custom":
            if isinstance(self._state.custom_observer_ephemeris, dict):
                observer_time = str(
                    self._state.custom_observer_ephemeris.get("obs_date")
                    or self._state.custom_observer_ephemeris.get("obs_time")
                    or ""
                )
        elif anchor is not None:
            try:
                observer_time = anchor.isot
            except Exception:
                observer_time = ""
        elif model_time:
            observer_time = model_time
        if observer_time:
            base_text = f"{base_text}\nobserver_time={observer_time}"
        save_issue = self.fov_persistence_issue()
        if save_issue is not None:
            base_text = (
                f"{base_text}\nfov_save_note=FOV unchanged in "
                f"{self._observer_label_for_key(self._state.fov_definition_observer_key)} frame; "
                f"save will ask to realign for "
                f"{self._observer_label_for_key(self._state.display_observer_key)} or clear FOV"
            )
        if self._normalize_observer_key(self._state.display_observer_key) == "custom":
            base_text = f"{base_text}\ncustom_label={self._state.custom_observer_label or 'Custom'}"
            if self._state.custom_observer_source:
                base_text = f"{base_text}\ncustom_source={self._state.custom_observer_source}"
        self._last_status_base_text = base_text
        self._emit_status_text()

    def _refresh_map_info(self) -> None:
        if self._state is None:
            self._set_map_info_text("Map info: <uninitialized>")
            return
        context_summary = self._single_map_summary(self._state.selected_context_id, role="Context", bottom=False)
        bottom_summary = self._single_map_summary(self._state.selected_bottom_id, role="Base", bottom=True)
        self._set_map_info_text(f"{context_summary}\n\n{bottom_summary}")

    def _set_map_info_text(self, text: str) -> None:
        self._last_map_info_text = text
        if self._map_info_callback is not None:
            self._map_info_callback(text)

    def _auto_adjust_axes_margins(self, ax, *, top: float = 0.93, pad_px: float = 8.0) -> bool:
        """Expand subplot margins after rendering if WCS labels are clipped."""
        try:
            self._fig.subplots_adjust(top=top)
            self._canvas.draw()
            renderer = self._canvas.get_renderer()
            tight = ax.get_tightbbox(renderer)
            if tight is None:
                return False
            fig_bbox = self._fig.bbox
            fig_w = max(float(fig_bbox.width), 1.0)
            fig_h = max(float(fig_bbox.height), 1.0)
            sp = self._fig.subplotpars
            left = float(sp.left)
            right = float(sp.right)
            bottom = float(sp.bottom)

            left_over = max(0.0, (fig_bbox.x0 + pad_px) - float(tight.x0))
            right_over = max(0.0, float(tight.x1) + pad_px - fig_bbox.x1)
            bottom_over = max(0.0, (fig_bbox.y0 + pad_px) - float(tight.y0))

            new_left = min(0.30, left + (left_over / fig_w))
            new_right = max(0.70, right - (right_over / fig_w))
            new_bottom = min(0.22, bottom + (bottom_over / fig_h))

            if (
                abs(new_left - left) > 1e-4
                or abs(new_right - right) > 1e-4
                or abs(new_bottom - bottom) > 1e-4
            ):
                self._fig.subplots_adjust(left=new_left, right=new_right, bottom=new_bottom, top=top)
                return True
        except Exception:
            return False
        return False

    def _emit_observer_info(self) -> None:
        if self._observer_info_callback is not None:
            self._observer_info_callback(self.current_observer_info())

    def _single_map_summary(self, map_id: Optional[str], role: str, bottom: bool) -> str:
        if not map_id:
            return f"{role} map: <none>"
        source_token = self._map_source_cache_token()
        view_key = str(self._view_mode or "box_fov")
        cache_key = f"{role}:{source_token}:{view_key}:{map_id}"
        if cache_key in self._map_summary_cache:
            return self._map_summary_cache[cache_key]
        try:
            smap = self._selected_bottom_map() if bottom else self._selected_context_map()
            if smap is None:
                txt = f"{role} map ({self._display_map_label(map_id, bottom)}): unavailable"
                self._map_summary_cache[cache_key] = txt
                return txt
            data = np.asarray(smap.data)
            finite = np.isfinite(data)
            n_finite = int(finite.sum())
            stats = "all non-finite"
            if n_finite > 0:
                vals = data[finite]
                stats = (
                    f"min={float(np.nanmin(vals)):.3g}, "
                    f"max={float(np.nanmax(vals)):.3g}, "
                    f"mean={float(np.nanmean(vals)):.3g}"
                )
            obs_time = getattr(smap, "date", None)
            purpose = "bottom" if bottom else "context"
            txt = (
                f"{role} map ({self._display_map_label(map_id, bottom)})\n"
                f"source={self._map_source_label(map_id, purpose=purpose)}\n"
                f"shape={tuple(data.shape)}, finite={n_finite}/{data.size}\n"
                f"{stats}\n"
                f"obs_time={obs_time}"
            )
        except Exception as exc:
            purpose = "bottom" if bottom else "context"
            txt = f"{role} map ({map_id}) load failed:\n{self._map_source_label(map_id, purpose=purpose)}\n{exc}"
        self._map_summary_cache[cache_key] = txt
        return txt

    @staticmethod
    def _display_map_label(map_id: Optional[str], bottom: bool) -> Optional[str]:
        if map_id is None:
            return None
        if not bottom and map_id == "Bz":
            return "Blos"
        return map_id

    def _selected_context_map(self):
        if self._state is None:
            return None
        map_id = self._state.selected_context_id
        if not map_id:
            return None
        return self._map_for_id(map_id, purpose="context")

    def _selected_bottom_map(self):
        if self._state is None:
            return None
        map_id = self._state.selected_bottom_id
        if not map_id:
            return None
        return self._map_for_id(map_id, purpose="bottom")

    def _map_for_id(self, map_id: str, purpose: str):
        self._ensure_cache_initialized()
        canonical_key = self._canonical_map_key(map_id, purpose=purpose)
        display_key = self._display_prepared_cache_key(canonical_key, purpose)
        alias_key = (
            f"__{purpose}__:{self._map_source_cache_token()}:"
            f"{self._normalize_observer_key(self._state.display_observer_key if self._state else 'earth')}:"
            f"{self.__dict__.get('_view_mode', 'box_fov') or 'box_fov'}:{map_id}"
        )
        display_cache = self.__dict__.setdefault("_display_prepared_cache", {})
        with self._cache_lock:
            if alias_key in display_cache:
                return display_cache[alias_key].get("map")
            if display_key in display_cache:
                entry = display_cache[display_key]
                display_cache[alias_key] = entry
                return entry.get("map")

        smap = self._load_raw_map(canonical_key, purpose=purpose)
        if smap is None:
            return None
        use_native_crop = None
        if purpose == "context":
            use_native_crop = self.__dict__.get("_view_mode", "box_fov") == "box_fov"
        prepared_map, coverage_fov = self._prepare_map_for_display(
            canonical_key,
            smap,
            purpose=purpose,
            use_native_crop=use_native_crop,
        )
        entry = {"map": prepared_map, "coverage_fov": coverage_fov}
        loaded_cache = self.__dict__.setdefault("_loaded_map_cache", {})
        with self._cache_lock:
            display_cache[display_key] = entry
            display_cache[alias_key] = entry
            loaded_cache[display_key] = prepared_map
            loaded_cache[alias_key] = prepared_map
        return prepared_map

    def _context_map_for_id(self, map_id: str, canonical_key: str):
        return self._map_for_id(map_id, purpose="context")

    def _ensure_prepared_context_cache(
        self,
        canonical_key: str,
        source_mode: str,
        *,
        prepare_variant: str,
    ) -> None:
        del source_mode, prepare_variant
        self._ensure_native_crop_cache(canonical_key)

    def _schedule_context_prewarm(self) -> None:
        if self._state is None:
            return
        map_keys = tuple(self._iter_warmable_context_map_keys())
        if not map_keys:
            return
        with self._cache_lock:
            self._context_prewarm_generation += 1
            generation = self._context_prewarm_generation
        self._set_runtime_status(
            f"Preparing reference maps in background ({len(map_keys)} map(s))..."
        )
        thread = threading.Thread(
            target=self._context_prewarm_worker,
            args=(generation, map_keys),
            daemon=True,
        )
        with self._cache_lock:
            self._context_prewarm_thread = thread
        thread.start()

    def _context_prewarm_worker(self, generation: int, map_keys: tuple[str, ...]) -> None:
        self._context_prewarm_active = True
        prepared = 0
        try:
            for canonical_key in map_keys:
                with self._cache_lock:
                    if generation != self._context_prewarm_generation:
                        return
                try:
                    self._ensure_native_crop_cache(canonical_key)
                    prepared += 1
                except Exception:
                    continue
        finally:
            self._context_prewarm_active = False
        QTimer.singleShot(
            0,
            lambda: self._on_context_prewarm_finished(generation, prepared, len(map_keys)),
        )

    def _on_context_prewarm_finished(self, generation: int, prepared: int, total: int) -> None:
        with self._cache_lock:
            if generation != self._context_prewarm_generation:
                return
        if prepared >= total:
            self._set_runtime_status(f"Reference map cache ready ({prepared} map(s)).")
        else:
            self._set_runtime_status(
                f"Reference map cache partially ready ({prepared}/{total} map(s))."
            )
        self._refresh_status_text()

    def _invalidate_display_prepared_cache(self) -> None:
        self._ensure_cache_initialized()
        with self._cache_lock:
            self._display_prepared_cache.clear()
            self._loaded_map_cache.clear()
            self._background_cache_generation += 1

    def _invalidate_native_crop_cache(self) -> None:
        with self._cache_lock:
            self._native_crop_cache.clear()
            self._context_prewarm_generation += 1

    def _invalidate_display_on_source_mode_change(self) -> None:
        with self._cache_lock:
            self._display_prepared_cache.clear()
            self._loaded_map_cache.clear()
            self._native_crop_cache.clear()
            self._raw_map_cache.clear()
            self._background_cache_generation += 1
            self._context_prewarm_generation += 1

    def _invalidate_map_caches(self) -> None:
        with self._cache_lock:
            self._context_prewarm_generation += 1
            self._display_prepared_cache.clear()
            self._loaded_map_cache.clear()
            self._native_crop_cache.clear()
            self._raw_map_cache.clear()
            self._background_cache_generation += 1

    def _invalidate_display_map_cache(self) -> None:
        self._invalidate_display_prepared_cache()

    def _invalidate_geometry_dependent_display_maps(self) -> None:
        with self._cache_lock:
            self._native_crop_cache.clear()
            self._context_prewarm_generation += 1
            self._display_prepared_cache.clear()
            self._loaded_map_cache.clear()
            self._background_cache_generation += 1

    def _start_background_cache_build(self) -> None:
        if not self._background_cache_enabled:
            return
        if self._state is None:
            return
        map_ids = tuple(
            m for m in (self._state.session_input.map_ids or ())
            if m in {"Bz", "Ic", "B_rho", "B_theta", "B_phi", "disambig", "Br", "Bp", "Bt"}
        )
        if not map_ids:
            return
        with self._cache_lock:
            generation = self._background_cache_generation
            thread = self._background_cache_thread
            if thread is not None and thread.is_alive():
                return
            self._background_cache_thread = threading.Thread(
                target=self._background_cache_worker,
                args=(generation, map_ids),
                daemon=True,
            )
            self._background_cache_thread.start()

    def _background_cache_worker(self, generation: int, map_ids: tuple[str, ...]) -> None:
        for map_id in map_ids:
            with self._cache_lock:
                if generation != self._background_cache_generation:
                    return
            try:
                self._map_for_id(map_id, purpose="context")
                self._map_for_id(map_id, purpose="bottom")
            except Exception:
                continue

    @staticmethod
    def _canonical_map_key(map_id: str, *, purpose: str = "context") -> str:
        if purpose == "bottom":
            return _BOTTOM_DISPLAY_MAP_ALIASES.get(map_id, map_id)
        return _CONTEXT_DISPLAY_MAP_ALIASES.get(map_id, map_id)

    def _map_source_label(self, map_id: str, *, purpose: str = "context") -> str:
        canonical_key = self._canonical_map_key(map_id, purpose=purpose)
        if canonical_key in {"field", "inclination", "azimuth", "disambig"}:
            path = self._filesystem_path_for_key(canonical_key, purpose=purpose)
            if path:
                return Path(path).name
            return canonical_key
        path = self._filesystem_path_for_key(canonical_key, purpose=purpose)
        if path:
            return Path(path).name
        if self._embedded_base_key_for_map(canonical_key) and self._embedded_base_array(canonical_key, purpose=purpose) is not None:
            return f"embedded:base.{self._embedded_base_key_for_map(canonical_key)}"
        ref_key = self._embedded_refmap_key(canonical_key)
        if ref_key and self._embedded_payload_for_key(ref_key, purpose=purpose):
            return f"embedded:{ref_key}"
        return canonical_key

    def _filesystem_enabled(self, purpose: str = "context") -> bool:
        if purpose == "bottom":
            return False
        return self._state is not None and self._state.map_source_mode in {"auto", "filesystem"}

    def _embedded_enabled(self, purpose: str = "context") -> bool:
        if self._state is None:
            return False
        if purpose == "bottom":
            return bool(self._state.base_maps or self._state.refmaps)
        # In "filesystem" mode, prefer on-disk files when they exist, but still
        # allow fallback to embedded products for map types that have no
        # filesystem representation (e.g. Vert_current in saved HDF5 models).
        return bool(self._state.base_maps or self._state.refmaps)

    def _filesystem_path_for_key(self, map_key: str, purpose: str = "context") -> str | None:
        if not self._filesystem_enabled(purpose=purpose):
            return None
        return (self._state.map_files or {}).get(map_key) if self._state is not None else None

    def _embedded_payload_for_key(self, ref_key: str, purpose: str = "context"):
        if not self._embedded_enabled(purpose=purpose) or self._state is None:
            return None
        return (self._state.refmaps or {}).get(ref_key)

    @staticmethod
    def _embedded_base_key_for_map(map_key: str) -> str | None:
        key = str(map_key)
        key_l = key.lower()
        if key_l in {"bx", "by", "bz"}:
            return key_l
        if map_key == "magnetogram":
            return "bz"
        if key_l in {"continuum", "ic"}:
            return "ic"
        if key_l == "chromo_mask":
            return "chromo_mask"
        if key_l == "vert_current":
            return "vert_current"
        return None

    def _embedded_base_array(self, map_key: str, purpose: str = "context"):
        if not self._embedded_enabled(purpose=purpose) or self._state is None:
            return None
        base_key = self._embedded_base_key_for_map(map_key)
        if not base_key:
            return None
        base_maps = self._state.base_maps or {}
        if base_key not in base_maps:
            # Backward/forward compatibility for case variants in persisted keys.
            folded = {str(k).lower(): k for k in base_maps.keys()}
            if str(base_key).lower() not in folded:
                return None
            base_key = folded[str(base_key).lower()]
        arr = np.asarray(base_maps[base_key])
        if arr.ndim != 2:
            return None
        return arr

    def _load_embedded_base_map(
        self,
        map_key: str,
        purpose: str = "context",
        *,
        source_token: str | None = None,
    ):
        if self._state is None:
            return None
        source_token = source_token or self._map_source_cache_token()
        with self._cache_lock:
            cache_key = f"__base__:{source_token}:{purpose}:{map_key}"
            if cache_key in self._raw_map_cache:
                return self._raw_map_cache[cache_key]
        data = self._embedded_base_array(map_key, purpose=purpose)
        if data is None:
            return None
        try:
            header = self._model_geometry_earth_wcs_header()
            if header is None:
                return None
            smap = map_from_data_header_compat(np.asarray(data), header)
        except Exception:
            return None
        with self._cache_lock:
            self._raw_map_cache[cache_key] = smap
        return smap

    @staticmethod
    def _header_text_from_value(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", "ignore")
        if isinstance(value, np.ndarray) and value.shape == ():
            return MapBoxDisplayWidget._header_text_from_value(value.item())
        return str(value)

    @staticmethod
    def _normalize_embedded_header_text(header_text: str) -> str:
        text = str(header_text or "")
        # Embedded box files may persist FITS headers with literal "\\n"
        # separators instead of real newlines.
        if "\\n" in text and "\n" not in text:
            text = text.replace("\\n", "\n")
        return text

    @staticmethod
    def _copy_observer_cards_from_map(header, smap) -> None:
        if header is None or smap is None:
            return
        meta = getattr(smap, "meta", None)
        if meta is None:
            return
        # Copy observer ephemeris only; keep embedded DATE-OBS from the payload.
        # Reference maps (often from filesystem) can be from unrelated epochs and
        # must not overwrite the embedded observation-time anchor.
        for src_key, dst_key in (
            ("hgln_obs", "HGLN_OBS"),
            ("hglt_obs", "HGLT_OBS"),
            ("dsun_obs", "DSUN_OBS"),
            ("crln_obs", "CRLN_OBS"),
            ("crlt_obs", "CRLT_OBS"),
            ("rsun_ref", "RSUN_REF"),
        ):
            value = meta.get(src_key)
            if value is None:
                value = meta.get(dst_key)
            if value is not None:
                header[dst_key] = value

    @staticmethod
    def _embedded_refmap_key(map_key: str) -> str | None:
        key = str(map_key)
        key_l = key.lower()
        if key == "magnetogram":
            return "Bz_reference"
        if key == "continuum":
            return "Ic_reference"
        if key_l == "vert_current":
            return "Vert_current"
        if key.isdigit():
            return f"AIA_{key}"
        return key

    def _load_embedded_refmap(
        self,
        ref_key: str,
        purpose: str = "context",
        *,
        source_token: str | None = None,
    ):
        if self._state is None:
            return None
        source_token = source_token or self._map_source_cache_token()
        with self._cache_lock:
            cache_key = f"__embedded__:{source_token}:{purpose}:{ref_key}"
            if cache_key in self._raw_map_cache:
                return self._raw_map_cache[cache_key]
        payload = self._embedded_payload_for_key(ref_key, purpose=purpose)
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        header_text = self._normalize_embedded_header_text(
            self._header_text_from_value(payload.get("wcs_header"))
        )
        if data is None or not header_text.strip():
            return None
        try:
            header = fits.Header.fromstring(header_text, sep="\n")
            if bool(header.get("PYALIGN", False)):
                ref_map = self._earth_geometry_reference_map()
                if ref_map is not None:
                    self._copy_observer_cards_from_map(header, ref_map)
            self._ensure_embedded_header_obstime(header)
            header[_EMBEDDED_REFMAP_FLAG] = True
            smap = map_from_data_header_compat(np.asarray(data), header)
        except Exception:
            return None
        with self._cache_lock:
            self._raw_map_cache[cache_key] = smap
        return smap

    def _load_raw_map_for_source_mode(
        self,
        map_key: str,
        source_mode: str,
        *,
        purpose: str = "context",
    ):
        source_mode = str(source_mode or "auto").lower()
        with self._cache_lock:
            raw_cache_key = f"__rawmap__:{source_mode}:{purpose}:{map_key}"
            if raw_cache_key in self._raw_map_cache:
                return self._raw_map_cache[raw_cache_key]
        smap = None
        if source_mode in {"auto", "filesystem"}:
            path = (self._state.map_files or {}).get(map_key) if self._state is not None else None
            if path:
                smap = load_sunpy_map_compat(path)
                if map_key in _HMI_VECTOR_SEGMENTS:
                    smap = self._submap_to_geometry_fov(smap)
        if smap is None and (source_mode in {"auto", "embedded"} or purpose == "bottom"):
            smap = self._load_embedded_base_map(
                map_key,
                purpose=purpose,
                source_token=source_mode,
            )
            if smap is None:
                ref_key = self._embedded_refmap_key(map_key)
                smap = (
                    self._load_embedded_refmap(ref_key, purpose=purpose, source_token=source_mode)
                    if ref_key
                    else None
                )
        if smap is None:
            return None
        with self._cache_lock:
            self._raw_map_cache[raw_cache_key] = smap
        return smap

    def _load_raw_map(self, map_key: str, purpose: str = "context"):
        return self._load_raw_map_for_source_mode(
            map_key,
            self._map_source_cache_token(),
            purpose=purpose,
        )

    def _prepare_context_map(
        self,
        map_key: str,
        smap,
        *,
        prepare_variant: str = _CONTEXT_PREPARE_VARIANT_FULL_DISK,
    ):
        use_native_crop = prepare_variant == _CONTEXT_PREPARE_VARIANT_FOV_CROP
        return self._prepare_map_for_display(
            map_key,
            smap,
            purpose="context",
            use_native_crop=use_native_crop,
        )

    def _prepare_bottom_map(self, map_key: str, smap):
        prepared, _coverage = self._prepare_map_for_display(
            map_key,
            smap,
            purpose="bottom",
            use_native_crop=False,
        )
        return prepared

    def _model_obstime_for_geometry(self) -> Time | None:
        if self._state is None:
            return None
        anchor = self._display_obstime_anchor()
        if anchor is not None:
            return anchor
        fallback_iso = self._state.session_input.time_iso
        return self._parse_obstime(fallback_iso)

    def _geometry_stub_map(self, obstime, observer_key: str = "earth"):
        when = obstime if isinstance(obstime, Time) else Time(obstime)
        observer = self._resolve_display_observer_coord(observer_key, when)
        if observer is None:
            try:
                observer = get_earth(when)
            except Exception:
                observer = "earth"
        center = SkyCoord(
            0 * u.arcsec,
            0 * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=when),
        )
        data = np.zeros((2, 2), dtype=np.float32)
        header = make_fitswcs_header(
            data,
            center,
            scale=u.Quantity([1.0, 1.0], u.arcsec / u.pix),
        )
        header["DATE-OBS"] = when.isot
        header["DATE_OBS"] = when.isot
        return Map(data, header)

    def _stamp_model_time_and_observer_on_header(
        self,
        header: fits.Header,
        obstime,
        observer_key: str,
    ) -> None:
        if obstime is None:
            return
        when = obstime if isinstance(obstime, Time) else Time(obstime)
        header["DATE-OBS"] = when.isot
        header["DATE_OBS"] = when.isot
        metadata = self._resolve_display_observer_metadata(observer_key, when)
        if not isinstance(metadata, dict):
            return
        if metadata.get("hgln_obs_deg") is not None:
            header["HGLN_OBS"] = float(metadata["hgln_obs_deg"])
        if metadata.get("hglt_obs_deg") is not None:
            header["HGLT_OBS"] = float(metadata["hglt_obs_deg"])
        dsun_cm = metadata.get("dsun_cm")
        if dsun_cm is not None:
            try:
                header["DSUN_OBS"] = float(dsun_cm) * 0.01
            except Exception:
                pass
        rsun_cm = metadata.get("rsun_cm")
        if rsun_cm is not None:
            try:
                header["RSUN_REF"] = float(rsun_cm) * 0.01
            except Exception:
                pass

    @staticmethod
    def _header_only_map_from_wcs(header: fits.Header):
        try:
            naxis1 = int(header.get("NAXIS1", 0) or 0)
            naxis2 = int(header.get("NAXIS2", 0) or 0)
            if naxis1 <= 0 or naxis2 <= 0:
                return None
            data = np.full((naxis2, naxis1), np.nan, dtype=np.float32)
            return map_from_data_header_compat(data, header.copy())
        except Exception:
            return None

    def _model_geometry_earth_wcs_header(self) -> fits.Header | None:
        if self._state is None or self._state.geometry is None:
            return None
        obstime = self._model_obstime_for_geometry()
        if obstime is None:
            return None
        geometry_observer_key = self._normalize_observer_key(
            self._state.geometry_definition_observer_key
        )
        base_text = str(self._state.base_wcs_header or "").strip()
        if base_text:
            try:
                header = fits.Header.fromstring(base_text, sep="\n")
                naxis1 = int(header.get("NAXIS1", 0) or 0)
                naxis2 = int(header.get("NAXIS2", 0) or 0)
                if naxis1 > 0 and naxis2 > 0:
                    self._stamp_model_time_and_observer_on_header(
                        header,
                        obstime,
                        geometry_observer_key,
                    )
                    return header
            except Exception:
                pass
        stub = self._geometry_stub_map(obstime, geometry_observer_key)
        box = self._build_legacy_box(
            stub,
            geometry_observer_key=geometry_observer_key,
        )
        if box is None:
            return None
        header = fits.Header(box.bottom_cea_header)
        self._stamp_model_time_and_observer_on_header(
            header,
            obstime,
            geometry_observer_key,
        )
        return header

    def _model_geometry_scaffold_cache_key(self) -> str:
        display_key = (
            self._normalize_observer_key(self._state.display_observer_key)
            if self._state is not None
            else "earth"
        )
        return (
            f"__geom_scaffold__:{self._map_source_cache_token()}:"
            f"{self._display_obstime_cache_token()}:"
            f"{display_key}:{self._geometry_cache_token()}"
        )

    @staticmethod
    def _fov_selection_to_dict(fov: DisplayFovSelection) -> dict[str, float]:
        return {
            "xc_arcsec": float(fov.center_x_arcsec),
            "yc_arcsec": float(fov.center_y_arcsec),
            "xsize_arcsec": float(fov.width_arcsec),
            "ysize_arcsec": float(fov.height_arcsec),
        }

    def _empty_observer_scaffold_from_geometry(self, earth_map, display_key: str, obstime):
        from pyampp.io.refmap_crop import make_empty_observer_fov_map

        base_fov = self._model_fov_in_definition_frame(earth_map)
        if base_fov is None:
            return None
        projected = self._project_fov_between_observers(
            base_fov,
            self._state.geometry_definition_observer_key,
            display_key,
            obstime,
        )
        if projected is None:
            return None
        observer = self._resolve_display_observer_coord(display_key, obstime)
        if observer is None:
            return None
        try:
            return make_empty_observer_fov_map(
                earth_map,
                observer=observer,
                obstime=obstime,
                fov=self._fov_selection_to_dict(projected),
            )
        except Exception:
            return None

    def _model_geometry_scaffold_map(self):
        """Header-only model WCS scaffold at model time for the display observer."""
        if self._state is None or self._state.geometry is None:
            return None
        self._ensure_cache_initialized()
        cache_key = self._model_geometry_scaffold_cache_key()
        with self._cache_lock:
            cached = self._raw_map_cache.get(cache_key)
            if cached is not None:
                return cached
        header = self._model_geometry_earth_wcs_header()
        if header is None:
            return None
        earth_map = self._header_only_map_from_wcs(header)
        if earth_map is None:
            return None
        display_key = self._normalize_observer_key(self._state.display_observer_key)
        geometry_key = self._normalize_observer_key(self._state.geometry_definition_observer_key)
        obstime = self._model_obstime_for_geometry()
        # A shared LOS still has to be helioprojective. CEA / Carrington
        # base headers are the model grid; drawing boxes on them shows
        # Carrington longitude and latitude instead of the observer LOS.
        share_los = obstime is not None and self._observers_share_los(
            display_key, geometry_key, obstime
        )
        if share_los and not self._is_known_non_los_map(earth_map):
            scaffold = earth_map
        else:
            scaffold = self._empty_observer_scaffold_from_geometry(earth_map, display_key, obstime)
            if scaffold is None or self._is_known_non_los_map(scaffold):
                when = obstime or self._model_obstime_for_geometry()
                try:
                    scaffold = self._geometry_stub_map(when, display_key) if when is not None else None
                except Exception:
                    scaffold = None
            if (scaffold is None or self._is_known_non_los_map(scaffold)) and not self._is_known_non_los_map(earth_map):
                scaffold = earth_map
        if scaffold is None:
            return None
        with self._cache_lock:
            self._raw_map_cache[cache_key] = scaffold
        return scaffold

    @staticmethod
    def _is_helioprojective_map(smap) -> bool:
        try:
            frame = smap.coordinate_frame
        except Exception:
            frame = None
        name = str(getattr(frame, "name", "") or "").lower()
        if "helioprojective" in name:
            return True
        meta = getattr(smap, "meta", {}) or {}
        ctype = str(meta.get("ctype1") or meta.get("CTYPE1") or "").upper()
        return ctype.startswith("HPLN")

    @staticmethod
    def _is_known_non_los_map(smap) -> bool:
        """True when the map frame is identified and is not helioprojective.

        Unidentified test doubles are left alone. Carrington / CEA headers are
        rejected so they cannot become the selector axes.
        """
        try:
            frame = smap.coordinate_frame
        except Exception:
            frame = None
        name = str(getattr(frame, "name", "") or "").lower()
        if "helioprojective" in name:
            return False
        if name:
            return True
        meta = getattr(smap, "meta", {}) or {}
        ctype = str(meta.get("ctype1") or meta.get("CTYPE1") or "").upper()
        if ctype.startswith("HPLN"):
            return False
        return bool(ctype)

    def _earth_geometry_reference_map(self):
        header = self._model_geometry_earth_wcs_header()
        if header is None:
            return None
        return self._header_only_map_from_wcs(header)

    @staticmethod
    def _bottom_display_map_ids(session_input: SelectorSessionInput) -> set[str]:
        base_maps = dict(session_input.base_maps or {})
        aliases = {
            "bx": "Bx",
            "by": "By",
            "bz": "Bz",
            "ic": "Ic",
            "vert_current": "Vert_current",
            "chromo_mask": "chromo_mask",
        }
        out: set[str] = set()
        for base_key in base_maps:
            display_id = aliases.get(str(base_key).lower(), str(base_key))
            out.add(display_id)
        return out

    @staticmethod
    def _available_context_map_ids(session_input: SelectorSessionInput) -> list[str]:
        base_ids = list(session_input.map_ids or ())
        bottom_ids = MapBoxDisplayWidget._bottom_display_map_ids(session_input)
        bottom_only = {map_id for map_id in bottom_ids if map_id not in {"Bz", "Ic"}}
        bottom_only.update({"chromo_mask", "Bx", "By"})
        preferred = [
            "94", "131", "1600", "1700", "171", "193", "211", "304", "335",
            "Bz", "Ic", "B_rho", "B_theta", "B_phi", "disambig", "Vert_current",
            "Br", "Bp", "Bt",
        ]
        allowed = [map_id for map_id in base_ids if map_id not in bottom_only]
        ordered = [map_id for map_id in preferred if map_id in allowed]
        ordered.extend(map_id for map_id in allowed if map_id not in ordered)
        return ordered

    def _uses_geometry_scaffold_for_context(self) -> bool:
        if self._state is None:
            return False
        if not self._state.selected_context_id:
            return True
        return not self._available_context_map_ids(self._state.session_input)

    def _context_canvas_map(self):
        if self._uses_geometry_scaffold_for_context():
            return self._model_geometry_scaffold_map()
        return self._selected_context_map()

    def _geometry_anchor_coord(self, geom: BoxGeometrySelection, smap, observer_key: str | None = None):
        obstime = getattr(smap, "date", None)
        if observer_key is None and self._state is not None:
            observer_key = self._state.geometry_definition_observer_key
        observer_context = self._observer_context(observer_key, obstime)
        observer = getattr(observer_context, "observer_coordinate", None)
        if observer is None:
            observer = self._resolved_observer_for_map(smap, observer_key) or "earth"
        if geom.coord_mode == CoordMode.HPC:
            return SkyCoord(
                Tx=geom.coord_x * u.arcsec,
                Ty=geom.coord_y * u.arcsec,
                obstime=obstime,
                observer=observer,
                frame=Helioprojective,
            )
        if geom.coord_mode == CoordMode.HGC:
            return SkyCoord(
                lon=geom.coord_x * u.deg,
                lat=geom.coord_y * u.deg,
                radius=696 * u.Mm,
                obstime=obstime,
                observer=observer,
                frame=HeliographicCarrington,
            )
        return SkyCoord(
            lon=geom.coord_x * u.deg,
            lat=geom.coord_y * u.deg,
            radius=696 * u.Mm,
            obstime=obstime,
            observer=observer,
            frame=HeliographicStonyhurst,
        )

    def _build_legacy_box(
        self,
        smap,
        geom: BoxGeometrySelection | None = None,
        *,
        geometry_observer_key: str | None = None,
    ):
        if self._state is None:
            return None
        geom = geom or self._state.geometry
        if geom is None:
            return None
        box_dims = u.Quantity([geom.grid_x, geom.grid_y, geom.grid_z], u.pix)
        box_res = geom.dx_km * u.km
        box_origin = self._geometry_anchor_coord(geom, smap, observer_key=geometry_observer_key)
        observer = self._resolved_observer_for_map(
            smap,
            geometry_observer_key if geometry_observer_key is not None else (
                self._state.display_observer_key if self._state is not None else "earth"
            ),
        ) or "earth"
        obstime = getattr(smap, "date", None)
        frame_obs = Helioprojective(observer=observer, obstime=obstime)
        frame_hcc = Heliocentric(observer=box_origin, obstime=obstime)
        box_dimensions = box_dims / u.pix * box_res
        box_center = box_origin.transform_to(frame_hcc)
        box_center = SkyCoord(
            x=box_center.x,
            y=box_center.y,
            z=box_center.z + box_dimensions[2] / 2,
            frame=box_center.frame,
        )
        box = Box(frame_obs, box_origin, box_center, box_dims, box_res)
        if self._session_box_template is not None and isinstance(getattr(self._session_box_template, "b3d", None), dict):
            box.b3d = copy.deepcopy(self._session_box_template.b3d)
        self._apply_live_session_state(box, update_frame_obs=False)
        return box

    def _submap_to_box_bounds(self, smap, box, pad_frac: float | None = None):
        if box is None:
            return smap
        if pad_frac is None:
            pad_frac = float(self._state.session_input.pad_frac or 0.10) if self._state is not None else 0.10
        try:
            fov = box.bounds_coords_bl_tr(pad_frac=pad_frac)
            return smap.submap(fov[0], top_right=fov[1])
        except Exception:
            return smap

    def _submap_to_geometry_fov(self, smap):
        geometry_observer_key = self._state.geometry_definition_observer_key if self._state is not None else None
        return self._submap_to_box_bounds(
            smap,
            self._build_legacy_box(smap, geometry_observer_key=geometry_observer_key),
        )

    def _display_window_pixel_bounds(
        self,
        smap,
        *,
        pad_factor: float = 1.10,
    ) -> tuple[float, float, float, float] | None:
        if self._view_mode != "box_fov":
            return None
        projected_edges = self._fov_box_projected_edges(smap)
        projected_bbox = self._edge_pixel_bounds(smap, projected_edges) if projected_edges else None
        if projected_bbox is not None:
            x0, x1, y0, y1 = projected_bbox
        elif self._state is not None and self._state.fov is not None:
            rect = self._fov_selection_to_pixel_rect(smap, self._state.fov)
            x0 = rect.get_x()
            x1 = x0 + rect.get_width()
            y0 = rect.get_y()
            y1 = y0 + rect.get_height()
        else:
            geometry_observer_key = self._state.geometry_definition_observer_key if self._state is not None else None
            box = self._build_legacy_box(smap, geometry_observer_key=geometry_observer_key)
            if box is None:
                return None
            fov = self._box_bounds_to_fov_selection(box, smap)
            rect = self._fov_selection_to_pixel_rect(smap, fov)
            x0 = rect.get_x()
            x1 = x0 + rect.get_width()
            y0 = rect.get_y()
            y1 = y0 + rect.get_height()
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        half_w = 0.5 * max(abs(x1 - x0), 1e-6) * float(pad_factor)
        half_h = 0.5 * max(abs(y1 - y0), 1e-6) * float(pad_factor)
        return (
            float(cx - half_w),
            float(cx + half_w),
            float(cy - half_h),
            float(cy + half_h),
        )

    def _display_window_fov_selection(self, smap) -> DisplayFovSelection | None:
        pixel_bounds = self._display_window_pixel_bounds(smap, pad_factor=1.10)
        if pixel_bounds is not None:
            x0, x1, y0, y1 = pixel_bounds
            rect = Rectangle(
                (x0, y0),
                max(1e-6, x1 - x0),
                max(1e-6, y1 - y0),
                visible=False,
            )
            fov = self._pixel_rect_to_fov_selection(smap, rect)
            return DisplayFovSelection(
                center_x_arcsec=float(fov.center_x_arcsec),
                center_y_arcsec=float(fov.center_y_arcsec),
                width_arcsec=float(max(fov.width_arcsec, 1e-3)),
                height_arcsec=float(max(fov.height_arcsec, 1e-3)),
            )
        geometry_observer_key = self._state.geometry_definition_observer_key if self._state is not None else None
        box = self._build_legacy_box(smap, geometry_observer_key=geometry_observer_key)
        if box is None:
            return None
        fov = self._box_bounds_to_fov_selection(box, smap)
        return DisplayFovSelection(
            center_x_arcsec=float(fov.center_x_arcsec),
            center_y_arcsec=float(fov.center_y_arcsec),
            width_arcsec=float(max(fov.width_arcsec, 1e-3) * 1.10),
            height_arcsec=float(max(fov.height_arcsec, 1e-3) * 1.10),
        )

    def _submap_to_fov_selection_pixels(
        self,
        smap,
        fov: DisplayFovSelection,
        *,
        use_display_observer: bool = False,
    ):
        rect = self._fov_selection_to_pixel_rect(
            smap,
            fov,
            use_display_observer=use_display_observer,
        )
        if rect is None:
            return smap
        x0 = float(rect.get_x())
        y0 = float(rect.get_y())
        x1 = x0 + float(rect.get_width())
        y1 = y0 + float(rect.get_height())
        cropped = self._submap_to_pixel_bounds(smap, (x0, x1, y0, y1))
        if not self._valid_map_array(cropped):
            return smap
        return cropped

    def _submap_to_explicit_fov(
        self,
        smap,
        pad_factor: float = 1.10,
        *,
        fov_override: DisplayFovSelection | None = None,
        prefer_display_observer: bool = False,
    ):
        fov = fov_override if fov_override is not None else (self._state.fov if (self._state and self._state.fov) else None)
        if fov is None:
            geometry_observer_key = self._state.geometry_definition_observer_key if self._state is not None else None
            box = self._build_legacy_box(smap, geometry_observer_key=geometry_observer_key)
            if box is None:
                return smap
            fov = self._box_bounds_to_fov_selection(box, smap)
        if self._state is not None and not prefer_display_observer:
            return self._submap_to_fov_selection_pixels(
                smap,
                self._padded_fov_selection(fov, pad_factor) if float(pad_factor) != 1.0 else fov,
            )
        half_w = 0.5 * max(float(fov.width_arcsec), 1e-3) * float(pad_factor)
        half_h = 0.5 * max(float(fov.height_arcsec), 1e-3) * float(pad_factor)
        observer, obstime = self._fov_observer_coord_for_submap(
            smap,
            prefer_display_observer=prefer_display_observer,
        )
        bottom_left = SkyCoord(
            Tx=(float(fov.center_x_arcsec) - half_w) * u.arcsec,
            Ty=(float(fov.center_y_arcsec) - half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        top_right = SkyCoord(
            Tx=(float(fov.center_x_arcsec) + half_w) * u.arcsec,
            Ty=(float(fov.center_y_arcsec) + half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        try:
            cropped = _submap_with_fov_safe(smap, bottom_left, top_right)
            if not self._valid_map_array(cropped):
                return smap
            return cropped
        except Exception:
            return smap

    def _submap_to_pixel_bounds(
        self,
        smap,
        bounds: tuple[float, float, float, float],
        *,
        margin_pixels: float = 2.0,
    ):
        x0, x1, y0, y1 = bounds
        if not all(np.isfinite(v) for v in (x0, x1, y0, y1)):
            return smap
        x_lo = min(float(x0), float(x1)) - float(margin_pixels)
        x_hi = max(float(x0), float(x1)) + float(margin_pixels)
        y_lo = min(float(y0), float(y1)) - float(margin_pixels)
        y_hi = max(float(y0), float(y1)) + float(margin_pixels)
        try:
            bottom_left = smap.wcs.pixel_to_world(x_lo, y_lo)
            top_right = smap.wcs.pixel_to_world(x_hi, y_hi)
            cropped = smap.submap(bottom_left, top_right=top_right)
            if not MapBoxDisplayWidget._valid_map_array(cropped):
                return smap
            return cropped
        except Exception:
            return smap

    @staticmethod
    def _target_rsun_meters(target) -> float | None:
        try:
            if hasattr(target, "rsun_meters"):
                return float(target.rsun_meters.to_value(u.m))
        except Exception:
            pass
        try:
            if isinstance(target, dict):
                value = target.get("rsun_ref")
                if value is not None:
                    return float(value)
        except Exception:
            pass
        return None

    @staticmethod
    def _with_matching_rsun(smap, target):
        target_rsun_m = MapBoxDisplayWidget._target_rsun_meters(target)
        if not target_rsun_m or target_rsun_m <= 0:
            return smap
        try:
            current_rsun_m = float(smap.rsun_meters.to_value(u.m))
        except Exception:
            current_rsun_m = None
        if current_rsun_m is not None and abs(current_rsun_m - target_rsun_m) < 1e-3:
            return smap
        try:
            meta = smap.meta.copy()
            meta["rsun_ref"] = target_rsun_m
            return smap._new_instance(smap.data, meta, plot_settings=smap.plot_settings)
        except Exception:
            return smap

    @staticmethod
    def _apply_display_scaling(smap, map_key: str) -> None:
        data = np.asarray(smap.data)
        finite = np.isfinite(data)
        if not finite.any():
            return
        vals = data[finite]
        try:
            if map_key in _AIA_COLOR_KEYS:
                cmap = sunpy_colormaps.cm.cmlist.get(f"sdoaia{map_key}")
                if cmap is not None:
                    smap.plot_settings["cmap"] = cmap
                lo = float(np.nanpercentile(vals, 0.5))
                hi = float(np.nanpercentile(vals, 99.8))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    smap.plot_settings["norm"] = mcolors.Normalize(vmin=lo, vmax=hi)
            if map_key in _SIGNED_MAGNETIC_KEYS:
                if map_key in _TRANSVERSE_MAGNETIC_KEYS:
                    pct = 92.5
                elif map_key == "br":
                    pct = 97.5
                else:
                    pct = 99.0
                hi = float(np.nanpercentile(np.abs(vals), pct))
                if hi > 0:
                    smap.plot_settings["cmap"] = "gray"
                    smap.plot_settings["norm"] = mcolors.TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
            elif map_key in _VERT_CURRENT_KEYS:
                hi = float(np.nanpercentile(np.abs(vals), 99.0))
                if hi > 0:
                    smap.plot_settings["cmap"] = "RdBu_r"
                    smap.plot_settings["norm"] = mcolors.TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
            elif str(map_key).startswith(_EOVSA_REFMAP_PREFIX):
                lo = float(np.nanpercentile(vals, 0.5))
                hi = float(np.nanpercentile(vals, 99.5))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    smap.plot_settings["cmap"] = "hot"
                    smap.plot_settings["norm"] = mcolors.Normalize(vmin=lo, vmax=hi)
            elif map_key in _CHROMO_MASK_KEYS:
                cmap = mcolors.ListedColormap([
                    "#000000", "#1f77b4", "#ff7f0e", "#2ca02c",
                    "#d62728", "#9467bd", "#8c564b", "#e377c2",
                    "#7f7f7f",
                ])
                smap.plot_settings["cmap"] = cmap
                smap.plot_settings["norm"] = mcolors.BoundaryNorm(np.arange(-0.5, 9.5, 1.0), cmap.N)
            elif map_key == "continuum":
                lo = float(np.nanpercentile(vals, 1.0))
                hi = float(np.nanpercentile(vals, 99.5))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    smap.plot_settings["norm"] = mcolors.Normalize(vmin=lo, vmax=hi)
                    smap.plot_settings["cmap"] = "gray"
            elif map_key == "field":
                lo = float(np.nanpercentile(vals, 1.0))
                hi = float(np.nanpercentile(vals, 99.0))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    smap.plot_settings["norm"] = mcolors.Normalize(vmin=lo, vmax=hi)
                    smap.plot_settings["cmap"] = "magma"
            elif map_key in {"inclination", "azimuth"}:
                lo = float(np.nanpercentile(vals, 0.5))
                hi = float(np.nanpercentile(vals, 99.5))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    smap.plot_settings["norm"] = mcolors.Normalize(vmin=lo, vmax=hi)
                    smap.plot_settings["cmap"] = "twilight"
            elif map_key == "disambig":
                lo = float(np.nanmin(vals))
                hi = float(np.nanmax(vals))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    smap.plot_settings["norm"] = mcolors.Normalize(vmin=lo, vmax=hi)
                    smap.plot_settings["cmap"] = "viridis"
        except Exception:
            # Display scaling should not break rendering.
            pass

    def _refresh_plot(self, preserve_current_view: bool = False) -> None:
        prev_xlim = prev_ylim = None
        if preserve_current_view and self._current_axes is not None:
            try:
                prev_xlim = self._current_axes.get_xlim()
                prev_ylim = self._current_axes.get_ylim()
            except Exception:
                prev_xlim = prev_ylim = None
        self._clear_drag_preview_artists()
        self._fig.clear()
        self._current_map = None
        self._current_axes = None
        self._overlay_rect = None
        self._overlay_bbox_rect = None
        self._projected_box_bbox_rect = None
        self._projected_box_fov = None
        self._overlay_center_artist = None
        self._overlay_corner_artists = []
        self._overlay_line_artists = []
        self._zoom_anchor_px = None
        self._full_view_limits = None
        smap = overlay_map = None
        try:
            smap = self._context_canvas_map()
            overlay_map = self._selected_bottom_map()
        except Exception as exc:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Map load failed:\n{exc}", ha="center", va="center")
            ax.axis("off")
            self._canvas.draw_idle()
            return

        if smap is None:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "No local map available for selected map ID", ha="center", va="center")
            ax.axis("off")
            self._fig.subplots_adjust(left=0.12, right=0.985, bottom=0.10, top=0.96)
            self._canvas.draw_idle()
            return

        try:
            ax = self._fig.add_subplot(111, projection=smap)
            self._current_map = smap
            self._current_axes = ax
            self._emit_observer_info()
            ax.set_facecolor("black")
            try:
                ax.set_box_aspect(1)
            except Exception:
                pass
            try:
                ax.set_aspect("equal", adjustable="box")
            except Exception:
                pass
            if preserve_current_view and prev_xlim is not None and prev_ylim is not None:
                restore_after_draw = (prev_xlim, prev_ylim)
            else:
                restore_after_draw = None
            try:
                if self._uses_geometry_scaffold_for_context():
                    # Geometry-only canvas: keep extents without showing refmap pixels.
                    smap.plot(axes=ax, annotate=False, alpha=0.0)
                else:
                    smap.plot(axes=ax, annotate=False)
            except TypeError:
                smap.plot(axes=ax)
            try:
                context_xlim = ax.get_xlim()
                context_ylim = ax.get_ylim()
            except Exception:
                context_xlim = context_ylim = None
            try:
                ax.set_title("")
            except Exception:
                pass
            context_key = None
            bottom_key = None
            if self._state is not None:
                context_key = self._canonical_map_key(self._state.selected_context_id, purpose="context")
                bottom_key = self._canonical_map_key(self._state.selected_bottom_id, purpose="bottom")
            if overlay_map is not None and self._should_plot_bottom_overlay(context_key, bottom_key):
                try:
                    overlay_map.plot(axes=ax, autoalign=True, alpha=0.95, zorder=5)
                except Exception:
                    pass
            # These often improve readability and mimic the legacy gxbox style.
            try:
                smap.draw_grid(axes=ax, color="w", lw=0.5, annotate=False)
            except Exception:
                pass
            try:
                smap.draw_limb(axes=ax, color="w", lw=0.8)
            except Exception:
                pass
            self._plot_box_outline(ax, smap)
            try:
                ax.set_title("")
            except Exception:
                pass
            time_banner = self._format_display_time_banner(
                self._display_obstime_anchor(),
                self._state.session_input.time_iso,
            )
            if not time_banner:
                time_banner = str(getattr(smap, "date", "") or "")
            title = (
                f"{self._display_map_label(self._state.selected_context_id, bottom=False)} | "
                f"{self._display_map_label(self._state.selected_bottom_id, bottom=True)} | "
                f"{self._observer_label_for_key(self._state.display_observer_key)} | {time_banner}"
            )
            self._fig.text(0.02, 0.992, title, ha="left", va="top", fontsize=10)
            if context_xlim is not None and context_ylim is not None:
                self._full_view_limits = (context_xlim, context_ylim)
            else:
                self._full_view_limits = (ax.get_xlim(), ax.get_ylim())
            if restore_after_draw is not None:
                prev_xlim, prev_ylim = restore_after_draw
                self._restore_preserved_view(prev_xlim, prev_ylim)
            elif self._view_mode == "box_fov":
                self._set_view_to_projected_fov(pad_factor=1.10)
            elif self._view_mode == "full_sun":
                self._set_view_to_full_sun_disk()
                try:
                    self._full_view_limits = (ax.get_xlim(), ax.get_ylim())
                except Exception:
                    pass
        except Exception as exc:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot failed:\n{exc}", ha="center", va="center")
            ax.axis("off")

        self._fig.subplots_adjust(left=0.12, right=0.985, bottom=0.12, top=0.93)
        adjusted = self._auto_adjust_axes_margins(ax, top=0.93, pad_px=10.0)
        self._render_fieldlines()
        if adjusted:
            self._canvas.draw()
        else:
            self._canvas.draw_idle()
        self._pending_launch_margin_fix = True
        self._update_cursor_for_mode()

    def open_live_3d_viewer(self) -> None:
        self._check_viewer3d_state()
        if self._viewer3d is not None:
            try:
                self._refresh_live_3d_viewer_state()
                self._viewer3d.show()
                if hasattr(self._viewer3d, "app_window"):
                    self._viewer3d.app_window.show()
                    self._viewer3d.app_window.showNormal()
                    if hasattr(self._viewer3d, "ensure_window_visible"):
                        self._viewer3d.ensure_window_visible()
                    self._viewer3d.app_window.raise_()
                    self._viewer3d.app_window.activateWindow()
                if hasattr(self._viewer3d, "schedule_startup_los_view"):
                    self._viewer3d.schedule_startup_los_view()
                return
            except Exception:
                self._viewer3d = None
                self._viewer3d_temp_h5_path = None
                self._refresh_open_3d_state()
                self._emit_action_state()
        if self._entry_box_path is None:
            self._set_runtime_status("3D viewer unavailable: no entry box is attached to this selector.")
            return
        try:
            self._ensure_session_model_loaded()
            box, obs_time, b3dtype = self._clone_session_model()
            if box is None:
                raise RuntimeError("No in-memory session model is available for the embedded 3D viewer.")
            self._apply_live_session_state(box)
            box_norm_direction, box_view_up = _viewer_camera_basis(box, obs_time)
            self._fieldline_frame_hcc = getattr(getattr(box, "_center", None), "frame", None)
            self._fieldline_frame_obs = getattr(box, "_frame_obs", None)
            self._viewer3d_close_handled = False
            self._viewer3d = _magfield_viewer_cls()(
                box,
                time=obs_time,
                b3dtype=b3dtype,
                parent=self,
                box_norm_direction=box_norm_direction,
                box_view_up=box_view_up,
                session_mode="embedded",
                source_model_path=self._entry_box_path,
            )
            self._viewer3d_temp_h5_path = self._session_temp_h5_path
            if hasattr(self._viewer3d, "app_window"):
                self._viewer3d.app_window.setWindowTitle(f"GxBox 3D viewer - {self._entry_box_path.name}")
                self._viewer3d.app_window.destroyed.connect(self._on_viewer3d_closed)
            else:
                self._viewer3d.destroyed.connect(self._on_viewer3d_closed)
            self._refresh_open_3d_state()
            self._emit_action_state()
            self._viewer3d_watchdog.start()
            host = self.window()
            host.hide()
            self._hidden_for_live_3d = True
            self._viewer3d.show()
            if hasattr(self._viewer3d, "app_window"):
                self._viewer3d.app_window.show()
                self._viewer3d.app_window.showNormal()
                if hasattr(self._viewer3d, "ensure_window_visible"):
                    self._viewer3d.ensure_window_visible()
                self._viewer3d.app_window.raise_()
                self._viewer3d.app_window.activateWindow()
            if hasattr(self._viewer3d, "schedule_startup_los_view"):
                self._viewer3d.schedule_startup_los_view()
            self._set_runtime_status(f"Opened live 3D viewer for: {self._entry_box_path}")
        except Exception as exc:
            self._viewer3d = None
            self._viewer3d_temp_h5_path = None
            self._refresh_open_3d_state()
            self._emit_action_state()
            self._set_runtime_status(f"3D viewer launch failed: {exc}")

    def clear_fieldlines(self) -> None:
        self._fieldline_streamlines = []
        self._fieldline_z_base = 0.0
        while self._fieldline_artists:
            artist = self._fieldline_artists.pop()
            try:
                artist.remove()
            except Exception:
                pass
        self._can_clear_lines = False
        self._emit_action_state()
        self._canvas.draw_idle()

    @staticmethod
    def _should_plot_bottom_overlay(context_key: str | None, bottom_key: str | None) -> bool:
        context_key = str(context_key or "")
        bottom_key = str(bottom_key or "")
        if not context_key or not bottom_key:
            return False
        return context_key != bottom_key

    def plot_fieldlines(self, streamlines, z_base=0.0) -> None:
        self._fieldline_streamlines = list(streamlines or [])
        self._fieldline_z_base = float(z_base)
        rendered = self._render_fieldlines()
        self._can_clear_lines = bool(self._fieldline_streamlines)
        self._emit_action_state()
        self._canvas.draw_idle()
        if self._fieldline_streamlines:
            if rendered > 0:
                self._set_runtime_status(
                    f"Received {len(self._fieldline_streamlines)} field-line bundle(s) from 3D viewer; "
                    f"rendered {rendered} line(s)."
                )
            else:
                self._set_runtime_status(
                    f"Received {len(self._fieldline_streamlines)} field-line bundle(s) from 3D viewer, "
                    "but no line segments projected into the current 2D view."
                )

    def _render_fieldlines(self) -> int:
        while self._fieldline_artists:
            artist = self._fieldline_artists.pop()
            try:
                artist.remove()
            except Exception:
                pass
        if not self._fieldline_streamlines or self._current_axes is None or self._current_map is None:
            return 0
        rendered_count = 0
        try:
            frame_hcc = self._fieldline_frame_hcc
            current_observer = getattr(self._current_map, "observer_coordinate", None)
            current_obstime = getattr(self._current_map, "date", None)
            if frame_hcc is None or current_observer is None:
                self._set_runtime_status(
                    "Field-line overlay unavailable: no legacy-equivalent 3D viewer frames are attached."
                )
                return 0
            frame_obs = Helioprojective(observer=current_observer, obstime=current_obstime)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "selector_fieldlines",
                ["#4c9aff", "#f6c945", "#e85d3f"],
                N=256,
            )
            norm = mcolors.Normalize(vmin=0.0, vmax=1000.0)
            for streamlines_subset in self._fieldline_streamlines:
                for coord, field in self._extract_streamlines(streamlines_subset):
                    # Mirror the legacy GxBox field-line overlay behavior:
                    # convert streamline coords from HCC to observer HPC, project to
                    # map pixels, then render pixel-space LineCollection segments.
                    coord_world = local_cartesian_to_world(
                        coord,
                        frame=frame_hcc,
                        z_base_mm=self._fieldline_z_base,
                    )
                    coord_hpc = project_world_to_observer_hpc(
                        coord_world,
                        observer=current_observer,
                        obstime=current_obstime,
                    )
                    projected = project_world_to_pixel(coord_hpc, self._current_map)
                    if projected is None:
                        continue
                    x, y = projected
                    magnitude = np.asarray(field["magnitude"], dtype=float)
                    if x.size < 2 or y.size < 2 or magnitude.size < 2:
                        continue
                    finite = np.isfinite(x) & np.isfinite(y)
                    if np.count_nonzero(finite) < 2:
                        continue
                    segments = []
                    colors = []
                    for i in range(len(x) - 1):
                        if not (finite[i] and finite[i + 1]):
                            continue
                        segments.append(((x[i], y[i]), (x[i + 1], y[i + 1])))
                        color_idx = min(i, magnitude.size - 1)
                        colors.append(cmap(norm(magnitude[color_idx])))
                    if not segments:
                        continue
                    lc = LineCollection(segments, colors=colors, linewidths=0.7, alpha=0.7)
                    lc.set_zorder(20)
                    self._current_axes.add_collection(lc)
                    self._fieldline_artists.append(lc)
                    rendered_count += 1
        except Exception as exc:
            self._set_runtime_status(f"Field-line overlay failed: {exc}")
            return 0
        return rendered_count

    @staticmethod
    def _extract_streamlines(streamlines) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
        out = []
        lines_arr = np.asarray(streamlines.lines)
        points = np.asarray(streamlines.points)
        i = 0
        n_lines = int(lines_arr.shape[0])
        while i < n_lines:
            num_points = int(lines_arr[i])
            start_idx = int(lines_arr[i + 1])
            end_idx = start_idx + num_points
            coord = points[start_idx:end_idx]
            bx = np.asarray(streamlines["bx"][start_idx:end_idx])
            by = np.asarray(streamlines["by"][start_idx:end_idx])
            bz = np.asarray(streamlines["bz"][start_idx:end_idx])
            out.append(
                (
                    coord,
                    {
                        "bx": bx,
                        "by": by,
                        "bz": bz,
                        "magnitude": np.sqrt(bx ** 2 + by ** 2 + bz ** 2),
                    },
                )
            )
            i += num_points + 1
        return out

    def _set_runtime_status(self, message: str) -> None:
        self._last_status_base_text = message
        self._emit_status_text()

    @staticmethod
    def _padded_fov_selection(
        fov: DisplayFovSelection,
        pad_factor: float,
    ) -> DisplayFovSelection:
        return DisplayFovSelection(
            center_x_arcsec=float(fov.center_x_arcsec),
            center_y_arcsec=float(fov.center_y_arcsec),
            width_arcsec=max(float(fov.width_arcsec) * float(pad_factor), 1e-3),
            height_arcsec=max(float(fov.height_arcsec) * float(pad_factor), 1e-3),
        )

    @staticmethod
    def _fov_contains(
        outer: DisplayFovSelection | None,
        inner: DisplayFovSelection | None,
        *,
        margin_arcsec: float = 2.0,
    ) -> bool:
        if outer is None or inner is None:
            return False
        outer_half_w = 0.5 * float(outer.width_arcsec)
        outer_half_h = 0.5 * float(outer.height_arcsec)
        inner_half_w = 0.5 * float(inner.width_arcsec)
        inner_half_h = 0.5 * float(inner.height_arcsec)
        return (
            float(inner.center_x_arcsec) - inner_half_w >= float(outer.center_x_arcsec) - outer_half_w + margin_arcsec
            and float(inner.center_x_arcsec) + inner_half_w <= float(outer.center_x_arcsec) + outer_half_w - margin_arcsec
            and float(inner.center_y_arcsec) - inner_half_h >= float(outer.center_y_arcsec) - outer_half_h + margin_arcsec
            and float(inner.center_y_arcsec) + inner_half_h <= float(outer.center_y_arcsec) + outer_half_h - margin_arcsec
        )

    def _current_display_prepare_fov(
        self,
        observer_key: str,
        *,
        obstime=None,
        pad_factor: float = 1.6,
    ) -> DisplayFovSelection | None:
        if self._state is None or self._view_mode != "box_fov" or self._state.fov is None:
            return None
        compare_time = obstime if obstime is not None else self._state.session_input.time_iso
        if not self._observers_share_los(self._state.fov_definition_observer_key, observer_key, compare_time):
            return None
        return self._padded_fov_selection(self._state.fov, pad_factor)

    def _project_fov_between_observers(
        self,
        fov: DisplayFovSelection,
        from_observer_key: str,
        to_observer_key: str,
        obstime,
    ) -> DisplayFovSelection | None:
        if self._state is None:
            return None
        from_key = self._normalize_observer_key(from_observer_key)
        to_key = self._normalize_observer_key(to_observer_key)
        compare_time = obstime if obstime is not None else self._state.session_input.time_iso
        if self._observers_share_los(from_key, to_key, compare_time):
            return fov
        target_context = self._observer_context(to_key, compare_time)
        target_observer = getattr(target_context, "observer_coordinate", None)
        if target_observer is None:
            return None
        target_frame = Helioprojective(observer=target_observer, obstime=compare_time)
        source_context = self._observer_context(from_key, compare_time)
        source_observer = getattr(source_context, "observer_coordinate", None) or "earth"
        source_obstime = getattr(source_context, "date", None) or compare_time
        base_corners = observer_rectangle_to_hpc_corners(
            xc_arcsec=float(fov.center_x_arcsec),
            yc_arcsec=float(fov.center_y_arcsec),
            xsize_arcsec=float(fov.width_arcsec),
            ysize_arcsec=float(fov.height_arcsec),
            observer=source_observer,
            obstime=source_obstime,
        )
        if base_corners is None:
            return None
        try:
            projected = base_corners.transform_to(target_frame)
            tx = np.asarray(projected.Tx.to_value(u.arcsec), dtype=float).ravel()
            ty = np.asarray(projected.Ty.to_value(u.arcsec), dtype=float).ravel()
            finite = np.isfinite(tx) & np.isfinite(ty)
            if not np.any(finite):
                return None
            tx = tx[finite]
            ty = ty[finite]
            xmin, xmax = float(np.min(tx)), float(np.max(tx))
            ymin, ymax = float(np.min(ty)), float(np.max(ty))
            return DisplayFovSelection(
                center_x_arcsec=0.5 * (xmin + xmax),
                center_y_arcsec=0.5 * (ymin + ymax),
                width_arcsec=max(xmax - xmin, 4.0),
                height_arcsec=max(ymax - ymin, 4.0),
            )
        except Exception:
            return None

    def _fov_selection_projected_to_display_observer(
        self,
        fov: DisplayFovSelection,
        obstime,
    ) -> DisplayFovSelection | None:
        """Express a FOV rectangle from ``fov_definition_observer_key`` in display HPC."""
        if self._state is None:
            return None
        return self._project_fov_between_observers(
            fov,
            self._state.fov_definition_observer_key,
            self._state.display_observer_key,
            obstime,
        )

    def _display_observer_reproject_header_for_fov(self, smap, observer, obstime, pad_factor: float = 1.10):
        if self._state is None or self._state.fov is None:
            return None
        display_key = self._normalize_observer_key(self._state.display_observer_key)
        fov_key = self._normalize_observer_key(self._state.fov_definition_observer_key)
        if not self._observers_share_los(display_key, fov_key, obstime):
            return None
        fov = self._padded_fov_selection(self._state.fov, pad_factor)
        return self._display_observer_reproject_header_for_selection(smap, observer, obstime, fov)

    def _display_observer_reproject_header_for_selection(self, smap, observer, obstime, fov: DisplayFovSelection | None):
        if fov is None:
            return None
        try:
            scale_x = abs(float(smap.scale.axis1.to_value(u.arcsec / u.pix)))
            scale_y = abs(float(smap.scale.axis2.to_value(u.arcsec / u.pix)))
        except Exception:
            return None
        if not (np.isfinite(scale_x) and np.isfinite(scale_y) and scale_x > 0 and scale_y > 0):
            return None
        width = max(float(fov.width_arcsec), 4.0)
        height = max(float(fov.height_arcsec), 4.0)
        nx = max(32, int(np.ceil(width / scale_x)))
        ny = max(32, int(np.ceil(height / scale_y)))
        try:
            target_center = SkyCoord(
                Tx=float(fov.center_x_arcsec) * u.arcsec,
                Ty=float(fov.center_y_arcsec) * u.arcsec,
                frame=Helioprojective(observer=observer, obstime=obstime),
            )
            header = make_fitswcs_header(
                np.empty((ny, nx), dtype=np.float32),
                target_center,
                scale=u.Quantity([scale_x, scale_y], u.arcsec / u.pix),
            )
            try:
                header["rsun_ref"] = float(smap.rsun_meters.to_value(u.m))
            except Exception:
                pass
            return header
        except Exception:
            return None

    def save_current_plot(self, output_path: str) -> None:
        self._fig.savefig(output_path, dpi=150, bbox_inches="tight")

    def _restore_preserved_view(self, prev_xlim, prev_ylim) -> None:
        if self._current_axes is None:
            return
        prev_width = abs(float(prev_xlim[1] - prev_xlim[0]))
        prev_height = abs(float(prev_ylim[1] - prev_ylim[0]))
        if prev_width <= 0 or prev_height <= 0:
            return
        if self._state is not None and self._state.fov is not None and self._current_map is not None:
            try:
                observer = getattr(self._current_map, "observer_coordinate", None) or "earth"
                obstime = getattr(self._current_map, "date", None)
                fov = self._state.fov
                center_world = SkyCoord(
                    Tx=float(fov.center_x_arcsec) * u.arcsec,
                    Ty=float(fov.center_y_arcsec) * u.arcsec,
                    frame=Helioprojective(observer=observer, obstime=obstime),
                )
                cpx, cpy = self._current_map.wcs.world_to_pixel(center_world)
                if np.isfinite(cpx) and np.isfinite(cpy):
                    self._set_view_window(float(cpx), float(cpy), prev_width, prev_height)
                    return
            except Exception:
                pass
        self._current_axes.set_xlim(prev_xlim)
        self._current_axes.set_ylim(prev_ylim)

    @staticmethod
    def _default_context_id(session_input: SelectorSessionInput) -> Optional[str]:
        context_ids = MapBoxDisplayWidget._available_context_map_ids(session_input)
        if not context_ids:
            return None
        preferred = [
            "171", "193", "211", "304", "335", "1600",
            "Bz", "Ic", "B_rho", "B_theta", "B_phi", "disambig",
            # Backward-compatible legacy labels.
            "Br", "Bp", "Bt",
        ]
        for key in preferred:
            if key in context_ids:
                return key
        return context_ids[0]

    @staticmethod
    def _default_bottom_id(session_input: SelectorSessionInput) -> Optional[str]:
        base_maps = dict(session_input.base_maps or {})
        if "bz" in base_maps:
            return "Bz"
        if "ic" in base_maps:
            return "Ic"
        if "bx" in base_maps:
            return "Bx"
        if "by" in base_maps:
            return "By"
        if "vert_current" in base_maps:
            return "Vert_current"
        if "chromo_mask" in base_maps:
            return "chromo_mask"
        return None

    def _edge_pixel_bounds(self, smap, edges) -> tuple[float, float, float, float] | None:
        xs: list[float] = []
        ys: list[float] = []
        for edge in edges:
            try:
                px, py = smap.wcs.world_to_pixel(edge)
            except Exception:
                continue
            px = np.asarray(px, dtype=float).ravel()
            py = np.asarray(py, dtype=float).ravel()
            finite = np.isfinite(px) & np.isfinite(py)
            if np.any(finite):
                xs.extend(px[finite].tolist())
                ys.extend(py[finite].tolist())
        if not xs or not ys:
            return None
        return (
            float(np.nanmin(xs)),
            float(np.nanmax(xs)),
            float(np.nanmin(ys)),
            float(np.nanmax(ys)),
        )

    def _plot_box_outline(self, ax, smap) -> None:
        if self._state is None or self._state.geometry is None:
            return
        try:
            box = self._build_legacy_box(
                smap,
                geometry_observer_key=self._state.geometry_definition_observer_key,
            )
            if box is None:
                return
            self._overlay_line_artists = []

            for edge in box.bottom_edges:
                self._overlay_line_artists.extend(
                    ax.plot_coord(edge, color="tab:red", ls="--", marker="", lw=1.0, zorder=20)
                )
            for edge in box.non_bottom_edges:
                self._overlay_line_artists.extend(
                    ax.plot_coord(edge, color="tab:red", ls="-", marker="", lw=1.0, zorder=20)
                )

            full_bounds = self._edge_pixel_bounds(smap, list(box.bottom_edges) + list(box.non_bottom_edges))
            bottom_bounds = self._edge_pixel_bounds(smap, list(box.bottom_edges))
            if full_bounds is None or bottom_bounds is None:
                return
            x0, x1, y0, y1 = full_bounds
            bx0, bx1, by0, by1 = bottom_bounds
            self._projected_box_fov = self._box_bounds_to_fov_selection(box, smap)
            self._projected_box_bbox_rect = self._fov_selection_to_pixel_rect(smap, self._projected_box_fov)
            if self._state.fov is None:
                self._state.fov = DisplayFovSelection(
                    center_x_arcsec=self._projected_box_fov.center_x_arcsec,
                    center_y_arcsec=self._projected_box_fov.center_y_arcsec,
                    width_arcsec=self._projected_box_fov.width_arcsec,
                    height_arcsec=self._projected_box_fov.height_arcsec,
                )
                self._state.fov_definition_observer_key = self._normalize_observer_key(
                    self._state.display_observer_key
                )
                if self._fov_change_callback is not None:
                    self._fov_change_callback(self._state.fov)
            if self._state.fov_box is None:
                self._state.fov_box = self._compute_fov_box_from_geometry()
            fov_rect = self._fov_selection_to_pixel_rect(smap, self._state.fov)
            fx0, fy0 = fov_rect.get_x(), fov_rect.get_y()
            fw, fh = fov_rect.get_width(), fov_rect.get_height()
            projected_edges = self._fov_box_projected_edges(smap)
            for edge in projected_edges:
                try:
                    self._overlay_line_artists.extend(
                        ax.plot_coord(edge, color="deepskyblue", ls="-", marker="", lw=0.9, zorder=21)
                    )
                except Exception:
                    continue
            projected_face = self._fov_box_projected_face(smap)
            if projected_face is not None:
                try:
                    self._overlay_line_artists.extend(
                        ax.plot_coord(projected_face, color="deepskyblue", ls="-", marker="", lw=1.6, zorder=22)
                    )
                except Exception:
                    pass
            projected_bbox = self._edge_pixel_bounds(smap, projected_edges) if projected_edges else None
            if projected_bbox is not None:
                pfx0, pfx1, pfy0, pfy1 = projected_bbox
                fx0, fy0 = pfx0, pfy0
                fw, fh = max(1e-6, pfx1 - pfx0), max(1e-6, pfy1 - pfy0)

            # Invisible rectangles retained as the internal geometry extents for
            # button-driven manipulations and Box-FOV calculations.
            self._overlay_rect = Rectangle(
                (bx0, by0),
                max(1e-6, bx1 - bx0),
                max(1e-6, by1 - by0),
                visible=False,
            )
            self._overlay_bbox_rect = Rectangle(
                (fx0, fy0),
                max(1e-6, fw),
                max(1e-6, fh),
                visible=False,
            )
            self._zoom_anchor_px = (
                float(fx0 + 0.5 * max(1e-6, fw)),
                float(fy0 + 0.5 * max(1e-6, fh)),
            )

            anchor = self._geometry_anchor_coord(self._state.geometry, smap).transform_to(box._frame_obs)
            cpx, cpy = smap.wcs.world_to_pixel(anchor)
            self._overlay_center_artist = ax.plot(
                [float(cpx)], [float(cpy)],
                marker="+", color="yellow", ms=10, mew=1.5,
                transform=ax.get_transform("pixel"),
            )[0]
            self._overlay_line_artists.append(self._overlay_center_artist)
        except Exception:
            # Overlay failure should not break map display.
            return

    def _geometry_center_hpc_for_map(self, geom: BoxGeometrySelection, smap):
        return self._geometry_anchor_coord(geom, smap).transform_to(
            Helioprojective(
                observer=self._resolved_observer_for_map(
                    smap,
                    self._state.geometry_definition_observer_key if self._state is not None else "earth",
                ) or "earth",
                obstime=getattr(smap, "date", None),
            )
        )

    def _box_half_extent_arcsec(self, n_pix: int, dx_km: float, smap) -> float:
        # Approximate small-angle conversion using observer distance.
        dsun_km = None
        try:
            observer = self._resolved_observer_for_map(smap)
            if observer is not None:
                dsun_km = observer.radius.to_value(u.km)
        except Exception:
            try:
                dsun_km = smap.dsun.to_value(u.km)
            except Exception:
                try:
                    dsun_obs = smap.meta.get("dsun_obs")
                    if dsun_obs is not None:
                        dsun_km = (float(dsun_obs) * u.m).to_value(u.km)
                except Exception:
                    dsun_km = None
        if not dsun_km or dsun_km <= 0:
            dsun_km = 1.496e8  # fallback ~1 AU
        half_size_km = 0.5 * float(n_pix) * float(dx_km)
        return half_size_km / dsun_km * 206265.0

    def _geometry_pixel_arcsec(self, geom: BoxGeometrySelection, smap) -> float:
        dsun_km = self._dsun_km_from_map(smap)
        return float(max(geom.dx_km, 1e-6) / max(dsun_km, 1e-6) * 206265.0)

    def _geometry_from_world_center(self, geom: BoxGeometrySelection, world_center) -> BoxGeometrySelection:
        out = BoxGeometrySelection(
            coord_mode=geom.coord_mode,
            coord_x=geom.coord_x,
            coord_y=geom.coord_y,
            grid_x=geom.grid_x,
            grid_y=geom.grid_y,
            grid_z=geom.grid_z,
            dx_km=geom.dx_km,
        )
        try:
            if geom.coord_mode == CoordMode.HPC:
                observer_key = self._state.geometry_definition_observer_key if self._state is not None else "earth"
                source_context = self._observer_context(
                    observer_key,
                    getattr(self._current_map, "date", None),
                )
                c = world_center.transform_to(
                    Helioprojective(
                        obstime=getattr(source_context, "date", None) or getattr(self._current_map, "date", None),
                        observer=getattr(source_context, "observer_coordinate", None)
                        or self._resolved_observer_for_map(self._current_map, observer_key)
                        or "earth",
                    )
                )
                out.coord_x = float(c.Tx.to_value(u.arcsec))
                out.coord_y = float(c.Ty.to_value(u.arcsec))
            elif geom.coord_mode == CoordMode.HGC:
                c = world_center.transform_to(
                    HeliographicCarrington(
                        obstime=getattr(self._current_map, "date", None),
                        observer=self._resolved_observer_for_map(self._current_map, observer_key) or "earth",
                    )
                )
                out.coord_x = float(c.lon.to_value(u.deg))
                out.coord_y = float(c.lat.to_value(u.deg))
            else:
                c = world_center.transform_to(
                    HeliographicStonyhurst(obstime=getattr(self._current_map, "date", None))
                )
                out.coord_x = float(c.lon.to_value(u.deg))
                out.coord_y = float(c.lat.to_value(u.deg))
        except Exception:
            pass
        return out

    def show_full_sun_view(self) -> None:
        self._view_mode = "full_sun"
        self._map_summary_cache.clear()
        self._refresh_map_info()
        self._refresh_plot()

    def show_box_fov_view(self, pad_factor: float | None = None) -> None:
        # `pad_factor` is handled by the display-crop helper; keep the method
        # signature stable for existing button hookups.
        self._view_mode = "box_fov"
        self._map_summary_cache.clear()
        self._refresh_map_info()
        self._refresh_plot()

    def _clamp_view_to_limits(self, limits: tuple) -> None:
        if self._current_axes is None or limits is None:
            return
        try:
            ref_xlim, ref_ylim = limits
            xlim = self._current_axes.get_xlim()
            ylim = self._current_axes.get_ylim()
            ref_xw = abs(float(ref_xlim[1] - ref_xlim[0]))
            ref_yh = abs(float(ref_ylim[1] - ref_ylim[0]))
            cur_xw = abs(float(xlim[1] - xlim[0]))
            cur_yh = abs(float(ylim[1] - ylim[0]))
            if ref_xw <= 0 or ref_yh <= 0:
                return
            if cur_xw <= ref_xw + 1e-6 and cur_yh <= ref_yh + 1e-6:
                return
            cx = 0.5 * (float(xlim[0]) + float(xlim[1]))
            cy = 0.5 * (float(ylim[0]) + float(ylim[1]))
            x_dir = 1.0 if xlim[1] >= xlim[0] else -1.0
            y_dir = 1.0 if ylim[1] >= ylim[0] else -1.0
            half_w = 0.5 * ref_xw
            half_h = 0.5 * ref_yh
            self._current_axes.set_xlim(
                (cx - half_w, cx + half_w) if x_dir > 0 else (cx + half_w, cx - half_w)
            )
            self._current_axes.set_ylim(
                (cy - half_h, cy + half_h) if y_dir > 0 else (cy + half_h, cy - half_h)
            )
            self._canvas.draw_idle()
        except Exception:
            return

    def _set_view_window(self, cx: float, cy: float, width: float, height: float) -> None:
        if self._current_axes is None:
            return
        ax = self._current_axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_dir = 1.0 if xlim[1] >= xlim[0] else -1.0
        y_dir = 1.0 if ylim[1] >= ylim[0] else -1.0
        half_w = 0.5 * max(width, 4.0)
        half_h = 0.5 * max(height, 4.0)
        ax.set_xlim((cx - half_w, cx + half_w) if x_dir > 0 else (cx + half_w, cx - half_w))
        ax.set_ylim((cy - half_h, cy + half_h) if y_dir > 0 else (cy + half_h, cy - half_h))
        self._canvas.draw_idle()

    def _set_view_to_projected_fov(self, pad_factor: float = 1.10) -> None:
        if self._current_axes is None:
            return
        rect = self._overlay_bbox_rect or self._projected_box_bbox_rect
        if rect is None:
            return
        x0 = float(rect.get_x())
        y0 = float(rect.get_y())
        width = float(rect.get_width())
        height = float(rect.get_height())
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(width) and np.isfinite(height)):
            return
        width = max(width * float(pad_factor), 4.0)
        height = max(height * float(pad_factor), 4.0)
        self._set_view_window(
            cx=x0 + 0.5 * float(rect.get_width()),
            cy=y0 + 0.5 * float(rect.get_height()),
            width=width,
            height=height,
        )

    def _set_view_window_hpc(
        self,
        center_x_arcsec: float,
        center_y_arcsec: float,
        width_arcsec: float,
        height_arcsec: float,
    ) -> None:
        if self._current_axes is None or self._current_map is None:
            return
        half_w = 0.5 * max(float(width_arcsec), 1e-3)
        half_h = 0.5 * max(float(height_arcsec), 1e-3)
        observer = self._resolved_observer_for_map(self._current_map, self._state.display_observer_key) or "earth"
        obstime = getattr(self._current_map, "date", None)
        bottom_left = SkyCoord(
            Tx=(center_x_arcsec - half_w) * u.arcsec,
            Ty=(center_y_arcsec - half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        top_right = SkyCoord(
            Tx=(center_x_arcsec + half_w) * u.arcsec,
            Ty=(center_y_arcsec + half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        try:
            corners = SkyCoord([bottom_left, top_right])
            px, py = self._current_map.wcs.world_to_pixel(corners)
            px = np.asarray(px, dtype=float).ravel()
            py = np.asarray(py, dtype=float).ravel()
            finite = np.isfinite(px) & np.isfinite(py)
            if not np.any(finite):
                return
            px = px[finite]
            py = py[finite]
            x0, x1 = float(np.nanmin(px)), float(np.nanmax(px))
            y0, y1 = float(np.nanmin(py)), float(np.nanmax(py))
            self._set_view_window(
                cx=0.5 * (x0 + x1),
                cy=0.5 * (y0 + y1),
                width=max(x1 - x0, 4.0),
                height=max(y1 - y0, 4.0),
            )
        except Exception:
            return

    def _scale_view(self, factor: float, center_px: tuple[float, float] | None = None) -> None:
        if self._current_axes is None:
            return
        ax = self._current_axes
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if self._zoom_anchor_px is not None:
            try:
                cx, cy = self._zoom_anchor_px
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    raise ValueError("non-finite zoom center")
            except Exception:
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
        else:
            rect = self._overlay_bbox_rect or self._projected_box_bbox_rect
            if rect is not None:
                try:
                    cx = float(rect.get_x()) + 0.5 * float(rect.get_width())
                    cy = float(rect.get_y()) + 0.5 * float(rect.get_height())
                    if not (np.isfinite(cx) and np.isfinite(cy)):
                        raise ValueError("non-finite zoom center")
                except Exception:
                    cx = 0.5 * (x0 + x1)
                    cy = 0.5 * (y0 + y1)
            elif center_px is not None:
                cx, cy = center_px
            else:
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
        width = abs(x1 - x0) * float(factor)
        height = abs(y1 - y0) * float(factor)
        self._set_view_window(cx, cy, max(width, 4.0), max(height, 4.0))

    def _nudge_box_size(self, axis: str, sign: int) -> None:
        if not self._geometry_edit_enabled or self._state is None or self._state.geometry is None:
            return
        geom = self._state.geometry
        step = self._coarse_box_grid_step(axis)
        new_geom = BoxGeometrySelection(
            coord_mode=geom.coord_mode,
            coord_x=geom.coord_x,
            coord_y=geom.coord_y,
            grid_x=geom.grid_x,
            grid_y=geom.grid_y,
            grid_z=geom.grid_z,
            dx_km=geom.dx_km,
        )
        if axis == "x":
            new_geom.grid_x = max(1, geom.grid_x + int(sign) * step)
        elif axis == "y":
            new_geom.grid_y = max(1, geom.grid_y + int(sign) * step)
        else:
            return
        self.set_geometry_selection(new_geom)

    def _nudge_box_size_xy(self, sign: int) -> None:
        if not self._geometry_edit_enabled or self._state is None or self._state.geometry is None:
            return
        geom = self._state.geometry
        step_x = self._coarse_box_grid_step("x")
        step_y = self._coarse_box_grid_step("y")
        new_geom = BoxGeometrySelection(
            coord_mode=geom.coord_mode,
            coord_x=geom.coord_x,
            coord_y=geom.coord_y,
            grid_x=max(1, geom.grid_x + int(sign) * step_x),
            grid_y=max(1, geom.grid_y + int(sign) * step_y),
            grid_z=geom.grid_z,
            dx_km=geom.dx_km,
        )
        self.set_geometry_selection(new_geom)

    def _nudge_fov_size(self, axis: str, sign: int) -> None:
        if self._state is None or self._state.fov is None:
            return
        fov = self._state.fov
        step = self._coarse_fov_step(axis)
        width = float(fov.width_arcsec)
        height = float(fov.height_arcsec)
        if axis == "x":
            width = max(step, width + int(sign) * step)
            if self._state.square_fov:
                height = width
        elif axis == "y":
            if self._state.square_fov:
                return
            height = max(step, height + int(sign) * step)
        else:
            return
        self.set_fov_selection(
            DisplayFovSelection(
                center_x_arcsec=fov.center_x_arcsec,
                center_y_arcsec=fov.center_y_arcsec,
                width_arcsec=width,
                height_arcsec=height,
            )
        )

    def _nudge_fov_size_xy(self, sign: int) -> None:
        if self._state is None or self._state.fov is None:
            return
        fov = self._state.fov
        step_x = self._coarse_fov_step("x")
        step_y = self._coarse_fov_step("y")
        width = max(step_x, float(fov.width_arcsec) + int(sign) * step_x)
        height = max(step_y, float(fov.height_arcsec) + int(sign) * step_y)
        if self._state.square_fov:
            height = width
        self.set_fov_selection(
            DisplayFovSelection(
                center_x_arcsec=fov.center_x_arcsec,
                center_y_arcsec=fov.center_y_arcsec,
                width_arcsec=width,
                height_arcsec=height,
            )
        )

    def _nudge_fov_center(self, axis: str, sign: int) -> None:
        if self._state is None or self._state.fov is None:
            return
        fov = self._state.fov
        step = self._coarse_fov_step(axis)
        cx = float(fov.center_x_arcsec)
        cy = float(fov.center_y_arcsec)
        if axis == "x":
            cx += float(sign) * step
        elif axis == "y":
            cy += float(sign) * step
        else:
            return
        self.set_fov_selection(
            DisplayFovSelection(
                center_x_arcsec=cx,
                center_y_arcsec=cy,
                width_arcsec=fov.width_arcsec,
                height_arcsec=fov.height_arcsec,
            )
        )

    def _nudge_box_center(self, axis: str, sign: int) -> None:
        if not self._geometry_edit_enabled or self._state is None or self._state.geometry is None or self._current_map is None:
            return
        geom = self._state.geometry
        step_arcsec = self._coarse_box_center_step_arcsec(axis)
        if not np.isfinite(step_arcsec) or step_arcsec <= 0:
            return
        center_hpc = self._geometry_center_hpc_for_map(geom, self._current_map)
        tx = float(center_hpc.Tx.to_value(u.arcsec))
        ty = float(center_hpc.Ty.to_value(u.arcsec))
        if axis == "x":
            tx += float(sign) * step_arcsec
        elif axis == "y":
            ty += float(sign) * step_arcsec
        else:
            return
        nudged_center = SkyCoord(
            Tx=tx * u.arcsec,
            Ty=ty * u.arcsec,
            frame=Helioprojective(
                observer=getattr(self._current_map, "observer_coordinate", None) or "earth",
                obstime=getattr(self._current_map, "date", None),
            ),
        )
        new_geom = self._geometry_from_world_center(geom, nudged_center)
        self.set_geometry_selection(new_geom)

    def _coarse_box_grid_step(self, axis: str) -> int:
        if self._state is None or self._state.geometry is None:
            return 1
        geom = self._state.geometry
        dim = geom.grid_x if axis == "x" else geom.grid_y if axis == "y" else geom.grid_z
        return max(1, int(round(float(dim) * 0.10)))

    def _coarse_fov_step(self, axis: str) -> float:
        if self._state is None or self._state.fov is None:
            return 1.0
        fov = self._state.fov
        span = float(fov.width_arcsec) if axis == "x" else float(fov.height_arcsec)
        base_step = 1.0
        if self._state.geometry is not None and self._current_map is not None:
            try:
                base_step = max(1e-3, self._geometry_pixel_arcsec(self._state.geometry, self._current_map))
            except Exception:
                base_step = 1.0
        return max(base_step, abs(span) * 0.10)

    def _coarse_box_center_step_arcsec(self, axis: str) -> float:
        if self._state is None or self._state.geometry is None or self._current_map is None:
            return 0.0
        geom = self._state.geometry
        step_arcsec = self._geometry_pixel_arcsec(geom, self._current_map)
        if not np.isfinite(step_arcsec) or step_arcsec <= 0:
            return 0.0
        grid_step = self._coarse_box_grid_step(axis)
        return float(step_arcsec) * float(grid_step)

    def _compute_fov_box_from_geometry(self) -> Optional[DisplayFovBoxSelection]:
        if self._state is None or self._state.fov is None or self._current_map is None:
            return None
        obstime = getattr(self._current_map, "date", None)
        geometry_observer_key = self._state.geometry_definition_observer_key
        source_map = self._observer_context(geometry_observer_key, obstime) or self._current_map
        box = self._build_legacy_box(
            source_map,
            geometry_observer_key=geometry_observer_key,
        )
        if box is None:
            return None
        observer = self._resolved_observer_for_map(self._current_map, self._state.display_observer_key) or "earth"
        try:
            world = box.model_box_corners_world()
            if world is None:
                return None
            fov_box = build_fov_box_from_red_box_world(world, observer=observer, obstime=obstime)
            if fov_box is None:
                return None
            return DisplayFovBoxSelection(
                center_x_arcsec=float(fov_box["xc_arcsec"]),
                center_y_arcsec=float(fov_box["yc_arcsec"]),
                width_arcsec=float(fov_box["xsize_arcsec"]),
                height_arcsec=float(fov_box["ysize_arcsec"]),
                z_min_mm=float(fov_box["zmin_mm"]),
                z_max_mm=float(fov_box["zmax_mm"]),
                observer_key=self._normalize_observer_key(self._state.display_observer_key),
            )
        except Exception:
            return None

    def _fov_box_world_corners(self, smap, fov_box: DisplayFovBoxSelection) -> SkyCoord | None:
        if self._state is None:
            return None
        obstime = getattr(smap, "date", None)
        source_observer_key = self._normalize_observer_key(getattr(fov_box, "observer_key", None))
        source_context = self._observer_context(source_observer_key, obstime)
        source_map = source_context or smap
        box = self._build_legacy_box(
            source_map,
            geometry_observer_key=self._state.geometry_definition_observer_key,
        )
        if box is None:
            return None
        box_frame = getattr(getattr(box, "_center", None), "frame", None)
        if box_frame is None:
            return None
        source_observer = self._resolved_observer_for_map(source_map, source_observer_key) or "earth"
        meta = fov_box.as_observer_metadata(square=bool(self._state.square_fov))
        try:
            return observer_fov_box_to_world_corners(
                xc_arcsec=float(meta["xc_arcsec"]),
                yc_arcsec=float(meta["yc_arcsec"]),
                xsize_arcsec=float(meta["xsize_arcsec"]),
                ysize_arcsec=float(meta["ysize_arcsec"]),
                zmin_mm=float(meta["zmin_mm"]),
                zmax_mm=float(meta["zmax_mm"]),
                observer=source_observer,
                obstime=getattr(source_map, "date", None),
                target_frame=box_frame,
            )
        except Exception:
            return None

    def recompute_fov_from_box(self) -> None:
        if self._state is None or self._projected_box_fov is None:
            return
        self._state.fov_definition_observer_key = self._normalize_observer_key(self._state.display_observer_key)
        self._state.fov_box = self._compute_fov_box_from_geometry()
        if self._state.fov_box is not None:
            width = float(self._state.fov_box.width_arcsec)
            height = float(self._state.fov_box.height_arcsec)
            if self._state.square_fov:
                side = max(width, height)
                width = side
                height = side
            selection = DisplayFovSelection(
                center_x_arcsec=float(self._state.fov_box.center_x_arcsec),
                center_y_arcsec=float(self._state.fov_box.center_y_arcsec),
                width_arcsec=width,
                height_arcsec=height,
            )
        else:
            width = self._projected_box_fov.width_arcsec
            height = self._projected_box_fov.height_arcsec
            if self._state.square_fov:
                height = width
            selection = DisplayFovSelection(
                center_x_arcsec=self._projected_box_fov.center_x_arcsec,
                center_y_arcsec=self._projected_box_fov.center_y_arcsec,
                width_arcsec=width,
                height_arcsec=height,
            )
        self.set_fov_selection(
            selection
        )
        self._refresh_status_text()

    def _on_scroll(self, event) -> None:
        if self._current_axes is None or event.inaxes is not self._current_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        if getattr(event, "button", None) == "up":
            self._scale_view(1 / 1.12, center_px=(event.xdata, event.ydata))
        elif getattr(event, "button", None) == "down":
            self._scale_view(1.12, center_px=(event.xdata, event.ydata))

    def _overlay_rect_bounds(self):
        if self._overlay_rect is None:
            return None
        x0, y0 = self._overlay_rect.get_x(), self._overlay_rect.get_y()
        w, h = self._overlay_rect.get_width(), self._overlay_rect.get_height()
        return x0, y0, x0 + w, y0 + h

    def _set_static_overlay_visible(self, visible: bool) -> None:
        for artist in self._overlay_line_artists:
            try:
                artist.set_visible(bool(visible))
            except Exception:
                continue

    def _clear_drag_preview_artists(self) -> None:
        for artist_name in (
            "_drag_preview_box_artist",
            "_drag_preview_fov_artist",
            "_drag_preview_center_artist",
        ):
            artist = getattr(self, artist_name, None)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, artist_name, None)
        self._drag_preview_background = None
        self._drag_preview_active = False

    def _ensure_drag_preview(self) -> bool:
        if self._current_axes is None or self._current_map is None:
            return False
        if self._drag_preview_active:
            return True
        self._set_static_overlay_visible(False)
        self._canvas.draw()
        try:
            self._drag_preview_background = self._canvas.copy_from_bbox(self._current_axes.bbox)
        except Exception:
            self._set_static_overlay_visible(True)
            self._canvas.draw_idle()
            return False
        pixel_transform = self._current_axes.get_transform("pixel")
        self._drag_preview_box_artist = Rectangle(
            (0.0, 0.0), 1.0, 1.0,
            fill=False, ec="tab:red", ls="--", lw=1.2,
            transform=pixel_transform, animated=True, visible=True,
        )
        self._drag_preview_fov_artist = Rectangle(
            (0.0, 0.0), 1.0, 1.0,
            fill=False, ec="deepskyblue", ls="-", lw=0.9,
            transform=pixel_transform, animated=True, visible=True,
        )
        self._drag_preview_center_artist = self._current_axes.plot(
            [0.0], [0.0],
            marker="+", color="yellow", ms=10, mew=1.5,
            transform=pixel_transform,
            animated=True,
        )[0]
        self._current_axes.add_patch(self._drag_preview_box_artist)
        self._current_axes.add_patch(self._drag_preview_fov_artist)
        self._drag_preview_active = True
        return True

    def _update_drag_preview(self, box_rect: Rectangle, center_px: tuple[float, float]) -> None:
        if not self._ensure_drag_preview():
            return
        fov_rect = self._overlay_bbox_rect
        try:
            self._drag_preview_box_artist.set_bounds(
                float(box_rect.get_x()),
                float(box_rect.get_y()),
                float(box_rect.get_width()),
                float(box_rect.get_height()),
            )
            if fov_rect is not None:
                self._drag_preview_fov_artist.set_bounds(
                    float(fov_rect.get_x()),
                    float(fov_rect.get_y()),
                    float(fov_rect.get_width()),
                    float(fov_rect.get_height()),
                )
                self._drag_preview_fov_artist.set_visible(True)
            else:
                self._drag_preview_fov_artist.set_visible(False)
            self._drag_preview_center_artist.set_data([float(center_px[0])], [float(center_px[1])])
            self._canvas.restore_region(self._drag_preview_background)
            self._current_axes.draw_artist(self._drag_preview_box_artist)
            if self._drag_preview_fov_artist.get_visible():
                self._current_axes.draw_artist(self._drag_preview_fov_artist)
            self._current_axes.draw_artist(self._drag_preview_center_artist)
            self._canvas.blit(self._current_axes.bbox)
        except Exception:
            self._end_drag_preview(restore_static=True)

    def _end_drag_preview(self, *, restore_static: bool) -> None:
        self._clear_drag_preview_artists()
        if restore_static:
            self._set_static_overlay_visible(True)
            self._canvas.draw_idle()

    def _geometry_preview_overlay(self, geom: BoxGeometrySelection) -> tuple[Rectangle, tuple[float, float]] | None:
        if self._current_map is None or self._state is None:
            return None
        box = self._build_legacy_box(
            self._current_map,
            geom=geom,
            geometry_observer_key=self._state.geometry_definition_observer_key,
        )
        if box is None:
            return None
        bottom_bounds = self._edge_pixel_bounds(self._current_map, list(box.bottom_edges))
        if bottom_bounds is None:
            return None
        bx0, bx1, by0, by1 = bottom_bounds
        anchor = self._geometry_anchor_coord(geom, self._current_map).transform_to(box._frame_obs)
        cpx, cpy = self._current_map.wcs.world_to_pixel(anchor)
        return (
            Rectangle(
                (bx0, by0),
                max(1e-6, bx1 - bx0),
                max(1e-6, by1 - by0),
                visible=False,
            ),
            (float(cpx), float(cpy)),
        )

    def _fov_selection_to_pixel_rect(
        self,
        smap,
        fov: DisplayFovSelection | None,
        *,
        use_display_observer: bool = False,
    ) -> Rectangle | None:
        if fov is None:
            if self._projected_box_bbox_rect is not None:
                return Rectangle(
                    (self._projected_box_bbox_rect.get_x(), self._projected_box_bbox_rect.get_y()),
                    self._projected_box_bbox_rect.get_width(),
                    self._projected_box_bbox_rect.get_height(),
                    visible=False,
                )
            return Rectangle((0.0, 0.0), 10.0, 10.0, visible=False)
        if use_display_observer and self._state is not None:
            observer_key = self._state.display_observer_key
        else:
            observer_key = self._state.fov_definition_observer_key if self._state is not None else "earth"
        source_context = self._observer_context(observer_key, getattr(smap, "date", None))
        observer = getattr(source_context, "observer_coordinate", None) or "earth"
        obstime = getattr(source_context, "date", None) or getattr(smap, "date", None)
        half_w = 0.5 * max(float(fov.width_arcsec), 1e-3)
        half_h = 0.5 * max(float(fov.height_arcsec), 1e-3)
        bottom_left = SkyCoord(
            Tx=(fov.center_x_arcsec - half_w) * u.arcsec,
            Ty=(fov.center_y_arcsec - half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        top_right = SkyCoord(
            Tx=(fov.center_x_arcsec + half_w) * u.arcsec,
            Ty=(fov.center_y_arcsec + half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        try:
            px, py = smap.wcs.world_to_pixel(SkyCoord([bottom_left, top_right]))
            px = np.asarray(px, dtype=float).ravel()
            py = np.asarray(py, dtype=float).ravel()
            finite = np.isfinite(px) & np.isfinite(py)
            if np.any(finite):
                px = px[finite]
                py = py[finite]
                x0, x1 = float(np.nanmin(px)), float(np.nanmax(px))
                y0, y1 = float(np.nanmin(py)), float(np.nanmax(py))
                return Rectangle((x0, y0), max(1e-6, x1 - x0), max(1e-6, y1 - y0), visible=False)
        except Exception:
            pass
        if use_display_observer:
            return None
        # Fallback to projected-box bounds if corner projection fails.
        if self._projected_box_bbox_rect is not None:
            return Rectangle(
                (self._projected_box_bbox_rect.get_x(), self._projected_box_bbox_rect.get_y()),
                self._projected_box_bbox_rect.get_width(),
                self._projected_box_bbox_rect.get_height(),
                visible=False,
            )
        return Rectangle((0.0, 0.0), 10.0, 10.0, visible=False)

    def _fov_box_projected_edges(self, smap) -> list[SkyCoord]:
        if self._state is None:
            return []
        fov_box = self._state.fov_box
        fov_rect = self._state.fov
        if fov_box is None and fov_rect is None:
            return []
        observer = self._resolved_observer_for_map(smap, self._state.display_observer_key) or "earth"
        obstime = getattr(smap, "date", None)
        frame_obs = Helioprojective(observer=observer, obstime=obstime)

        if fov_box is not None:
            corners_world = self._fov_box_world_corners(smap, fov_box)
            if corners_world is not None and len(corners_world) == 8:
                projected_edges = project_coordinate_edges_to_observer_hpc(
                    corners_world,
                    edge_pairs=_BOX_EDGE_INDEX_PAIRS,
                    frame_obs=frame_obs,
                )
                if projected_edges is not None:
                    return projected_edges

        fov_like = fov_rect or fov_box
        if fov_like is None:
            return []
        source_observer_key = self._state.fov_definition_observer_key
        source_context = self._observer_context(source_observer_key, obstime)
        source_observer = getattr(source_context, "observer_coordinate", None) or observer
        source_obstime = getattr(source_context, "date", None) or obstime
        source_frame = Helioprojective(observer=source_observer, obstime=source_obstime)
        half_w = 0.5 * max(float(fov_like.width_arcsec), 1e-3)
        half_h = 0.5 * max(float(fov_like.height_arcsec), 1e-3)
        base_corners = observer_rectangle_to_hpc_corners(
            xc_arcsec=float(fov_like.center_x_arcsec),
            yc_arcsec=float(fov_like.center_y_arcsec),
            xsize_arcsec=2.0 * half_w,
            ysize_arcsec=2.0 * half_h,
            observer=source_observer,
            obstime=source_obstime,
        )
        if base_corners is None or len(base_corners) != 4:
            return []
        corners = SkyCoord(list(base_corners) + list(base_corners))
        projected_edges = project_coordinate_edges_to_observer_hpc(
            corners,
            edge_pairs=_BOX_EDGE_INDEX_PAIRS,
            frame_obs=frame_obs,
        )
        return projected_edges or []

    def _fov_box_projected_face(self, smap) -> SkyCoord | None:
        if self._state is None or self._state.fov_box is None:
            return None
        fov_box = self._state.fov_box
        observer = self._resolved_observer_for_map(smap, self._state.display_observer_key) or "earth"
        obstime = getattr(smap, "date", None)
        frame_obs = Helioprojective(observer=observer, obstime=obstime)

        corners_world = self._fov_box_world_corners(smap, fov_box)
        if corners_world is None or len(corners_world) != 8:
            return None
        return project_box_front_face_to_observer_hpc(corners_world, frame_obs=frame_obs)

    def _box_bounds_to_fov_selection(self, box, smap) -> DisplayFovSelection:
        bounds = box.bounds_coords.transform_to(
            Helioprojective(
                observer=self._resolved_observer_for_map(smap, self._state.display_observer_key) or "earth",
                obstime=getattr(smap, "date", None),
            )
        )
        tx = np.asarray(bounds.Tx.to_value(u.arcsec), dtype=float).ravel()
        ty = np.asarray(bounds.Ty.to_value(u.arcsec), dtype=float).ravel()
        finite = np.isfinite(tx) & np.isfinite(ty)
        if not np.any(finite):
            return DisplayFovSelection(0.0, 0.0, 10.0, 10.0)
        tx = tx[finite]
        ty = ty[finite]
        xmin, xmax = float(np.nanmin(tx)), float(np.nanmax(tx))
        ymin, ymax = float(np.nanmin(ty)), float(np.nanmax(ty))
        return DisplayFovSelection(
            center_x_arcsec=0.5 * (xmin + xmax),
            center_y_arcsec=0.5 * (ymin + ymax),
            width_arcsec=max(1e-3, xmax - xmin),
            height_arcsec=max(1e-3, ymax - ymin),
        )

    def _pixel_rect_to_fov_selection(self, smap, rect: Rectangle) -> DisplayFovSelection:
        x0, y0 = rect.get_x(), rect.get_y()
        w, h = rect.get_width(), rect.get_height()
        cx = x0 + 0.5 * w
        cy = y0 + 0.5 * h
        world = smap.wcs.pixel_to_world(cx, cy)
        observer_key = self._state.fov_definition_observer_key if self._state is not None else "earth"
        source_context = self._observer_context(observer_key, getattr(smap, "date", None))
        hpc = world.transform_to(
            Helioprojective(
                observer=getattr(source_context, "observer_coordinate", None)
                or self._resolved_observer_for_map(smap, observer_key)
                or "earth",
                obstime=getattr(source_context, "date", None) or getattr(smap, "date", None),
            )
        )
        scale_x = self._map_pixel_scale_arcsec(smap, axis=0)
        scale_y = self._map_pixel_scale_arcsec(smap, axis=1)
        return DisplayFovSelection(
            center_x_arcsec=float(hpc.Tx.to_value(u.arcsec)),
            center_y_arcsec=float(hpc.Ty.to_value(u.arcsec)),
            width_arcsec=float(abs(w) * scale_x),
            height_arcsec=float(abs(h) * scale_y),
        )

    @staticmethod
    def _map_pixel_scale_arcsec(smap, axis: int) -> float:
        try:
            if axis == 0:
                return abs(float(smap.scale.axis1.to_value(u.arcsec / u.pix)))
            return abs(float(smap.scale.axis2.to_value(u.arcsec / u.pix)))
        except Exception:
            return 0.6

    def _hit_test_overlay(self, ex: float, ey: float):
        bounds = self._overlay_rect_bounds()
        if bounds is None:
            return None
        x0, y0, x1, y1 = bounds
        w, h = x1 - x0, y1 - y0
        cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
        tol = max(6.0, 0.03 * max(w, h))

        if abs(ex - cx) <= tol and abs(ey - cy) <= tol:
            return {"kind": "move"}

        corners = {"bl": (x0, y0), "br": (x1, y0), "tr": (x1, y1), "tl": (x0, y1)}
        for corner_name, (hx, hy) in corners.items():
            if abs(ex - hx) <= tol and abs(ey - hy) <= tol:
                return {"kind": "resize_corner", "corner": corner_name}

        if y0 - tol <= ey <= y1 + tol and abs(ex - x0) <= tol:
            return {"kind": "resize_x", "side": "left"}
        if y0 - tol <= ey <= y1 + tol and abs(ex - x1) <= tol:
            return {"kind": "resize_x", "side": "right"}
        if x0 - tol <= ex <= x1 + tol and abs(ey - y0) <= tol:
            return {"kind": "resize_y", "side": "bottom"}
        if x0 - tol <= ex <= x1 + tol and abs(ey - y1) <= tol:
            return {"kind": "resize_y", "side": "top"}

        if x0 <= ex <= x1 and y0 <= ey <= y1:
            return {"kind": "inside"}
        return None

    def _build_drag_state_from_click(self, ex: float, ey: float):
        bounds = self._overlay_rect_bounds()
        if bounds is None:
            return None
        x0, y0, x1, y1 = bounds
        w, h = x1 - x0, y1 - y0
        cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
        hit = self._hit_test_overlay(ex, ey)

        if self._interaction_mode == "auto":
            if hit is None:
                return None
            if hit["kind"] in {"move", "inside"}:
                return {"mode": "move", "dx": ex - cx, "dy": ey - cy}
            if hit["kind"] == "resize_corner":
                corner_name = hit["corner"]
                return {
                    "mode": "resize",
                    "corner": corner_name,
                    "anchor_x": x1 if "l" in corner_name else x0,
                    "anchor_y": y1 if "b" in corner_name else y0,
                }
            if hit["kind"] == "resize_x":
                return {"mode": "resize_x", "anchor_x": x1 if hit["side"] == "left" else x0}
            if hit["kind"] == "resize_y":
                return {"mode": "resize_y", "anchor_y": y1 if hit["side"] == "bottom" else y0}
            return None

        if hit is None and self._interaction_mode in {"move", "resize_xy", "resize_x", "resize_y"}:
            return None

        if self._interaction_mode == "move":
            return {"mode": "move", "dx": ex - cx, "dy": ey - cy}
        if self._interaction_mode == "resize_xy":
            # Pick the active corner by click quadrant around current center.
            return {
                "mode": "resize",
                "corner": ("t" if ey >= cy else "b") + ("r" if ex >= cx else "l"),
                "anchor_x": x0 if ex >= cx else x1,
                "anchor_y": y0 if ey >= cy else y1,
            }
        if self._interaction_mode == "resize_x":
            return {"mode": "resize_x", "anchor_x": x0 if ex >= cx else x1}
        if self._interaction_mode == "resize_y":
            return {"mode": "resize_y", "anchor_y": y0 if ey >= cy else y1}
        return None

    def _update_cursor_for_mode(self) -> None:
        if self._drag_state is not None:
            return
        if self._interaction_mode == "move":
            self._canvas.setCursor(Qt.SizeAllCursor)
        elif self._interaction_mode == "resize_x":
            self._canvas.setCursor(Qt.SizeHorCursor)
        elif self._interaction_mode == "resize_y":
            self._canvas.setCursor(Qt.SizeVerCursor)
        elif self._interaction_mode == "resize_xy":
            self._canvas.setCursor(Qt.SizeFDiagCursor)
        else:
            self._canvas.setCursor(Qt.ArrowCursor)

    def _update_hover_cursor(self, event) -> None:
        if self._interaction_mode != "auto":
            self._update_cursor_for_mode()
            return
        if event is None or event.inaxes is not self._current_axes or event.xdata is None or event.ydata is None:
            self._canvas.setCursor(Qt.ArrowCursor)
            return
        hit = self._hit_test_overlay(float(event.xdata), float(event.ydata))
        if hit is None:
            self._canvas.setCursor(Qt.ArrowCursor)
            return
        if hit["kind"] in {"move", "inside"}:
            self._canvas.setCursor(Qt.SizeAllCursor)
        elif hit["kind"] == "resize_x":
            self._canvas.setCursor(Qt.SizeHorCursor)
        elif hit["kind"] == "resize_y":
            self._canvas.setCursor(Qt.SizeVerCursor)
        elif hit["kind"] == "resize_corner":
            self._canvas.setCursor(Qt.SizeFDiagCursor if hit["corner"] in {"bl", "tr"} else Qt.SizeBDiagCursor)
        else:
            self._canvas.setCursor(Qt.ArrowCursor)

    def _on_mouse_press(self, event) -> None:
        if not self._mouse_actions_enabled:
            return
        if event.button != 1 or event.inaxes is None:
            return
        if self._state is None or self._state.geometry is None or self._current_map is None or self._overlay_rect is None:
            return
        if event.inaxes is not self._current_axes:
            return
        ex, ey = event.xdata, event.ydata
        if ex is None or ey is None:
            return

        self._drag_state = self._build_drag_state_from_click(float(ex), float(ey))
        if self._drag_state is not None:
            self._drag_preview_geometry = None
            self._update_cursor_for_mode()

    def _on_mouse_move(self, event) -> None:
        if not self._mouse_actions_enabled:
            return
        if self._drag_state is None:
            self._update_hover_cursor(event)
            return
        if event.inaxes is not self._current_axes or event.xdata is None or event.ydata is None:
            return
        if self._state is None or self._state.geometry is None or self._current_map is None:
            return
        geom = self._state.geometry
        smap = self._current_map
        x = float(event.xdata)
        y = float(event.ydata)

        try:
            if self._drag_state["mode"] == "move":
                new_cx = x - self._drag_state["dx"]
                new_cy = y - self._drag_state["dy"]
                new_geom = self._geometry_from_pixel_edit(
                    geom,
                    center_px=(new_cx, new_cy),
                )
            elif self._drag_state["mode"] == "resize":
                ax_x = float(self._drag_state["anchor_x"])
                ax_y = float(self._drag_state["anchor_y"])
                cx = 0.5 * (ax_x + x)
                cy = 0.5 * (ax_y + y)
                new_geom = self._geometry_from_pixel_edit(
                    geom,
                    center_px=(cx, cy),
                    size_px=(abs(x - ax_x), abs(y - ax_y)),
                )
            elif self._drag_state["mode"] == "resize_x":
                rect = self._overlay_rect
                y0, h = rect.get_y(), rect.get_height()
                ax_x = float(self._drag_state["anchor_x"])
                cx = 0.5 * (ax_x + x)
                cy = y0 + 0.5 * h
                new_geom = self._geometry_from_pixel_edit(
                    geom,
                    center_px=(cx, cy),
                    size_px=(abs(x - ax_x), h),
                )
            elif self._drag_state["mode"] == "resize_y":
                rect = self._overlay_rect
                x0, w = rect.get_x(), rect.get_width()
                ax_y = float(self._drag_state["anchor_y"])
                cx = x0 + 0.5 * w
                cy = 0.5 * (ax_y + y)
                new_geom = self._geometry_from_pixel_edit(
                    geom,
                    center_px=(cx, cy),
                    size_px=(w, abs(y - ax_y)),
                )
            else:
                return
        except Exception:
            return

        preview = self._geometry_preview_overlay(new_geom)
        if preview is None:
            return
        box_rect, center_px = preview
        self._drag_preview_geometry = new_geom
        self._update_drag_preview(box_rect, center_px)

    def _on_mouse_release(self, event) -> None:
        if not self._mouse_actions_enabled:
            return
        pending_geom = self._drag_preview_geometry
        self._drag_state = None
        self._drag_preview_geometry = None
        if pending_geom is not None:
            self._end_drag_preview(restore_static=False)
            self.set_geometry_selection(pending_geom)
        else:
            self._end_drag_preview(restore_static=True)
        self._update_hover_cursor(event)

    def _geometry_from_pixel_edit(self, geom: BoxGeometrySelection, center_px=None, size_px=None) -> BoxGeometrySelection:
        smap = self._current_map
        out = BoxGeometrySelection(
            coord_mode=geom.coord_mode,
            coord_x=geom.coord_x,
            coord_y=geom.coord_y,
            grid_x=geom.grid_x,
            grid_y=geom.grid_y,
            grid_z=geom.grid_z,
            dx_km=geom.dx_km,
        )

        if center_px is not None:
            world = smap.wcs.pixel_to_world(float(center_px[0]), float(center_px[1]))
            try:
                if geom.coord_mode == CoordMode.HPC:
                    c = world.transform_to(Helioprojective(
                        obstime=getattr(smap, "date", None),
                        observer=self._resolved_observer_for_map(
                            smap,
                            self._state.geometry_definition_observer_key if self._state is not None else "earth",
                        ) or "earth",
                    ))
                    out.coord_x = float(c.Tx.to_value(u.arcsec))
                    out.coord_y = float(c.Ty.to_value(u.arcsec))
                elif geom.coord_mode == CoordMode.HGC:
                    c = world.transform_to(HeliographicCarrington(
                        obstime=getattr(smap, "date", None),
                        observer=self._resolved_observer_for_map(
                            smap,
                            self._state.geometry_definition_observer_key if self._state is not None else "earth",
                        ) or "earth",
                    ))
                    out.coord_x = float(c.lon.to_value(u.deg))
                    out.coord_y = float(c.lat.to_value(u.deg))
                else:
                    c = world.transform_to(HeliographicStonyhurst(obstime=getattr(smap, "date", None)))
                    out.coord_x = float(c.lon.to_value(u.deg))
                    out.coord_y = float(c.lat.to_value(u.deg))
            except Exception:
                pass

        if size_px is not None:
            try:
                center_hpc = self._geometry_center_hpc_for_map(out, smap)
                cpx, cpy = smap.wcs.world_to_pixel(center_hpc)
                wpx = max(1.0, float(size_px[0]))
                hpx = max(1.0, float(size_px[1]))
                x0 = float(cpx) - 0.5 * wpx
                x1 = float(cpx) + 0.5 * wpx
                y0 = float(cpy) - 0.5 * hpx
                y1 = float(cpy) + 0.5 * hpx
                wx = smap.wcs.pixel_to_world([x0, x1], [float(cpy), float(cpy)])
                wy = smap.wcs.pixel_to_world([float(cpx), float(cpx)], [y0, y1])
                dx_arcsec = abs(wx[1].Tx.to_value(u.arcsec) - wx[0].Tx.to_value(u.arcsec))
                dy_arcsec = abs(wy[1].Ty.to_value(u.arcsec) - wy[0].Ty.to_value(u.arcsec))
                dsun_km = self._dsun_km_from_map(smap)
                width_km = dx_arcsec / 206265.0 * dsun_km
                height_km = dy_arcsec / 206265.0 * dsun_km
                out.grid_x = max(1, int(round(width_km / max(out.dx_km, 1e-6))))
                out.grid_y = max(1, int(round(height_km / max(out.dx_km, 1e-6))))
            except Exception:
                pass

        return out

    def _dsun_km_from_map(self, smap) -> float:
        try:
            observer = self._resolved_observer_for_map(smap)
            if observer is not None:
                return float(observer.radius.to_value(u.km))
        except Exception:
            try:
                return float(smap.dsun.to_value(u.km))
            except Exception:
                dsun_obs = smap.meta.get("dsun_obs")
                if dsun_obs is not None:
                    return float((float(dsun_obs) * u.m).to_value(u.km))
        return 1.496e8
