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
from astropy.coordinates import SkyCoord
from astropy.time import Time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QSize
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
    SphericalScreen,
    sun,
)
from sunpy.visualization import colormaps as sunpy_colormaps

from pyampp.geometry import (
    local_cartesian_to_world,
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
_AIA_REFERENCE_IDS = ("171", "193", "211", "304", "335", "1600", "1700", "131", "94")
_HMI_VECTOR_DISPLAY_KEYS = {"field", "inclination", "azimuth", "disambig"}
_AIA_COLOR_KEYS = {"94", "131", "1600", "1700", "171", "193", "211", "304", "335"}
_BOTTOM_OVERLAY_CONTEXT_KEYS = _AIA_COLOR_KEYS | _HMI_VECTOR_DISPLAY_KEYS
_EMBEDDED_REFMAP_FLAG = "PYEMBED"
_BOX_EDGE_INDEX_PAIRS = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_DISPLAY_OBSERVER_OPTIONS = (
    ("earth", "Earth"),
    ("sdo", "SDO"),
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
        self._prepared_context_map_cache = {}
        self._raw_map_cache = {}
        self._cache_lock = threading.RLock()
        self._background_cache_enabled = False
        self._background_cache_generation = 0
        self._background_cache_thread = None
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
            "sdo": "sdo",
            "sdo/aia": "sdo",
            "sdo/hmi": "sdo",
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

    @staticmethod
    def _obstime_for_map(smap, fallback_iso: str | None = None):
        obstime = getattr(smap, "date", None)
        if obstime is not None:
            return obstime
        if fallback_iso:
            try:
                return Time(fallback_iso)
            except Exception:
                return None
        return None

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
                if coord is None and key == "sdo":
                    raw_map = None
                    try:
                        raw_map = self._load_raw_map(
                            self._canonical_map_key(
                                self._state.selected_context_id if self._state is not None else None,
                                purpose="context",
                            ),
                            purpose="context",
                        ) if (self._state is not None and self._state.selected_context_id) else None
                    except Exception:
                        raw_map = None
                    if raw_map is None:
                        try:
                            raw_map = self._reference_context_map()
                        except Exception:
                            raw_map = None
                    if raw_map is not None:
                        try:
                            raw_observer = getattr(raw_map, "observer_coordinate", None)
                            if raw_observer is not None:
                                coord = raw_observer.transform_to(HeliographicStonyhurst(obstime=when))
                                warning = None
                                used_key = "sdo"
                        except Exception:
                            pass
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
            metadata = {
                "observer_coordinate": coord,
                "observer_key": used_key,
                "obs_time": when,
                "b0_deg": float(hgs.lat.to_value(u.deg)),
                "l0_deg": float(hgs.lon.to_value(u.deg)),
                "p_deg": float(sun.P(when).to_value(u.deg)) if used_key == "earth" else None,
                "dsun_cm": float(coord.radius.to_value(u.cm)),
                "rsun_cm": None,
                "rsun_arcsec": None,
                "source": "session",
            }

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
        obstime = self._obstime_for_map(smap, self._state.session_input.time_iso if self._state is not None else None)
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

    def _reproject_map_for_display_observer(
        self,
        smap,
        *,
        fov_override: DisplayFovSelection | None = None,
    ):
        if self._state is None:
            return smap, None
        display_key = self._normalize_observer_key(self._state.display_observer_key)
        if display_key == "earth":
            return smap, None
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
                    atol=1e-6,
                )
                same_lat = np.isclose(
                    float(current_observer.lat.to_value(u.deg)),
                    float(observer.lat.to_value(u.deg)),
                    atol=1e-6,
                )
                if same_lon and same_lat:
                    return smap, None
        except Exception:
            pass
        ny, nx = np.asarray(smap.data).shape[:2]
        cx = 0.5 * max(0, nx - 1)
        cy = 0.5 * max(0, ny - 1)
        try:
            center_world = smap.wcs.pixel_to_world(cx, cy)
            source_label = str(
                getattr(smap, "detector", None)
                or getattr(smap, "observatory", None)
                or getattr(smap, "nickname", None)
                or "map"
            )
            target_fov = fov_override if fov_override is not None else self._current_display_prepare_fov(
                display_key,
                obstime=obstime,
            )
            header = self._display_observer_reproject_header_for_selection(smap, observer, obstime, target_fov)
            if header is not None:
                self._record_prepare_event(f"observer reproj: {source_label} -> {display_key} [roi]")
                coverage_fov = target_fov
            else:
                target_center = center_world.transform_to(Helioprojective(observer=observer, obstime=obstime))
                self._record_prepare_event(f"observer reproj: {source_label} -> {display_key} [full]")
                header = make_fitswcs_header(
                    smap.data,
                    target_center,
                    scale=u.Quantity([
                        abs(smap.scale.axis1.to_value(u.arcsec / u.pix)),
                        abs(smap.scale.axis2.to_value(u.arcsec / u.pix)),
                    ], u.arcsec / u.pix),
                )
                try:
                    header["rsun_ref"] = float(smap.rsun_meters.to_value(u.m))
                except Exception:
                    pass
                coverage_fov = None
            return smap.reproject_to(header, algorithm="adaptive", roundtrip_coords=False), coverage_fov
        except Exception:
            return smap, None

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
        self._state.session_input.map_ids = map_ids
        if self._state.selected_context_id not in map_ids:
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
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot(preserve_current_view=self._should_preserve_pixel_view())

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
        self._state.map_files = normalized
        self._map_summary_cache.clear()
        self._invalidate_map_caches()
        self._refresh_map_info()
        self._refresh_plot()

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
        self._invalidate_map_caches()
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot(preserve_current_view=self._should_preserve_pixel_view())

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
        if self._state is None or self._state.fov is None or self._state.fov_box is None:
            return
        new_fov_box = DisplayFovBoxSelection(
            center_x_arcsec=float(self._state.fov.center_x_arcsec),
            center_y_arcsec=float(self._state.fov.center_y_arcsec),
            width_arcsec=float(self._state.fov.width_arcsec),
            height_arcsec=float(self._state.fov.height_arcsec),
            z_min_mm=float(self._state.fov_box.z_min_mm),
            z_max_mm=float(self._state.fov_box.z_max_mm),
            observer_key=str(self._state.fov_box.observer_key or self._state.fov_definition_observer_key),
        )
        self._state.fov_box = new_fov_box

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
        if self._status_callback is not None:
            self._status_callback(self._last_status_text)

    def _clear_prepare_trace(self) -> None:
        self._prep_trace_counts.clear()
        self._prep_trace_order.clear()

    def _record_prepare_event(self, label: str) -> None:
        key = str(label or "").strip()
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
        model_time = str(self._state.session_input.time_iso or "")
        if model_time:
            base_text = f"{base_text}\nmodel_time={model_time}"
        observer_time = ""
        if self._normalize_observer_key(self._state.display_observer_key) == "custom":
            if isinstance(self._state.custom_observer_ephemeris, dict):
                observer_time = str(
                    self._state.custom_observer_ephemeris.get("obs_date")
                    or self._state.custom_observer_ephemeris.get("obs_time")
                    or ""
                )
        else:
            observer_time = model_time
        if observer_time:
            base_text = f"{base_text}\nobserver_time={observer_time}"
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
        cache_key = f"{role}:{map_id}"
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
        canonical_key = self._canonical_map_key(map_id, purpose=purpose)
        if purpose == "context":
            return self._context_map_for_id(map_id, canonical_key)
        smap = self._load_raw_map(canonical_key, purpose=purpose)
        if smap is None:
            return None
        observer_token = self._display_observer_cache_token(smap)
        view_key = str(self._view_mode or "box_fov")
        display_key = f"__{purpose}__:{observer_token}:{view_key}:{canonical_key}"
        alias_key = f"__{purpose}__:{observer_token}:{view_key}:{map_id}"
        with self._cache_lock:
            if alias_key in self._loaded_map_cache:
                return self._loaded_map_cache[alias_key]
            if display_key in self._loaded_map_cache:
                smap = self._loaded_map_cache[display_key]
                self._loaded_map_cache[alias_key] = smap
                return smap

        smap = self._prepare_bottom_map(canonical_key, smap)
        with self._cache_lock:
            self._loaded_map_cache[display_key] = smap
            self._loaded_map_cache[alias_key] = smap
        return smap

    def _context_map_for_id(self, map_id: str, canonical_key: str):
        smap = self._load_raw_map(canonical_key, purpose="context")
        if smap is None:
            return None
        observer_key = self._normalize_observer_key(
            self._state.display_observer_key if self._state is not None else "earth"
        )
        observer_token = self._display_observer_cache_token(smap, observer_key)
        prepared_key = f"__context_prepared__:{observer_token}:{canonical_key}"
        desired_prepare_fov = self._current_display_prepare_fov(
            observer_key,
            obstime=self._obstime_for_map(
                smap,
                self._state.session_input.time_iso if self._state is not None else None,
            ),
        )
        with self._cache_lock:
            cache_entry = self._prepared_context_map_cache.get(prepared_key)
        prepared_map = None
        coverage_fov = None
        if isinstance(cache_entry, dict):
            prepared_map = cache_entry.get("map")
            coverage_fov = cache_entry.get("coverage_fov")
        else:
            prepared_map = cache_entry
        if prepared_map is None or (desired_prepare_fov is None and coverage_fov is not None) or (
            desired_prepare_fov is not None and coverage_fov is not None and not self._fov_contains(coverage_fov, desired_prepare_fov)
        ):
            self._record_prepare_event(f"context prepare: {canonical_key} @ {observer_key}")
            prepared_map, coverage_fov = self._prepare_context_map(canonical_key, smap, target_fov=desired_prepare_fov)
            with self._cache_lock:
                self._prepared_context_map_cache[prepared_key] = {
                    "map": prepared_map,
                    "coverage_fov": coverage_fov,
                }

        if self._view_mode == "box_fov":
            display_bounds = self._display_window_pixel_bounds(prepared_map)
            if display_bounds is not None:
                return self._submap_to_pixel_bounds(prepared_map, display_bounds)
            display_fov = self._display_window_fov_selection(prepared_map)
            if display_fov is not None:
                return self._submap_to_explicit_fov(prepared_map, fov_override=display_fov, pad_factor=1.0)
            box = self._build_legacy_box(prepared_map)
            if box is not None:
                return self._submap_to_box_bounds(prepared_map, box)
        return prepared_map

    def _invalidate_map_caches(self) -> None:
        with self._cache_lock:
            self._loaded_map_cache.clear()
            self._prepared_context_map_cache.clear()
            self._raw_map_cache.clear()
            self._background_cache_generation += 1

    def _invalidate_display_map_cache(self) -> None:
        with self._cache_lock:
            self._loaded_map_cache.clear()
            self._background_cache_generation += 1

    def _invalidate_geometry_dependent_display_maps(self) -> None:
        with self._cache_lock:
            keys_to_drop = [
                key for key in self._loaded_map_cache.keys()
                if key.startswith("__bottom__:")
            ]
            if self._is_non_earth_display_observer() and self._view_mode == "box_fov":
                keys_to_drop.extend(
                    key for key in self._loaded_map_cache.keys()
                    if key.startswith("__context__:")
                )
            for key in set(keys_to_drop):
                self._loaded_map_cache.pop(key, None)
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

    def _load_embedded_base_map(self, map_key: str, purpose: str = "context"):
        if self._state is None:
            return None
        with self._cache_lock:
            cache_key = f"__base__:{purpose}:{map_key}"
            if cache_key in self._raw_map_cache:
                return self._raw_map_cache[cache_key]
        data = self._embedded_base_array(map_key, purpose=purpose)
        if data is None:
            return None
        try:
            ref_map = self._reference_context_map()
            if ref_map is None:
                return None
            base_geom = self._state.base_geometry or self._state.geometry
            box = self._build_legacy_box(ref_map, geom=base_geom)
            if box is None:
                return None
            header = box.bottom_cea_header
            self._copy_observer_cards_from_map(header, ref_map)
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
        for src_key, dst_key in (
            ("hgln_obs", "HGLN_OBS"),
            ("hglt_obs", "HGLT_OBS"),
            ("dsun_obs", "DSUN_OBS"),
            ("crln_obs", "CRLN_OBS"),
            ("crlt_obs", "CRLT_OBS"),
            ("rsun_ref", "RSUN_REF"),
            ("date-obs", "DATE-OBS"),
            ("date_obs", "DATE_OBS"),
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
        return None

    def _load_embedded_refmap(self, ref_key: str, purpose: str = "context"):
        if self._state is None:
            return None
        with self._cache_lock:
            cache_key = f"__embedded__:{purpose}:{ref_key}"
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
            ref_map = self._reference_context_map()
            self._copy_observer_cards_from_map(header, ref_map)
            header[_EMBEDDED_REFMAP_FLAG] = True
            smap = map_from_data_header_compat(np.asarray(data), header)
        except Exception:
            return None
        with self._cache_lock:
            self._raw_map_cache[cache_key] = smap
        return smap

    def _load_raw_map(self, map_key: str, purpose: str = "context"):
        with self._cache_lock:
            raw_cache_key = f"__rawmap__:{purpose}:{map_key}"
            if raw_cache_key in self._raw_map_cache:
                return self._raw_map_cache[raw_cache_key]
        path = self._filesystem_path_for_key(map_key, purpose=purpose)
        if path:
            smap = load_sunpy_map_compat(path)
            if map_key in _HMI_VECTOR_SEGMENTS:
                smap = self._submap_to_geometry_fov(smap)
        else:
            smap = self._load_embedded_base_map(map_key, purpose=purpose)
            if smap is not None:
                with self._cache_lock:
                    self._raw_map_cache[raw_cache_key] = smap
                return smap
            ref_key = self._embedded_refmap_key(map_key)
            smap = self._load_embedded_refmap(ref_key, purpose=purpose) if ref_key else None
            if smap is None:
                return None
        with self._cache_lock:
            self._raw_map_cache[raw_cache_key] = smap
        return smap

    def _prepare_context_map(
        self,
        map_key: str,
        smap,
        *,
        target_fov: DisplayFovSelection | None = None,
    ):
        display_map = smap
        if map_key in _HMI_DISPLAY_KEYS:
            try:
                display_map = display_map.rotate(order=3)
            except Exception:
                pass
            ref_map = self._reference_context_map()
            if ref_map is not None:
                try:
                    self._record_prepare_event(f"context reproj: {map_key} -> ref_wcs")
                    display_map = self._with_matching_rsun(display_map, ref_map)
                    display_map = display_map.reproject_to(ref_map.wcs)
                except Exception:
                    pass
        display_map, coverage_fov = self._reproject_map_for_display_observer(display_map, fov_override=target_fov)
        self._apply_display_scaling(display_map, map_key)
        return display_map, coverage_fov

    def _prepare_bottom_map(self, map_key: str, smap):
        display_map = smap
        box = self._build_legacy_box(display_map)
        if box is not None:
            if self._view_mode == "box_fov":
                display_bounds = self._display_window_pixel_bounds(display_map)
                if display_bounds is not None:
                    display_map = self._submap_to_pixel_bounds(display_map, display_bounds)
                else:
                    display_fov = self._display_window_fov_selection(display_map)
                    if display_fov is not None:
                        display_map = self._submap_to_explicit_fov(display_map, fov_override=display_fov, pad_factor=1.0)
                    else:
                        display_map = self._submap_to_box_bounds(display_map, box)
            if not self._is_non_earth_display_observer() and self._view_mode == "box_fov":
                try:
                    self._record_prepare_event(f"bottom reproj: {map_key} -> base_cea")
                    display_map = self._with_matching_rsun(display_map, box.bottom_cea_header)
                    display_map = display_map.reproject_to(
                        box.bottom_cea_header,
                        algorithm="adaptive",
                        roundtrip_coords=False,
                    )
                except Exception:
                    pass
        # Keep bottom maps in their native/base WCS and let SunPy autoalign
        # handle observer-frame plotting on the current axes.
        self._apply_display_scaling(display_map, map_key)
        return display_map

    def _reference_context_map(self):
        if self._state is None:
            return None
        for map_id in _AIA_REFERENCE_IDS:
            path = self._filesystem_path_for_key(map_id)
            cache_key = f"__ref__:{map_id}"
            with self._cache_lock:
                if cache_key in self._raw_map_cache:
                    return self._raw_map_cache[cache_key]
            try:
                if path:
                    ref_map = load_sunpy_map_compat(path)
                else:
                    ref_map = self._load_embedded_refmap(f"AIA_{map_id}")
                    if ref_map is None:
                        continue
                with self._cache_lock:
                    self._raw_map_cache[cache_key] = ref_map
                return ref_map
            except Exception:
                continue
        return None

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

    def _submap_to_explicit_fov(
        self,
        smap,
        pad_factor: float = 1.10,
        *,
        fov_override: DisplayFovSelection | None = None,
    ):
        fov = fov_override if fov_override is not None else (self._state.fov if (self._state and self._state.fov) else None)
        if fov is None:
            geometry_observer_key = self._state.geometry_definition_observer_key if self._state is not None else None
            box = self._build_legacy_box(smap, geometry_observer_key=geometry_observer_key)
            if box is None:
                return smap
            fov = self._box_bounds_to_fov_selection(box, smap)
        half_w = 0.5 * max(float(fov.width_arcsec), 1e-3) * float(pad_factor)
        half_h = 0.5 * max(float(fov.height_arcsec), 1e-3) * float(pad_factor)
        observer = getattr(smap, "observer_coordinate", None) or "earth"
        obstime = getattr(smap, "date", None)
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
            return smap.submap(bottom_left, top_right=top_right)
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
            return smap.submap(bottom_left, top_right=top_right)
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
                if bool(getattr(smap, "meta", {}).get(_EMBEDDED_REFMAP_FLAG, False)):
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
            smap = self._selected_context_map()
            overlay_map = self._selected_bottom_map()
        except Exception as exc:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Map load failed:\n{exc}", ha="center", va="center")
            ax.axis("off")
            self._canvas.draw_idle()
            return

        if smap is None:
            ref_map = self._reference_context_map()
            if ref_map is not None:
                smap, _coverage_fov = self._reproject_map_for_display_observer(ref_map)
            else:
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
            try:
                if self._state is not None and self._state.selected_context_id is None:
                    # Keep correct context-map extents even when context display is hidden.
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
            title = (
                f"{self._display_map_label(self._state.selected_context_id, bottom=False)} | "
                f"{self._display_map_label(self._state.selected_bottom_id, bottom=True)} | "
                f"{self._observer_label_for_key(self._state.display_observer_key)} | {getattr(smap, 'date', '')}"
            )
            self._fig.text(0.02, 0.992, title, ha="left", va="top", fontsize=10)
            if context_xlim is not None and context_ylim is not None:
                self._full_view_limits = (context_xlim, context_ylim)
                if self._view_mode == "full_sun":
                    ax.set_xlim(context_xlim)
                    ax.set_ylim(context_ylim)
            else:
                self._full_view_limits = (ax.get_xlim(), ax.get_ylim())
            if preserve_current_view and prev_xlim is not None and prev_ylim is not None:
                self._restore_preserved_view(prev_xlim, prev_ylim)
            elif self._view_mode == "box_fov":
                self._set_view_to_projected_fov(pad_factor=1.10)
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
        if context_key == bottom_key:
            return False
        # Bottom overlay is useful for image-like context maps, but it obscures
        # signed diagnostic maps such as Vert_current almost completely.
        return context_key in _BOTTOM_OVERLAY_CONTEXT_KEYS
        self._set_runtime_status("Cleared over-plotted field lines.")

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
        map_ids = list(session_input.map_ids or [])
        preferred = [
            "171", "193", "211", "304", "335", "1600",
            "Bz", "Ic", "B_rho", "B_theta", "B_phi", "disambig",
            # Backward-compatible legacy labels.
            "Br", "Bp", "Bt",
        ]
        for key in preferred:
            if key in map_ids:
                return key
        return map_ids[0] if map_ids else None

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
        self._refresh_plot()

    def show_box_fov_view(self, pad_factor: float | None = None) -> None:
        # `pad_factor` is handled by the display-crop helper; keep the method
        # signature stable for existing button hookups.
        self._view_mode = "box_fov"
        self._refresh_plot()

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
            fov_box = box.model_box_inscribing_fov_box(observer=observer, obstime=obstime)
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

    def _fov_selection_to_pixel_rect(self, smap, fov: DisplayFovSelection | None) -> Rectangle:
        if fov is None:
            if self._projected_box_bbox_rect is not None:
                return Rectangle(
                    (self._projected_box_bbox_rect.get_x(), self._projected_box_bbox_rect.get_y()),
                    self._projected_box_bbox_rect.get_width(),
                    self._projected_box_bbox_rect.get_height(),
                    visible=False,
                )
            return Rectangle((0.0, 0.0), 10.0, 10.0, visible=False)
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
            source_observer_key = self._normalize_observer_key(getattr(fov_box, "observer_key", None))
            source_context = self._observer_context(source_observer_key, obstime)
            source_map = source_context or smap
            box = self._build_legacy_box(
                source_map,
                geometry_observer_key=self._state.geometry_definition_observer_key,
            )
            if box is not None:
                corners_world = box.fov_box_corners_world(
                    fov_box.as_observer_metadata(square=bool(self._state.square_fov))
                )
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

        source_observer_key = self._normalize_observer_key(getattr(fov_box, "observer_key", None))
        source_context = self._observer_context(source_observer_key, obstime)
        source_map = source_context or smap
        box = self._build_legacy_box(
            source_map,
            geometry_observer_key=self._state.geometry_definition_observer_key,
        )
        if box is None:
            return None
        corners_world = box.fov_box_corners_world(
            fov_box.as_observer_metadata(square=bool(self._state.square_fov))
        )
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
