from __future__ import annotations

from pathlib import Path
from typing import Optional

from astropy.time import Time
from PyQt5.QtCore import QEvent, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from .observer_restore import (
    normalize_observer_key,
    probe_observer_availability,
    resolve_observer_parameters_from_ephemeris,
)
from .boxutils import observer_ephemeris_from_reference_file, observer_reference_details_from_file
from .map_box_view import MapBoxDisplayWidget
from .selector_api import (
    BoxGeometrySelection,
    CoordMode,
    DisplayFovSelection,
    SelectorDialogResult,
    SelectorSessionInput,
)


class _GuardedComboBox(QComboBox):
    """Ignore wheel-based selection changes unless the popup is open."""

    def wheelEvent(self, event) -> None:
        view = self.view()
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


class _ObserverAvailabilityWorker(QObject):
    finished = pyqtSignal(object)

    def __init__(
        self,
        *,
        session_input: SelectorSessionInput,
        observer_keys: tuple[str, ...],
        skip_keys: tuple[str, ...] = (),
    ):
        super().__init__()
        self._session_input = session_input
        self._observer_keys = observer_keys
        self._skip_keys = {normalize_observer_key(key) for key in skip_keys}

    def run(self) -> None:
        enabled_keys = {"earth"}
        statuses: dict[str, str] = {"earth": "available"}
        notice = ""
        offline = False
        try:
            when = Time(self._session_input.time_iso)
        except Exception:
            when = None
        saved_keys = {
            normalize_observer_key(getattr(self._session_input, "display_observer_key", "earth")),
            normalize_observer_key(getattr(getattr(self._session_input, "fov_box", None), "observer_key", None)),
        }
        saved_keys.discard("earth")
        saved_keys.discard("")
        source_b3d: dict = {}
        if isinstance(self._session_input.refmaps, dict) and self._session_input.refmaps:
            source_b3d["refmaps"] = dict(self._session_input.refmaps)
        if isinstance(self._session_input.custom_observer_ephemeris, dict):
            source_b3d["observer"] = {
                "name": "custom",
                "ephemeris": dict(self._session_input.custom_observer_ephemeris),
            }
        if when is not None:
            for key in self._observer_keys:
                if key == "earth" or normalize_observer_key(key) in self._skip_keys:
                    statuses[key] = "saved"
                    enabled_keys.add(key)
                    continue
                status, _detail = probe_observer_availability(source_b3d, key, when)
                statuses[key] = str(status)
                if status == "available":
                    enabled_keys.add(key)
                    continue
                if status == "offline":
                    offline = True
                    notice = (
                        "Observer availability scan could not reach remote ephemeris services. "
                        "Non-Earth observer selections were disabled for this session."
                    )
                    break
        if offline:
            enabled_keys = {"earth"} | {key for key in saved_keys if key in self._observer_keys}
            for key in self._observer_keys:
                if key != "earth" and key not in enabled_keys:
                    statuses[key] = "offline"
        else:
            enabled_keys |= {key for key in saved_keys if key in self._observer_keys}
        self.finished.emit(
            {
                "enabled_keys": tuple(key for key in self._observer_keys if key in enabled_keys or key == "earth"),
                "statuses": statuses,
                "notice": notice,
            }
        )


class FovBoxSelectorDialog(QDialog):
    """
    Standalone post-download FOV/box selection GUI scaffold.

    This dialog is intentionally minimal at this stage:
    - it accepts/returns the finalized geometry contract (`BoxGeometrySelection`)
    - it hosts a reusable `MapBoxDisplayWidget`
    - it does not yet implement interactive plotting/dragging
    """

    def __init__(
        self,
        session_input: SelectorSessionInput,
        parent: Optional[QWidget] = None,
        entry_box_path: Optional[str | Path] = None,
    ):
        super().__init__(parent)
        self._session_input = session_input
        self._accepted_selection: Optional[SelectorDialogResult] = None
        self._entry_box_path = Path(entry_box_path).expanduser().resolve() if entry_box_path else None
        self._pending_session_input: Optional[SelectorSessionInput] = session_input
        self._session_loaded = False
        self._session_load_scheduled = False
        self._custom_observer_active = False
        self._last_standard_observer_info: dict[str, str] = {}
        self._last_custom_observer_info: dict[str, str] = {}
        self._observer_action_serial = 0
        self._observer_availability_statuses: dict[str, str] = {}
        self._suspend_status_updates = False
        self._availability_thread: Optional[QThread] = None
        self._availability_worker: Optional[_ObserverAvailabilityWorker] = None
        self.setWindowTitle("FOV / Box Selector")
        self.resize(1320, 760)

        self._build_ui()
        self.map_meta_label.setPlainText("Preparing viewer data...")
        self.selector_status_label.setPlainText("Preparing viewer data...\nPlease wait until maps are ready.")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._session_load_scheduled and not self._session_loaded and self._pending_session_input is not None:
            self._session_load_scheduled = True
            QTimer.singleShot(50, self._load_pending_session_input)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        body = QGridLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setHorizontalSpacing(4)
        body.setVerticalSpacing(0)
        root.addLayout(body, stretch=1)

        self.context_map_combo = _GuardedComboBox()
        self.context_map_combo.setMaximumWidth(160)
        self.context_map_combo.currentIndexChanged.connect(self._on_context_map_changed)
        self.bottom_map_combo = _GuardedComboBox()
        self.bottom_map_combo.setMaximumWidth(120)
        self.bottom_map_combo.currentIndexChanged.connect(self._on_bottom_map_changed)
        self.map_source_combo = _GuardedComboBox()
        self.map_source_combo.setMaximumWidth(110)
        self.map_source_combo.addItem("Auto", "auto")
        self.map_source_combo.addItem("Filesystem", "filesystem")
        self.map_source_combo.addItem("Embedded", "embedded")
        self.map_source_combo.currentIndexChanged.connect(self._on_map_source_changed)

        left_group = QGroupBox("")
        left_group.setFlat(True)
        left_group.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_group)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(2)
        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(6)
        for label_text, widget in (
            ("Context Maps", self.context_map_combo),
            ("Base Maps", self.bottom_map_combo),
            ("Map Source", self.map_source_combo),
        ):
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(2)
            col.addWidget(QLabel(label_text))
            col.addWidget(widget)
            selector_row.addLayout(col)
        left_layout.addLayout(selector_row)
        self.map_box_widget = MapBoxDisplayWidget()
        self.map_box_widget.set_entry_box_path(self._entry_box_path, load_session_model=False)
        left_layout.addWidget(self.map_box_widget, stretch=1)
        body.addWidget(left_group, 0, 0)

        self.coord_mode_combo = _GuardedComboBox()
        self.coord_mode_combo.addItems([m.value for m in CoordMode])
        self.coord_x_edit = QLineEdit()
        self.coord_y_edit = QLineEdit()
        self.grid_x_edit = QLineEdit()
        self.grid_y_edit = QLineEdit()
        self.grid_z_edit = QLineEdit()
        self.res_edit = QLineEdit()
        self.fov_x_edit = QLineEdit()
        self.fov_y_edit = QLineEdit()
        self.fov_w_edit = QLineEdit()
        self.fov_h_edit = QLineEdit()
        self.square_fov_box = QCheckBox()
        self.coord_mode_combo.setMaximumWidth(72)
        compact_fields = (
            self.coord_x_edit,
            self.coord_y_edit,
            self.grid_x_edit,
            self.grid_y_edit,
            self.grid_z_edit,
            self.res_edit,
            self.fov_x_edit,
            self.fov_y_edit,
            self.fov_w_edit,
            self.fov_h_edit,
        )
        for edit in compact_fields:
            edit.setMaximumWidth(92)

        observer_group = QGroupBox("")
        observer_group.setFlat(True)
        observer_outer = QVBoxLayout(observer_group)
        observer_outer.setContentsMargins(2, 16, 2, 2)
        observer_outer.setSpacing(4)

        observer_ephem_group = QGroupBox("Observer")
        observer_layout = QGridLayout(observer_ephem_group)
        observer_layout.setContentsMargins(4, 6, 4, 4)
        observer_layout.setHorizontalSpacing(4)
        observer_layout.setVerticalSpacing(4)
        self._observer_button_group = QButtonGroup(self)
        self._observer_buttons: dict[str, QRadioButton] = {}
        self._observer_status_labels: dict[str, QLabel] = {}
        observer_choice_grid = QGridLayout()
        observer_choice_grid.setContentsMargins(0, 0, 0, 0)
        observer_choice_grid.setHorizontalSpacing(10)
        observer_choice_grid.setVerticalSpacing(4)
        observer_choice_grid.setColumnStretch(0, 0)
        observer_choice_grid.setColumnStretch(1, 1)
        observer_row_height = 24
        observer_button_row = 0
        for key, label in self.map_box_widget.observer_options():
            row_widget = QWidget()
            row_widget.setFixedHeight(observer_row_height)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(QLabel(label))
            row_layout.addStretch(1)
            button = QRadioButton("")
            button.toggled.connect(lambda checked, observer_key=key: self._on_observer_radio_toggled(observer_key, checked))
            self._observer_button_group.addButton(button)
            self._observer_buttons[key] = button
            row_layout.addWidget(button)
            observer_choice_grid.addWidget(row_widget, observer_button_row, 0)
            status_label = QLabel("")
            status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self._observer_status_labels[key] = status_label
            observer_choice_grid.addWidget(status_label, observer_button_row, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
            observer_button_row += 1
        custom_row = observer_button_row
        custom_row_widget = QWidget()
        custom_row_widget.setFixedHeight(observer_row_height)
        custom_row_layout = QHBoxLayout(custom_row_widget)
        custom_row_layout.setContentsMargins(0, 0, 0, 0)
        custom_row_layout.setSpacing(6)
        custom_row_layout.addWidget(QLabel("Custom"))
        custom_row_layout.addStretch(1)
        self._observer_custom_button = QRadioButton("")
        self._observer_button_group.addButton(self._observer_custom_button)
        self._observer_custom_button.toggled.connect(self._on_custom_observer_toggled)
        custom_row_layout.addWidget(self._observer_custom_button)
        observer_choice_grid.addWidget(custom_row_widget, custom_row, 0)
        self._custom_mode_group = QButtonGroup(self)
        self._custom_manual_button = QRadioButton("Manual")
        self._custom_upload_button = QRadioButton("Upload Ref")
        self._custom_mode_group.addButton(self._custom_manual_button)
        self._custom_mode_group.addButton(self._custom_upload_button)
        self._custom_manual_button.setChecked(True)
        self._custom_manual_button.toggled.connect(self._on_custom_mode_changed)
        self._custom_upload_button.toggled.connect(self._on_custom_mode_changed)
        custom_mode_widget = QWidget()
        custom_mode_widget.setFixedHeight(observer_row_height)
        custom_mode_row = QHBoxLayout(custom_mode_widget)
        custom_mode_row.setContentsMargins(0, 0, 0, 0)
        custom_mode_row.setSpacing(6)
        custom_mode_row.addWidget(self._custom_manual_button)
        custom_mode_row.addWidget(self._custom_upload_button)
        custom_mode_row.addStretch(1)
        observer_choice_grid.addWidget(custom_mode_widget, custom_row, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        for row in range(custom_row + 1):
            observer_choice_grid.setRowMinimumHeight(row, observer_row_height)
        observer_layout.addLayout(observer_choice_grid, 0, 0, 1, 2)
        observer_button_row += 1
        observer_layout.setRowMinimumHeight(observer_button_row, 10)
        self._observer_fields: dict[str, QLineEdit] = {}
        observer_rows = [
            ("model_time", "MODEL-DATE"),
            ("obs_date", "OBS-DATE"),
            ("label", "LABEL"),
            ("b0_deg", "B0"),
            ("l0_deg", "L0"),
            ("rsun_arcsec", "RSUN_ARCSEC"),
            ("p_deg", "P"),
        ]
        for row, (key, label) in enumerate(observer_rows, start=observer_button_row + 1):
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setEnabled(False)
            edit.setMaximumWidth(190)
            edit.setAlignment(Qt.AlignRight)
            observer_layout.addWidget(QLabel(label), row, 0)
            observer_layout.addWidget(edit, row, 1)
            self._observer_fields[key] = edit
        for time_key in ("model_time", "obs_date"):
            time_edit = self._observer_fields[time_key]
            time_edit.setMaximumWidth(190)
            time_edit.setMinimumWidth(190)
        self._apply_custom_observer_edit_state()
        observer_outer.addWidget(observer_ephem_group, stretch=0)

        row0 = len(observer_rows) + 2
        fov_group = QGroupBox("FOV")
        fov_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        fov_layout = QGridLayout(fov_group)
        fov_layout.setContentsMargins(4, 6, 4, 4)
        fov_layout.setHorizontalSpacing(4)
        fov_layout.setVerticalSpacing(3)
        fov_layout.addWidget(QLabel("fov_xc"), 0, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        fov_layout.addWidget(self.fov_x_edit, 0, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        fov_layout.addWidget(QLabel("fov_yc"), 1, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        fov_layout.addWidget(self.fov_y_edit, 1, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        fov_layout.addWidget(QLabel("fov_xsize"), 2, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        fov_layout.addWidget(self.fov_w_edit, 2, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        fov_layout.addWidget(QLabel("fov_ysize"), 3, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        fov_layout.addWidget(self.fov_h_edit, 3, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        fov_layout.addWidget(QLabel("Square FOV"), 4, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        fov_layout.addWidget(self.square_fov_box, 4, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        observer_outer.addWidget(fov_group, stretch=0)

        box_group = QGroupBox("Box")
        box_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        box_layout = QGridLayout(box_group)
        box_layout.setContentsMargins(4, 6, 4, 4)
        box_layout.setHorizontalSpacing(4)
        box_layout.setVerticalSpacing(3)
        box_layout.addWidget(QLabel("mode"), 0, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(self.coord_mode_combo, 0, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(QLabel("box_xc"), 1, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(self.coord_x_edit, 1, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(QLabel("box_yc"), 2, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(self.coord_y_edit, 2, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(QLabel("grid_nx"), 3, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(self.grid_x_edit, 3, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(QLabel("grid_ny"), 4, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(self.grid_y_edit, 4, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(QLabel("grid_nz"), 5, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(self.grid_z_edit, 5, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(QLabel("dx_km"), 6, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        box_layout.addWidget(self.res_edit, 6, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        observer_outer.addWidget(box_group, stretch=0)
        observer_outer.addStretch(1)
        body.addWidget(observer_group, 0, 1)

        right_group = QGroupBox("")
        right_group.setFlat(True)
        right_layout = QVBoxLayout(right_group)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)
        self.map_meta_label = QPlainTextEdit()
        self.map_meta_label.setReadOnly(True)
        meta_group = QGroupBox("Selected Map Metadata")
        meta_layout = QVBoxLayout(meta_group)
        meta_layout.setContentsMargins(4, 4, 4, 4)
        meta_layout.addWidget(self.map_meta_label)
        right_layout.addWidget(meta_group, stretch=1)

        self.selector_status_label = QPlainTextEdit()
        self.selector_status_label.setReadOnly(True)
        status_group = QGroupBox("Interaction Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(4, 4, 4, 4)
        status_layout.addWidget(self.selector_status_label)
        right_layout.addWidget(status_group, stretch=1)
        mono_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.map_meta_label.setFont(mono_font)
        self.selector_status_label.setFont(mono_font)
        body.addWidget(right_group, 0, 2)
        body.setColumnStretch(0, 0)
        body.setColumnStretch(1, 0)
        body.setColumnStretch(2, 1)

        for edit in (
            self.coord_x_edit,
            self.coord_y_edit,
            self.grid_x_edit,
            self.grid_y_edit,
            self.grid_z_edit,
            self.res_edit,
            self.fov_x_edit,
            self.fov_y_edit,
            self.fov_w_edit,
            self.fov_h_edit,
            self._observer_fields["label"],
            self._observer_fields["b0_deg"],
            self._observer_fields["l0_deg"],
            self._observer_fields["rsun_arcsec"],
        ):
            edit.editingFinished.connect(self._push_form_to_view_state)
            edit.installEventFilter(self)
        self._observer_fields["label"].editingFinished.connect(self._apply_custom_observer_identity)
        self._observer_fields["b0_deg"].editingFinished.connect(self._apply_custom_manual_observer)
        self._observer_fields["l0_deg"].editingFinished.connect(self._apply_custom_manual_observer)
        self._observer_fields["rsun_arcsec"].editingFinished.connect(self._apply_custom_manual_observer)

        self.coord_mode_combo.currentTextChanged.connect(self._push_form_to_view_state)
        self.square_fov_box.toggled.connect(self._on_square_fov_toggled)
        self.map_box_widget.set_geometry_change_callback(self._apply_selection_to_form)
        self.map_box_widget.set_fov_change_callback(self._apply_fov_to_form)
        self.map_box_widget.set_map_info_callback(self._on_map_info_changed)
        self.map_box_widget.set_status_callback(self._on_selector_status_changed)
        self.map_box_widget.set_observer_info_callback(self._on_observer_info_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self._ok_button = buttons.button(QDialogButtonBox.Ok)
        self._cancel_button = buttons.button(QDialogButtonBox.Cancel)
        if self._ok_button is not None:
            self._ok_button.setText("Apply && Close")
            self._ok_button.setDefault(False)
            self._ok_button.setAutoDefault(False)
        if self._cancel_button is not None:
            self._cancel_button.setText("Close")
            self._cancel_button.setDefault(False)
            self._cancel_button.setAutoDefault(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._open_3d_button = QPushButton("Open 3D Viewer")
        self._open_3d_button.clicked.connect(self.map_box_widget.open_live_3d_viewer)
        button_row = QHBoxLayout()
        button_row.addWidget(self._open_3d_button)
        button_row.addStretch()
        button_row.addWidget(buttons)
        root.addLayout(button_row)
        self.map_box_widget.set_action_state_callback(self._on_map_action_state_changed)

    def _load_pending_session_input(self) -> None:
        if self._session_loaded:
            return
        session_input = self._pending_session_input
        if session_input is None:
            return
        self._prepare_session_input_ui(session_input)
        self._start_observer_availability_scan(session_input)

    def set_accept_button_text(self, text: str) -> None:
        """Allow wrapper entrypoints to make accept action intent explicit."""
        if self._ok_button is not None and isinstance(text, str) and text.strip():
            self._ok_button.setText(text.strip())

    def _start_observer_availability_scan(self, session_input: SelectorSessionInput) -> None:
        observer_keys = tuple(key for key, _label in self.map_box_widget.observer_options())
        thread = QThread(self)
        restored_key = self.map_box_widget.current_display_observer_key()
        worker = _ObserverAvailabilityWorker(
            session_input=session_input,
            observer_keys=observer_keys,
            skip_keys=(restored_key,),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_observer_availability_ready)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._availability_thread = thread
        self._availability_worker = worker
        thread.start()

    def _on_observer_availability_ready(self, payload: object) -> None:
        session_input = self._pending_session_input
        self._pending_session_input = None
        self._availability_thread = None
        self._availability_worker = None
        if session_input is None:
            return
        result = payload if isinstance(payload, dict) else {}
        enabled_keys = result.get("enabled_keys")
        if isinstance(enabled_keys, tuple):
            session_input.available_observer_keys = enabled_keys
        elif isinstance(enabled_keys, list):
            session_input.available_observer_keys = tuple(enabled_keys)
        else:
            session_input.available_observer_keys = None
        statuses = result.get("statuses")
        self._observer_availability_statuses = {
            normalize_observer_key(key): str(value)
            for key, value in statuses.items()
        } if isinstance(statuses, dict) else {}
        notice = str(result.get("notice", "")).strip()
        session_input.observer_availability_notice = notice or None
        self.map_box_widget.set_available_observer_keys(
            session_input.available_observer_keys,
            notice=session_input.observer_availability_notice,
        )
        self._sync_observer_buttons_from_widget()
        self._finalize_session_input_ui(session_input)
        self._session_loaded = True

    def _prepare_session_input_ui(self, session_input: SelectorSessionInput) -> None:
        self._suspend_status_updates = True
        self.map_box_widget.initialize(session_input)
        self.map_box_widget.set_geometry_edit_enabled(bool(session_input.allow_geometry_edit))
        self._sync_observer_buttons_from_widget()

        state = self.map_box_widget.state()
        context_ids = self._context_map_ids(session_input)
        bottom_ids = self._bottom_map_ids(session_input)

        self.context_map_combo.blockSignals(True)
        self.context_map_combo.clear()
        self.context_map_combo.addItem("none", None)
        for map_id in context_ids:
            self.context_map_combo.addItem(self._context_display_text(map_id), map_id)
        if state is not None and state.selected_context_id:
            idx = self.context_map_combo.findData(state.selected_context_id)
            if idx >= 0:
                self.context_map_combo.setCurrentIndex(idx)
            elif context_ids:
                self.context_map_combo.setCurrentIndex(1)
        else:
            self.context_map_combo.setCurrentIndex(0 if not context_ids else 1)
        self.context_map_combo.blockSignals(False)

        self.bottom_map_combo.blockSignals(True)
        self.bottom_map_combo.clear()
        self.bottom_map_combo.addItem("none", None)
        for map_id in bottom_ids:
            self.bottom_map_combo.addItem(map_id, map_id)
        if state is not None and state.selected_bottom_id:
            idx = self.bottom_map_combo.findData(state.selected_bottom_id)
            if idx >= 0:
                self.bottom_map_combo.setCurrentIndex(idx)
            else:
                self.bottom_map_combo.setCurrentIndex(0)
                self.map_box_widget.set_bottom_map_id(None)
        else:
            self.bottom_map_combo.setCurrentIndex(0)
            self.map_box_widget.set_bottom_map_id(None)
        self.bottom_map_combo.blockSignals(False)

        self.map_source_combo.blockSignals(True)
        idx = self.map_source_combo.findData(str(session_input.map_source_mode or "auto"))
        self.map_source_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.map_source_combo.blockSignals(False)

        self._apply_selection_to_form(session_input.geometry)
        if state is not None and state.fov is not None:
            self._apply_fov_to_form(state.fov)
        self.square_fov_box.blockSignals(True)
        self.square_fov_box.setChecked(bool(session_input.square_fov))
        self.square_fov_box.blockSignals(False)
        self.map_box_widget.set_square_fov(bool(session_input.square_fov), refresh=False)
        self._apply_fov_edit_enabled_state()
        self._apply_square_fov_state_to_form()
        self._apply_geometry_edit_enabled_state(bool(session_input.allow_geometry_edit))
        self._pending_session_input = session_input

    def _finalize_session_input_ui(self, session_input: SelectorSessionInput) -> None:
        self.map_box_widget.refresh_session_view()
        state = self.map_box_widget.state()
        if state is None or state.fov is None:
            inferred_fov = self.map_box_widget.current_fov_selection() or self.map_box_widget.projected_box_fov()
            if inferred_fov is not None:
                self.map_box_widget.set_fov_selection(inferred_fov)
        self._suspend_status_updates = False
        self.selector_status_label.setPlainText(self.map_box_widget.current_status_text())

    def _on_map_action_state_changed(self, can_open_3d: bool, _can_clear_lines: bool) -> None:
        self._open_3d_button.setEnabled(bool(can_open_3d))

    @staticmethod
    def _context_map_ids(session_input: SelectorSessionInput) -> list[str]:
        base_ids = list(session_input.map_ids or [])
        bottom_ids = set(FovBoxSelectorDialog._bottom_map_ids(session_input))
        bottom_only = {map_id for map_id in bottom_ids if map_id not in {"Bz", "Ic"}}
        bottom_only.update({"chromo_mask", "Bx", "By"})
        preferred = [
            "94", "131", "1600", "1700", "171", "193", "211", "304", "335",
            "Bz", "Ic", "B_rho", "B_theta", "B_phi", "disambig", "Vert_current",
            # Backward-compatible legacy labels.
            "Br", "Bp", "Bt",
        ]
        allowed = [map_id for map_id in base_ids if map_id not in bottom_only]
        ordered = [map_id for map_id in preferred if map_id in allowed]
        ordered.extend(map_id for map_id in allowed if map_id not in ordered)
        return ordered

    @staticmethod
    def _bottom_map_ids(session_input: SelectorSessionInput) -> list[str]:
        base_maps = dict(session_input.base_maps or {})
        out: list[str] = []

        def _display_id(base_key: str) -> str:
            key = str(base_key).lower()
            aliases = {
                "bx": "Bx",
                "by": "By",
                "bz": "Bz",
                "ic": "Ic",
                "vert_current": "Vert_current",
                "chromo_mask": "chromo_mask",
            }
            return aliases.get(key, key)

        preferred_order = ("bx", "by", "bz", "ic", "vert_current", "chromo_mask")
        for base_key in preferred_order:
            if base_key in base_maps:
                display_id = _display_id(base_key)
                if display_id not in out:
                    out.append(display_id)
        for base_key in sorted(base_maps.keys()):
            display_id = _display_id(base_key)
            if display_id not in out:
                out.append(display_id)
        return out

    @staticmethod
    def _context_display_text(map_id: str) -> str:
        return "Blos" if map_id == "Bz" else map_id

    def _on_context_map_changed(self, _index: int) -> None:
        map_id = self.context_map_combo.currentData()
        self.map_box_widget.set_context_map_id(str(map_id) if map_id else None)

    def _on_bottom_map_changed(self, _index: int) -> None:
        map_id = self.bottom_map_combo.currentData()
        self.map_box_widget.set_bottom_map_id(str(map_id) if map_id else None)

    def _on_map_source_changed(self, _index: int) -> None:
        mode = self.map_source_combo.currentData()
        self.map_box_widget.set_map_source_mode(str(mode or "auto"))

    def _on_map_info_changed(self, text: str) -> None:
        self.map_meta_label.setPlainText(text)

    def _on_selector_status_changed(self, text: str) -> None:
        if self._suspend_status_updates:
            return
        self.selector_status_label.setPlainText(text)

    def _on_observer_info_changed(self, info: dict[str, str]) -> None:
        if str(info.get("name", "")).strip().lower() == "custom":
            self._last_custom_observer_info = {key: str(value) for key, value in info.items()}
        else:
            self._last_standard_observer_info = {key: str(value) for key, value in info.items()}
        if not self._custom_observer_active:
            for key, edit in self._observer_fields.items():
                edit.setText(str(info.get(key, "")))
        self._sync_observer_buttons_from_widget()

    def _sync_observer_buttons_from_widget(self) -> None:
        current_key = self.map_box_widget.current_display_observer_key()
        enabled_keys = self.map_box_widget.observer_enabled_keys()
        if current_key == "custom":
            self._custom_observer_active = True
        elif not self._observer_custom_button.isChecked():
            self._custom_observer_active = False
        for key, button in self._observer_buttons.items():
            button.blockSignals(True)
            try:
                button.setEnabled(key in enabled_keys)
                button.setChecked((not self._custom_observer_active) and key == current_key)
            finally:
                button.blockSignals(False)
            status_label = self._observer_status_labels.get(key)
            if status_label is not None:
                status = self._observer_availability_statuses.get(normalize_observer_key(key), "")
                if key in enabled_keys or status in {"", "available", "saved"}:
                    status_label.setText("")
                elif status == "offline":
                    status_label.setText("Offline")
                else:
                    status_label.setText("Unavailable")
        self._observer_custom_button.blockSignals(True)
        try:
            self._observer_custom_button.setChecked(self._custom_observer_active or current_key == "custom")
        finally:
            self._observer_custom_button.blockSignals(False)
        self._apply_custom_observer_edit_state()

    def _next_observer_action_serial(self) -> int:
        self._observer_action_serial += 1
        return self._observer_action_serial

    def _defer_observer_action(self, serial: int, callback) -> None:
        def _run() -> None:
            if serial != self._observer_action_serial:
                return
            callback()

        QTimer.singleShot(0, _run)

    def _on_observer_radio_toggled(self, observer_key: str, checked: bool) -> None:
        if not checked:
            return
        serial = self._next_observer_action_serial()
        self._custom_observer_active = False
        self._reset_custom_mode_selection()
        self._apply_custom_observer_edit_state()
        self.update()
        self._defer_observer_action(serial, lambda: self.map_box_widget.set_display_observer_key(observer_key))

    def _on_custom_observer_toggled(self, checked: bool) -> None:
        serial = self._next_observer_action_serial()
        self._custom_observer_active = bool(checked)
        if checked:
            self._seed_custom_observer_fields()
        self._apply_custom_observer_edit_state()
        self._sync_observer_buttons_from_widget()
        self.update()
        if checked and self._custom_manual_button.isChecked():
            self._defer_observer_action(serial, self._apply_custom_manual_observer)

    def _on_custom_mode_changed(self, _checked: bool) -> None:
        serial = self._next_observer_action_serial()
        if self._custom_observer_active and self._custom_manual_button.isChecked():
            self._seed_custom_observer_fields()
        self._apply_custom_observer_edit_state()
        self.update()
        if self._custom_observer_active and self._custom_manual_button.isChecked():
            self._defer_observer_action(serial, self._apply_custom_manual_observer)
        elif self._custom_observer_active and self._custom_upload_button.isChecked():
            self._defer_observer_action(serial, self._apply_custom_upload_observer)

    def _reset_custom_mode_selection(self) -> None:
        self._custom_manual_button.blockSignals(True)
        self._custom_upload_button.blockSignals(True)
        try:
            self._custom_manual_button.setChecked(True)
        finally:
            self._custom_manual_button.blockSignals(False)
            self._custom_upload_button.blockSignals(False)

    def _current_custom_seed_info(self) -> dict[str, str]:
        if self._last_custom_observer_info:
            return self._last_custom_observer_info
        return self._last_standard_observer_info

    def _seed_custom_observer_fields(self) -> None:
        seed_info = self._current_custom_seed_info()
        if not seed_info:
            return
        for key, edit in self._observer_fields.items():
            if key in {"model_time", "obs_date", "label", "rsun_arcsec", "b0_deg", "l0_deg", "p_deg"}:
                edit.setText(seed_info.get(key, ""))
            elif not edit.text():
                edit.setText(seed_info.get(key, ""))

    def _apply_custom_observer_edit_state(self) -> None:
        custom_mode_enabled = self._custom_observer_active
        self._custom_manual_button.setEnabled(custom_mode_enabled)
        self._custom_upload_button.setEnabled(custom_mode_enabled)
        manual_edit = custom_mode_enabled and self._custom_manual_button.isChecked()
        label_edit = self._observer_fields["label"]
        label_edit.setReadOnly(not custom_mode_enabled)
        label_edit.setEnabled(custom_mode_enabled)
        for key in ("rsun_arcsec", "b0_deg", "l0_deg"):
            edit = self._observer_fields[key]
            edit.setReadOnly(not manual_edit)
            edit.setEnabled(manual_edit)

    def _apply_custom_observer_identity(self) -> None:
        if not self._custom_observer_active:
            return
        label = self._observer_fields["label"].text().strip() or "Custom"
        self._observer_fields["label"].setText(label)
        self._last_custom_observer_info["label"] = label
        self.map_box_widget.set_custom_observer_identity(label=label)

    def _apply_custom_manual_observer(self) -> None:
        if not (self._custom_observer_active and self._custom_manual_button.isChecked()):
            return
        try:
            b0_deg = float(self._observer_fields["b0_deg"].text())
            l0_deg = float(self._observer_fields["l0_deg"].text())
            rsun_arcsec = float(self._observer_fields["rsun_arcsec"].text())
        except Exception:
            return
        seed_info = self._current_custom_seed_info()
        obs_date = self._observer_fields["obs_date"].text().strip() or seed_info.get("obs_date") or None
        rsun_cm = None
        rsun_cm_text = seed_info.get("rsun_cm", "")
        if rsun_cm_text:
            try:
                rsun_cm = float(rsun_cm_text)
            except Exception:
                rsun_cm = None
        label = self._observer_fields["label"].text().strip() or seed_info.get("label") or "Custom"
        source = seed_info.get("source") or "Manual"
        self._last_custom_observer_info["label"] = label
        self._last_custom_observer_info["source"] = source
        self.map_box_widget.set_custom_display_observer_pb0r(
            b0_deg=b0_deg,
            l0_deg=l0_deg,
            rsun_arcsec=rsun_arcsec,
            obs_date=obs_date,
            rsun_cm=rsun_cm,
            label=label,
            source=source,
        )

    def _set_custom_mode_manual(self) -> None:
        self._custom_manual_button.blockSignals(True)
        self._custom_upload_button.blockSignals(True)
        try:
            self._custom_manual_button.setChecked(True)
            self._custom_upload_button.setChecked(False)
        finally:
            self._custom_manual_button.blockSignals(False)
            self._custom_upload_button.blockSignals(False)
        self._apply_custom_observer_edit_state()

    def _restore_custom_upload_fallback(self, *, message: str | None = None) -> None:
        self._set_custom_mode_manual()
        self._seed_custom_observer_fields()
        if message:
            QMessageBox.warning(self, "Upload Ref Failed", message)

    def _apply_custom_upload_observer(self) -> None:
        if not (self._custom_observer_active and self._custom_upload_button.isChecked()):
            return
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select Reference FITS or SAV File",
            str(self._entry_box_path.parent if self._entry_box_path is not None else Path.cwd()),
            (
                "Reference Files (*.fits *.fit *.fts *.fits.gz *.fit.gz *.fts.gz *.sav);;"
                "FITS Files (*.fits *.fit *.fts *.fits.gz *.fit.gz *.fts.gz);;"
                "SAV Files (*.sav);;"
                "All Files (*)"
            ),
        )
        if not selected_path:
            self._restore_custom_upload_fallback()
            return
        try:
            ephemeris, missing = observer_ephemeris_from_reference_file(selected_path)
            details = observer_reference_details_from_file(selected_path)
        except Exception as exc:
            self._restore_custom_upload_fallback(
                message=f"Could not read observer metadata from the selected reference file:\n{exc}"
            )
            return
        if missing:
            missing_text = ", ".join(missing)
            self._restore_custom_upload_fallback(
                message=(
                    "The selected reference file does not contain enough observer metadata.\n\n"
                    f"Missing cards: {missing_text}"
                )
            )
            return
        params = resolve_observer_parameters_from_ephemeris(
            ephemeris,
            observer_key="custom",
            obs_time=ephemeris.get("obs_date"),
        )
        if params is None:
            self._restore_custom_upload_fallback(
                message="The selected FITS header could not be converted into a complete observer record."
            )
            return
        field_values = {
            "model_time": str(self._session_input.time_iso or ""),
            "obs_date": str(ephemeris.get("obs_date", "")),
            "label": str(details.get("label", "") or "Custom"),
            "b0_deg": f"{float(params.get('b0_deg', 0.0)):.6f}",
            "l0_deg": f"{float(params.get('l0_deg', 0.0)):.6f}",
            "rsun_arcsec": f"{float(params.get('rsun_arcsec', 0.0)):.2f}",
            "p_deg": (
                f"{float(params['p_deg']):.6f}"
                if params.get("p_deg") is not None
                else ""
            ),
            "hgln_obs_deg": f"{float(ephemeris.get('hgln_obs_deg', 0.0)):.6f}",
            "hglt_obs_deg": f"{float(ephemeris.get('hglt_obs_deg', 0.0)):.6f}",
            "dsun_cm": f"{float(ephemeris.get('dsun_cm', 0.0)):.6e}",
            "rsun_cm": f"{float(ephemeris.get('rsun_cm', 0.0)):.6e}",
            "name": "Custom",
            "source": f"Upload Ref: {details.get('source', Path(selected_path).name)}",
        }
        self._last_custom_observer_info = dict(field_values)
        for key in ("model_time", "obs_date", "label", "b0_deg", "l0_deg", "rsun_arcsec", "p_deg"):
            self._observer_fields[key].setText(field_values.get(key, ""))
        applied = self.map_box_widget.set_custom_display_observer_pb0r(
            b0_deg=float(params["b0_deg"]),
            l0_deg=float(params["l0_deg"]),
            rsun_arcsec=float(params["rsun_arcsec"]),
            obs_date=str(ephemeris["obs_date"]),
            rsun_cm=float(ephemeris["rsun_cm"]),
            label=field_values["label"],
            source=field_values["source"],
        )
        if not applied:
            self._restore_custom_upload_fallback(
                message="The selected reference file could not be applied as a custom observer."
            )
            return
        self._set_custom_mode_manual()

    def _commit_pending_observer_state(self) -> None:
        if self._observer_custom_button.isChecked():
            self._custom_observer_active = True
            if self._custom_manual_button.isChecked():
                self._apply_custom_manual_observer()
            elif self._custom_upload_button.isChecked():
                return
            return
        self._custom_observer_active = False
        for key, button in self._observer_buttons.items():
            if button.isChecked():
                self.map_box_widget.set_display_observer_key(key)
                return

    def _apply_selection_to_form(self, selection: BoxGeometrySelection) -> None:
        self.coord_mode_combo.blockSignals(True)
        self.coord_mode_combo.setCurrentText(selection.coord_mode.value)
        self.coord_mode_combo.blockSignals(False)

        fields = selection.as_gui_text_fields()
        self.coord_x_edit.setText(fields["coord_x_edit"])
        self.coord_y_edit.setText(fields["coord_y_edit"])
        self.grid_x_edit.setText(fields["grid_x_edit"])
        self.grid_y_edit.setText(fields["grid_y_edit"])
        self.grid_z_edit.setText(fields["grid_z_edit"])
        self.res_edit.setText(fields["res_edit"])

    def _push_form_to_view_state(self) -> None:
        try:
            if self.square_fov_box.isChecked():
                self.fov_h_edit.setText(self.fov_w_edit.text())
            selection = self._selection_from_form() if self._session_input.allow_geometry_edit else None
            fov = self._fov_from_form() if self._fov_edit_enabled() else None
        except ValueError:
            return
        state = self.map_box_widget.state()
        if selection is not None and (state is None or state.geometry != selection):
            self.map_box_widget.set_geometry_selection(selection)
        if fov is not None and (state is None or state.fov != fov):
            self.map_box_widget.set_fov_selection(fov)

    def _apply_geometry_edit_enabled_state(self, enabled: bool) -> None:
        editable = bool(enabled)
        self.coord_mode_combo.setEnabled(editable)
        for widget in (
            self.coord_x_edit,
            self.coord_y_edit,
            self.grid_x_edit,
            self.grid_y_edit,
            self.grid_z_edit,
            self.res_edit,
        ):
            widget.setEnabled(editable)

    def _fov_edit_enabled(self) -> bool:
        return self._entry_box_path is not None

    def _apply_fov_edit_enabled_state(self) -> None:
        editable = self._fov_edit_enabled()
        for widget in (
            self.fov_x_edit,
            self.fov_y_edit,
            self.fov_w_edit,
        ):
            widget.setEnabled(editable)
        self.square_fov_box.setEnabled(editable)
        self.fov_h_edit.setEnabled(editable and not self.square_fov_box.isChecked())

    def _apply_square_fov_state_to_form(self) -> None:
        square = self.square_fov_box.isChecked()
        if square:
            self.fov_h_edit.setText(self.fov_w_edit.text())
        self.fov_h_edit.setEnabled(self._fov_edit_enabled() and not square)

    def _on_square_fov_toggled(self, checked: bool) -> None:
        self.map_box_widget.set_square_fov(bool(checked))
        self._apply_square_fov_state_to_form()
        self._push_form_to_view_state()

    def _selection_from_form(self) -> BoxGeometrySelection:
        return BoxGeometrySelection(
            coord_mode=CoordMode(self.coord_mode_combo.currentText()),
            coord_x=float(self.coord_x_edit.text()),
            coord_y=float(self.coord_y_edit.text()),
            grid_x=int(float(self.grid_x_edit.text())),
            grid_y=int(float(self.grid_y_edit.text())),
            grid_z=int(float(self.grid_z_edit.text())),
            dx_km=float(self.res_edit.text()),
        )

    def _fov_from_form(self) -> DisplayFovSelection:
        return DisplayFovSelection(
            center_x_arcsec=float(self.fov_x_edit.text()),
            center_y_arcsec=float(self.fov_y_edit.text()),
            width_arcsec=max(1e-3, float(self.fov_w_edit.text())),
            height_arcsec=max(1e-3, float(self.fov_h_edit.text())),
        )

    def _apply_fov_to_form(self, selection: DisplayFovSelection) -> None:
        fields = selection.as_gui_text_fields()
        self.fov_x_edit.setText(fields["fov_x_edit"])
        self.fov_y_edit.setText(fields["fov_y_edit"])
        self.fov_w_edit.setText(fields["fov_w_edit"])
        self.fov_h_edit.setText(fields["fov_w_edit"] if self.square_fov_box.isChecked() else fields["fov_h_edit"])

    def eventFilter(self, obj, event):
        # Prevent Enter/Return in input fields from closing the dialog through the default button.
        # Instead, treat Enter as "apply field edit" only.
        if obj in {
            self.coord_x_edit,
            self.coord_y_edit,
            self.grid_x_edit,
            self.grid_y_edit,
            self.grid_z_edit,
            self.res_edit,
            self.fov_x_edit,
            self.fov_y_edit,
            self.fov_w_edit,
            self.fov_h_edit,
        } and event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._push_form_to_view_state()
            return True
        if obj in {
            self._observer_fields["b0_deg"],
            self._observer_fields["l0_deg"],
            self._observer_fields["rsun_arcsec"],
        } and event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._apply_custom_manual_observer()
            return True
        return super().eventFilter(obj, event)

    def accept(self) -> None:
        self._push_form_to_view_state()
        self._commit_pending_observer_state()
        self._accepted_selection = SelectorDialogResult(
            geometry=self._selection_from_form(),
            fov=self._fov_from_form(),
            square_fov=bool(self.square_fov_box.isChecked()),
        )
        super().accept()

    def accepted_selection(self) -> Optional[SelectorDialogResult]:
        return self._accepted_selection

    def committed_line_seeds(self):
        return self.map_box_widget.committed_line_seeds()

    def current_fov_box_selection(self):
        return self.map_box_widget.current_fov_box_selection()

    def current_observer_persistence_state(self):
        return self.map_box_widget.current_observer_persistence_state()


def run_fov_box_selector(
    session_input: SelectorSessionInput,
    parent: Optional[QWidget] = None,
    entry_box_path: Optional[str | Path] = None,
) -> Optional[SelectorDialogResult]:
    dialog = FovBoxSelectorDialog(session_input=session_input, parent=parent, entry_box_path=entry_box_path)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.accepted_selection()
    return None
