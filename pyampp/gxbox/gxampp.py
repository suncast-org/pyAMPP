import sys
import os
import queue
import re
import shlex
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
                             QRadioButton,
                             QCheckBox, QGridLayout, QGroupBox, QButtonGroup, QVBoxLayout, QHBoxLayout, QDateTimeEdit,
                             QCalendarWidget, QTextEdit, QMessageBox, QDockWidget, QToolButton, QMenu,
                             QFileDialog, QStyle)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QDateTime, Qt, QTimer, QSettings, QSize
from datetime import datetime
from PyQt5 import uic

from pyampp.util.config import *
import pyampp
from pathlib import Path
from pyampp.gxbox.boxutils import read_b3d_h5, validate_number
from pyampp.gxbox.gx_fov2box import (
    _load_entry_box_any,
    _entry_stage_from_loaded,
    _extract_execute_paths,
)
from pyampp.gxbox.gxbox_selector_view import _discover_external_ref_map_files, _discover_filesystem_maps
from pyampp.gxbox.fov_selector_gui import run_fov_box_selector
from pyampp.gxbox.selector_api import (
    BoxGeometrySelection,
    CoordMode,
    DisplayFovSelection,
    SelectorDialogResult,
    SelectorSessionInput,
)
from pyampp.util.idl_execute_to_gxfov2box import _parse_idl_call, _build_gx_fov2box_command
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from sunpy.coordinates import get_earth, HeliographicStonyhurst, HeliographicCarrington, Helioprojective
from sunpy.sun import constants as sun_consts
import numpy as np
import typer
import subprocess

app = typer.Typer(help="Launch the PyAmpp application.")

base_dir = Path(pyampp.__file__).parent


def _split_process_output_text(partial_line: str, text: str) -> tuple[list[str], str]:
    merged = partial_line + text.replace("\r\n", "\n").replace("\r", "\n")
    if merged.endswith("\n"):
        return merged.split("\n")[:-1], ""
    parts = merged.split("\n")
    return parts[:-1], parts[-1]


def _contains_entry_field_payload(boxdata: dict) -> bool:
    if not isinstance(boxdata, dict):
        return False

    for key in ("corona", "nlfff", "pot"):
        group = boxdata.get(key)
        if isinstance(group, dict) and any(name in group for name in ("bx", "by", "bz", "bcube")):
            return True

    chromo = boxdata.get("chromo")
    if isinstance(chromo, dict):
        if any(name in chromo for name in ("bx", "by", "bz", "bcube", "chromo_bcube")):
            return True

    return False


def _has_template_execute_metadata(boxdata: dict) -> bool:
    if not isinstance(boxdata, dict):
        return False
    metadata = boxdata.get("metadata")
    if not isinstance(metadata, dict):
        return False
    execute = metadata.get("execute")
    if execute is None:
        return False
    if isinstance(execute, (bytes, bytearray)):
        execute = execute.decode("utf-8", "ignore")
    return bool(str(execute).strip())


class CustomQLineEdit(QLineEdit):
    def setTextL(self, text):
        """
        Sets the text of the QLineEdit and moves the cursor to the beginning.

        :param text: str
            The text to set.
        """
        self.setText(text)
        self.setCursorPosition(0)


class PyAmppGUI(QMainWindow):
    """
    Main application GUI for the Solar Data Model.

    This class creates the main window and sets up the user interface for managing solar data and model configurations.

    Attributes
    ----------
    central_widget : QWidget
        The central widget of the main window.
    main_layout : QVBoxLayout
        The main layout for the central widget.

    Methods
    -------
    initUI():
        Initializes the user interface.
    add_data_repository_section():
        Adds the data repository section to the UI.
    update_sdo_data_dir():
        Updates the SDO data directory path.
    update_gxmodel_dir():
        Updates the GX model directory path.
    update_external_box_dir():
        Updates the external box directory path.
    update_dir(new_path, default_path):
        Updates the specified directory path.
    open_sdo_file_dialog():
        Opens a file dialog for selecting the SDO data directory.
    open_gx_file_dialog():
        Opens a file dialog for selecting the GX model directory.
    open_external_file_dialog():
        Opens a file dialog for selecting the external box directory.
    add_model_configuration_section():
        Adds the model configuration section to the UI.
    add_options_section():
        Adds the options section to the UI.
    add_cmd_display():
        Adds the command display section to the UI.
    add_cmd_buttons():
        Adds command buttons to the UI.
    add_status_log():
        Adds the status log section to the UI.
    update_command_display():
        Updates the command display with the current command.
    update_hpc_state(checked):
        Updates the UI when Helioprojective coordinates are selected.
    update_hgc_state(checked):
        Updates the UI when Heliographic Carrington coordinates are selected.
    get_command():
        Constructs the command based on the current UI settings.
    execute_command():
        Executes the constructed command.
    save_command():
        Saves the current command.
    refresh_command():
        Refreshes the current session.
    clear_command():
        Clears the status log.
    """

    def __init__(self):
        """
        Initializes the PyAmppGUI class.
        """
        super().__init__()
        self._gxbox_proc = None
        self._view2d_proc = None
        self._view2d_target_path = None
        self._view2d_adopt_on_close = False
        self._view2d_launch_pending = False
        self._view3d_proc = None
        self._view3d_target_path = None
        self._view3d_launch_pending = False
        self._proc_output_queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._proc_reader_thread = None
        self._proc_partial_line = ""
        self._proc_command = None
        self._pending_stop_after = None
        self._selector_fov: DisplayFovSelection | None = None
        self._selector_square_fov = False
        self._selector_observer_name = "earth"
        self._selector_unsaved_session_active = False
        self.info_only_box = None
        self._last_model_path = None
        self._last_valid_entry_box = ""
        self._entry_stage_detected = None
        self._entry_type_detected = None
        self._entry_is_template_only = False
        self._suppress_entry_dialog_on_restore = False
        self._hydrating_entry = False
        self._proc_timer = QTimer(self)
        self._proc_timer.setInterval(500)
        self._proc_timer.timeout.connect(self._check_gxbox_process)
        self._view2d_timer = QTimer(self)
        self._view2d_timer.setInterval(500)
        self._view2d_timer.timeout.connect(self._check_view2d_process)
        self._view3d_timer = QTimer(self)
        self._view3d_timer.setInterval(500)
        self._view3d_timer.timeout.connect(self._check_view3d_process)
        self._settings = QSettings("SUNCAST", "pyAMPP")
        self.model_time_orig = None
        # self.rotate_to_time_button = None
        self.rotate_revert_button = None
        self.coords_center = None
        self.coords_center_orig = None
        self.initUI()

    @staticmethod
    def _default_test_model_state():
        return {
            "time_iso": "2025-11-26T15:34:31",
            "coord_mode": "hpc",
            "coord_x": "-280.0",
            "coord_y": "-250.0",
            "projection": "cea",
            "grid_x": "150",
            "grid_y": "75",
            "grid_z": "150",
            "dx_km": "1400.000",
            "pad_percent": "10",
        }

    def _has_entry_box(self) -> bool:
        return bool(self.external_box_edit.text().strip()) and not bool(self._entry_is_template_only)

    def _stage_to_jump_action(self, stage: str) -> str:
        s = (stage or "").upper()
        return {
            "NONE": "none",
            "POT": "potential",
            "BND": "bounds",
            "NAS": "nlfff",
            "GEN": "lines",
            "CHR": "chromo",
        }.get(s, "none")

    def _set_model_params_enabled(self, enabled: bool) -> None:
        widgets = [
            self.model_time_edit,
            self.coord_x_edit,
            self.coord_y_edit,
            self.hpc_radio_button,
            self.hgc_radio_button,
            self.hgs_radio_button,
            self.proj_cea_radio,
            self.proj_top_radio,
            self.grid_x_edit,
            self.grid_y_edit,
            self.grid_z_edit,
            self.res_edit,
            self.padding_size_edit,
            self.disambig_hmi_radio,
            self.disambig_sfq_radio,
        ]
        for w in widgets:
            w.setEnabled(enabled)

    def initUI(self):
        """
        Sets up the initial user interface for the main window.
        """
        # Main widget and layout
        uic.loadUi(Path(__file__).parent / "UI" / "gxampp.ui", self)
        self.setWindowTitle("GX Automatic Production Pipeline Interface")

        # Adding different sections
        self.add_data_repository_section()
        self.add_model_configuration_section()
        self.add_options_section()
        self.add_cmd_display()
        self.add_cmd_buttons()
        self.add_status_log()
        self._restore_or_apply_default_session_state()
        self._refresh_after_session_state_apply()
        self.show()

    def _restore_or_apply_default_session_state(self):
        """
        Restore last-used GUI state from QSettings if present.
        Otherwise, initialize a known-good test configuration.
        """
        try:
            has_saved = bool(self._settings.value("session/model_time_iso", "", type=str).strip())
            if has_saved:
                self._restore_session_state_from_settings()
            else:
                self._apply_default_test_model_state()
        except Exception as exc:
            # Fall back to deterministic defaults if restore fails.
            self.status_log_edit.append(f"Settings restore warning: {exc}")
            self._apply_default_test_model_state()

    def _apply_default_test_model_state(self):
        cfg = self._default_test_model_state()
        dt = datetime.strptime(cfg["time_iso"], "%Y-%m-%dT%H:%M:%S")
        self.model_time_edit.setDateTime(QDateTime(dt))
        self.coord_x_edit.setText(cfg["coord_x"])
        self.coord_y_edit.setText(cfg["coord_y"])
        self.grid_x_edit.setText(cfg["grid_x"])
        self.grid_y_edit.setText(cfg["grid_y"])
        self.grid_z_edit.setText(cfg["grid_z"])
        self.res_edit.setText(cfg["dx_km"])
        self.padding_size_edit.setText(cfg["pad_percent"])
        self.hpc_radio_button.setChecked(cfg["coord_mode"] == "hpc")
        self.hgc_radio_button.setChecked(cfg["coord_mode"] == "hgc")
        self.hgs_radio_button.setChecked(cfg["coord_mode"] == "hgs")
        self.proj_cea_radio.setChecked(cfg["projection"] == "cea")
        self.proj_top_radio.setChecked(cfg["projection"] == "top")
        self._set_download_backend("drms")
        self._set_use_cached_downloads(True)

    def _restore_session_state_from_settings(self):
        # Core geometry / coordinate state
        time_iso = self._settings.value("session/model_time_iso", "", type=str)
        if time_iso:
            try:
                dt = datetime.strptime(time_iso, "%Y-%m-%dT%H:%M:%S")
                self.model_time_edit.setDateTime(QDateTime(dt))
            except Exception:
                pass

        self.coord_x_edit.setText(self._settings.value("session/coord_x", self.coord_x_edit.text(), type=str))
        self.coord_y_edit.setText(self._settings.value("session/coord_y", self.coord_y_edit.text(), type=str))
        self.grid_x_edit.setText(self._settings.value("session/grid_x", self.grid_x_edit.text(), type=str))
        self.grid_y_edit.setText(self._settings.value("session/grid_y", self.grid_y_edit.text(), type=str))
        self.grid_z_edit.setText(self._settings.value("session/grid_z", self.grid_z_edit.text(), type=str))
        self.res_edit.setText(self._settings.value("session/dx_km", self.res_edit.text(), type=str))
        self.padding_size_edit.setText(self._settings.value("session/pad_percent", self.padding_size_edit.text(), type=str))

        coord_mode = self._settings.value("session/coord_mode", "hpc", type=str).lower()
        self.hpc_radio_button.setChecked(coord_mode == "hpc")
        self.hgc_radio_button.setChecked(coord_mode == "hgc")
        self.hgs_radio_button.setChecked(coord_mode == "hgs")

        projection = self._settings.value("session/projection", "cea", type=str).lower()
        self.proj_cea_radio.setChecked(projection == "cea")
        self.proj_top_radio.setChecked(projection == "top")
        self._set_download_backend(self._settings.value("session/download_backend", "drms", type=str))
        self._set_use_cached_downloads(self._settings.value("session/use_cached_downloads", True, type=bool))

        # Entry mode
        self._set_jump_action(self._settings.value("session/entry_mode", "continue", type=str))

        # Workflow toggles (best effort; names map to widget attrs)
        bool_widgets = {
            "download_aia_uv": self.download_aia_uv,
            "download_aia_euv": self.download_aia_euv,
            "stop_early": self.stop_early_box,
            "save_empty": self.save_empty_box,
            "save_potential": self.save_potential_box,
            "save_bounds": self.save_bounds_box,
            "save_nas": self.save_nas_box,
            "save_gen": self.save_gen_box,
            "stop_none": self.empty_box_only_box,
            "stop_pot": self.stop_after_potential_box,
            "stop_nas": self.nlfff_only_box,
            "stop_gen": self.generic_only_box,
            "skip_nlfff": self.skip_nlfff_extrapolation,
            "skip_lines": self.skip_line_computation_box,
            "center_vox": self.center_vox_box,
            "save_chromo": self.add_save_chromo_box,
            "disambig_sfq": self.disambig_sfq_radio,
        }
        for key, widget in bool_widgets.items():
            try:
                value = self._settings.value(f"session/{key}", widget.isChecked(), type=bool)
                widget.setChecked(bool(value))
            except Exception:
                pass

        # Optional persisted entry box path (kept separate from data/model dir persistence)
        entry_box = self._settings.value("session/entry_box_path", "", type=str).strip()
        if entry_box:
            self.external_box_edit.setText(entry_box)
            # Re-trigger entry box processing to rebuild entry type and pipeline state
            # Suppress the template dialog on restore since it's not a new user action
            try:
                self._suppress_entry_dialog_on_restore = True
                self.read_external_box()
            except Exception:
                # If entry box path is stale or invalid, silently ignore
                # User can fix it manually
                self._entry_is_template_only = False
            finally:
                self._suppress_entry_dialog_on_restore = False

    def _save_session_state_to_settings(self):
        try:
            dt = self.model_time_edit.dateTime().toPyDateTime()
            self._settings.setValue("session/model_time_iso", dt.strftime("%Y-%m-%dT%H:%M:%S"))
            self._settings.setValue("session/coord_x", self.coord_x_edit.text().strip())
            self._settings.setValue("session/coord_y", self.coord_y_edit.text().strip())
            self._settings.setValue("session/grid_x", self.grid_x_edit.text().strip())
            self._settings.setValue("session/grid_y", self.grid_y_edit.text().strip())
            self._settings.setValue("session/grid_z", self.grid_z_edit.text().strip())
            self._settings.setValue("session/dx_km", self.res_edit.text().strip())
            self._settings.setValue("session/pad_percent", self.padding_size_edit.text().strip())
            self._settings.setValue("session/coord_mode", self._current_coord_mode().value)
            self._settings.setValue("session/projection", "cea" if self.proj_cea_radio.isChecked() else "top")
            self._settings.setValue("session/download_backend", self._selected_download_backend())
            self._settings.setValue("session/use_cached_downloads", self._use_cached_downloads())
            self._settings.setValue("session/entry_mode", self._get_jump_action())
            self._settings.setValue("session/entry_box_path", self.external_box_edit.text().strip())
            self._settings.setValue("paths/refmaps_path", self.refmaps_path_edit.text().strip())
            self._settings.setValue("session/entry_is_template_only", self._entry_is_template_only)

            bool_widgets = {
                "download_aia_uv": self.download_aia_uv,
                "download_aia_euv": self.download_aia_euv,
                "stop_early": self.stop_early_box,
                "save_empty": self.save_empty_box,
                "save_potential": self.save_potential_box,
                "save_bounds": self.save_bounds_box,
                "save_nas": self.save_nas_box,
                "save_gen": self.save_gen_box,
                "stop_none": self.empty_box_only_box,
                "stop_pot": self.stop_after_potential_box,
                "stop_nas": self.nlfff_only_box,
                "stop_gen": self.generic_only_box,
                "skip_nlfff": self.skip_nlfff_extrapolation,
                "skip_lines": self.skip_line_computation_box,
                "center_vox": self.center_vox_box,
                "save_chromo": self.add_save_chromo_box,
                "disambig_sfq": self.disambig_sfq_radio,
            }
            for key, widget in bool_widgets.items():
                self._settings.setValue(f"session/{key}", bool(widget.isChecked()))
            self._settings.sync()
        except Exception:
            # Settings persistence should never break GUI actions.
            pass

    def _refresh_after_session_state_apply(self):
        try:
            self.update_coords_center()
        except Exception:
            pass
        self._refresh_viewer_button_state()
        self._sync_pipeline_options()
        self.update_command_display()

    def on_reset_to_test_defaults_clicked(self):
        self._apply_default_test_model_state()
        self._refresh_after_session_state_apply()
        self.status_log_edit.append("Reset GUI fields to test-model defaults.")

    def on_restore_last_saved_clicked(self):
        if not bool(self._settings.value("session/model_time_iso", "", type=str).strip()):
            self.status_log_edit.append("No saved GUI session found; applying test-model defaults.")
            self._apply_default_test_model_state()
        else:
            self._restore_session_state_from_settings()
            self.status_log_edit.append("Restored last saved GUI session.")
        self._refresh_after_session_state_apply()

    def on_open_fov_selector_clicked(self):
        if self._selector_unsaved_session_active:
            self._launch_download_fov_selector_placeholder()
            return
        model_path = self._current_viewable_model_path()
        if model_path is not None:
            self._launch_box_view2d(model_path)
            return
        self._launch_download_fov_selector_placeholder()

    def closeEvent(self, event):
        self._save_session_state_to_settings()
        super().closeEvent(event)

    def add_data_repository_section(self):
        layout = self.data_repository_section.layout()
        if layout is not None:
            layout.setColumnStretch(0, 0)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 0)
            spacer = layout.itemAtPosition(3, 2)
            if spacer is not None and spacer.widget() is None:
                layout.removeItem(spacer)
        self.sdo_data_edit.setText(self._settings.value("paths/data_dir", DOWNLOAD_DIR, type=str))
        self.sdo_data_edit.setMinimumWidth(520)
        self.sdo_data_edit.returnPressed.connect(self.update_sdo_data_dir)
        self.sdo_data_edit.textChanged.connect(self._persist_data_dir)
        self.sdo_browse_button.clicked.connect(self.open_sdo_file_dialog)
        self.sdo_browse_button.setText("")
        self.sdo_browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.sdo_browse_button.setToolTip("Select SDO data repository")
        self.sdo_browse_button.setFixedWidth(28)

        self.gx_model_edit.setText(self._settings.value("paths/gxmodel_dir", GXMODEL_DIR, type=str))
        self.gx_model_edit.setMinimumWidth(520)
        self.gx_model_edit.returnPressed.connect(self.update_gxmodel_dir)
        self.gx_model_edit.textChanged.connect(self._persist_gxmodel_dir)
        self.gx_browse_button.clicked.connect(self.open_gx_file_dialog)
        self.gx_browse_button.setText("")
        self.gx_browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.gx_browse_button.setToolTip("Select GX model repository")
        self.gx_browse_button.setFixedWidth(28)

        notify_email = os.environ.get("PYAMPP_JSOC_NOTIFY_EMAIL", JSOC_NOTIFY_EMAIL)
        self.jsoc_notify_email_edit.setText(notify_email)
        self.jsoc_notify_email_edit.returnPressed.connect(self.update_jsoc_notify_email)

        self.external_box_edit.setMinimumWidth(520)
        self.external_box_edit.returnPressed.connect(self.update_external_box_dir)
        self.external_browse_button.clicked.connect(self.open_external_file_dialog)
        self.external_browse_button.setText("")
        self.external_browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.external_browse_button.setToolTip("Select entry model file (.h5/.sav)")
        self.external_browse_button.setFixedWidth(28)
        if layout is not None:
            layout.addWidget(self.label_external_box, 3, 0)
            layout.addWidget(self.external_box_edit, 3, 1)
            layout.addWidget(self.external_browse_button, 3, 2)

        if layout is not None:
            layout.addWidget(self.label_jsoc_notify_email, 4, 0)
            layout.addWidget(self.jsoc_notify_email_edit, 4, 1)

        self.refmaps_path_label = QLabel("Reference Map Path:")
        self.refmaps_path_edit = QLineEdit()
        self.refmaps_path_edit.setText(self._settings.value("paths/refmaps_path", "", type=str))
        self.refmaps_path_edit.setMinimumWidth(520)
        self.refmaps_path_edit.setToolTip("Optional FITS file or directory of external reference maps")
        self.refmaps_path_edit.returnPressed.connect(self.update_refmaps_path)
        self.refmaps_path_edit.textChanged.connect(self._persist_refmaps_path)
        self.refmaps_browse_button = QPushButton()
        self.refmaps_browse_button.clicked.connect(self.open_refmaps_path_dialog)
        self.refmaps_browse_button.setText("")
        self.refmaps_browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.refmaps_browse_button.setToolTip("Select external reference-map FITS directory")
        self.refmaps_browse_button.setFixedWidth(28)
        if layout is not None:
            layout.addWidget(self.refmaps_path_label, 2, 0)
            layout.addWidget(self.refmaps_path_edit, 2, 1)
            layout.addWidget(self.refmaps_browse_button, 2, 2)

        self.entry_stage_label = QLabel("Detected Entry Type:")
        self.entry_stage_edit = QLineEdit("N/A")
        self.entry_stage_edit.setReadOnly(True)
        self.entry_stage_edit.setFixedWidth(130)
        self.entry_stage_edit.setToolTip("Detected entry type, e.g. POT.GEN.SAV or NAS.CHR.H5")
        self.continue_radio = QRadioButton("Continue")
        self.rebuild_none_radio = QRadioButton("Rebuild from NONE")
        self.rebuild_obs_radio = QRadioButton("Rebuild from OBS")
        self.modify_radio = QRadioButton("Modify")
        self.continue_radio.setChecked(True)
        self.continue_radio.toggled.connect(self._sync_pipeline_options)
        self.rebuild_none_radio.toggled.connect(self._sync_pipeline_options)
        self.rebuild_obs_radio.toggled.connect(self._sync_pipeline_options)
        self.modify_radio.toggled.connect(self._sync_pipeline_options)
        self.entry_mode_widget = QWidget()
        mode_layout = QHBoxLayout(self.entry_mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(self.entry_stage_edit)
        mode_layout.addWidget(self.continue_radio)
        mode_layout.addWidget(self.rebuild_none_radio)
        mode_layout.addWidget(self.rebuild_obs_radio)
        mode_layout.addWidget(self.modify_radio)
        mode_layout.addStretch()
        if layout is not None:
            layout.addWidget(self.entry_stage_label, 5, 0)
            layout.addWidget(self.entry_mode_widget, 5, 1, 1, 2)

    def update_sdo_data_dir(self):
        """
        Updates the SDO data directory path based on the user input.
        """
        new_path = self.sdo_data_edit.text()
        self.update_dir(new_path, DOWNLOAD_DIR, self.sdo_data_edit)
        self._persist_data_dir(self.sdo_data_edit.text())
        self.update_command_display()

    def update_gxmodel_dir(self):
        """
        Updates the GX model directory path based on the user input.
        """
        new_path = self.gx_model_edit.text()
        self.update_dir(new_path, GXMODEL_DIR, self.gx_model_edit)
        self._persist_gxmodel_dir(self.gx_model_edit.text())
        self.update_command_display()

    def _persist_data_dir(self, text):
        self._settings.setValue("paths/data_dir", (text or "").strip())

    def _persist_gxmodel_dir(self, text):
        self._settings.setValue("paths/gxmodel_dir", (text or "").strip())

    def update_refmaps_path(self):
        self._persist_refmaps_path(self.refmaps_path_edit.text())
        self.update_command_display()

    def _persist_refmaps_path(self, text):
        self._settings.setValue("paths/refmaps_path", (text or "").strip())

    def update_jsoc_notify_email(self):
        """
        Updates the JSOC notify email via the PYAMPP_JSOC_NOTIFY_EMAIL environment variable.
        """
        new_email = self.jsoc_notify_email_edit.text().strip()
        if new_email:
            os.environ["PYAMPP_JSOC_NOTIFY_EMAIL"] = new_email
        else:
            os.environ.pop("PYAMPP_JSOC_NOTIFY_EMAIL", None)
            self.jsoc_notify_email_edit.setText(JSOC_NOTIFY_EMAIL)

    def read_external_box(self):
        """
        Reads the external box path based on the user input.
        """
        boxfile = self.external_box_edit.text()
        self._hydrating_entry = True
        try:
            boxdata = _load_entry_box_any(Path(boxfile))
            has_payload = _contains_entry_field_payload(boxdata)
            template_only = (not has_payload)
            if template_only and not _has_template_execute_metadata(boxdata):
                raise ValueError(
                    "Entry box is incompatible: missing 3D field payload (corona/chromo) "
                    "and no metadata.execute template was found."
                )
            self._entry_is_template_only = template_only
            entry_stage = _entry_stage_from_loaded(boxdata, Path(boxfile))
            self._entry_stage_detected = entry_stage
            entry_type = self._derive_entry_type(boxdata, Path(boxfile), entry_stage)
            if self._entry_is_template_only:
                entry_type = f"{entry_type}.TEMPLATE"
            self._entry_type_detected = entry_type
            self.entry_stage_edit.setText(entry_type)

            execute_text = self._decode_meta_value(boxdata.get("metadata", {}).get("execute", ""))
            self._apply_execute_defaults(execute_text, boxdata)
            # Canonical time comes from entry model identity (id/path), not stale execute text.
            # This guarantees "Rebuild from OBS" uses the uploaded model timestamp.
            entry_dt = self._infer_entry_datetime(boxdata, Path(boxfile))
            if entry_dt is not None:
                self.model_time_edit.setDateTime(QDateTime(entry_dt))
            exec_data_dir, exec_model_dir = _extract_execute_paths(execute_text)
            warnings = []
            if exec_data_dir:
                p = Path(exec_data_dir).expanduser()
                if p.exists():
                    self.sdo_data_edit.setText(str(p))
                else:
                    self.sdo_data_edit.setText(DOWNLOAD_DIR)
                    warnings.append(f"Invalid execute data-dir on this system, using default: {DOWNLOAD_DIR}")
            if exec_model_dir:
                p = Path(exec_model_dir).expanduser()
                if p.exists():
                    self.gx_model_edit.setText(str(p))
                else:
                    self.gx_model_edit.setText(GXMODEL_DIR)
                    warnings.append(f"Invalid execute gxmodel-dir on this system, using default: {GXMODEL_DIR}")
            self._persist_data_dir(self.sdo_data_edit.text())
            self._persist_gxmodel_dir(self.gx_model_edit.text())

            # Keep command state predictable when importing an entry box.
            self._reset_pipeline_checks_for_entry()
            self._set_jump_action("continue")
            self._last_valid_entry_box = boxfile
            if self._entry_is_template_only and not self._suppress_entry_dialog_on_restore:
                QMessageBox.information(
                    self,
                    "Template Entry Loaded",
                    "Loaded metadata-only entry template.\n\n"
                    "pyAMPP will use metadata.execute fields to prefill GUI parameters and build a fresh model from scratch.\n"
                    "Resume/continue-from-box mode is disabled for this template.",
                )
            if warnings:
                QMessageBox.warning(self, "Entry Box Path Warnings", "\n".join(warnings))
        finally:
            self._hydrating_entry = False
        self._set_model_params_enabled(not self._has_entry_box())
        self._sync_pipeline_options()
        self.update_command_display()

    def _derive_entry_type(self, boxdata: dict, entry_path: Path, entry_stage: str) -> str:
        meta_id = self._decode_meta_value(boxdata.get("metadata", {}).get("id", "")).strip()
        stage_path = entry_stage
        if ".CEA." in meta_id:
            stage_path = meta_id.split(".CEA.", 1)[1]
        elif ".TOP." in meta_id:
            stage_path = meta_id.split(".TOP.", 1)[1]
        stage_path = self._infer_stage_path_from_content(boxdata, stage_path)
        suffix = entry_path.suffix.lower()
        file_tag = "SAV" if suffix == ".sav" else "H5"
        return f"{stage_path}.{file_tag}".upper()

    @staticmethod
    def _infer_entry_datetime(boxdata: dict, entry_path: Path):
        """
        Infer observation datetime from metadata id or filename token YYYYMMDD_HHMMSS.
        """
        meta = boxdata.get("metadata", {}) if isinstance(boxdata, dict) else {}
        meta_id = PyAmppGUI._decode_meta_value(meta.get("id", "")).strip()
        candidates = [meta_id, entry_path.stem, entry_path.name]
        for text in candidates:
            m = re.search(r"(\d{8}_\d{6})", str(text))
            if not m:
                continue
            try:
                return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            except Exception:
                continue
        return None

    @staticmethod
    def _has_lines_metadata(boxdata: dict) -> bool:
        required = {"codes", "apex_idx", "start_idx", "end_idx", "seed_idx", "av_field", "phys_length", "voxel_status"}
        lines = boxdata.get("lines")
        if isinstance(lines, dict) and required.issubset(set(lines.keys())):
            return True
        # Backward compatibility: old files stored line arrays under chromo.
        chromo = boxdata.get("chromo")
        if isinstance(chromo, dict) and required.issubset(set(chromo.keys())):
            return True
        return False

    def _infer_stage_path_from_content(self, boxdata: dict, stage_path_hint: str) -> str:
        hint = (stage_path_hint or "").upper()
        corona = boxdata.get("corona", {}) if isinstance(boxdata, dict) else {}
        model_type = ""
        if isinstance(corona, dict):
            attrs = corona.get("attrs", {})
            if isinstance(attrs, dict):
                model_type = str(attrs.get("model_type", "")).strip().lower()
        has_lines = self._has_lines_metadata(boxdata)
        has_chromo = isinstance(boxdata.get("chromo"), dict)

        # First derive POT/NAS branch from data, fallback to hint.
        if model_type == "pot" or hint.startswith("POT"):
            prefix = "POT"
        elif model_type in ("nlfff", "nas") or hint.startswith("NAS"):
            prefix = "NAS"
        elif hint.startswith("BND") or model_type in ("bnd", "bounds"):
            return "BND"
        elif hint.startswith("NONE") or model_type == "none":
            return "NONE"
        else:
            prefix = "NAS"

        # Then derive stage detail from content (not ID text).
        if has_chromo:
            return f"{prefix}.GEN.CHR" if has_lines else f"{prefix}.CHR"
        if has_lines:
            return f"{prefix}.GEN"
        if prefix == "POT":
            return "POT"
        return "NAS"

    @staticmethod
    def _decode_meta_value(v):
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", errors="ignore")
        if hasattr(v, "item"):
            try:
                vv = v.item()
                if isinstance(vv, (bytes, bytearray)):
                    return vv.decode("utf-8", errors="ignore")
                return str(vv)
            except Exception:
                pass
        return str(v or "")

    def _apply_execute_defaults(self, execute_text: str, boxdata: dict):
        """
        Populate GUI fields from entry model metadata/execute.
        """
        parsed_cmd = []
        text = (execute_text or "").strip()
        if text:
            # Native Python execute string stored in metadata.
            if text.startswith("gx-fov2box") or " --" in text:
                try:
                    parsed_cmd = shlex.split(text)
                except Exception:
                    parsed_cmd = []
            # IDL execute string fallback.
            if not parsed_cmd:
                try:
                    _, kw = _parse_idl_call(text)
                    parsed = _build_gx_fov2box_command(kw)
                    parsed_cmd = parsed.command
                except Exception:
                    parsed_cmd = []

        # Parse translated command tokens into a simple option map.
        flag_arity = {
            "--time": 1,
            "--coords": 2,
            "--box-dims": 3,
            "--dx-km": 1,
            "--pad-frac": 1,
            "--data-dir": 1,
            "--gxmodel-dir": 1,
            "--refmaps-path": 1,
            "--refmap-path": 1,
        }
        opts = {}
        i = 0
        while i < len(parsed_cmd):
            tok = parsed_cmd[i]
            if tok in flag_arity:
                n = flag_arity[tok]
                vals = parsed_cmd[i + 1:i + 1 + n]
                if len(vals) == n:
                    opts[tok] = vals
                i += 1 + n
                continue
            opts[tok] = True
            i += 1

        # Fallback extraction for robustness if parser misses.
        if "--time" not in opts:
            m = re.search(r"'(\d{1,2}-[A-Za-z]{3}-\d{2,4}\s+\d{2}:\d{2}:\d{2})'", text)
            if m:
                try:
                    _, kw = _parse_idl_call(f"gx_fov2box, '{m.group(1)}'")
                    parsed = _build_gx_fov2box_command(kw)
                    if "--time" in parsed.command:
                        ti = parsed.command.index("--time")
                        if ti + 1 < len(parsed.command):
                            opts["--time"] = [parsed.command[ti + 1]]
                except Exception:
                    pass
        if "--coords" not in opts:
            m = re.search(r"CENTER_ARCSEC\s*=\s*\[\s*([^\],]+)\s*,\s*([^\]]+)\s*\]", text, flags=re.IGNORECASE)
            if m:
                opts["--coords"] = [m.group(1).strip(), m.group(2).strip()]
                opts["--hpc"] = True
        if "--box-dims" not in opts:
            m = re.search(r"SIZE_PIX\s*=\s*\[\s*([^\],]+)\s*,\s*([^\],]+)\s*,\s*([^\]]+)\s*\]", text, flags=re.IGNORECASE)
            if m:
                opts["--box-dims"] = [m.group(1).strip(), m.group(2).strip(), m.group(3).strip()]

        # Time
        if "--time" in opts:
            try:
                iso = str(opts["--time"][0]).strip()
                dt = datetime.fromisoformat(iso)
                self.model_time_edit.setDateTime(QDateTime(dt))
            except Exception:
                pass
        # Frame first (without firing conversion handlers that could overwrite imported coords).
        if "--hgc" in opts:
            target_frame = "hgc"
        elif "--hgs" in opts:
            target_frame = "hgs"
        else:
            # default to HPC if execute didn't specify.
            target_frame = "hpc"

        for rb in (self.hpc_radio_button, self.hgc_radio_button, self.hgs_radio_button):
            rb.blockSignals(True)
        self.hpc_radio_button.setChecked(target_frame == "hpc")
        self.hgc_radio_button.setChecked(target_frame == "hgc")
        self.hgs_radio_button.setChecked(target_frame == "hgs")
        for rb in (self.hpc_radio_button, self.hgc_radio_button, self.hgs_radio_button):
            rb.blockSignals(False)

        # Coordinates (apply after frame selection so imported values are not overwritten).
        if "--coords" in opts:
            try:
                cx, cy = opts["--coords"]
                self.coord_x_edit.setText(str(cx))
                self.coord_y_edit.setText(str(cy))
            except Exception:
                pass

        # Projection
        if "--top" in opts:
            self.proj_top_radio.setChecked(True)
        else:
            self.proj_cea_radio.setChecked(True)

        # Box dimensions
        if "--box-dims" in opts:
            try:
                nx, ny, nz = opts["--box-dims"]
                self.grid_x_edit.setText(str(nx))
                self.grid_y_edit.setText(str(ny))
                self.grid_z_edit.setText(str(nz))
            except Exception:
                pass
        else:
            corona = boxdata.get("corona", {})
            if isinstance(corona, dict) and "bx" in corona:
                try:
                    nz, ny, nx = corona["bx"].shape
                    self.grid_x_edit.setText(str(nx))
                    self.grid_y_edit.setText(str(ny))
                    self.grid_z_edit.setText(str(nz))
                except Exception:
                    pass

        # Resolution (dx_km)
        if "--dx-km" in opts:
            try:
                self.res_edit.setText(f"{float(opts['--dx-km'][0]):.3f}")
            except Exception:
                pass
        else:
            corona = boxdata.get("corona", {})
            if isinstance(corona, dict) and "dr" in corona:
                try:
                    dr0 = float(corona["dr"][0])
                    rsun_km = sun_consts.radius.to(u.km).value
                    self.res_edit.setText(f"{dr0 * rsun_km:.3f}")
                except Exception:
                    pass

        # Padding fraction
        if "--pad-frac" in opts:
            try:
                pad_frac = float(opts["--pad-frac"][0])
                self.padding_size_edit.setText(f"{pad_frac * 100:.1f}")
            except Exception:
                pass

        # Context map toggles
        self.download_aia_euv.setChecked("--euv" in opts and "--no-euv" not in opts)
        self.download_aia_uv.setChecked("--uv" in opts and "--no-uv" not in opts)
        refmaps_opt = opts.get("--refmaps-path") or opts.get("--refmap-path")
        if refmaps_opt and hasattr(self, "refmaps_path_edit"):
            self.refmaps_path_edit.setText(str(refmaps_opt[0]))

        # Disambiguation
        disambig = self._decode_meta_value(boxdata.get("metadata", {}).get("disambiguation", "")).strip().upper()
        if "--sfq" in opts or disambig == "SFQ":
            self.disambig_sfq_radio.setChecked(True)
        else:
            self.disambig_hmi_radio.setChecked(True)

    def update_external_box_dir(self):
        """
        Updates the external box directory path based on the user input.
        """
        new_path = self.external_box_edit.text()
        if not new_path.strip():
            self._entry_stage_detected = None
            self._entry_type_detected = None
            self._entry_is_template_only = False
            self.entry_stage_edit.setText("N/A")
            self._set_jump_action("continue")
            self._set_model_params_enabled(True)
            self._refresh_viewer_button_state()
            self._sync_pipeline_options()
            self.update_command_display()
            return
        if not os.path.isfile(new_path):
            QMessageBox.critical(self, "Invalid Entry Box", f"Path is not a file:\n{new_path}")
            self._restore_last_valid_entry_box()
            return
        try:
            self.read_external_box()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Entry Box", f"Could not read entry box:\n{exc}")
            self._restore_last_valid_entry_box()
            return
        self._refresh_viewer_button_state()
        self.update_command_display()

    def _restore_last_valid_entry_box(self):
        self.external_box_edit.blockSignals(True)
        self.external_box_edit.setText(self._last_valid_entry_box)
        self.external_box_edit.blockSignals(False)
        self.update_command_display()

    def update_dir(self, new_path, default_path, target_edit=None):
        """
        Updates the specified directory path.

        :param new_path: The new directory path.
        :type new_path: str
        :param default_path: The default directory path.
        :type default_path: str
        """
        if new_path != default_path:
            # Normalize the path whether it's absolute or relative
            if not os.path.isabs(new_path):
                new_path = os.path.abspath(new_path)

            if not os.path.exists(new_path):  # Checks if the path does not exist
                # Ask user if they want to create the directory
                reply = QMessageBox.question(self, 'Create Directory?',
                                             "The directory does not exist. Do you want to create it?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                if reply == QMessageBox.Yes:
                    try:
                        os.makedirs(new_path)
                        # QMessageBox.information(self, "Directory Created", "The directory was successfully created.")
                    except PermissionError:
                        QMessageBox.critical(self, "Permission Denied",
                                             "You do not have permission to create this directory.")
                    except OSError as e:
                        QMessageBox.critical(self, "Error", f"Failed to create directory: {str(e)}")
                else:
                    # User chose not to create the directory, revert to the original path
                    if target_edit is not None:
                        target_edit.setText(default_path)
        # else:
        #     QMessageBox.warning(self, "Invalid Path", "The specified path is not a valid absolute path.")

    def open_sdo_file_dialog(self):
        """
        Opens a file dialog for selecting the SDO data directory.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        start_dir = self.sdo_data_edit.text().strip() or DOWNLOAD_DIR
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        if file_name:
            self.sdo_data_edit.setText(file_name)
            self.update_sdo_data_dir()

    def open_gx_file_dialog(self):
        """
        Opens a file dialog for selecting the GX model directory.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        start_dir = self.gx_model_edit.text().strip() or GXMODEL_DIR
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        if file_name:
            self.gx_model_edit.setText(file_name)
            self.update_gxmodel_dir()

    def open_external_file_dialog(self):
        """
        Opens a file dialog for selecting the external box directory.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", os.getcwd(), "Model Files (*.h5 *.sav)")
        # file_name = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        if file_name:
            self.external_box_edit.setText(file_name)
            self.update_external_box_dir()

    def open_refmaps_path_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        start_dir = self.refmaps_path_edit.text().strip() or os.getcwd()
        file_name = QFileDialog.getExistingDirectory(self, "Select Reference Map Directory", start_dir)
        if file_name:
            self.refmaps_path_edit.setText(file_name)
            self.update_refmaps_path()

    def add_model_configuration_section(self):
        # Hide legacy jump controls; workflow is linear with optional rebuild.
        self.jump_to_action_combo.setVisible(False)
        self.label_jumpToAction.setVisible(False)
        if hasattr(self, "jumpToActionLayout") and self.jumpToActionLayout is not None:
            while self.jumpToActionLayout.count():
                item = self.jumpToActionLayout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
            if self.verticalLayout_2.count() > 0:
                first_item = self.verticalLayout_2.itemAt(0)
                if first_item is not None and first_item.layout() is self.jumpToActionLayout:
                    self.verticalLayout_2.takeAt(0)
        self.model_time_edit.setDateTime(QDateTime.currentDateTimeUtc())
        self.model_time_edit.setDateTimeRange(QDateTime(2010, 1, 1, 0, 0, 0), QDateTime(QDateTime.currentDateTimeUtc()))
        self.model_time_edit.dateTimeChanged.connect(self.on_time_input_changed)
        self.download_backend_label = QLabel("Downloader:")
        self.download_backend_fido_radio = QRadioButton("Fido")
        self.download_backend_drms_radio = QRadioButton("DRMS")
        self.download_use_cache_box = QCheckBox("Use cache")
        self.download_backend_group = QButtonGroup(self)
        self.download_backend_group.addButton(self.download_backend_fido_radio)
        self.download_backend_group.addButton(self.download_backend_drms_radio)
        self.download_backend_drms_radio.setChecked(True)
        self.download_use_cache_box.setChecked(True)
        self.download_backend_fido_radio.setToolTip("Use the existing SunPy/Fido downloader path.")
        self.download_backend_drms_radio.setToolTip("Use the direct DRMS/JSOC downloader path for comparison.")
        self.download_use_cache_box.setToolTip("Reuse matching local files when available. Uncheck to force redownload for benchmarking.")
        self.download_backend_fido_radio.toggled.connect(self.update_command_display)
        self.download_backend_drms_radio.toggled.connect(self.update_command_display)
        self.download_use_cache_box.toggled.connect(self.update_command_display)
        if hasattr(self, "modelTimeLayout") and self.modelTimeLayout is not None:
            insert_idx = self.modelTimeLayout.indexOf(self.spacer_modelTime) if hasattr(self, "spacer_modelTime") else -1
            if insert_idx < 0:
                insert_idx = self.modelTimeLayout.count()
            for widget in (
                self.download_backend_label,
                self.download_backend_drms_radio,
                self.download_backend_fido_radio,
                self.download_use_cache_box,
            ):
                self.modelTimeLayout.insertWidget(insert_idx, widget)
                insert_idx += 1

        self.coord_x_edit.returnPressed.connect(lambda: self.on_coord_x_input_return_pressed(self.coord_x_edit))
        self.coord_y_edit.returnPressed.connect(lambda: self.on_coord_y_input_return_pressed(self.coord_y_edit))

        self.hpc_radio_button.toggled.connect(self.update_hpc_state)
        self.hgc_radio_button.toggled.connect(self.update_hgc_state)
        self.hgs_radio_button.toggled.connect(self.update_hgs_state)

        self.grid_x_edit.returnPressed.connect(lambda: self.on_grid_x_input_return_pressed(self.grid_x_edit))
        self.grid_y_edit.returnPressed.connect(lambda: self.on_grid_y_input_return_pressed(self.grid_y_edit))
        self.grid_z_edit.returnPressed.connect(lambda: self.on_grid_z_input_return_pressed(self.grid_z_edit))
        self.res_edit.returnPressed.connect(lambda: self.on_res_input_return_pressed(self.res_edit))
        self.padding_size_edit.returnPressed.connect(
            lambda: self.on_padding_size_input_return_pressed(self.padding_size_edit))
        self.hpc_radio_button.setText("Heliocentric")
        self.hgc_radio_button.setText("Carrington")
        self.hgs_radio_button.setText("Stonyhurst")
        self.label_padding.setText("Pad (%):")

        for edit, width in (
            (self.grid_x_edit, 72),
            (self.grid_y_edit, 72),
            (self.grid_z_edit, 72),
            (self.res_edit, 88),
            (self.padding_size_edit, 68),
        ):
            edit.setMaximumWidth(width)
            edit.setMinimumWidth(width)

        grid_row_layout = None
        for i in range(self.verticalLayout_2.count()):
            item = self.verticalLayout_2.itemAt(i)
            layout = item.layout() if item is not None else None
            if layout is not None and layout.indexOf(self.grid_x_edit) >= 0:
                grid_row_layout = layout
                break

        if grid_row_layout is not None and hasattr(self, "spacer_grid") and self.spacer_grid is not None:
            idx = grid_row_layout.indexOf(self.spacer_grid)
            if idx >= 0:
                grid_row_layout.takeAt(idx)
        if hasattr(self, "spacer_resPadding") and self.spacer_resPadding is not None:
            idx = self.resPaddingLayout.indexOf(self.spacer_resPadding)
            if idx >= 0:
                self.resPaddingLayout.takeAt(idx)

        merged_widgets = [
            self.label_resolution,
            self.res_edit,
            self.label_padding,
            self.padding_size_edit,
        ]
        if grid_row_layout is not None:
            for widget in merged_widgets:
                self.resPaddingLayout.removeWidget(widget)
                grid_row_layout.addWidget(widget)
            grid_row_layout.addStretch()
        if hasattr(self, "resPaddingLayout") and self.resPaddingLayout is not None:
            for i in range(self.verticalLayout_2.count()):
                item = self.verticalLayout_2.itemAt(i)
                if item is not None and item.layout() is self.resPaddingLayout:
                    self.verticalLayout_2.takeAt(i)
                    break

        self.proj_group = QGroupBox("Geometrical Projection")
        self.proj_cea_radio = QRadioButton("CEA")
        self.proj_top_radio = QRadioButton("TOP")
        self.proj_cea_radio.setChecked(True)
        self.proj_button_group = QButtonGroup(self.proj_group)
        self.proj_button_group.addButton(self.proj_cea_radio)
        self.proj_button_group.addButton(self.proj_top_radio)
        proj_layout = QHBoxLayout()
        proj_layout.addWidget(self.proj_cea_radio)
        proj_layout.addWidget(self.proj_top_radio)
        proj_layout.addStretch()
        self.proj_group.setLayout(proj_layout)
        self.proj_cea_radio.toggled.connect(self.update_command_display)
        self.proj_top_radio.toggled.connect(self.update_command_display)

        # Standalone disambiguation group in model configuration (not part of workflow options).
        self.disambig_group = QGroupBox("Pi-disambiguation")
        self.disambig_hmi_radio = QRadioButton("HMI")
        self.disambig_sfq_radio = QRadioButton("SFQ")
        self.disambig_hmi_radio.setChecked(True)
        self.disambig_button_group = QButtonGroup(self.disambig_group)
        self.disambig_button_group.addButton(self.disambig_hmi_radio)
        self.disambig_button_group.addButton(self.disambig_sfq_radio)
        disambig_layout = QHBoxLayout()
        disambig_layout.addWidget(self.disambig_hmi_radio)
        disambig_layout.addWidget(self.disambig_sfq_radio)
        disambig_layout.addStretch()
        self.disambig_group.setLayout(disambig_layout)
        self.disambig_hmi_radio.toggled.connect(self.update_command_display)
        self.disambig_sfq_radio.toggled.connect(self.update_command_display)

        # Keep projection and disambiguation on one row.
        proj_disambig_row = QHBoxLayout()
        proj_disambig_row.addWidget(self.proj_group)
        proj_disambig_row.addWidget(self.disambig_group)
        self.verticalLayout_2.addLayout(proj_disambig_row)

    def _get_jump_action(self):
        if self.modify_radio.isChecked():
            return "modify"
        if self.rebuild_obs_radio.isChecked():
            return "rebuild_obs"
        if self.rebuild_none_radio.isChecked():
            return "rebuild_none"
        return "continue"

    def _set_jump_action(self, action):
        mode = (action or "continue").lower()
        self.continue_radio.blockSignals(True)
        self.rebuild_none_radio.blockSignals(True)
        self.rebuild_obs_radio.blockSignals(True)
        self.modify_radio.blockSignals(True)
        self.continue_radio.setChecked(mode == "continue")
        self.rebuild_none_radio.setChecked(mode == "rebuild_none")
        self.rebuild_obs_radio.setChecked(mode == "rebuild_obs")
        self.modify_radio.setChecked(mode == "modify")
        self.continue_radio.blockSignals(False)
        self.rebuild_none_radio.blockSignals(False)
        self.rebuild_obs_radio.blockSignals(False)
        self.modify_radio.blockSignals(False)

    def _reset_pipeline_checks_for_entry(self):
        boxes = [
            self.download_hmi_box,
            self.download_aia_uv,
            self.download_aia_euv,
            self.stop_early_box,
            self.save_empty_box,
            self.save_potential_box,
            self.save_bounds_box,
            self.save_nas_box,
            self.save_gen_box,
            self.empty_box_only_box,
            self.stop_after_bnd_box,
            self.potential_only_box,
            self.stop_after_potential_box,
            self.nlfff_only_box,
            self.generic_only_box,
            self.add_save_chromo_box,
            self.skip_nlfff_extrapolation,
            self.skip_line_computation_box,
            self.center_vox_box,
        ]
        for b in boxes:
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)

    def add_options_section(self):
        """
        Adds the options section to the main layout.
        """
        self.optionsGroupBox.setTitle("Pipeline Workflow")
        self.download_hmi_box = QCheckBox("Download HMI Vector Magnetograms")
        self.download_hmi_box.setChecked(True)
        self.download_hmi_box.setEnabled(False)
        self.stop_early_box = QCheckBox("Stop after data download")
        self.download_aia_euv.setChecked(True)
        self.download_aia_uv.setChecked(True)
        self.save_empty_box.setChecked(False)
        self.save_potential_box.setChecked(False)
        self.save_bounds_box.setChecked(False)
        self.skip_nlfff_extrapolation.setChecked(False)
        self.stop_after_potential_box.setChecked(False)
        self.stop_after_potential_box.setVisible(True)
        self.stop_after_potential_box.setEnabled(True)
        self.stop_after_potential_box.setText("Stop after POT")
        self.skip_nlfff_extrapolation.setText("Skip NLFFF stage")
        self.download_aia_uv.setText("Download AIA/UV")
        self.download_aia_euv.setText("Download AIA/EUV")
        self.save_empty_box.setText("Save Empty Box (NONE)")
        self.save_potential_box.setText("Save Potential Box (POT)")
        self.save_bounds_box.setText("Save Bounds Box (BND)")

        # Additional CLI parity controls (added programmatically to preserve .ui compatibility)
        options_layout = self.optionsGroupBox.layout()
        while options_layout.count():
            item = options_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        self.save_nas_box = QCheckBox("Save NLFFF Box (NAS)")
        self.save_gen_box = QCheckBox("Save Lines (GEN)")
        self.empty_box_only_box = QCheckBox("Stop after NONE")
        self.stop_after_bnd_box = QCheckBox("Stop after BND")
        self.potential_only_box = QCheckBox("Potential Only")
        self.nlfff_only_box = QCheckBox("Stop after NAS")
        self.generic_only_box = QCheckBox("Stop after GEN")
        self.skip_line_computation_box = QCheckBox("Skip Line Computation")
        self.center_vox_box = QCheckBox("Center Box Tracing")
        self.add_save_chromo_box = QCheckBox("Add and save Chromo Model (CHR)")
        self.add_save_chromo_box.setChecked(True)

        # Stage-ordered two-column workflow layout.
        # Two columns, 8 rows each, strict requested order (fill column 1 then column 2).
        options_layout.addWidget(self.download_hmi_box, 0, 0)           # 1
        options_layout.addWidget(self.download_aia_uv, 1, 0)            # 2
        options_layout.addWidget(self.download_aia_euv, 2, 0)           # 3
        options_layout.addWidget(self.stop_early_box, 3, 0)             # 4
        options_layout.addWidget(self.save_empty_box, 4, 0)             # 5
        options_layout.addWidget(self.empty_box_only_box, 5, 0)         # 6
        options_layout.addWidget(self.save_potential_box, 6, 0)         # 7
        options_layout.addWidget(self.save_bounds_box, 7, 0)            # 8
        options_layout.addWidget(self.stop_after_potential_box, 0, 1)   # 9
        options_layout.addWidget(self.skip_nlfff_extrapolation, 1, 1)   # 10
        options_layout.addWidget(self.save_nas_box, 2, 1)               # 11
        options_layout.addWidget(self.nlfff_only_box, 3, 1)             # 12
        options_layout.addWidget(self.skip_line_computation_box, 4, 1)  # 13
        options_layout.addWidget(self.save_gen_box, 5, 1)               # 14
        options_layout.addWidget(self.generic_only_box, 6, 1)           # 15
        options_layout.addWidget(self.add_save_chromo_box, 7, 1)        # 16
        self.center_vox_box.setToolTip("Line tracing mode: Fast is default; enable for center-voxel tracing.")
        self.center_vox_box.setVisible(False)

        # Keep only for CLI backward compatibility.
        self.potential_only_box.setVisible(False)

        # Update command/state when options change
        dynamic_widgets = [
            self.download_aia_euv,
            self.download_aia_uv,
            self.download_hmi_box,
            self.stop_early_box,
            self.save_empty_box,
            self.save_potential_box,
            self.save_bounds_box,
            self.stop_after_potential_box,
            self.skip_nlfff_extrapolation,
            self.save_nas_box,
            self.save_gen_box,
            self.empty_box_only_box,
            self.stop_after_bnd_box,
            self.nlfff_only_box,
            self.generic_only_box,
            self.skip_line_computation_box,
            self.add_save_chromo_box,
        ]
        for w in dynamic_widgets:
            w.toggled.connect(self._sync_pipeline_options)
        self.external_box_edit.textChanged.connect(self.update_command_display)
        self.sdo_data_edit.textChanged.connect(self.update_command_display)
        self.gx_model_edit.textChanged.connect(self.update_command_display)
        self._sync_pipeline_options()

    def _set_checkbox_state(self, box, enabled):
        box.blockSignals(True)
        if not enabled and box.isChecked():
            box.setChecked(False)
        box.setEnabled(enabled)
        box.blockSignals(False)

    def _sync_pipeline_options(self, *_):
        """
        Enforce linear stage workflow:
        - from scratch (no entry): full pipeline options
        - entry selected: forward-only from detected stage
        - rebuild selected: restart from NONE/OBS using restored entry parameters
        - modify selected: unlock parameters and build a fresh script
        """
        try:
            stage_rank = {"NONE": 0, "POT": 1, "BND": 2, "NAS": 3, "GEN": 4, "CHR": 5}
            save_stage_boxes = [
                (self.save_empty_box, 0),
                (self.save_potential_box, 1),
                (self.save_bounds_box, 2),
                (self.save_nas_box, 3),
                (self.save_gen_box, 4),
            ]
            stop_stage_boxes = [
                (self.stop_early_box, -1, None),
                (self.empty_box_only_box, 0, self.save_empty_box),
                (self.stop_after_potential_box, 1, self.save_potential_box),
                (self.nlfff_only_box, 3, self.save_nas_box),
                (self.generic_only_box, 4, self.save_gen_box),
            ]

            has_entry = self._has_entry_box()
            mode = self._get_jump_action()
            modify_mode = mode == "modify"
            rebuild_obs = mode == "rebuild_obs"
            rebuild_none = mode == "rebuild_none"
            start_stage = 0 if (not has_entry or rebuild_obs or rebuild_none or modify_mode) else stage_rank.get(self._entry_stage_detected or "NONE", 0)
            allow_download_stage = (not has_entry) or rebuild_obs or rebuild_none or modify_mode

            if not has_entry and mode != "continue":
                self._set_jump_action("continue")
                mode = "continue"
                modify_mode = False
                rebuild_obs = False
                rebuild_none = False
            self.rebuild_none_radio.setEnabled(has_entry)
            self.rebuild_obs_radio.setEnabled(has_entry)
            self.modify_radio.setEnabled(has_entry)
            self._set_model_params_enabled((not has_entry) or modify_mode)

            # Strict no-going-back for map downloads when resuming from entry box.
            if has_entry and not modify_mode:
                for box in (self.download_hmi_box, self.download_aia_uv, self.download_aia_euv):
                    box.blockSignals(True)
                    box.setChecked(False)
                    box.setEnabled(False)
                    box.blockSignals(False)
            else:
                self.download_hmi_box.blockSignals(True)
                self.download_hmi_box.setChecked(True)
                self.download_hmi_box.setEnabled(False)
                self.download_hmi_box.blockSignals(False)
                if modify_mode:
                    for box in (self.download_aia_uv, self.download_aia_euv):
                        box.blockSignals(True)
                        box.setChecked(True)
                        box.blockSignals(False)
                self.download_aia_uv.setEnabled(True)
                self.download_aia_euv.setEnabled(True)

            # Keep stop controls mutually exclusive in stage order.
            stop_stage = None
            chosen_box = None
            for box, stage, _save in stop_stage_boxes:
                if box.isChecked():
                    stop_stage = stage
                    chosen_box = box
                    break
            if stop_stage is not None:
                for box, _stage, _save in stop_stage_boxes:
                    if box is not chosen_box and box.isChecked():
                        box.blockSignals(True)
                        box.setChecked(False)
                        box.blockSignals(False)

            use_potential = self.skip_nlfff_extrapolation.isChecked()
            skip_lines = self.skip_line_computation_box.isChecked()
            # Allow skip-NLFFF from NONE/POT/BND starts.
            if start_stage > 2:
                self._set_checkbox_state(self.skip_nlfff_extrapolation, False)
                use_potential = False
            else:
                # Skip-NLFFF is only meaningful if pipeline can proceed beyond POT.
                skip_enabled = stop_stage is None or stop_stage >= 2
                self._set_checkbox_state(self.skip_nlfff_extrapolation, skip_enabled)
                use_potential = self.skip_nlfff_extrapolation.isChecked()

            skip_lines_enabled = start_stage <= 4 and (stop_stage is None or stop_stage >= 4)
            self._set_checkbox_state(self.skip_line_computation_box, skip_lines_enabled)
            skip_lines = self.skip_line_computation_box.isChecked()

            for box, stage, _save in stop_stage_boxes:
                enabled = (stage == -1 and allow_download_stage) or (stage >= start_stage)
                if stop_stage is not None and stage > stop_stage:
                    enabled = False
                if use_potential and stage in (2, 3):
                    enabled = False
                if skip_lines and stage == 4:
                    enabled = False
                self._set_checkbox_state(box, enabled)

            for box, stage in save_stage_boxes:
                enabled = stage >= start_stage
                if stop_stage is not None and stage > stop_stage:
                    enabled = False
                if use_potential and stage in (2, 3):
                    enabled = False
                if skip_lines and stage == 4:
                    enabled = False
                # Requested behavior: when stopping after POT, POT save is default
                # while Save BND stays available as an optional bonus.
                if stop_stage == 1 and stage == 2:
                    enabled = True
                self._set_checkbox_state(box, enabled)

            # Continue mode from GEN/CHR defaults to a direct CHR completion path.
            # Until the user explicitly selects "Stop after GEN", GEN-only controls
            # should stay off because the generated command will jump straight to CHR.
            continue_direct_chr = (
                has_entry
                and not modify_mode
                and mode == "continue"
                and (self._entry_stage_detected or "NONE") in ("GEN", "CHR")
                and stop_stage is None
            )
            if continue_direct_chr:
                self._set_checkbox_state(self.save_gen_box, False)
                self._set_checkbox_state(self.skip_line_computation_box, False)
                self._set_checkbox_state(self.center_vox_box, False)

            # When a stop is selected, corresponding save is automatic.
            if stop_stage is not None:
                for _stop_box, stage, save_box in stop_stage_boxes:
                    if stage == stop_stage and save_box is not None:
                        save_box.blockSignals(True)
                        save_box.setChecked(True)
                        save_box.setEnabled(False)
                        save_box.blockSignals(False)
                        break

            # CHR save is automatic when pipeline is not stopped earlier.
            # Keep it disabled in all cases; uncheck only when a stop is selected.
            self.add_save_chromo_box.blockSignals(True)
            self.add_save_chromo_box.setChecked(stop_stage is None)
            self.add_save_chromo_box.setEnabled(False)
            self.add_save_chromo_box.blockSignals(False)

            # center-vox matters only when lines are computed.
            center_vox_enabled = True
            if start_stage > 4:
                center_vox_enabled = False
            if stop_stage is not None and stop_stage < 4:
                center_vox_enabled = False
            if skip_lines:
                center_vox_enabled = False
            self._set_checkbox_state(self.center_vox_box, center_vox_enabled)
            self.update_command_display()
        except Exception as exc:
            self.status_log_edit.append(f"GUI workflow sync error: {exc}")

    def add_cmd_display(self):
        """
        Adds the command display section to the main layout.
        """
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(11)
        self.cmd_display_edit.setFont(mono)
        self.cmd_display_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        self.cmd_display_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.cmd_display_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.cmd_display_edit.setMinimumHeight(72)
        self.cmd_display_edit.setMaximumHeight(88)

    def add_cmd_buttons(self):
        """
        Adds the command buttons to the main layout.
        """
        icon_size = QSize(20, 20)
        button_width = 34
        icon_dir = self._resolve_svg_dir()
        self.reset_test_defaults_button = QPushButton("Reset to Test Defaults")
        self.restore_last_saved_button = QPushButton("Restore Last Saved")
        self.reset_test_defaults_button.setToolTip(
            "Reset GUI geometry/time/projection fields to the built-in test configuration."
        )
        self.restore_last_saved_button.setToolTip("Restore the last saved GUI session from local settings.")
        self.reset_test_defaults_button.clicked.connect(self.on_reset_to_test_defaults_clicked)
        self.restore_last_saved_button.clicked.connect(self.on_restore_last_saved_clicked)
        self.reset_test_defaults_button.setText("")
        self.restore_last_saved_button.setText("")
        self.reset_test_defaults_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.restore_last_saved_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.reset_test_defaults_button.setFixedWidth(button_width)
        self.restore_last_saved_button.setFixedWidth(button_width)
        self.reset_test_defaults_button.setIconSize(icon_size)
        self.restore_last_saved_button.setIconSize(icon_size)
        self.info_only_box = QCheckBox("Info Only")
        self.info_only_box.toggled.connect(self.update_command_display)
        self.command_menu_button = QToolButton()
        self.command_menu_button.setToolTip("gx-fov2box command options")
        self.command_menu_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.command_menu_button.setFixedWidth(button_width)
        self.command_menu_button.setIconSize(icon_size)
        self.command_menu_button.setPopupMode(QToolButton.InstantPopup)
        command_menu = QMenu(self.command_menu_button)
        command_menu.addAction("Copy Command", self.copy_command)
        command_menu.addAction("Save Script As...", self.save_command)
        self.command_menu_button.setMenu(command_menu)
        self.open_fov_selector_button = QPushButton("")
        self.open_fov_selector_button.setToolTip(
            "Open gxbox-view2d on demand (2D map/FOV viewer using current GUI fields and local cached maps if available)."
        )
        monitor_icon = icon_dir / "monitor.svg"
        if monitor_icon.exists():
            self.open_fov_selector_button.setIcon(QIcon(str(monitor_icon)))
        else:
            self.open_fov_selector_button.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.open_fov_selector_button.setFixedWidth(button_width)
        self.open_fov_selector_button.setIconSize(icon_size)
        self.open_fov_selector_button.clicked.connect(self.on_open_fov_selector_clicked)
        self.execute_button.clicked.connect(self.execute_command)
        self.stop_button.clicked.connect(self.stop_command)
        self.stop_button.setEnabled(False)
        self.save_button.setVisible(False)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_command)
        self.send_to_viewer_button = QPushButton("")
        self.send_to_viewer_button.setToolTip("Open the latest generated model in gxbox-view3d")
        box_icon = icon_dir / "box.svg"
        if box_icon.exists():
            self.send_to_viewer_button.setIcon(QIcon(str(box_icon)))
        else:
            self.send_to_viewer_button.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.send_to_viewer_button.setFixedWidth(button_width)
        self.send_to_viewer_button.setIconSize(icon_size)
        self.send_to_viewer_button.setEnabled(False)
        self.send_to_viewer_button.clicked.connect(self.send_to_gxbox_view)
        self.exit_button = QPushButton("")
        self.exit_button.setToolTip("Close pyAMPP")
        self.exit_button.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        self.exit_button.setFixedWidth(button_width)
        self.exit_button.setIconSize(icon_size)
        self.exit_button.clicked.connect(self.close)
        self.clear_button_refresh.clicked.connect(self.refresh_command)
        self.clear_button_clear.setVisible(False)
        self.clear_button_clear.setEnabled(False)
        self.clear_button_clear.clicked.connect(self.clear_command)
        if self.cmd_button_layout is not None:
            spacer_idx = max(0, self.cmd_button_layout.count() - 1)
            for widget in (
                self.reset_test_defaults_button,
                self.restore_last_saved_button,
                self.info_only_box,
                self.command_menu_button,
                self.open_fov_selector_button,
                self.send_to_viewer_button,
                self.exit_button,
            ):
                self.cmd_button_layout.insertWidget(spacer_idx, widget)
                spacer_idx += 1

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

    def add_status_log(self):
        """
        Adds the status log section to the main layout.
        """
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)
        self.status_log_edit.setFont(mono)
        # Move console to a right-side dock panel to keep the main workflow visible.
        for i in range(self.main_layout.count()):
            item = self.main_layout.itemAt(i)
            if item is not None and item.widget() is self.status_log_edit:
                self.main_layout.takeAt(i)
                break
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setObjectName("consoleDock")
        self.console_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.console_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.console_dock.setMinimumWidth(340)
        self.console_dock.setMaximumWidth(520)
        self.console_dock.setWidget(self.status_log_edit)
        title_bar = QWidget(self.console_dock)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(6, 2, 6, 2)
        title_layout.setSpacing(4)
        title_label = QLabel("Console", title_bar)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        self.console_menu_button = QToolButton(title_bar)
        self.console_menu_button.setText("⋮")
        self.console_menu_button.setToolTip("Console options")
        self.console_menu_button.setPopupMode(QToolButton.InstantPopup)
        console_menu = QMenu(self.console_menu_button)
        console_menu.addAction("Clear", self.clear_console)
        console_menu.addAction("Copy All", self.copy_console)
        console_menu.addAction("Save As...", self.save_console)
        self.console_menu_button.setMenu(console_menu)
        title_layout.addWidget(self.console_menu_button)
        self.console_dock.setTitleBarWidget(title_bar)
        self.addDockWidget(Qt.RightDockWidgetArea, self.console_dock)

    @validate_number
    def on_coord_x_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_coord_y_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_grid_x_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_grid_y_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_grid_z_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_res_input_return_pressed(self, widget):
        self.update_command_display(widget)

    def on_padding_size_input_return_pressed(self, widget):
        self.update_command_display(widget)

    def _remove_stretch_from_layout(self, layout):
        """
        Removes the last stretch item from the given layout if it exists.

        This method checks the last item in the layout and removes it if it is a spacer item.
        It is useful for dynamically managing layout items, especially when adding or removing widgets.

        Parameters
        ----------
        layout : QLayout
            The layout from which the stretch item should be removed.
        """
        count = layout.count()
        if count > 0 and layout.itemAt(count - 1).spacerItem():
            layout.takeAt(count - 1)

    def on_time_input_changed(self):
        self.coords_center = self._coords_center
        if self.model_time_orig is not None:
            time = Time(self.model_time_edit.dateTime().toPyDateTime()).mjd
            model_time_orig = self.model_time_orig.mjd
            time_sec_diff = (time - model_time_orig) * 24 * 3600
            print(time_sec_diff)
            if np.abs(time_sec_diff) >= 0.5:
                self.on_rotate_model_to_time()
                if self.rotate_revert_button is None:
                    self._remove_stretch_from_layout(self.model_time_layout)
                    self.rotate_revert_button = QPushButton("Revert")
                    self.rotate_revert_button.setToolTip("Revert the model to the original time")
                    self.rotate_revert_button.clicked.connect(self.on_rotate_revert_button_clicked)
                    self.model_time_layout.addWidget(self.rotate_revert_button)
                    self.model_time_layout.addStretch()
            else:
                if self.rotate_revert_button is not None:
                    self.update_coords_center(revert=True)
                    self.rotate_revert()
                    self._remove_stretch_from_layout(self.model_time_layout)
                    self.model_time_layout.removeWidget(self.rotate_revert_button)
                    self.rotate_revert_button.deleteLater()
                    self.rotate_revert_button = None
                    self.model_time_layout.addStretch()
        self.update_command_display()

    def on_rotate_revert_button_clicked(self):
        self.model_time_edit.setDateTime(QDateTime(self.model_time_orig.to_datetime()))
        self.rotate_revert()

    def rotate_revert(self):
        if self.hpc_radio_button.isChecked():
            self.update_hpc_state(True)
        elif self.hgc_radio_button.isChecked():
            self.update_hgc_state(True)
        elif self.hgs_radio_button.isChecked():
            self.update_hgs_state(True)

    def on_rotate_model_to_time(self):
        """
        Rotates the model to the specified time.
        """
        from sunpy.coordinates import RotatedSunFrame
        point = self.coords_center_orig
        time = Time(self.model_time_edit.dateTime().toPyDateTime()).mjd
        model_time_orig = self.model_time_orig.mjd
        time_sec_diff = (time - model_time_orig) * 24 * 3600
        diffrot_point = SkyCoord(RotatedSunFrame(base=point, duration=time_sec_diff * u.s))
        self.coords_center = diffrot_point.transform_to(self._coords_center.frame)
        print(self.coords_center_orig, self.coords_center)
        # self.status_log_edit.append("Model rotated to the specified time")
        if self.hpc_radio_button.isChecked():
            self.update_hpc_state(True, self.coords_center)
        elif self.hgc_radio_button.isChecked():
            self.update_hgc_state(True, self.coords_center)
        elif self.hgs_radio_button.isChecked():
            self.update_hgs_state(True, self.coords_center)
        self.update_command_display()

    def update_command_display(self, widget=None):
        """
        Updates the command display with the current command.
        """
        try:
            if getattr(self, "_hydrating_entry", False):
                return
            self.coords_center = self._coords_center
            command = self.get_command()
            self.cmd_display_edit.setPlainText(" ".join(command))
        except Exception as exc:
            self.status_log_edit.append(f"GUI update error: {exc}")

    def apply_geometry_selection(self, selection: BoxGeometrySelection) -> None:
        """
        Apply an accepted geometry selection (e.g., from a future standalone FOV selector)
        to the existing pyAMPP GUI input fields.
        """
        # Set coordinate mode first without firing conversion handlers.
        target_frame = str(selection.coord_mode)
        for rb in (self.hpc_radio_button, self.hgc_radio_button, self.hgs_radio_button):
            rb.blockSignals(True)
        self.hpc_radio_button.setChecked(target_frame == CoordMode.HPC.value)
        self.hgc_radio_button.setChecked(target_frame == CoordMode.HGC.value)
        self.hgs_radio_button.setChecked(target_frame == CoordMode.HGS.value)
        for rb in (self.hpc_radio_button, self.hgc_radio_button, self.hgs_radio_button):
            rb.blockSignals(False)

        # Apply field values in the same format used by the GUI widgets.
        field_values = selection.as_gui_text_fields()
        self.coord_x_edit.setText(field_values["coord_x_edit"])
        self.coord_y_edit.setText(field_values["coord_y_edit"])
        self.grid_x_edit.setText(field_values["grid_x_edit"])
        self.grid_y_edit.setText(field_values["grid_y_edit"])
        self.grid_z_edit.setText(field_values["grid_z_edit"])
        self.res_edit.setText(field_values["res_edit"])

        # Refresh labels/tooltips for the selected coordinate mode and recompute internal center.
        self.update_coords_center()
        if selection.coord_mode == CoordMode.HPC:
            self.update_hpc_state(True)
        elif selection.coord_mode == CoordMode.HGC:
            self.update_hgc_state(True)
        else:
            self.update_hgs_state(True)

        # Keep pipeline/UI state synchronized with the new geometry values.
        self._sync_pipeline_options()
        self.update_command_display()
        self.status_log_edit.append("Applied geometry selection from selector API.")

    def apply_selector_result(self, result: SelectorDialogResult) -> None:
        """Apply a full selector dialog result to GUI-visible and hidden state."""
        self.apply_geometry_selection(result.geometry)
        self._selector_fov = result.fov
        self._selector_square_fov = bool(result.square_fov)
        self._selector_unsaved_session_active = True
        self.update_command_display()
        if result.fov is not None:
            self.status_log_edit.append("Stored observer FOV metadata from selector API.")

    def update_hpc_state(self, checked, coords_center=None):
        """
        Updates the UI when Helioprojective coordinates are selected.

        :param checked: Whether the Helioprojective radio button is checked.
        :type checked: bool
        """
        if checked:
            self.coord_x_edit.setToolTip("Solar X coordinate of the model center in arcsec")
            self.coord_y_edit.setToolTip("Solar Y coordinate of the model center in arcsec")
            self.coord_label.setText("Center Coords  in arcsec")
            self.coord_x_label.setText("X:")
            self.coord_y_label.setText("Y:")
            if coords_center is None:
                obstime = Time(self.model_time_edit.dateTime().toPyDateTime())
                observer = get_earth(obstime)
                coords_center = self.coords_center.transform_to(Helioprojective(obstime=obstime, observer=observer))
            self.coord_x_edit.setText(f'{coords_center.Tx.to(u.arcsec).value}')
            self.coord_y_edit.setText(f'{coords_center.Ty.to(u.arcsec).value}')
            self.update_command_display()

    def update_hgc_state(self, checked, coords_center=None):
        """
        Updates the UI when Heliographic Carrington coordinates are selected.

        :param checked: Whether the Heliographic Carrington radio button is checked.
        :type checked: bool
        """
        if checked:
            self.coord_x_edit.setToolTip("Heliographic Carrington Longitude of the model center in deg")
            self.coord_y_edit.setToolTip("Heliographic Carrington Latitude of the model center in deg")
            self.coord_label.setText("Center Coords in deg")
            self.coord_x_label.setText("lon:")
            self.coord_y_label.setText("lat:")
            if coords_center is None:
                print(f'coords_center: {self.coords_center}')
                obstime = Time(self.model_time_edit.dateTime().toPyDateTime())
                observer = get_earth(obstime)
                coords_center = self.coords_center.transform_to(
                    HeliographicCarrington(obstime=obstime, observer=observer))
            print(f'new coords_center: {coords_center}')
            self.coord_x_edit.setText(f'{coords_center.lon.to(u.deg).value}')
            self.coord_y_edit.setText(f'{coords_center.lat.to(u.deg).value}')
            self.update_command_display()

    def update_hgs_state(self, checked, coords_center=None):
        """
        Updates the UI when Heliographic Stonyhurst coordinates are selected.

        :param checked: Whether the Heliographic Stonyhurst radio button is checked.
        :type checked: bool
        """
        if checked:
            self.coord_x_edit.setToolTip("Heliographic Stonyhurst Longitude of the model center in deg")
            self.coord_y_edit.setToolTip("Heliographic Stonyhurst Latitude of the model center in deg")
            self.coord_label.setText("Center Coords in deg")
            self.coord_x_label.setText("lon:")
            self.coord_y_label.setText("lat:")
            if coords_center is None:
                obstime = Time(self.model_time_edit.dateTime().toPyDateTime())
                # observer = get_earth(obstime)
                coords_center = self.coords_center.transform_to(
                    HeliographicStonyhurst(obstime=obstime))
            self.coord_x_edit.setText(f'{coords_center.lon.to(u.deg).value}')
            self.coord_y_edit.setText(f'{coords_center.lat.to(u.deg).value}')
            self.update_command_display()

    def update_coords_center(self, revert=False):
        if revert:
            self.coords_center = self.coords_center_orig
        else:
            self.coords_center = self._coords_center

    @property
    def _coords_center(self):
        time = Time(self.model_time_edit.dateTime().toPyDateTime())
        coords = [float(self.coord_x_edit.text()), float(self.coord_y_edit.text())]
        observer = get_earth(time)
        if self.hpc_radio_button.isChecked():
            coords_center = SkyCoord(coords[0] * u.arcsec, coords[1] * u.arcsec, obstime=time, observer=observer,
                                     rsun=696 * u.Mm, frame='helioprojective')
        elif self.hgc_radio_button.isChecked():
            coords_center = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time, observer=observer,
                                     radius=696 * u.Mm,
                                     frame='heliographic_carrington')
        elif self.hgs_radio_button.isChecked():
            coords_center = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time, observer=observer,
                                     radius=696 * u.Mm,
                                     frame='heliographic_stonyhurst')
        return coords_center

    def _current_coord_mode(self) -> CoordMode:
        if self.hpc_radio_button.isChecked():
            return CoordMode.HPC
        if self.hgc_radio_button.isChecked():
            return CoordMode.HGC
        return CoordMode.HGS

    def _current_geometry_selection(self) -> BoxGeometrySelection:
        return BoxGeometrySelection(
            coord_mode=self._current_coord_mode(),
            coord_x=float(self.coord_x_edit.text()),
            coord_y=float(self.coord_y_edit.text()),
            grid_x=int(float(self.grid_x_edit.text())),
            grid_y=int(float(self.grid_y_edit.text())),
            grid_z=int(float(self.grid_z_edit.text())),
            dx_km=float(self.res_edit.text()),
        )

    def _selector_map_ids_from_gui(self) -> tuple[str, ...]:
        # Display names are intentionally user-facing here.
        map_ids = ["Bz", "Ic", "Br", "Bp", "Bt"]
        if self.download_aia_uv.isChecked():
            map_ids.append("1600")
        if self.download_aia_euv.isChecked():
            map_ids.extend(["171", "193", "211", "304", "335"])
        # Deduplicate while preserving order.
        out = []
        seen = set()
        for m in map_ids:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return tuple(out)

    def _build_download_selector_session_input(self) -> SelectorSessionInput:
        time_iso = Time(self.model_time_edit.dateTime().toPyDateTime()).to_datetime().strftime('%Y-%m-%dT%H:%M:%S')
        try:
            pad_frac = float(self.padding_size_edit.text()) / 100.0
        except Exception:
            pad_frac = None
        map_files = {}
        try:
            map_files = _discover_filesystem_maps(time_iso, self.sdo_data_edit.text())
        except Exception as exc:
            self.status_log_edit.append(f"Selector map-file discovery warning: {exc}")
        refmaps_path = self.refmaps_path_edit.text().strip() if hasattr(self, "refmaps_path_edit") else ""
        if refmaps_path:
            try:
                map_files.update(_discover_external_ref_map_files([refmaps_path]))
            except Exception as exc:
                self.status_log_edit.append(f"Selector reference-map discovery warning: {exc}")
        requested_map_ids = self._selector_map_ids_from_gui()
        map_ids = []
        availability = {
            "Bz": "magnetogram" in map_files,
            "Ic": "continuum" in map_files,
            "Br": all(k in map_files for k in ("field", "inclination", "azimuth", "disambig")),
            "Bp": all(k in map_files for k in ("field", "inclination", "azimuth", "disambig")),
            "Bt": all(k in map_files for k in ("field", "inclination", "azimuth", "disambig")),
        }
        for map_id in requested_map_ids:
            if map_id in availability:
                if availability[map_id]:
                    map_ids.append(map_id)
                continue
            if map_id in map_files:
                map_ids.append(map_id)
        internal_source_keys = {"field", "inclination", "azimuth", "disambig", "magnetogram", "continuum"}
        for map_id in map_files:
            if map_id not in internal_source_keys and map_id not in map_ids:
                map_ids.append(map_id)
        if not map_ids:
            map_ids = list(requested_map_ids)
        return SelectorSessionInput(
            time_iso=time_iso,
            data_dir=self.sdo_data_edit.text(),
            geometry=self._current_geometry_selection(),
            fov=self._selector_fov,
            square_fov=bool(self._selector_square_fov),
            map_ids=map_ids,
            map_files=map_files,
            display_observer_key=self._selector_observer_name or "earth",
            initial_map_id="171" if "171" in map_ids else ("Bz" if "Bz" in map_ids else (map_ids[0] if map_ids else None)),
            pad_frac=pad_frac,
        )

    def get_command(self):
        """
        Constructs the command based on the current UI settings.

        Returns
        -------
        list
            The command as a list of strings.
        """
        import astropy.time
        import astropy.units as u

        command = ['gx-fov2box']
        has_entry = self._has_entry_box()
        jump_action = self._get_jump_action()
        final_stage = "chr"

        # Entry-based modes keep CLI minimal; modify mode builds a fresh script from GUI fields.
        if has_entry and jump_action != "modify":
            command += ['--data-dir', self.sdo_data_edit.text()]
            command += ['--gxmodel-dir', self.gx_model_edit.text()]
            command += ['--entry-box', self.external_box_edit.text()]
        else:
            time = astropy.time.Time(self.model_time_edit.dateTime().toPyDateTime())
            command += ['--time', time.to_datetime().strftime('%Y-%m-%dT%H:%M:%S')]
            command += ['--coords', self.coord_x_edit.text(), self.coord_y_edit.text()]
            if self.hpc_radio_button.isChecked():
                command += ['--hpc']
            elif self.hgc_radio_button.isChecked():
                command += ['--hgc']
            else:
                command += ['--hgs']
            if self.proj_top_radio.isChecked():
                command += ['--top']
            else:
                command += ['--cea']

            command += ['--box-dims', self.grid_x_edit.text(), self.grid_y_edit.text(), self.grid_z_edit.text()]
            command += ['--dx-km', f'{float(self.res_edit.text()):.3f}']
            command += ['--pad-frac', f'{float(self.padding_size_edit.text()) / 100:.2f}']
            command += ['--data-dir', self.sdo_data_edit.text()]
            command += ['--gxmodel-dir', self.gx_model_edit.text()]

        if self.download_aia_euv.isChecked():
            command += ['--euv']
        if self.download_aia_uv.isChecked():
            command += ['--uv']
        if self._selected_download_backend() == "fido":
            command += ['--use-fido']
        if not self._use_cached_downloads():
            command += ['--force-download']
        refmaps_path = self.refmaps_path_edit.text().strip() if hasattr(self, "refmaps_path_edit") else ""
        if refmaps_path:
            command += ['--refmaps-path', refmaps_path]

        if self.save_empty_box.isChecked():
            command += ['--save-empty-box']
        if self.save_potential_box.isChecked():
            command += ['--save-potential']
        if self.save_bounds_box.isChecked():
            command += ['--save-bounds']
        if self.save_nas_box.isChecked():
            command += ['--save-nas']
        if self.save_gen_box.isChecked():
            command += ['--save-gen']
        if self.add_save_chromo_box.isChecked():
            command += ['--save-chr']

        if self.stop_early_box.isChecked():
            final_stage = "dl"
        elif self.empty_box_only_box.isChecked():
            final_stage = "none"
        elif self.stop_after_potential_box.isChecked():
            final_stage = "pot"
        elif self.nlfff_only_box.isChecked():
            final_stage = "nas"
        elif self.generic_only_box.isChecked() or not self.add_save_chromo_box.isChecked():
            final_stage = "gen"
        command += ['--stop-after', final_stage]

        if self.skip_nlfff_extrapolation.isChecked():
            command += ['--use-potential']
        if self.skip_line_computation_box.isChecked():
            command += ['--skip-lines']
        if self.center_vox_box.isChecked():
            command += ['--center-vox']

        if jump_action == 'rebuild_obs':
            command += ['--rebuild']
        elif jump_action == 'rebuild_none':
            command += ['--rebuild-from-none']
        elif (
            has_entry
            and jump_action == "continue"
            and final_stage == "chr"
            and (self._entry_stage_detected or "").upper() in ("GEN", "CHR")
        ):
            command += ['--jump2chromo']

        # Disambiguation affects fresh scripts (new or modify mode) only.
        if ((not has_entry) or jump_action == "modify") and self.disambig_sfq_radio.isChecked():
            command += ['--sfq']

        if self.info_only_box is not None and self.info_only_box.isChecked():
            command += ['--info']

        if self._selector_fov is not None:
            command += ['--observer-name', self._selector_observer_name]
            command += ['--fov-xc', f'{self._selector_fov.center_x_arcsec:.2f}']
            command += ['--fov-yc', f'{self._selector_fov.center_y_arcsec:.2f}']
            command += ['--fov-xsize', f'{self._selector_fov.width_arcsec:.2f}']
            command += ['--fov-ysize', f'{self._selector_fov.height_arcsec:.2f}']
            if self._selector_square_fov:
                command += ['--square-fov']

        return command

    def _selected_download_backend(self) -> str:
        if hasattr(self, "download_backend_drms_radio") and self.download_backend_drms_radio.isChecked():
            return "drms"
        return "fido"

    def _set_download_backend(self, backend: str) -> None:
        key = str(backend or "drms").strip().lower()
        if hasattr(self, "download_backend_drms_radio"):
            self.download_backend_drms_radio.setChecked(key == "drms")
        if hasattr(self, "download_backend_fido_radio"):
            self.download_backend_fido_radio.setChecked(key != "drms")

    def _use_cached_downloads(self) -> bool:
        return not hasattr(self, "download_use_cache_box") or self.download_use_cache_box.isChecked()

    def _set_use_cached_downloads(self, use_cache: bool) -> None:
        if hasattr(self, "download_use_cache_box"):
            self.download_use_cache_box.setChecked(bool(use_cache))

    def execute_command(self):
        """
        Executes the constructed command.
        """
        if self._gxbox_proc is not None and self._gxbox_proc.poll() is None:
            QMessageBox.warning(self, "GXbox Running", "A GXbox process is already running.")
            return

        self._save_session_state_to_settings()
        modify_mode = self._get_jump_action() == "modify"
        command = self.get_command()
        if modify_mode and self._has_entry_box():
            self.external_box_edit.blockSignals(True)
            self.external_box_edit.clear()
            self.external_box_edit.blockSignals(False)
            self._last_valid_entry_box = ""
            self._entry_stage_detected = None
            self._entry_type_detected = None
            self._selector_unsaved_session_active = False
            self._sync_pipeline_options()
            self.update_command_display()
            self.status_log_edit.append("Modify mode consumed the entry box; cleared entry-box field before run.")
        self._proc_command = list(command)
        self._pending_stop_after = self._command_stop_after(command)
        self._last_model_path = None
        self.send_to_viewer_button.setEnabled(False)
        try:
            self._gxbox_proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self._proc_output_queue = queue.SimpleQueue()
            self._proc_partial_line = ""
            if self._gxbox_proc.stdout is not None:
                self._proc_reader_thread = threading.Thread(
                    target=self._read_process_output,
                    args=(self._gxbox_proc.stdout, self._proc_output_queue),
                    daemon=True,
                )
                self._proc_reader_thread.start()
            self.execute_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_log_edit.append("Command started: " + " ".join(command))
            self._proc_timer.start()
        except Exception as e:
            self._proc_command = None
            self._pending_stop_after = None
            QMessageBox.critical(self, "Execution Error", f"Failed to start command: {e}")
            self.status_log_edit.append("Command failed to start")

    @staticmethod
    def _read_process_output(stream, output_queue: queue.SimpleQueue[str | None]) -> None:
        try:
            for line in iter(stream.readline, ""):
                output_queue.put(line)
        except Exception as exc:
            output_queue.put(f"\nGUI process-reader error: {exc}\n")
        finally:
            try:
                stream.close()
            except Exception:
                pass
            output_queue.put(None)

    @staticmethod
    def _command_stop_after(command) -> str | None:
        try:
            for i, token in enumerate(command[:-1]):
                if token == "--stop-after":
                    return str(command[i + 1])
        except Exception:
            return None
        return None

    def _launch_download_fov_selector_placeholder(self):
        """
        Post-download hook for the standalone FOV/box selector GUI.

        The visualization backend is still scaffold-level, but the launch/apply
        contract is fully wired:
        - build session input from current pyAMPP GUI fields
        - open selector dialog
        - apply accepted geometry back to the pyAMPP GUI
        """
        try:
            session_input = self._build_download_selector_session_input()
            result = run_fov_box_selector(session_input=session_input, parent=self)
            if result is None:
                self.status_log_edit.append("FOV/box selector cancelled.")
                return
            self.apply_selector_result(result)
            self.status_log_edit.append("FOV/box selector accepted.")
        except Exception as exc:
            self.status_log_edit.append(f"FOV/box selector error: {exc}")

    def _drain_process_output(self):
        if self._gxbox_proc is None:
            return
        chunks = []
        while True:
            try:
                chunk = self._proc_output_queue.get_nowait()
            except queue.Empty:
                break
            if chunk is None:
                continue
            chunks.append(chunk)
        if not chunks:
            return
        complete_lines, self._proc_partial_line = _split_process_output_text(
            self._proc_partial_line,
            "".join(chunks),
        )
        for line in complete_lines:
            if line.strip():
                self.status_log_edit.append(line)

    def stop_command(self):
        """
        Stops the running GXbox process if any.
        """
        if self._gxbox_proc is None or self._gxbox_proc.poll() is not None:
            self.status_log_edit.append("No running command to stop")
            self.stop_button.setEnabled(False)
            self.execute_button.setEnabled(True)
            return

        self.status_log_edit.append("Stopping command...")
        try:
            self._gxbox_proc.terminate()
            self._gxbox_proc.wait(timeout=5)
            self._drain_process_output()
            self.status_log_edit.append("Command stopped")
        except subprocess.TimeoutExpired:
            self._gxbox_proc.kill()
            self._drain_process_output()
            self.status_log_edit.append("Command killed")
        finally:
            self._gxbox_proc = None
            self._proc_command = None
            self._pending_stop_after = None
            self.stop_button.setEnabled(False)
            self.execute_button.setEnabled(True)
            self._proc_timer.stop()

    def _check_gxbox_process(self):
        try:
            if self._gxbox_proc is None:
                self._proc_timer.stop()
                return

            self._drain_process_output()
            if self._gxbox_proc.poll() is None:
                return

            self._drain_process_output()
            if self._proc_partial_line.strip():
                self.status_log_edit.append(self._proc_partial_line)
                self._proc_partial_line = ""
            exit_code = self._gxbox_proc.returncode
            if exit_code == 0:
                self.status_log_edit.append("Command finished successfully")
                if self._pending_stop_after == "dl":
                    self._last_model_path = None
                    self._refresh_viewer_button_state()
                    self._launch_download_fov_selector_placeholder()
                else:
                    self._selector_unsaved_session_active = False
                    self._update_last_model_path()
                if self._pending_stop_after is not None and self._pending_stop_after != "dl":
                    self._launch_stop_stage_box_view2d()
            else:
                self.status_log_edit.append(f"Command exited with code {exit_code}")
        except Exception as exc:
            self.status_log_edit.append(f"GUI process-monitor error: {exc}")
        finally:
            if self._gxbox_proc is not None and self._gxbox_proc.poll() is not None:
                self._gxbox_proc = None
                self._proc_reader_thread = None
                self._proc_command = None
                self._pending_stop_after = None
                self.stop_button.setEnabled(False)
                self.execute_button.setEnabled(True)
                self._proc_timer.stop()

    def save_command(self):
        """
        Saves the current gx-fov2box command as a shell script.
        """
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save gx-fov2box Script",
            str(Path.cwd() / "gx_fov2box.sh"),
            "Shell Scripts (*.sh);;Text Files (*.txt);;All Files (*)",
        )
        if not file_name:
            return

        command_text = " ".join(self.get_command()).strip()
        script_text = f"#!/bin/sh\n{command_text}\n"
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(script_text)
            if str(file_name).lower().endswith(".sh"):
                os.chmod(file_name, 0o755)
            self.status_log_edit.append(f"Saved gx-fov2box script: {file_name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not save gx-fov2box script:\n{exc}")

    def copy_command(self):
        """
        Copies the current gx-fov2box command line to the clipboard.
        """
        QApplication.clipboard().setText(" ".join(self.get_command()).strip())
        self.status_log_edit.append("Copied gx-fov2box command to clipboard.")

    def refresh_command(self):
        """
        Refreshes the current session.
        """
        # Placeholder for refreshing command
        self.status_log_edit.append("Command refreshed")

    def clear_command(self):
        """
        Clears the status log.
        """
        # Placeholder for clearing command
        self.status_log_edit.clear()

    def clear_console(self):
        """
        Clears the console panel.
        """
        self.status_log_edit.clear()
        self._last_model_path = None
        self._refresh_viewer_button_state()

    def copy_console(self):
        """
        Copies the full console text to clipboard.
        """
        QApplication.clipboard().setText(self.status_log_edit.toPlainText())

    def save_console(self):
        """
        Saves the console output to a text file.
        """
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Console Output",
            str(Path.cwd() / "pyampp_console.txt"),
            "Text Files (*.txt);;Log Files (*.log);;All Files (*)",
        )
        if not file_name:
            return
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(self.status_log_edit.toPlainText())
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not save console output:\n{exc}")

    def _update_last_model_path(self):
        text = self.status_log_edit.toPlainText()
        candidates = re.findall(r"^- (.+\.h5)\s*$", text, flags=re.MULTILINE)
        for raw in reversed(candidates):
            p = Path(raw).expanduser()
            if p.exists():
                self._last_model_path = str(p)
                self._refresh_viewer_button_state()
                return
        root = Path(self.gx_model_edit.text()).expanduser()
        if not root.exists():
            return
        newest = None
        newest_mtime = -1.0
        for p in root.rglob("*.h5"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest = p
        if newest is not None:
            self._last_model_path = str(newest)
            self._refresh_viewer_button_state()

    def _current_viewable_model_path(self) -> Path | None:
        if self._last_model_path:
            model_path = Path(self._last_model_path).expanduser()
            if model_path.exists() and model_path.is_file():
                return model_path
        if self._has_entry_box():
            entry_path = Path(self.external_box_edit.text().strip()).expanduser()
            if entry_path.exists() and entry_path.is_file():
                return entry_path
        return None

    def _refresh_viewer_button_state(self) -> None:
        model_path = self._current_viewable_model_path()
        self.send_to_viewer_button.setEnabled(model_path is not None)

    def send_to_gxbox_view(self):
        model_path = self._current_viewable_model_path()
        if model_path is None:
            QMessageBox.information(self, "No Model", "No generated model was found to send.")
            return
        self._launch_box_view3d(model_path)

    def _adopt_entry_box_for_continue(self, model_path: Path) -> None:
        path_text = str(model_path.expanduser())
        self.external_box_edit.blockSignals(True)
        self.external_box_edit.setText(path_text)
        self.external_box_edit.blockSignals(False)
        self._selector_unsaved_session_active = False
        try:
            self.update_external_box_dir()
            self._set_jump_action("continue")
            self._sync_pipeline_options()
            self.update_command_display()
            self.status_log_edit.append(f"Adopted stop-stage box for continue: {path_text}")
        except Exception as exc:
            self.status_log_edit.append(f"Could not adopt stop-stage box for continue: {exc}")

    def _check_view2d_process(self):
        try:
            if self._view2d_proc is None:
                self._view2d_timer.stop()
                return
            if self._view2d_proc.poll() is None:
                if self._view2d_launch_pending:
                    self._view2d_launch_pending = False
                    if self._view2d_target_path:
                        self.status_log_edit.append(f"Launched gxbox-view2d with: {self._view2d_target_path}")
                return
            exit_code = self._view2d_proc.returncode
            startup_output = self._consume_view2d_output(self._view2d_proc)
            target_path = self._view2d_target_path
            adopt_on_close = bool(self._view2d_adopt_on_close)
            launch_pending = bool(self._view2d_launch_pending)
            self._view2d_proc = None
            self._view2d_target_path = None
            self._view2d_adopt_on_close = False
            self._view2d_launch_pending = False
            self._view2d_timer.stop()
            if launch_pending:
                self.status_log_edit.append(f"gxbox-view2d exited during startup (code {exit_code}).")
                if startup_output:
                    self.status_log_edit.append(f"gxbox-view2d startup output: {startup_output}")
                return
            if adopt_on_close and target_path is not None and Path(target_path).exists():
                self._adopt_entry_box_for_continue(Path(target_path))
            elif exit_code not in (None, 0):
                self.status_log_edit.append(f"gxbox-view2d exited with code {exit_code}.")
                if startup_output:
                    self.status_log_edit.append(f"gxbox-view2d output: {startup_output}")
        except Exception as exc:
            self.status_log_edit.append(f"gxbox-view2d monitor error: {exc}")
            self._view2d_proc = None
            self._view2d_target_path = None
            self._view2d_adopt_on_close = False
            self._view2d_launch_pending = False
            self._view2d_timer.stop()

    @staticmethod
    def _consume_view2d_output(proc: subprocess.Popen) -> str:
        try:
            stdout_text, stderr_text = proc.communicate(timeout=0.2)
        except Exception:
            return ""
        parts = []
        if stdout_text:
            text = stdout_text.strip()
            if text:
                parts.append(text)
        if stderr_text:
            text = stderr_text.strip()
            if text:
                parts.append(text)
        return " | ".join(parts)

    @staticmethod
    def _consume_view3d_output(proc: subprocess.Popen) -> str:
        try:
            stdout_text, stderr_text = proc.communicate(timeout=0.2)
        except Exception:
            return ""
        parts = []
        if stdout_text:
            text = stdout_text.strip()
            if text:
                parts.append(text)
        if stderr_text:
            text = stderr_text.strip()
            if text:
                parts.append(text)
        return " | ".join(parts)

    def _launch_box_view2d(self, model_path: Path, adopt_on_close: bool = False):
        try:
            self._view2d_proc = subprocess.Popen([
                "gxbox-view2d",
                str(model_path.expanduser()),
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self._view2d_target_path = str(model_path.expanduser())
            self._view2d_adopt_on_close = bool(adopt_on_close)
            self._view2d_launch_pending = True
            self._view2d_timer.start()
            self.status_log_edit.append(f"Starting gxbox-view2d for: {model_path}")
        except Exception as exc:
            self._view2d_proc = None
            self._view2d_target_path = None
            self._view2d_adopt_on_close = False
            self._view2d_launch_pending = False
            self.status_log_edit.append(f"Could not launch gxbox-view2d: {exc}")

    def _launch_box_view3d(self, model_path: Path):
        try:
            self._view3d_proc = subprocess.Popen([
                "gxbox-view3d",
                "--pipeline-child",
                str(model_path.expanduser()),
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self._view3d_target_path = str(model_path.expanduser())
            self._view3d_launch_pending = True
            self._view3d_timer.start()
            self.status_log_edit.append(f"Starting gxbox-view3d for: {model_path}")
        except Exception as exc:
            self._view3d_proc = None
            self._view3d_target_path = None
            self._view3d_launch_pending = False
            QMessageBox.critical(self, "Launch Failed", f"Could not launch gxbox-view3d:\n{exc}")

    def _launch_stop_stage_box_view2d(self):
        if not self._last_model_path:
            self.status_log_edit.append("Stop-stage completed, but no generated box file was found for gxbox-view2d.")
            return
        model_path = Path(self._last_model_path).expanduser()
        if not model_path.exists():
            self.status_log_edit.append(f"Stop-stage box missing for gxbox-view2d: {model_path}")
            return
        self._launch_box_view2d(model_path, adopt_on_close=True)

    def _check_view3d_process(self):
        try:
            if self._view3d_proc is None:
                self._view3d_timer.stop()
                return
            if self._view3d_proc.poll() is None:
                if self._view3d_launch_pending:
                    self._view3d_launch_pending = False
                    if self._view3d_target_path:
                        self.status_log_edit.append(f"Launched gxbox-view3d with: {self._view3d_target_path}")
                return
            exit_code = self._view3d_proc.returncode
            startup_output = self._consume_view3d_output(self._view3d_proc)
            launch_pending = bool(self._view3d_launch_pending)
            self._view3d_proc = None
            self._view3d_target_path = None
            self._view3d_launch_pending = False
            self._view3d_timer.stop()
            if launch_pending:
                self.status_log_edit.append(f"gxbox-view3d exited during startup (code {exit_code}).")
                if startup_output:
                    self.status_log_edit.append(f"gxbox-view3d startup output: {startup_output}")
                return
            if exit_code not in (None, 0):
                self.status_log_edit.append(f"gxbox-view3d exited with code {exit_code}.")
                if startup_output:
                    self.status_log_edit.append(f"gxbox-view3d output: {startup_output}")
        except Exception as exc:
            self.status_log_edit.append(f"gxbox-view3d monitor error: {exc}")
            self._view3d_proc = None
            self._view3d_target_path = None
            self._view3d_launch_pending = False
            self._view3d_timer.stop()


@app.command()
def main(
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug mode with an interactive IPython session."
        )
):
    """
    Entry point for the PyAmppGUI application.

    This function initializes the PyQt application and displays the main GUI window.
    Session/default field state is established inside ``PyAmppGUI`` during startup.

    :param debug: Enable debug mode with an interactive IPython session, defaults to False
    :type debug: bool, optional
    :raises SystemExit: Exits the application loop when the GUI is closed
    :return: None
    :rtype: NoneType

    Examples
    --------
    .. code-block:: bash

        pyampp
    """

    app_qt = QApplication([])
    pyampp = PyAmppGUI()

    if debug:
        # Start an interactive IPython session for debugging
        import IPython
        IPython.embed()

        # If any matplotlib plots are created, show them
        import matplotlib.pyplot as plt
        plt.show()
    sys.exit(app_qt.exec_())

if __name__ == '__main__':
    app()
