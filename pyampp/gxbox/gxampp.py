import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
                             QRadioButton,
                             QCheckBox, QGridLayout, QGroupBox, QVBoxLayout, QHBoxLayout, QDateTimeEdit,
                             QCalendarWidget, QTextEdit, QMessageBox,
                             QFileDialog)
from PyQt5.QtGui import QIcon,QFont
from PyQt5.QtCore import QSize, QDateTime, Qt
from pyampp.util.config import *
import pyampp
from pathlib import Path

base_dir = Path(pyampp.__file__).parent
svg_dir = base_dir / 'gxbox' / 'UI'


class PyAmppGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Adding different sections
        self.add_data_repository_section()
        self.add_model_configuration_section()
        self.add_options_section()
        self.add_cmd_display()
        self.add_cmd_buttons()
        self.add_status_log()

        # Set window properties
        self.setWindowTitle('Solar Data Model GUI')
        self.setGeometry(100, 100, 800, 600)  # Modify as needed
        self.update_command_display()
        self.show()

    def add_data_repository_section(self):
        # Group box for data repositories
        group_box = QGroupBox("Data Repositories")
        layout = QGridLayout()

        # SDO Data Repository
        layout.addWidget(QLabel("SDO Data Repository:"), 0, 0)
        self.sdo_data_edit = QLineEdit()
        self.sdo_data_edit.setText(DOWNLOAD_DIR)
        self.sdo_data_edit.setToolTip("Path to the SDO data repository")
        self.sdo_data_edit.returnPressed.connect(self.update_sdo_data_dir)
        layout.addWidget(self.sdo_data_edit, 0, 1)
        sdo_browse_button = QPushButton("Browse")
        sdo_browse_button.clicked.connect(self.open_sdo_file_dialog)
        layout.addWidget(sdo_browse_button, 0, 2)

        # GX Model Repository
        layout.addWidget(QLabel("GX Model Repository:"), 1, 0)
        self.gx_model_edit = QLineEdit()
        self.gx_model_edit.setText(GXMODEL_DIR)
        self.gx_model_edit.setToolTip("Path to the GX model repository")
        self.gx_model_edit.returnPressed.connect(self.update_gxmodel_dir)
        layout.addWidget(self.gx_model_edit, 1, 1)
        gx_browse_button = QPushButton("Browse")
        gx_browse_button.clicked.connect(self.open_gx_file_dialog)
        layout.addWidget(gx_browse_button, 1, 2)

        # External Box Path
        layout.addWidget(QLabel("External Box Path:"), 2, 0)
        self.external_box_edit = QLineEdit()
        self.external_box_edit.setText(os.getcwd())
        self.external_box_edit.setToolTip("Path to the external box. Default is the current directory.")
        self.external_box_edit.returnPressed.connect(self.update_external_box_dir)
        layout.addWidget(self.external_box_edit, 2, 1)
        external_browse_button = QPushButton("Browse")
        external_browse_button.clicked.connect(self.open_external_file_dialog)
        layout.addWidget(external_browse_button, 2, 2)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def update_sdo_data_dir(self):
        new_path = self.sdo_data_edit.text()
        self.update_dir(new_path, DOWNLOAD_DIR)
        self.update_command_display()

    def update_gxmodel_dir(self):
        new_path = self.gx_model_edit.text()
        self.update_dir(new_path, GXMODEL_DIR)
        self.update_command_display()

    def update_external_box_dir(self):
        new_path = self.external_box_edit.text()
        self.update_dir(new_path, os.getcwd())
        self.update_command_display()

    def update_dir(self, new_path, default_path):
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
                    self.sdo_data_edit.setText(DOWNLOAD_DIR)
        # else:
        #     QMessageBox.warning(self, "Invalid Path", "The specified path is not a valid absolute path.")

    def open_sdo_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory", DOWNLOAD_DIR)
        if file_name:
            self.sdo_data_edit.setText(file_name)

    def open_gx_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory", GXMODEL_DIR)
        if file_name:
            self.gx_model_edit.setText(file_name)

    def open_external_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        if file_name:
            self.external_box_edit.setText(file_name)

    def add_model_configuration_section(self):
        group_box = QGroupBox("Model Configuration")
        main_layout = QVBoxLayout()

        # Jump-to Action
        jump_to_action_layout = QHBoxLayout()
        jump_to_action_layout.addWidget(QLabel("Jump-to Action:"))
        self.jump_to_action_combo = QComboBox()
        self.jump_to_action_combo.addItems(['none', 'potential', 'nIff', 'lines', 'chromo'])
        jump_to_action_layout.addWidget(self.jump_to_action_combo)
        jump_to_action_layout.addStretch()  # Add stretch to push elements to the left
        main_layout.addLayout(jump_to_action_layout)

        # Model Time
        model_time_layout = QHBoxLayout()
        model_time_layout.addWidget(QLabel("Time [UT]:"))
        self.model_time_edit = QDateTimeEdit()
        self.model_time_edit.setDateTime(QDateTime.currentDateTimeUtc())
        self.model_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.model_time_edit.setCalendarPopup(True)
        self.model_time_edit.setDateTimeRange(QDateTime(2010, 1, 1, 0, 0, 0),QDateTime(QDateTime.currentDateTimeUtc()))
        self.model_time_edit.setCalendarWidget(QCalendarWidget())
        self.model_time_edit.setToolTip("Model time in UT")
        self.model_time_edit.dateTimeChanged.connect(self.update_command_display)

        model_time_layout.addWidget(self.model_time_edit)
        model_time_layout.addStretch()  # Add stretch
        main_layout.addLayout(model_time_layout)

        # Model Coordinates
        coords_layout = QHBoxLayout()
        self.coord_label = QLabel("Center Coords in arcsec:")
        coords_layout.addWidget(self.coord_label)
        self.coord_x_label = QLabel("X:")
        coords_layout.addWidget(self.coord_x_label)
        self.coord_x_edit = QLineEdit("0.0")
        self.coord_x_edit.setToolTip("Solar X coordinate of the model center in arcsec")
        self.coord_x_edit.returnPressed.connect(self.update_command_display)
        coords_layout.addWidget(self.coord_x_edit)
        self.coord_y_label = QLabel("Y:")
        coords_layout.addWidget(self.coord_y_label)
        self.coord_y_edit = QLineEdit("0.0")
        self.coord_y_edit.setToolTip("Solar Y coordinate of the model center in arcsec")
        self.coord_y_edit.returnPressed.connect(self.update_command_display)
        self.coord_x_label.setFixedWidth(30)  # Adjust the width as necessary
        self.coord_y_label.setFixedWidth(30)  # Adjust the width as necessary
        self.coord_label.setFixedWidth(150)  # Adjust the width for "Center Coords in deg"
        self.coord_x_edit.setFixedWidth(100)
        self.coord_y_edit.setFixedWidth(100)
        coords_layout.addWidget(self.coord_y_edit)

        # Coordinate System
        self.hpc_radio_button = QRadioButton("Helioprojective")
        self.hpc_radio_button.setChecked(True)
        self.hpc_radio_button.setToolTip("Use Helioprojective coordinates frame to define the model center")
        self.hpc_radio_button.toggled.connect(self.update_hpc_state)
        coords_layout.addWidget(self.hpc_radio_button)
        self.hgs_radio_button = QRadioButton("Stonyhurst")
        self.hgs_radio_button.setToolTip("Use Heliographic Stonyhurst coordinates frame to define the model center")
        self.hgs_radio_button.toggled.connect(self.update_hgs_state)
        coords_layout.addWidget(self.hgs_radio_button)
        coords_layout.addStretch()  # Add stretch
        main_layout.addLayout(coords_layout)
        coords_layout.addStretch(1)  # Adds stretchable space at the end of the layout

        # Model Gridpoints
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid Size in pix"))
        grid_layout.addWidget(QLabel("X:"))
        self.grid_x_edit = QLineEdit("64")
        self.grid_x_edit.setToolTip("Number of grid points in the x-direction")
        self.grid_x_edit.returnPressed.connect(self.update_command_display)
        self.grid_x_edit.setFixedWidth(100)
        grid_layout.addWidget(self.grid_x_edit)
        grid_layout.addWidget(QLabel("Y:"))
        self.grid_y_edit = QLineEdit("64")
        self.grid_y_edit.setToolTip("Number of grid points in the y-direction")
        self.grid_y_edit.returnPressed.connect(self.update_command_display)
        self.grid_y_edit.setFixedWidth(100)
        grid_layout.addWidget(self.grid_y_edit)
        grid_layout.addWidget(QLabel("Z:"))
        self.grid_z_edit = QLineEdit("64")
        self.grid_z_edit.setToolTip("Number of grid points in the z-direction")
        self.grid_z_edit.returnPressed.connect(self.update_command_display)
        self.grid_z_edit.setFixedWidth(100)
        grid_layout.addWidget(self.grid_z_edit)
        grid_layout.addStretch()  # Add stretch
        main_layout.addLayout(grid_layout)

        # Resolution and Padding Zone Size
        res_padding_layout = QHBoxLayout()
        res_padding_layout.addWidget(QLabel("Res. [km]:"))
        self.res_edit = QLineEdit('1400')
        self.res_edit.setFixedWidth(100)
        self.res_edit.setToolTip("Resolution in km")
        self.res_edit.returnPressed.connect(self.update_command_display)
        res_padding_layout.addWidget(self.res_edit)
        res_padding_layout.addWidget(QLabel("Padding (%):"))
        self.padding_size_edit = QLineEdit()
        self.padding_size_edit.setFixedWidth(100)
        self.padding_size_edit.setToolTip(
            "Padding as a percentage of box dimensions, increases each side of the box for extended margins.")
        self.padding_size_edit.returnPressed.connect(self.update_command_display)
        self.padding_size_edit.setText("25")
        res_padding_layout.addWidget(self.padding_size_edit)
        res_padding_layout.addStretch()  # Add stretch
        main_layout.addLayout(res_padding_layout)

        # Set the main layout for the group box
        group_box.setLayout(main_layout)
        self.main_layout.addWidget(group_box)

    def add_options_section(self):
        group_box = QGroupBox("Options")
        layout = QGridLayout()

        # Save Options
        self.save_empty_box = QCheckBox("Save Empty Box")
        layout.addWidget(self.save_empty_box, 1, 0)
        self.save_potential_box = QCheckBox("Save Potential Box")
        layout.addWidget(self.save_potential_box, 1, 1)
        self.save_bounds_box = QCheckBox("Save Bounds Box")
        layout.addWidget(self.save_bounds_box, 1, 2)

        # Download Maps
        self.download_aia_euv = QCheckBox("Download AIA/EUV contextual maps")
        layout.addWidget(self.download_aia_euv, 0, 0)
        self.download_aia_uv = QCheckBox("Download AIA/UV contextual maps")
        layout.addWidget(self.download_aia_uv, 0, 1)

        # Execution Controls
        self.stop_after_potential_box = QCheckBox("Stop after the potential box is generated")
        layout.addWidget(self.stop_after_potential_box, 2, 0)
        self.skip_nlfff_extrapolation = QCheckBox("Skip NLFFF extrapolation")
        layout.addWidget(self.skip_nlfff_extrapolation, 2, 1)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def add_cmd_display(self):
        # Command Line Equivalent Display
        self.cmd_display_edit = QTextEdit()
        self.cmd_display_edit.setReadOnly(True)
        self.cmd_display_edit.setMaximumHeight(75)  # Adjusted smaller height for the command display area

        # Setting a monospace font and appropriate size
        font = QFont('Arial', 12)
        font.setWeight(QFont.Light)
        self.cmd_display_edit.setFont(font)
        self.main_layout.addWidget(self.cmd_display_edit)

    def add_cmd_buttons(self):
        # Command Buttons
        cmd_button_layout = QHBoxLayout()
        self.execute_button = QPushButton("Execute")
        self.execute_button.setText("")
        self.execute_button.setIcon(QIcon(str(svg_dir / 'play.svg')))
        self.execute_button.clicked.connect(self.execute_command)
        self.execute_button.setToolTip("Create GXbox with the given parameters")
        self.execute_button.setFixedWidth(50)
        cmd_button_layout.addWidget(self.execute_button)

        self.save_button = QPushButton("Save")
        self.save_button.setText("")
        self.save_button.setIcon(QIcon(str(svg_dir / 'save.svg')))
        self.save_button.clicked.connect(self.save_command)
        self.save_button.setToolTip("Save the GXbox")
        self.save_button.setFixedWidth(50)
        cmd_button_layout.addWidget(self.save_button)

        self.clear_button = QPushButton("Refresh")
        self.clear_button.setText("")
        self.clear_button.setIcon(QIcon(str(svg_dir / 'refresh.svg')))
        self.clear_button.clicked.connect(self.refresh_command)
        self.clear_button.setToolTip("Refresh the session")
        self.clear_button.setFixedWidth(50)
        cmd_button_layout.addWidget(self.clear_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setText("")
        self.clear_button.setIcon(QIcon(str(svg_dir / 'clear.svg')))
        self.clear_button.clicked.connect(self.clear_command)
        self.clear_button.setToolTip("Clear the status log")
        self.clear_button.setFixedWidth(50)
        cmd_button_layout.addWidget(self.clear_button)
        cmd_button_layout.addStretch()  # Add stretch
        self.main_layout.addLayout(cmd_button_layout)

    def add_status_log(self):
        # Status Log
        self.status_log_edit = QTextEdit()
        self.status_log_edit.setReadOnly(True)
        self.status_log_edit.setMinimumHeight(200)  # Adjusted smaller height for the command display area
        self.main_layout.addWidget(self.status_log_edit)

    def update_command_display(self):
        command = self.get_command()
        self.cmd_display_edit.clear()
        self.cmd_display_edit.append(" ".join(command))

    def update_hpc_state(self, checked):
        if checked:
            self.coord_x_edit.setToolTip("Solar X coordinate of the model center in arcsec")
            self.coord_y_edit.setToolTip("Solar Y coordinate of the model center in arcsec")
            self.coord_label.setText("Center Coords  in arcsec")
            self.coord_x_label.setText("X:")
            self.coord_y_label.setText("Y:")
            self.update_command_display()

    def update_hgs_state(self, checked):
        if checked:
            self.coord_x_edit.setToolTip("Heliographic Stonyhurst Longitude of the model center in deg")
            self.coord_y_edit.setToolTip("Heliographic Stonyhurst Latitude of the model center in deg")
            self.coord_label.setText("Center Coords in deg")
            self.coord_x_label.setText("lon:")
            self.coord_y_label.setText("lat:")
            self.update_command_display()

    def get_command(self):
        import astropy.time
        import astropy.units as u
        command = ['python', os.path.join(base_dir, 'gxbox', 'gxbox_factory.py')]
        time = astropy.time.Time(self.model_time_edit.dateTime().toPyDateTime())
        command += ['--time',time.to_datetime().strftime('%Y-%m-%dT%H:%M:%S')]

        if self.hpc_radio_button.isChecked():
            command += ['--coords', self.coord_x_edit.text(), self.coord_y_edit.text(), '--hpc']
        else:
            command += ['--coords', self.coord_x_edit.text(), self.coord_y_edit.text(), '--hgs']

        command += ['--box_dims', self.grid_x_edit.text(), self.grid_y_edit.text(), self.grid_z_edit.text()]
        command += ['--box_res', f'{((int(self.res_edit.text()) * u.km).to(u.Mm)).value:.3f}']
        command += ['--pad_frac', f'{float(self.padding_size_edit.text()) / 100:.2f}']
        command += ['--data_dir', self.sdo_data_edit.text()]
        command += ['--gxmodel_dir', self.gx_model_edit.text()]
        command += ['--external_box', self.external_box_edit.text()]
        # print(command)
        return command

    def execute_command(self):
        self.status_log_edit.append("Command executed")
        import subprocess
        command = self.get_command()
        subprocess.run(command, check=True)

    def save_command(self):
        # Placeholder for saving command
        self.status_log_edit.append("Command saved")

    def refresh_command(self):
        # Placeholder for refreshing command
        self.status_log_edit.append("Command refreshed")

    def clear_command(self):
        # Placeholder for clearing command
        self.status_log_edit.clear()


def main():
    app = QApplication(sys.argv)
    pyampp = PyAmppGUI()
    pyampp.model_time_edit.setDateTime(QDateTime(2014, 11, 1, 16, 40))
    pyampp.coord_x_edit.setText('-632')
    pyampp.coord_y_edit.setText('-135')
    pyampp.update_command_display()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
