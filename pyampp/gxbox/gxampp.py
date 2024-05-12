import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QComboBox, QRadioButton,
                             QCheckBox, QGridLayout, QGroupBox, QVBoxLayout, QHBoxLayout, QDateTimeEdit, QTextEdit, QFileDialog)
from pyampp.util.config import *


class SolarDataGUI(QMainWindow):
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
        self.add_command_display()
        self.add_status_log()

        # Set window properties
        self.setWindowTitle('Solar Data Model GUI')
        self.setGeometry(100, 100, 800, 600)  # Modify as needed
        self.show()

    def add_data_repository_section(self):
        # Group box for data repositories
        group_box = QGroupBox("Data Repositories")
        layout = QGridLayout()

        # SDO Data Repository
        layout.addWidget(QLabel("SDO Data Repository:"), 0, 0)
        self.sdo_data_edit = QLineEdit()
        self.sdo_data_edit.setText(DOWNLOAD_DIR)
        layout.addWidget(self.sdo_data_edit, 0, 1)
        sdo_browse_button = QPushButton("Browse")
        sdo_browse_button.clicked.connect(self.open_sdo_file_dialog)
        layout.addWidget(sdo_browse_button, 0, 2)

        # GX Model Repository
        layout.addWidget(QLabel("GX Model Repository:"), 1, 0)
        self.gx_model_edit = QLineEdit()
        self.gx_model_edit.setText(GXMODEL_DIR)
        layout.addWidget(self.gx_model_edit, 1, 1)
        gx_browse_button = QPushButton("Browse")
        gx_browse_button.clicked.connect(self.open_gx_file_dialog)
        layout.addWidget(gx_browse_button, 1, 2)

        # External Box Path
        layout.addWidget(QLabel("External Box Path:"), 2, 0)
        self.external_box_edit = QLineEdit()
        layout.addWidget(self.external_box_edit, 2, 1)
        external_browse_button = QPushButton("Browse")
        external_browse_button.clicked.connect(self.open_external_file_dialog)
        layout.addWidget(external_browse_button, 2, 2)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def open_sdo_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory", DOWNLOAD_DIR)
        if file_name:
            self.sdo_data_edit.setText(file_name)

    def open_gx_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory",GXMODEL_DIR)
        if file_name:
            self.gx_model_edit.setText(file_name)

    def open_external_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory")
        if file_name:
            self.external_box_edit.setText(file_name)

    def add_model_configuration_section(self):
        group_box = QGroupBox("Model Configuration")
        layout = QGridLayout()

        # Jump-to Action
        layout.addWidget(QLabel("Jump-to Action:"), 0, 0)
        self.jump_to_action_combo = QComboBox()
        self.jump_to_action_combo.addItems(['none', 'potential', 'nIff', 'lines', 'chromo'])
        layout.addWidget(self.jump_to_action_combo, 0, 1)

        # Model Time
        layout.addWidget(QLabel("Model Time:"), 1, 0)
        self.model_time_edit = QDateTimeEdit()
        layout.addWidget(self.model_time_edit, 1, 1)

        # Model Coordinates
        layout.addWidget(QLabel("Model Coordinates X:"), 2, 0)
        self.coord_x_edit = QLineEdit()
        layout.addWidget(self.coord_x_edit, 2, 1)
        layout.addWidget(QLabel("Y:"), 2, 2)
        self.coord_y_edit = QLineEdit()
        layout.addWidget(self.coord_y_edit, 2, 3)

        # Coordinate System
        self.heliocentric_radio = QRadioButton("Heliocentric")
        self.carrington_radio = QRadioButton("Carrington")
        layout.addWidget(self.heliocentric_radio, 3, 1)
        layout.addWidget(self.carrington_radio, 3, 2)

        # Model Gridpoints
        layout.addWidget(QLabel("Model Gridpoints X:"), 4, 0)
        self.grid_x_edit = QLineEdit()
        layout.addWidget(self.grid_x_edit, 4, 1)
        layout.addWidget(QLabel("Y:"), 4, 2)
        self.grid_y_edit = QLineEdit()
        layout.addWidget(self.grid_y_edit, 4, 3)
        layout.addWidget(QLabel("Z:"), 4, 4)
        self.grid_z_edit = QLineEdit()
        layout.addWidget(self.grid_z_edit, 4, 5)


        # Resolution
        layout.addWidget(QLabel("Res. (km):"), 4, 6)
        self.resolution_edit = QLineEdit()
        layout.addWidget(self.resolution_edit, 4, 7)

        # Buffer Zone Size
        layout.addWidget(QLabel("Buffer Zone Size (%):"), 5, 0)
        self.buffer_size_edit = QLineEdit()
        layout.addWidget(self.buffer_size_edit, 5, 1)

        group_box.setLayout(layout)
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

    def add_command_display(self):
        # Command Line Equivalent Display
        self.command_display_edit = QTextEdit()
        self.command_display_edit.setReadOnly(True)
        self.command_display_edit.setMaximumHeight(75)  # Adjusted smaller height for the command display area
        self.main_layout.addWidget(self.command_display_edit)

    def add_status_log(self):
        # Status Log
        self.status_log_edit = QTextEdit()
        self.status_log_edit.setReadOnly(True)
        self.main_layout.addWidget(self.status_log_edit)

def main():
    app = QApplication(sys.argv)
    ex = SolarDataGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
