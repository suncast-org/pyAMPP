from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QLabel, \
    QPushButton, QSlider, QLineEdit, QCheckBox, QMessageBox, QMenu, QHeaderView, QFileDialog, QAction, QToolButton, \
    QToolBar
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from astropy.time import Time
from pyampp.gxbox.boxutils import validate_number, read_b3d_h5, write_b3d_h5
import pickle
import vtk

import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTreeView, \
    QGroupBox
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import numpy as np

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


class MagFieldViewer(BackgroundPlotter):
    """
    A class to visualize the magnetic field of a box using PyVista. It inherits from the BackgroundPlotter class.

    :param box: object
        The box containing magnetic field data.
    :param parent: object, optional
        The parent object (default is None).
    """

    def __init__(self, box, parent=None, box_norm_direction=None, box_view_up=None, time=None, b3dtype='nlfff', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box = box
        self.parent = parent
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
        self.streamlines_actor = None
        self.streamlines = None
        self.sphere_visible = True
        self.plane_visible = True
        self.scalar = 'bz'
        self.previous_params = {}
        self.previous_valid_values = {}
        self.scalar_selector = None
        self.scalar_selector_items = []
        self.center_x_input = None
        self.center_y_input = None
        self.center_z_input = None
        self.radius_input = None
        self.n_points_input = None
        self.slice_z_input = None
        self.vmin_input = None
        self.vmax_input = None
        self.update_button = None
        self.send_button = None
        self.parallel_proj_button = None
        self.timestr = time.to_datetime().strftime("_%Y%m%dT%H%M%S") if time is not None else ''
        self.b3dtype = b3dtype
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
        self.default_sph_cen_x = np.mean(self.grid_x)
        self.default_sph_cen_y = np.mean(self.grid_y)
        self.default_sph_cen_z = self.grid_zmin + self.grid_z.ptp() * 0.1

        # self.init_ui()
        self.init_grid()
        self.add_widgets_to_window()
        self.init_plot()
        self.show_axes_all()
        self.view_isometric()
        self.plane_checkbox.setChecked(False)
        self.app_window.setWindowTitle("GxBox 3D viewer")
        self.add_menu_options()  # Add this line to include menu options
        self.add_parallel_projection_button() # Add parallel projection button
        if self.box_norm_direction is not None and self.box_view_up is not None:
            self.add_observer_cam_button()  # Add this line to include the observer cam button

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
        Sets the camera to align with the normal direction vector of the observer's line of sight (LoS).

        This function reorients the camera so that it points along the observer's normal direction
        while ensuring the 'view up' vector is aligned to the target y-axis (0, 1, 0).

        Steps:
            1. Normalize the provided view up vector.
            2. Calculate the rotation axis using the cross product of the normalized view up vector
               and the target y-axis.
            3. Compute the angle between the normalized view up vector and the target y-axis.
            4. Generate a rotation matrix using Rodrigues' rotation formula.
            5. Apply the rotation matrix to adjust the view up vector.
            6. Set the camera position to align with the normal direction and update the focal point.
            7. Enable parallel projection for the camera.

        :raises ValueError: If the norm of the view up vector is zero.
        """

        def normalize(v):
            """
            Normalizes a vector.

            :param v: array_like
                The vector to normalize.
            :return: array_like
                The normalized vector.
            """
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm

        # Normalize the view up vector
        view_up_normalized = normalize(self.box_view_up)

        # Define the target y-axis
        target_y = np.array([0, 1, 0])

        # Compute the axis of rotation (cross product)
        axis = np.cross(view_up_normalized, target_y)
        axis_normalized = normalize(axis)

        angle = np.arccos(np.dot(view_up_normalized, target_y))

        # Compute the rotation matrix using Rodrigues' rotation formula
        K = np.array([
            [0, -axis_normalized[2], axis_normalized[1]],
            [axis_normalized[2], 0, -axis_normalized[0]],
            [-axis_normalized[1], axis_normalized[0], 0]
        ])
        I = np.eye(3)
        rotation_matrix = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

        # Apply the rotation matrix to adjust the view up vector
        view_up_rot = np.dot(view_up_normalized, rotation_matrix)
        self.camera.up = [view_up_rot[0], view_up_rot[1], -view_up_rot[2]]

        # Set the camera position to align with the normal direction
        camera_position = [-self.box_norm_direction[0], -self.box_norm_direction[1], self.box_norm_direction[2]]
        self.camera.position = camera_position
        self.camera.focal_point = [0, 0, 0]
        self.reset_camera()
        self.parallel_proj_button.setChecked(True)


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

    def load_state(self):
        """
        Loads the state of spheres from a file. Prompts the user to select a file.
        """
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

    def add_widgets_to_window(self):
        """
        Adds the input widgets to the window.
        """
        # Get the central widget's layout
        central_widget = self.app_window.centralWidget()
        main_layout = central_widget.layout()

        # if main_layout is None:
        #     main_layout = QHBoxLayout()
        #     central_widget.setLayout(main_layout)

        control_layout = QHBoxLayout()

        field_lines_control_group = QGroupBox("Field Line Browser")
        field_lines_control_layout = QVBoxLayout()
        field_lines_control_group.setLayout(field_lines_control_layout)
        control_layout.addWidget(field_lines_control_group)

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
        field_lines_control_layout.addWidget(self.tree_view)

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
        field_lines_control_layout.addLayout(spheres_manage_layout)

        # Create and add the properties panel
        properties_panel = QWidget()
        properties_layout = QVBoxLayout()
        properties_panel.setLayout(properties_layout)

        control_layout.addWidget(properties_panel)

        # Add widgets to the layout
        # Slice Control Group
        slice_control_group = QGroupBox("Slice Z")
        slice_control_layout = QHBoxLayout()

        slice_z_label = QLabel("Z [Mm]:")
        slice_z_label.setToolTip(f"Enter the Z coordinate for the slice in the range of 0 to {self.grid_zmax:.2f} Mm.")
        self.slice_z_input = QLineEdit(
            f"{0:.2f}")
        self.slice_z_input.returnPressed.connect(lambda: self._on_slice_z_input_returnPressed(self.slice_z_input))
        self.slice_z_input.setToolTip(
            f"Enter the Z coordinate for the slice in the range of 0 to {self.grid_zmax:.2f} Mm.")
        slice_control_layout.addWidget(slice_z_label)
        slice_control_layout.addWidget(self.slice_z_input)

        scalar_label = QLabel("Select Scalar:")
        scalar_label.setToolTip("Select the scalar field to display on the slice.")
        self.scalar_selector = QComboBox()
        self.scalar_selector.addItems(self.scalar_selector_items)
        self.scalar_selector.setCurrentText(self.scalar)
        self.scalar_selector.currentTextChanged.connect(self.update_plot)
        slice_control_layout.addWidget(scalar_label)
        slice_control_layout.addWidget(self.scalar_selector)

        vmin_vmax_label = QLabel("Vmin/Vmax [G]:")
        vmin_vmax_label.setToolTip("Enter the minimum and maximum values for the color scale.")
        self.vmin_input = QLineEdit("-1000")
        self.vmin_input.setToolTip("Enter the minimum value for the color scale.")
        self.vmin_input.returnPressed.connect(lambda: self._on_vmin_input_returnPressed(self.vmin_input))
        slice_control_layout.addWidget(vmin_vmax_label)
        slice_control_layout.addWidget(self.vmin_input)

        self.vmax_input = QLineEdit("1000")
        self.vmax_input.setToolTip("Enter the maximum value for the color scale.")
        self.vmax_input.returnPressed.connect(lambda: self._on_vmax_input_returnPressed(self.vmax_input))
        slice_control_layout.addWidget(self.vmax_input)

        self.plane_checkbox = QCheckBox("Show Plane")
        self.plane_checkbox.setChecked(True)
        self.plane_checkbox.stateChanged.connect(self.toggle_plane_visibility)
        slice_control_layout.addWidget(self.plane_checkbox)
        slice_control_layout.addStretch()

        slice_control_group.setLayout(slice_control_layout)
        properties_layout.addWidget(slice_control_group)

        # Sphere Control Group
        sphere_control_group = QGroupBox("Sphere")
        sphere_control_layout = QHBoxLayout()
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
        sphere_control_layout.addWidget(center_label)
        sphere_control_layout.addWidget(self.center_x_input)
        sphere_control_layout.addWidget(self.center_y_input)
        sphere_control_layout.addWidget(self.center_z_input)

        self.lock_z_checkbox = QCheckBox("Lock Z")
        self.lock_z_checkbox.setChecked(False)
        self.lock_z_checkbox.stateChanged.connect(self.on_lock_z_changed)
        sphere_control_layout.addWidget(self.lock_z_checkbox)

        # Add a separator after the Lock Z button
        sphere_control_layout.addWidget(QLabel(" | "))

        radius_label = QLabel("Radius [Mm]:")
        radius_label.setToolTip(
            f"Enter the radius of the sphere.")
        self.radius_input = QLineEdit(
            f"{min(self.grid_x.ptp(), self.grid_y.ptp(), self.grid_z.ptp()) * 0.05:.2f}")
        self.radius_input.setToolTip(
            f"Enter the radius of the sphere in Mm.")
        self.radius_input.returnPressed.connect(lambda: self._on_radius_input_returnPressed(self.radius_input))
        sphere_control_layout.addWidget(radius_label)
        sphere_control_layout.addWidget(self.radius_input)

        n_points_label = QLabel("# of Field Lines:")
        n_points_label.setToolTip(
            "Enter the number of seed points for the field lines.")
        self.n_points_input = QLineEdit("100")
        self.n_points_input.setToolTip(
            "Enter the number of seed points for the field lines.")
        self.n_points_input.returnPressed.connect(lambda: self._on_n_points_input_returnPressed(self.n_points_input))
        sphere_control_layout.addWidget(n_points_label)
        sphere_control_layout.addWidget(self.n_points_input)

        sphere_control_group.setLayout(sphere_control_layout)
        properties_layout.addWidget(sphere_control_group)

        action_layout = QHBoxLayout()

        self.send_button = QPushButton("Send Field Lines")
        if self.parent is None:
            self.send_button.setEnabled(False)
            self.send_button.setToolTip("No parent object to send the field lines to.")
        else:
            self.send_button.setToolTip(f"Send the field lines to {self.parent.__class__}.")
        self.send_button.clicked.connect(self.send_streamlines)

        self.load_box_button = QPushButton("Load Box")
        self.load_box_button.setToolTip("Load the box data from a .hd5 file.")
        self.load_box_button.clicked.connect(self.load_box)


        self.save_box_button = QPushButton("Save Box")
        self.save_box_button.setToolTip("Save the box data to a .hd5 file.")
        self.save_box_button.clicked.connect(self.save_box)


        action_layout.addWidget(self.send_button)
        action_layout.addWidget(self.load_box_button)
        action_layout.addWidget(self.save_box_button)

        # self.update_button = QPushButton("Update")
        # self.update_button.clicked.connect(self.update_plot)
        # action_layout.addWidget(self.update_button)

        # self.sphere_checkbox = QCheckBox("Show Sphere")
        # self.sphere_checkbox.setChecked(True)
        # self.sphere_checkbox.stateChanged.connect(self.toggle_sphere_visibility)
        # action_layout.addWidget(self.sphere_checkbox)

        properties_layout.addLayout(action_layout)

        main_layout.addLayout(control_layout)

    def _on_add_sphere(self):
        """
        Adds a new sphere to the viewer and tree view, hiding the current sphere.
        """
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
        # center_z = self.grid_zmin + self.grid_z.ptp() * 0.1
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
            value = float(widget.text())
            if not min_val <= value <= max_val:
                original_value = min_val if value < min_val else max_val
                raise ValueError

            if paired_widget:
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

            widget.setText(str(original_value))
            return original_value

    def init_grid(self):
        x = self.grid_x
        y = self.grid_y
        z = self.grid_z

        bx = self.box.b3d[self.b3dtype]['bx']
        by = self.box.b3d[self.b3dtype]['by']
        bz = self.box.b3d[self.b3dtype]['bz']


        self.grid = pv.ImageData()
        self.grid.dimensions = (len(x), len(y), len(z))
        self.grid.spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
        self.grid.origin = (x.min(), y.min(), z.min())

        self.grid['bx'] = bx.ravel(order='F')
        self.grid['by'] = by.ravel(order='F')
        self.grid['bz'] = bz.ravel(order='F')
        self.grid['vectors'] = np.c_[self.grid['bx'] , self.grid['by'], self.grid['bz']]
        self.scalar_selector_items.extend(['bx', 'by', 'bz'])

        self.grid_bottom = pv.ImageData()
        self.grid_bottom.dimensions = (len(x), len(y), 1)
        self.grid_bottom.spacing = (x[1] - x[0], y[1] - y[0], 0)
        self.grid_bottom.origin = (x.min(), y.min(), z.min())
        self.bottom_name = self.parent.mapBottomSelector.currentText()
        self.grid_bottom[self.bottom_name] = self.parent.map_bottom.data.T.ravel(order='F')
        self.scalar_selector_items.append(self.bottom_name)


    def init_plot(self):
        """
        Initializes and displays the plot with the magnetic field data.
        """

        self.previous_valid_values = {
            self.center_x_input: float(self.center_x_input.text()),
            self.center_y_input: float(self.center_y_input.text()),
            self.center_z_input: float(self.center_z_input.text()),
            self.radius_input: float(self.radius_input.text()),
            self.slice_z_input: float(self.slice_z_input.text()),
            self.n_points_input: int(self.n_points_input.text()),
            self.vmin_input: float(self.vmin_input.text()),
            self.vmax_input: float(self.vmax_input.text())
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
        radius = self.validate_input(self.radius_input, 0, min(self.grid_x.ptp(), self.grid_y.ptp(), self.grid_z.ptp()),
                                     self.previous_valid_values[self.radius_input])
        n_points = self.validate_input(self.n_points_input, 1, 1000, self.previous_valid_values[self.n_points_input],
                                       to_int=True)

        if not init:
            self.update_sphere()

        self.update_plane()
        slice_z = self.validate_input(self.slice_z_input, 0, self.grid_zmax,
                                      self.previous_valid_values[self.slice_z_input])
        vmin = self.validate_input(self.vmin_input, -5e4, 5e4, self.previous_valid_values[self.vmin_input],
                                   paired_widget=self.vmax_input, paired_type='vmin')
        vmax = self.validate_input(self.vmax_input, -5e4, 5e4, self.previous_valid_values[self.vmax_input],
                                   paired_widget=self.vmin_input, paired_type='vmax')

        scalar = self.scalar_selector.currentText()
        sphere_visible = self.viz_sphere_button.isChecked()
        plane_visible = self.plane_visible

        # Create a dictionary of current parameters
        current_params = {
            "center_x": center_x,
            "center_y": center_y,
            "center_z": center_z,
            "radius": radius,
            "slice_z": slice_z,
            "n_points": n_points,
            "vmin": vmin,
            "vmax": vmax,
            "scalar": scalar,
            "sphere_visible": sphere_visible,
            "plane_visible": plane_visible
        }

        # Check if parameters have changed
        if current_params == self.previous_params:
            self.updating_flag = False  # Reset the flag
            return

        # Update only relevant objects based on parameter changes
        if current_params['slice_z'] != self.previous_params.get('slice_z') or \
                current_params['scalar'] != self.previous_params.get('scalar') or \
                current_params['vmin'] != self.previous_params.get('vmin') or \
                current_params['vmax'] != self.previous_params.get('vmax'):
            self.update_slice(current_params['slice_z'], current_params['scalar'], current_params['vmin'],
                              current_params['vmax'])

        if current_params['plane_visible'] != self.previous_params.get('plane_visible'):
            self.update_plane_visibility(current_params['plane_visible'])

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

    def update_slice(self, slice_z, scalar, vmin, vmax):
        """
        Updates the slice plot based on the given parameters.

        :param slice_z: float
            The Z coordinate for the slice.
        :param scalar: str
            The scalar field to use for the slice.
        :param vmin: float
            The minimum value for the color scale.
        :param vmax: float
            The maximum value for the color scale.
        """
        if scalar == self.bottom_name:
            # Display the new scalar data as a 2D plane
            if self.bottom_slice_actor is None:
                self.bottom_slice_actor = self.add_mesh(self.grid_bottom, scalars=scalar, clim=(vmin, vmax),
                                                        show_edges=False,
                                                        cmap='gray', pickable=False, show_scalar_bar=False)
            else:
                self.remove_actor(self.bottom_slice_actor)
                self.bottom_slice_actor = self.add_mesh(self.grid_bottom, scalars=scalar, clim=(vmin, vmax),
                                                        show_edges=False,
                                                        cmap='gray', pickable=False, reset_camera=False,
                                                        show_scalar_bar=False)
        else:
            if slice_z==0:
                slice_z = 1.0e-6
            new_slice = self.grid.slice(normal='z', origin=(self.grid.origin[0], self.grid.origin[1], slice_z))
            if self.bottom_slice_actor is None:
                self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalar, clim=(vmin, vmax), show_edges=False,
                                                        cmap='gray', pickable=False, show_scalar_bar=False)
            else:
                self.remove_actor(self.bottom_slice_actor)
                self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalar, clim=(vmin, vmax), show_edges=False,
                                                        cmap='gray', pickable=False, reset_camera=False,
                                                        show_scalar_bar=False)

    def create_streamlines(self, center_x, center_y, center_z, radius, n_points):
        self.streamlines = self.grid.streamlines(vectors='vectors', source_center=(center_x, center_y, center_z),
                                                 source_radius=radius, n_points=n_points, integration_direction='both',
                                                 max_time=5000, progress_bar=False)
        if self.streamlines.n_points > 0:
            if self.streamlines_actor is None:
                self.streamlines_actor = self.add_mesh(self.streamlines.tube(radius=0.1), pickable=False,
                                                       reset_camera=False, show_scalar_bar=False)
            else:
                self.remove_actor(self.streamlines_actor)
                self.streamlines_actor = self.add_mesh(self.streamlines.tube(radius=0.1), pickable=False,
                                                       reset_camera=False, show_scalar_bar=False)
        else:
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
                                            max_time=5000, progress_bar=False)
        self.spheres[self.current_sphere_id]['streamlines'] = streamlines
        if streamlines.n_points > 0:
            if streamlines_actor is None:
                streamlines_actor = self.add_mesh(streamlines.tube(radius=0.1), pickable=False,
                                                  reset_camera=False, show_scalar_bar=False)
            else:
                self.remove_actor(streamlines_actor)
                streamlines_actor = self.add_mesh(streamlines.tube(radius=0.1), pickable=False,
                                                  reset_camera=False, show_scalar_bar=False)
            self.spheres[self.current_sphere_id]['streamlines_actor'] = streamlines_actor
        else:
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

    def _on_sphere_moved(self, center):
        """
        Handles the event when the sphere widget is moved.

        :param center: list of float
            The new center coordinates of the sphere.
        """
        print('calling _on_sphere_moved')
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
            origin = self.grid_x.ptp() / 2, self.grid_y.ptp() / 2
            slice_z = float(self.slice_z_input.text())
            self.plane_actor.SetOrigin([origin[0], origin[1], slice_z])
            self.update_plot()

    def update_plane_visibility(self, plane_visible):
        """
        Updates the visibility of the plane widget.

        :param plane_visible: bool
            Whether the plane widget is visible.
        """
        if plane_visible:
            if self.plane_actor is None:
                origin = self.grid_x.ptp() / 2, self.grid_y.ptp() / 2
                slice_z = float(self.slice_z_input.text())
                self.plane_actor = self.add_plane_widget(self._on_plane_moved, normal='z',
                                                         origin=(origin[0], origin[1], slice_z), bounds=(
                        self.grid_xmin, self.grid_xmax, self.grid_ymin, self.grid_ymax, self.grid_zmin, self.grid_zmax),
                                                         normal_rotation=False)
            else:
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
        self.slice_z_input.setText(f"{origin[2]:.2f}")
        self.update_plane()

    def toggle_plane_visibility(self, state):
        """
        Toggles the visibility of the plane widget.

        :param state: int
            The state of the checkbox (checked or unchecked).
        """
        self.plane_visible = state == Qt.Checked
        self.update_plot()

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


    def save_box(self):
        box_dims_str = 'x'.join(map(str, self.box.dims_pix))
        default_filename = f'b3d_data_{self.box._frame_obs.obstime.to_datetime().strftime("%Y%m%dT%H%M%S")}_dim{box_dims_str}.h5'
        filename = QFileDialog.getSaveFileName(self, "Save Box", default_filename, "HDF5 Files (*.h5)")[0]
        write_b3d_h5(filename, self.box.b3d)

    def load_box(self):
        default_filename = "b3d_data.h5"
        filename = QFileDialog.getOpenFileName(self, "Load Box", default_filename, "HDF5 Files (*.h5)")[0]
        self.box.b3d = read_b3d_h5(filename)
        self.init_grid()
        self.update_plot()