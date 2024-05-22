from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QLabel, \
    QPushButton, QSlider, QLineEdit, QCheckBox, QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from astropy.time import Time

import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
import numpy as np


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

def validate_number(func):
    """
    Decorator to validate if the input in the widget is a number.

    :param func: function
        The function to wrap.
    :return: function
        The wrapped function.
    """
    def wrapper(self, widget, *args, **kwargs):
        try:
            float(widget.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")
            return
        return func(self, widget, *args, **kwargs)

    return wrapper


class MagFieldViewer(BackgroundPlotter):
    """
    A class to visualize the magnetic field of a box using PyVista. It inherits from the BackgroundPlotter class.

    :param box: object
        The box containing magnetic field data.
    :param parent: object, optional
        The parent object (default is None).
    """
    def __init__(self, box, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box = box
        self.parent = parent
        self.updating = False  # Flag to avoid recursion
        self.sphere_actor = None
        self.plane_actor = None
        self.bottom_slice_actor = None
        self.streamlines_actor = None
        self.sphere_visible = True
        self.plane_visible = True
        self.scalar = 'bz'
        self.previous_params = {}
        self.previous_valid_values = {}
        self.scalar_selector = None
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
        self.sphere_checkbox = None
        self.grid_x = self.box.grid_coords['x'].value
        self.grid_y = self.box.grid_coords['y'].value
        self.grid_z = self.box.grid_coords['z'].value
        self.grid_xmin, self.grid_xmax = minval(self.grid_x.min()), maxval(self.grid_x.max())
        self.grid_ymin, self.grid_ymax = minval(self.grid_y.min()), maxval(self.grid_y.max())
        self.grid_zmin, self.grid_zmax = minval(self.grid_z.min()), maxval(self.grid_z.max())

        self.add_widgets_to_window()
        self.show_plot()
        self.show_axes_all()
        self.view_isometric()
        self.plane_checkbox.setChecked(False)
        self.app_window.setWindowTitle("GxBox 3D viewer")



    def add_widgets_to_window(self):
        """
        Adds the input widgets to the window.
        """
        # Get the central widget's layout
        central_widget = self.app_window.centralWidget()
        layout = central_widget.layout()

        if layout is None:
            layout = QVBoxLayout()
            central_widget.setLayout(layout)

        # Add widgets to the layout
        slice_control_layout = QHBoxLayout()

        slice_z_label = QLabel("Slice Z [Mm]:")
        self.slice_z_input = QLineEdit(
            f"{np.min(self.box.grid_coords['z'].value) + self.box.grid_coords['z'].value.ptp() * 0.005:.2f}")
        self.slice_z_input.returnPressed.connect(lambda: self.on_slice_z_input_returnPressed(self.slice_z_input))
        slice_control_layout.addWidget(slice_z_label)
        slice_control_layout.addWidget(self.slice_z_input)

        vmin_label = QLabel("Vmin:")
        self.vmin_input = QLineEdit("-1000")
        self.vmin_input.returnPressed.connect(lambda: self.on_vmin_input_returnPressed(self.vmin_input))
        slice_control_layout.addWidget(vmin_label)
        slice_control_layout.addWidget(self.vmin_input)

        vmax_label = QLabel("Vmax:")
        self.vmax_input = QLineEdit("1000")
        self.vmax_input.returnPressed.connect(lambda: self.on_vmax_input_returnPressed(self.vmax_input))
        slice_control_layout.addWidget(vmax_label)
        slice_control_layout.addWidget(self.vmax_input)
        layout.addLayout(slice_control_layout)

        scalar_label = QLabel("Select Slice Scalar:")
        self.scalar_selector = QComboBox()
        self.scalar_selector.addItems(['bx', 'by', 'bz'])
        self.scalar_selector.setCurrentText(self.scalar)
        self.scalar_selector.currentTextChanged.connect(self.update_plot)
        slice_control_layout.addWidget(scalar_label)
        slice_control_layout.addWidget(self.scalar_selector)

        sphere_control_layout = QHBoxLayout()
        center_label = QLabel("Center (x, y, z):")
        self.center_x_input = QLineEdit(f"{np.mean(self.box.grid_coords['x'].value):.2f}")
        self.center_y_input = QLineEdit(f"{np.mean(self.box.grid_coords['y'].value):.2f}")
        self.center_z_input = QLineEdit(
            f"{np.min(self.box.grid_coords['z'].value) + self.box.grid_coords['z'].value.ptp() * 0.1:.2f}")
        self.center_x_input.returnPressed.connect(lambda: self.on_center_x_input_returnPressed(self.center_x_input))
        self.center_y_input.returnPressed.connect(lambda: self.on_center_y_input_returnPressed(self.center_y_input))
        self.center_z_input.returnPressed.connect(lambda: self.on_center_z_input_returnPressed(self.center_z_input))
        sphere_control_layout.addWidget(center_label)
        sphere_control_layout.addWidget(self.center_x_input)
        sphere_control_layout.addWidget(self.center_y_input)
        sphere_control_layout.addWidget(self.center_z_input)

        radius_label = QLabel("Sphere Radius [Mm]:")
        self.radius_input = QLineEdit(
            f"{min(self.box.grid_coords['x'].value.ptp(), self.box.grid_coords['y'].value.ptp(), self.box.grid_coords['z'].value.ptp()) * 0.05:.2f}")
        self.radius_input.returnPressed.connect(lambda: self.on_radius_input_returnPressed(self.radius_input))
        sphere_control_layout.addWidget(radius_label)
        sphere_control_layout.addWidget(self.radius_input)

        n_points_label = QLabel("Number of Seeds:")
        self.n_points_input = QLineEdit("10")
        self.n_points_input.returnPressed.connect(lambda: self.on_n_points_input_returnPressed(self.n_points_input))
        sphere_control_layout.addWidget(n_points_label)
        sphere_control_layout.addWidget(self.n_points_input)
        layout.addLayout(sphere_control_layout)


        action_layout = QHBoxLayout()

        self.send_button = QPushButton("Update Map")
        self.send_button.clicked.connect(self.send_streamlines)
        action_layout.addWidget(self.send_button)

        # self.update_button = QPushButton("Update")
        # self.update_button.clicked.connect(self.update_plot)
        # action_layout.addWidget(self.update_button)


        self.sphere_checkbox = QCheckBox("Show Sphere")
        self.sphere_checkbox.setChecked(True)
        self.sphere_checkbox.stateChanged.connect(self.toggle_sphere_visibility)
        action_layout.addWidget(self.sphere_checkbox)

        self.plane_checkbox = QCheckBox("Show Plane")
        self.plane_checkbox.setChecked(True)
        self.plane_checkbox.stateChanged.connect(self.toggle_plane_visibility)
        action_layout.addWidget(self.plane_checkbox)
        layout.addLayout(action_layout)



    @validate_number
    def on_center_x_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the center X input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_sphere()

    @validate_number
    def on_center_y_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the center Y input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_sphere()

    @validate_number
    def on_center_z_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the center Z input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_sphere()

    @validate_number
    def on_radius_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the radius input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_sphere()

    @validate_number
    def on_n_points_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the number of seeds input.

        :param widget: QLineEdit
            The input widget.
        """

        self.update_sphere()

    @validate_number
    def on_slice_z_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the slice Z input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_plot()

    @validate_number
    def on_vmin_input_returnPressed(self, widget):
        """
        Handles the return pressed event for the Vmin input.

        :param widget: QLineEdit
            The input widget.
        """
        self.update_plot()

    @validate_number
    def on_vmax_input_returnPressed(self, widget):
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

    def show_plot(self):
        """
        Initializes and displays the plot with the magnetic field data.
        """
        x = self.grid_x
        y = self.grid_y
        z = self.grid_z

        bx = self.box.b3d['nlfff']['bx']
        by = self.box.b3d['nlfff']['by']
        bz = self.box.b3d['nlfff']['bz']
        vectors = np.c_[bx.ravel(order='F'), by.ravel(order='F'), bz.ravel(order='F')]

        self.grid = pv.ImageData()
        self.grid.dimensions = (len(x), len(y), len(z))
        self.grid.spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
        self.grid.origin = (x.min(), y.min(), z.min())
        self.grid['vectors'] = vectors
        self.grid['bx'] = bx.ravel(order='F')
        self.grid['by'] = by.ravel(order='F')
        self.grid['bz'] = bz.ravel(order='F')

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

        self.update_plot()



    def update_plot(self):
        """
        Updates the plot based on the current input parameters.
        """
        print(f"Initial cam clip range: {self.camera.clipping_range}")
        if self.updating:  # Check if already updating
            return

        self.updating = True  # Set the flag


        # Get current parameters
        center_x = self.validate_input(self.center_x_input, self.grid_xmin, self.grid_xmax, self.previous_valid_values[self.center_x_input])
        center_y = self.validate_input(self.center_y_input, self.grid_ymin, self.grid_ymax, self.previous_valid_values[self.center_y_input])
        center_z = self.validate_input(self.center_z_input, self.grid_zmin, self.grid_zmax, self.previous_valid_values[self.center_z_input])
        radius = self.validate_input(self.radius_input, 0, min(self.grid_x.ptp(), self.grid_y.ptp(), self.grid_z.ptp()),
                                     self.previous_valid_values[self.radius_input])
        n_points = self.validate_input(self.n_points_input, 1, 500, self.previous_valid_values[self.n_points_input],
                                       to_int=True)
        self.update_sphere()
        slice_z = self.validate_input(self.slice_z_input, self.grid_zmin, self.grid_zmax, self.previous_valid_values[self.slice_z_input])
        self.update_plane()
        vmin = self.validate_input(self.vmin_input, -5e4, 5e4, self.previous_valid_values[self.vmin_input],
                                   paired_widget=self.vmax_input, paired_type='vmin')
        vmax = self.validate_input(self.vmax_input, -5e4, 5e4, self.previous_valid_values[self.vmax_input],
                                   paired_widget=self.vmin_input, paired_type='vmax')

        scalar = self.scalar_selector.currentText()
        sphere_visible = self.sphere_visible
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
            self.updating = False  # Reset the flag
            return


        # Update only relevant objects based on parameter changes
        if current_params['slice_z'] != self.previous_params.get('slice_z') or \
                current_params['scalar'] != self.previous_params.get('scalar') or \
                current_params['vmin'] != self.previous_params.get('vmin') or \
                current_params['vmax'] != self.previous_params.get('vmax'):
            self.update_slice(current_params['slice_z'], current_params['scalar'], current_params['vmin'],
                              current_params['vmax'])

        if current_params['center_x'] != self.previous_params.get('center_x') or \
                current_params['center_y'] != self.previous_params.get('center_y') or \
                current_params['center_z'] != self.previous_params.get('center_z') or \
                current_params['radius'] != self.previous_params.get('radius') or \
                current_params['n_points'] != self.previous_params.get('n_points'):
            self.update_streamlines(current_params['center_x'], current_params['center_y'], current_params['center_z'],
                                    current_params['radius'], current_params['n_points'])

        if current_params['sphere_visible'] != self.previous_params.get('sphere_visible'):
            self.update_sphere_visibility(current_params['sphere_visible'])

        if current_params['plane_visible'] != self.previous_params.get('plane_visible'):
            self.update_plane_visibility(current_params['plane_visible'])

        # Update previous parameters
        self.previous_params = current_params

        # self.plotter.show()
        self.updating = False  # Reset the flag
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
        new_slice = self.grid.slice(normal='z', origin=(self.grid.origin[0], self.grid.origin[1], slice_z))
        if self.bottom_slice_actor is None:
            self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalar, clim=(vmin, vmax), show_edges=False,
                                                    cmap='gray', pickable=False, show_scalar_bar=False)
        else:
            self.remove_actor(self.bottom_slice_actor)
            self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalar, clim=(vmin, vmax), show_edges=False,
                                                    cmap='gray', pickable=False, reset_camera=False,
                                                    show_scalar_bar=False)

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
        new_streamlines = self.grid.streamlines(vectors='vectors', source_center=(center_x, center_y, center_z),
                                                source_radius=radius, n_points=n_points, integration_direction='both',
                                                max_time=5000, progress_bar=True)
        if new_streamlines.n_points > 0:
            if self.streamlines_actor is None:
                self.streamlines_actor = self.add_mesh(new_streamlines.tube(radius=0.1), pickable=False,
                                                       show_scalar_bar=False)
            else:
                self.remove_actor(self.streamlines_actor)
                self.streamlines_actor = self.add_mesh(new_streamlines.tube(radius=0.1), pickable=False,
                                                       reset_camera=False, show_scalar_bar=False)
        else:
            print("No streamlines generated.")

    def update_sphere(self):
        """
        Updates the sphere widget based on the current input parameters.
        """
        if self.sphere_actor is not None:
            self.sphere_actor.SetCenter([float(self.center_x_input.text()), float(self.center_y_input.text()),
                                         float(self.center_z_input.text())])
            self.sphere_actor.SetRadius(float(self.radius_input.text()))
            self.update_plot()

    def update_sphere_visibility(self, sphere_visible):
        """
        Updates the visibility of the sphere widget.

        :param sphere_visible: bool
            Whether the sphere widget is visible.
        """
        if sphere_visible:
            if self.sphere_actor is None:
                center_x = float(self.center_x_input.text())
                center_y = float(self.center_y_input.text())
                center_z = float(self.center_z_input.text())
                radius = float(self.radius_input.text())
                self.sphere_actor = self.add_sphere_widget(self.on_sphere_moved,
                                                           center=(center_x, center_y, center_z),
                                                           radius=radius, theta_resolution=18, phi_resolution=18,
                                                           style='wireframe')
            else:
                self.sphere_actor.On()
        else:
            if self.sphere_actor is not None:
                self.sphere_actor.Off()

    def on_sphere_moved(self, center):
        """
        Handles the event when the sphere widget is moved.

        :param center: list of float
            The new center coordinates of the sphere.
        """
        self.center_x_input.setText(f"{center[0]:.2f}")
        self.center_y_input.setText(f"{center[1]:.2f}")
        self.center_z_input.setText(f"{center[2]:.2f}")
        self.update_sphere()

    def toggle_sphere_visibility(self, state):
        """
        Toggles the visibility of the sphere widget.

        :param state: int
            The state of the checkbox (checked or unchecked).
        """
        self.sphere_visible = state == Qt.Checked
        self.update_plot()

    def update_plane(self):
        """
        Updates the plane widget based on the current input parameters.
        """
        if self.plane_actor is not None:
            origin = self.box.grid_coords['x'].value.ptp() / 2, self.box.grid_coords['y'].value.ptp() / 2
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
                origin = self.box.grid_coords['x'].value.ptp() / 2, self.box.grid_coords['y'].value.ptp() / 2
                slice_z = float(self.slice_z_input.text())
                self.plane_actor = self.add_plane_widget(self.on_plane_moved, normal='z',
                                                         origin=(origin[0], origin[1], slice_z), normal_rotation=False)
            else:
                self.plane_actor.On()
        else:
            if self.plane_actor is not None:
                self.plane_actor.Off()

    def on_plane_moved(self, normal, origin):
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
        Sends the streamline data to the parent object (if any).
        """
        print("Sending streamlines to gxbox...")
        if self.parent is not None and self.streamlines_actor is not None:
            streamlines = self.grid.streamlines(vectors='vectors', source_center=(
                float(self.center_x_input.text()), float(self.center_y_input.text()),
                float(self.center_z_input.text())),
                                                source_radius=float(self.radius_input.text()),
                                                n_points=int(self.n_points_input.text()), integration_direction='both',
                                                max_time=5000, progress_bar=True)
            if streamlines.n_lines > 0:
                self.parent.plot_fieldlines(streamlines)

