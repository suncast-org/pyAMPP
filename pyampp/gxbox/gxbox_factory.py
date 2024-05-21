import itertools
import astropy.units as u
import astropy.time
import numpy as np
import matplotlib.pyplot as plt
from sunpy.map import Map, make_fitswcs_header, all_pixel_indices_from_map, coordinate_is_on_solar_disk
import sunpy.sun.constants
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Heliocentric, Helioprojective, get_earth, HeliographicStonyhurst, HeliographicCarrington, \
    sun
from datetime import datetime, timedelta
import os
import glob
from pyampp.util.config import *
from pyampp.data import downloader
from pyampp.gxbox.boxutils import hmi_disambig, hmi_b2ptr
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QLabel, \
    QPushButton, QSlider, QLineEdit, QCheckBox,QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import argparse
from astropy.time import Time
from pathlib import Path
import locale
from streamtracer import StreamTracer, VectorGrid
import pyampp
from pyampp.util.lff import mf_lfff
from pyampp.util.MagFieldWrapper import MagFieldWrapper
from pyampp.util.radio import GXRadioImageComputing
import pickle
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import pyvista as pv
from pyvistaqt import QtInteractor
from sunkit_pyvista import SunpyPlotter

base_dir = Path(pyampp.__file__).parent
nlfff_libpath = Path(base_dir / 'lib' / 'nlfff' / 'binaries' / 'WWNLFFFReconstruction.so').resolve()
radio_libpath = Path(base_dir / 'lib' / 'grff' / 'binaries' / 'RenderGRFF.so').resolve()

os.environ['OMP_NUM_THREADS'] = '16'  # number of parallel threads
locale.setlocale(locale.LC_ALL, "C");

import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
import numpy as np
from qtrangeslider import QRangeSlider

class MagneticFieldVisualizer(BackgroundPlotter):
    '''
    A class to visualize the magnetic field of a box using PyVista. It inherits from the BackgroundPlotter class.
    '''
    def __init__(self, box, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box = box
        self.updating = False  # Flag to avoid recursion
        self.sphere_actor = None
        self.bottom_slice_actor = None
        self.streamlines_actor = None
        self.sphere_visible = True
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
        self.sphere_checkbox = None
        self.add_widgets_to_window()

    def add_widgets_to_window(self):
        # Get the central widget's layout
        central_widget = self.app_window.centralWidget()
        layout = central_widget.layout()

        if layout is None:
            layout = QVBoxLayout()
            central_widget.setLayout(layout)

        # Add widgets to the layout
        scalar_control_layout = QHBoxLayout()
        scalar_label = QLabel("Select Scalar:")
        self.scalar_selector = QComboBox()
        self.scalar_selector.addItems(['bx', 'by', 'bz'])
        self.scalar_selector.setCurrentText(self.scalar)
        self.scalar_selector.currentTextChanged.connect(self.update_plot)
        scalar_control_layout.addWidget(scalar_label)
        scalar_control_layout.addWidget(self.scalar_selector)
        layout.addLayout(scalar_control_layout)

        sphere_control_layout = QHBoxLayout()
        center_label = QLabel("Center (x, y, z):")
        self.center_x_input = QLineEdit(f"{np.mean(self.box.grid_coords['x'].value):.2f}")
        self.center_y_input = QLineEdit(f"{np.mean(self.box.grid_coords['y'].value):.2f}")
        self.center_z_input = QLineEdit(
            f"{np.min(self.box.grid_coords['z'].value) + self.box.grid_coords['z'].value.ptp() * 0.1:.2f}")
        self.center_x_input.returnPressed.connect(self.update_sphere)
        self.center_y_input.returnPressed.connect(self.update_sphere)
        self.center_z_input.returnPressed.connect(self.update_sphere)
        sphere_control_layout.addWidget(center_label)
        sphere_control_layout.addWidget(self.center_x_input)
        sphere_control_layout.addWidget(self.center_y_input)
        sphere_control_layout.addWidget(self.center_z_input)

        radius_label = QLabel("Seed Radius [Mm]:")
        self.radius_input = QLineEdit(
            f"{min(self.box.grid_coords['x'].value.ptp(), self.box.grid_coords['y'].value.ptp(), self.box.grid_coords['z'].value.ptp()) * 0.05:.2f}")
        self.radius_input.returnPressed.connect(self.update_sphere)
        sphere_control_layout.addWidget(radius_label)
        sphere_control_layout.addWidget(self.radius_input)

        n_points_label = QLabel("Number of Seeds:")
        self.n_points_input = QLineEdit("10")
        self.n_points_input.returnPressed.connect(self.update_sphere)
        sphere_control_layout.addWidget(n_points_label)
        sphere_control_layout.addWidget(self.n_points_input)
        layout.addLayout(sphere_control_layout)

        slice_vmin_vmax_layout = QHBoxLayout()
        slice_z_label = QLabel("Slice Z [Mm]:")
        self.slice_z_input = QLineEdit(
            f"{np.min(self.box.grid_coords['z'].value) + self.box.grid_coords['z'].value.ptp() * 0.01:.2f}")
        self.slice_z_input.returnPressed.connect(self.update_plot)
        slice_vmin_vmax_layout.addWidget(slice_z_label)
        slice_vmin_vmax_layout.addWidget(self.slice_z_input)

        vmin_label = QLabel("Vmin:")
        self.vmin_input = QLineEdit("-1000")
        self.vmin_input.returnPressed.connect(self.update_plot)
        slice_vmin_vmax_layout.addWidget(vmin_label)
        slice_vmin_vmax_layout.addWidget(self.vmin_input)

        vmax_label = QLabel("Vmax:")
        self.vmax_input = QLineEdit("1000")
        self.vmax_input.returnPressed.connect(self.update_plot)
        slice_vmin_vmax_layout.addWidget(vmax_label)
        slice_vmin_vmax_layout.addWidget(self.vmax_input)
        layout.addLayout(slice_vmin_vmax_layout)

        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_plot)
        layout.addWidget(self.update_button)

        self.sphere_checkbox = QCheckBox("Show Sphere")
        self.sphere_checkbox.setChecked(True)
        self.sphere_checkbox.stateChanged.connect(self.toggle_sphere_visibility)
        layout.addWidget(self.sphere_checkbox)

        self.show_plot()
        self.show_axes_all()
        self.view_isometric(negative=True)

    def validate_input(self, widget, min_val, max_val, original_value, to_int=False, paired_widget=None,
                       paired_type=None):
        try:
            value = float(widget.text())
            if value < min_val or value > max_val:
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
            if paired_type == 'vmin':
                QMessageBox.warning(self, "Invalid Input",
                                    f"Please enter a number between {min_val:.3f} and {max_val:.3f} that is less than the corresponding max value.")
            elif paired_type == 'vmax':
                QMessageBox.warning(self, "Invalid Input",
                                    f"Please enter a number between {min_val:.3f} and {max_val:.3f} that is greater than the corresponding min value.")
            else:
                QMessageBox.warning(self, "Invalid Input",
                                    f"Please enter a number between {min_val:.3f} and {max_val:.3f}.")

            widget.setText(str(original_value))
            return original_value
    def show_plot(self):
        x = self.box.grid_coords['x'].value
        y = self.box.grid_coords['y'].value
        z = self.box.grid_coords['z'].value

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
        if self.updating:  # Check if already updating
            return

        self.updating = True  # Set the flag

        x = self.box.grid_coords['x'].value
        y = self.box.grid_coords['y'].value
        z = self.box.grid_coords['z'].value

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        zmin, zmax = z.min(), z.max()

        # Get current parameters
        center_x = self.validate_input(self.center_x_input, xmin, xmax, self.previous_valid_values[self.center_x_input])
        center_y = self.validate_input(self.center_y_input, ymin, ymax, self.previous_valid_values[self.center_y_input])
        center_z = self.validate_input(self.center_z_input, zmin, zmax, self.previous_valid_values[self.center_z_input])
        radius = self.validate_input(self.radius_input, 0, min(x.ptp(), y.ptp(), z.ptp()),
                                     self.previous_valid_values[self.radius_input])
        slice_z = self.validate_input(self.slice_z_input, zmin, zmax, self.previous_valid_values[self.slice_z_input])
        n_points = self.validate_input(self.n_points_input, 1, 500, self.previous_valid_values[self.n_points_input],
                                       to_int=True)

        vmin = self.validate_input(self.vmin_input, -5e4, 5e4, self.previous_valid_values[self.vmin_input],
                                   paired_widget=self.vmax_input, paired_type='vmin')
        vmax = self.validate_input(self.vmax_input, -5e4, 5e4, self.previous_valid_values[self.vmax_input],
                                   paired_widget=self.vmin_input, paired_type='vmax')

        scalar = self.scalar_selector.currentText()
        sphere_visible = self.sphere_visible

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
            "sphere_visible": sphere_visible
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

        # Update previous parameters
        self.previous_params = current_params

        # self.plotter.show()
        self.updating = False  # Reset the flag

    def update_slice(self, slice_z, scalar, vmin, vmax):
        new_slice = self.grid.slice(normal='z', origin=(self.grid.origin[0], self.grid.origin[1], slice_z))
        if self.bottom_slice_actor is None:
            self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalar, clim=(vmin, vmax), show_edges=False, cmap='gray', pickable=False, show_scalar_bar=False)
        else:
            self.remove_actor(self.bottom_slice_actor)
            self.bottom_slice_actor = self.add_mesh(new_slice, scalars=scalar, clim=(vmin, vmax), show_edges=False, cmap='gray', pickable=False, reset_camera=False, show_scalar_bar=False)

    def update_streamlines(self, center_x, center_y, center_z, radius, n_points):
        new_streamlines = self.grid.streamlines(vectors='vectors', source_center=(center_x, center_y, center_z),
                                                source_radius=radius, n_points=n_points, integration_direction='both',
                                                max_time=5000, progress_bar=True)
        if new_streamlines.n_points > 0:
            if self.streamlines_actor is None:
                self.streamlines_actor = self.add_mesh(new_streamlines.tube(radius=0.1), pickable=False, show_scalar_bar=False)
            else:
                self.remove_actor(self.streamlines_actor)
                self.streamlines_actor = self.add_mesh(new_streamlines.tube(radius=0.1), pickable=False, reset_camera=False, show_scalar_bar=False)
        else:
            print("No streamlines generated.")

    def update_sphere(self):
        if self.sphere_actor is not None:
            self.sphere_actor.SetCenter([float(self.center_x_input.text()), float(self.center_y_input.text()), float(self.center_z_input.text())])
            self.sphere_actor.SetRadius(float(self.radius_input.text()))
            self.update_plot()

    def update_sphere_visibility(self, sphere_visible):
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
        self.center_x_input.setText(f"{center[0]:.2f}")
        self.center_y_input.setText(f"{center[1]:.2f}")
        self.center_z_input.setText(f"{center[2]:.2f}")
        self.update_sphere()

    def toggle_sphere_visibility(self, state):
        self.sphere_visible = state == Qt.Checked
        self.update_plot()





class Box:
    """
    Represents a 3D box in solar or observer coordinates defined by its origin, center, dimensions, and resolution.

    This class calculates and stores the coordinates of the box's edges, differentiating between bottom edges and other edges.
    It is designed to integrate with solar physics data analysis frameworks such as SunPy and Astropy.

    :param frame_obs: The observer's frame of reference as a `SkyCoord` object.
    :param box_origin: The origin point of the box in the specified coordinate frame as a `SkyCoord`.
    :param box_center: The geometric center of the box as a `SkyCoord`.
    :param box_dims: The dimensions of the box specified as an `astropy.units.Quantity` array-like in the order (x, y, z).
    :param box_res: The resolution of the box, given as an `astropy.units.Quantity` typically in units of megameters.

    Attributes
    ----------
    corners : list of tuple
        List containing tuples representing the corner points of the box in the specified units.
    edges : list of tuple
        List containing tuples that represent the edges of the box by connecting the corners.
    bottom_edges : list of `SkyCoord`
        A list containing the bottom edges of the box calculated based on the minimum z-coordinate value.
    non_bottom_edges : list of `SkyCoord`
        A list containing all edges of the box that are not classified as bottom edges.

    Methods
    -------
    bl_tr_coords(pad_frac=0.0)
        Calculates and returns the bottom left and top right coordinates of the box in the observer frame.
        Optionally applies a padding factor to expand the box dimensions symmetrically.

    Example
    -------
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> time = Time('2024-05-09T17:12:00')
    >>> box_origin = SkyCoord(450 * u.arcsec, -256 * u.arcsec, obstime=time, observer="earth", frame='helioprojective')
    >>> box_center = SkyCoord(500 * u.arcsec, -200 * u.arcsec, obstime=time, observer="earth", frame='helioprojective')
    >>> box_dims = u.Quantity([100, 100, 50], u.Mm)
    >>> box_res = 1.4 * u.Mm
    >>> box = Box(frame_obs=box_origin.frame, box_origin=box_origin, box_center=box_center, box_dims=box_dims, box_res=box_res)
    >>> print(box.bounds_coords_bl_tr())
    """

    def __init__(self, frame_obs, box_origin, box_center, box_dims, box_res):
        '''
        Initializes the Box instance with origin, dimensions, and computes the corners and edges.

        :param box_center: SkyCoord, the origin point of the box in a given coordinate frame.
        :param box_dims: u.Quantity, the dimensions of the box (x, y, z) in specified units. x and y are in the solar frame, z is the height above the solar surface.
        '''
        self._frame_obs = frame_obs
        with Helioprojective.assume_spherical_screen(frame_obs.observer):
            self._origin = box_origin
            self._center = box_center
        self._dims = box_dims
        self._res = box_res
        self._dims_pix = np.int_(np.round(self._dims / self._res.to(self._dims.unit)))
        # Generate corner points based on the dimensions
        self.corners = list(itertools.product(self._dims[0] / 2 * [-1, 1],
                                              self._dims[1] / 2 * [-1, 1],
                                              self._dims[2] / 2 * [-1, 1]))

        # Identify edges as pairs of corners differing by exactly one dimension
        self.edges = [edge for edge in itertools.combinations(self.corners, 2)
                      if np.count_nonzero(u.Quantity(edge[0]) - u.Quantity(edge[1])) == 1]
        # Initialize properties to store categorized edges
        self._bottom_edges = None
        self._non_bottom_edges = None
        self._calculate_edge_types()  # Categorize edges upon initialization
        self.b3dtype = ['lfff', 'nlfff']
        self.b3d = {b3dtype: None for b3dtype in self.b3dtype}

    @property
    def dims_pix(self):
        return self._dims_pix

    @property
    def grid_coords(self):
        return self._get_grid_coords(self._center)

    def _get_grid_coords(self, grid_center):
        grid_coords = {}
        grid_coords['x'] = np.linspace(grid_center.x.to(self._dims.unit) - self._dims[0] / 2,
                                       grid_center.x.to(self._dims.unit) + self._dims[0] / 2, self._dims_pix[0])
        grid_coords['y'] = np.linspace(grid_center.y.to(self._dims.unit) - self._dims[1] / 2,
                                       grid_center.y.to(self._dims.unit) + self._dims[1] / 2, self._dims_pix[1])
        grid_coords['z'] = np.linspace(grid_center.z.to(self._dims.unit) - self._dims[2] / 2,
                                       grid_center.z.to(self._dims.unit) + self._dims[2] / 2, self._dims_pix[2])
        grid_coords['frame'] = self._frame_obs
        return grid_coords

    def _get_edge_coords(self, edges, box_center):
        """
        Translates edge corner points to their corresponding SkyCoord based on the box's origin.

        :param edges: List of tuples, each tuple contains two corner points defining an edge.
        :type edges: list of tuple
        :param box_center: The origin point of the box in the specified coordinate frame as a `SkyCoord`.
        :type box_center: `~astropy.coordinates.SkyCoord`
        :return: List of `SkyCoord` coordinates of edges in the box's frame.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        return [SkyCoord(x=box_center.x + u.Quantity([edge[0][0], edge[1][0]]),
                         y=box_center.y + u.Quantity([edge[0][1], edge[1][1]]),
                         z=box_center.z + u.Quantity([edge[0][2], edge[1][2]]),
                         frame=box_center.frame) for edge in edges]

    # def _get_bottom_bl_tr_coords(self,box_center):
    #     return [SkyCoord(x=box_center.x - self._box_dims[0] / 2,
    def _get_bottom_cea_header(self):
        """
        Generates a CEA header for the bottom of the box.

        :return: The FITS WCS header for the bottom of the box.
        :rtype: dict
        """
        origin = self._origin.transform_to(HeliographicStonyhurst)
        shape = self._dims[:-1][::-1] / self._res.to(self._dims.unit)
        shape = list(shape.value)
        shape = [int(np.ceil(s)) for s in shape]
        rsun = origin.rsun.to(self._res.unit)
        scale = np.arcsin(self._res / rsun).to(u.deg) / u.pix
        scale = u.Quantity((scale, scale))
        # bottom_cea_header = make_fitswcs_header(shape, origin,
        #                                         scale=scale, observatory=self._origin.observer, projection_code='CEA')
        bottom_cea_header = make_fitswcs_header(shape, origin,
                                                scale=scale, projection_code='CEA')
        bottom_cea_header['OBSRVTRY'] = str(origin.observer)
        return bottom_cea_header

    def _calculate_edge_types(self):
        """
        Separates the box's edges into bottom edges and non-bottom edges. This is done in a single pass to improve efficiency.
        """
        min_z = min(corner[2] for corner in self.corners)
        bottom_edges, non_bottom_edges = [], []
        for edge in self.edges:
            if edge[0][2] == min_z and edge[1][2] == min_z:
                bottom_edges.append(edge)
            else:
                non_bottom_edges.append(edge)
        self._bottom_edges = self._get_edge_coords(bottom_edges, self._center)
        self._non_bottom_edges = self._get_edge_coords(non_bottom_edges, self._center)

    def _get_bounds_coords(self, edges, bltr=False, pad_frac=0.0):
        """
        Provides the bounding box of the edges in solar x and y.

        :param edges: List of tuples, each tuple contains two corner points defining an edge.
        :type edges: list of tuple
        :param bltr: If True, returns bottom left and top right coordinates, otherwise returns minimum and maximum coordinates.
        :type bltr: bool, optional
        :param pad_frac: Fractional padding applied to each side of the box, expressed as a decimal, defaults to 0.0.
        :type pad_frac: float, optional

        :return: Coordinates of the box's bounds.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        xx = []
        yy = []
        for edge in edges:
            xx.append(edge.transform_to(self._frame_obs).Tx)
            yy.append(edge.transform_to(self._frame_obs).Ty)
        unit = xx[0][0].unit
        min_x = np.min(xx)
        max_x = np.max(xx)
        min_y = np.min(yy)
        max_y = np.max(yy)
        if pad_frac > 0:
            _pad = pad_frac * np.max([max_x - min_x, max_y - min_y, 20])
            min_x -= _pad
            max_x += _pad
            min_y -= _pad
            max_y += _pad
        if bltr:
            bottom_left = SkyCoord(min_x * unit, min_y * unit, frame=self._frame_obs)
            top_right = SkyCoord(max_x * unit, max_y * unit, frame=self._frame_obs)
            return [bottom_left, top_right]
        else:
            coords = SkyCoord(Tx=[min_x, max_x] * unit, Ty=[min_y, max_y] * unit,
                              frame=self._frame_obs)
            return coords

    def bounds_coords_bl_tr(self, pad_frac=0.0):
        """
        Calculates and returns the bottom left and top right coordinates of the box in the observer frame.
        Optionally applies a padding factor to expand the box dimensions symmetrically.

        :param pad_frac: Fractional padding applied to each side of the box, expressed as a decimal, defaults to 0.0.
        :type pad_frac: float, optional
        :return: Bottom left and top right coordinates of the box in the observer frame.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        return self._get_bounds_coords(self.all_edges, bltr=True, pad_frac=pad_frac)

    @property
    def bounds_coords(self):
        """
        Provides access to the box's bounds in the observer frame.

        :return: Coordinates of the box's bounds.
        :rtype: `~astropy.coordinates.SkyCoord`
        """
        return self._get_bounds_coords(self.all_edges)

    @property
    def bottom_bounds_coords(self):
        """
        Provides access to the box's bottom bounds in the observer frame.

        :return: Coordinates of the box's bottom bounds.
        :rtype: `~astropy.coordinates.SkyCoord`
        """
        return self._get_bounds_coords(self.bottom_edges)

    @property
    def bottom_cea_header(self):
        """
        Provides access to the box's bottom WCS CEA header.

        :return: The WCS CEA header for the box's bottom.
        :rtype: dict
        """
        return self._get_bottom_cea_header()

    @property
    def bottom_edges(self):
        """
        Provides access to the box's bottom edge coordinates.

        :return: Coordinates of the box's bottom edges.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        return self._bottom_edges

    @property
    def non_bottom_edges(self):
        """
        Provides access to the box's non-bottom edge coordinates.

        :return: Coordinates of the box's non-bottom edges.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        return self._non_bottom_edges

    @property
    def all_edges(self):
        """
        Provides access to all the edge coordinates of the box, combining both bottom and non-bottom edges.

        :return: Coordinates of all the edges of the box.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        return self._bottom_edges + self._non_bottom_edges

    @property
    def box_origin(self):
        """
        Provides read-only access to the box's origin coordinates.

        :return: The origin of the box in the specified frame.
        :rtype: `~astropy.coordinates.SkyCoord`
        """
        return self._center

    @property
    def box_dims(self):
        """
        Provides read-only access to the box's dimensions.

        :return: The dimensions of the box (length, width, height) in specified units.
        :rtype: `~astropy.units.Quantity`
        """
        return self._dims


class GxBox(QMainWindow):
    def __init__(self, time, observer, box_orig, box_dims=u.Quantity([100, 100, 100]) * u.Mm,
                 box_res=1.4 * u.Mm, pad_frac=0.25, data_dir=DOWNLOAD_DIR, gxmodel_dir=GXMODEL_DIR, external_box=None):
        """
        Main application window for visualizing and interacting with solar data in a 3D box.

        :param time: Observation time.
        :type time: `~astropy.time.Time`
        :param observer: Observer location.
        :type observer: `~astropy.coordinates.SkyCoord`
        :param box_orig: The origin of the box (center of the box bottom).
        :type box_orig: `~astropy.coordinates.SkyCoord`
        :param box_dims: Dimensions of the box in heliocentric coordinates, defaults to 100x100x100 Mm.
        :type box_dims: `~astropy.units.Quantity`
        :param box_res: Spatial resolution of the box, defaults to 1.4 Mm.
        :type box_res: `~astropy.units.Quantity`
        :param pad_frac: Fractional padding applied to each side of the box, expressed as a decimal, defaults to 0.25.
        :type pad_frac: float
        :param data_dir: Directory for storing data.
        :type data_dir: str
        :param gxmodel_dir: Directory for storing model outputs.
        :type gxmodel_dir: str
        :param external_box: Path to external box file (optional).
        :type external_box: str

        Methods
        -------
        loadmap(mapname, fov_coords=None)
            Loads a map from the available data.
        init_ui()
            Initializes the user interface.
        update_bottom_map(map_name)
            Updates the bottom map displayed in the UI.
        update_context_map(map_name)
            Updates the context map displayed in the UI.
        update_plot()
            Updates the plot with the current data and settings.
        create_lines_of_sight()
            Creates lines of sight for the visualization.
        visualize()
            Visualizes the data in the UI.

        Example
        -------
        >>> from astropy.time import Time
        >>> from astropy.coordinates import SkyCoord
        >>> import astropy.units as u
        >>> from pyampp.gxbox import GxBox
        >>> time = Time('2024-05-09T17:12:00')
        >>> observer = SkyCoord(0 * u.deg, 0 * u.deg, obstime=time, frame='heliographic_carrington')
        >>> box_orig = SkyCoord(450 * u.arcsec, -256 * u.arcsec, obstime=time, observer="earth", frame='helioprojective')
        >>> box_dims = u.Quantity([100, 100, 100], u.Mm)
        >>> box_res = 1.4 * u.Mm
        >>> gxbox = GxBox(time, observer, box_orig, box_dims, box_res)
        >>> gxbox.show()
        """

        super(GxBox, self).__init__()
        self.time = time
        self.observer = observer
        self.box_dimensions = box_dims
        self.box_res = box_res
        self.pad_frac = pad_frac
        ## this is the origin of the box, i.e., the center of the box bottom
        self.box_origin = box_orig
        self.sdofitsfiles = None
        self.frame_hcc = Heliocentric(observer=self.box_origin, obstime=self.time)
        self.frame_obs = Helioprojective(observer=self.observer, obstime=self.time)
        self.frame_hgs = HeliographicStonyhurst(obstime=self.time)
        self.lines_of_sight = []
        self.edge_coords = []
        self.axes = None
        self.fig = None
        self.init_map_context_name = '171'
        self.init_map_bottom_name = 'field'
        self.external_box = external_box

        ## this is a dummy map. it should be replaced by a real map from inputs.
        self.instrument_map = self.make_dummy_map(self.box_origin.transform_to(self.frame_obs))

        box_center = box_orig.transform_to(self.frame_hcc)
        box_center = SkyCoord(x=box_center.x,
                              y=box_center.y,
                              z=box_center.z + box_dims[2] / 2,
                              frame=box_center.frame)
        ## this is the center of the box
        self.box_center = box_center

        self.box = Box(self.frame_obs, self.box_origin, self.box_center, self.box_dimensions, self.box_res)
        self.box_bounds = self.box.bounds_coords
        self.bottom_wcs_header = self.box.bottom_cea_header

        self.fov_coords = self.box.bounds_coords_bl_tr(pad_frac=self.pad_frac)
        # print(f"Bottom left: {self.fov_coords[0]}; Top right: {self.fov_coords[1]}")

        if not all([coordinate_is_on_solar_disk(coord) for coord in self.fov_coords]):
            print("Warning: Some of the box corners are not on the solar disk. Please check the box dimensions.")

        download_sdo = downloader.SDOImageDownloader(time, data_dir=data_dir)
        self.sdofitsfiles = download_sdo.download_images()
        self.sdomaps = {}

        self.sdomaps[self.init_map_context_name] = self.loadmap(self.init_map_context_name)
        self.map_context = self.sdomaps[self.init_map_context_name]
        self.bottom_wcs_header['rsun_ref'] = self.map_context.meta['rsun_ref']
        self.sdomaps[self.init_map_bottom_name] = self.loadmap(self.init_map_bottom_name)

        self.map_bottom = self.sdomaps[self.init_map_bottom_name].reproject_to(self.bottom_wcs_header,
                                                                               algorithm="adaptive",
                                                                               roundtrip_coords=False)

        self.init_ui()

    def load_gxbox(self, boxfile):
        with open(boxfile, 'rb') as f:
            gxboxdata = pickle.load(f)
            for b3dtype in self.box.b3dtype:
                self.box.b3d[b3dtype] = gxboxdata['b3d'][b3dtype] if b3dtype in gxboxdata['b3d'].keys() else None

    @property
    def avaliable_maps(self):
        """
        Lists the available maps.

        :return: A list of available map keys.
        :rtype: list
        """
        if all(key in self.sdofitsfiles.keys() for key in HMI_B_SEGMENTS):
            return list(self.sdofitsfiles.keys()) + HMI_B_PRODUCTS
        else:
            return self.sdofitsfiles.keys()

    def corr_fov_coords(self, sunpymap, fov_coords):
        '''
        Corrects the field of view coordinates using the given map.
        :param sunpymap: The map to use for correction.
        :type sunpymap: sunpy.map.Map
        :param fov_coords: The field of view coordinates (bottom left and top right) as SkyCoord objects.
        :type fov_coords: list

        :return: Corrected field of view coordinates.
        :rtype: list
        '''
        fov_coords = [SkyCoord(Tx=fov_coords[0].Tx, Ty=fov_coords[0].Ty, frame=sunpymap.coordinate_frame),
                      SkyCoord(Tx=fov_coords[1].Tx, Ty=fov_coords[1].Ty, frame=sunpymap.coordinate_frame)]
        return fov_coords

    def _load_hmi_b_seg_maps(self, mapname, fov_coords):
        """
        Load specific HMI B segment maps required for the magnetic field vector data products.

        :param mapname: Name of the map to load.
        :type mapname: str
        :param fov_coords: The field of view coordinates (bottom left and top right) as SkyCoord objects.
        :type fov_coords: list
        :return: Loaded map object.
        :rtype: sunpy.map.Map
        :raises ValueError: If the map name is not in the expected HMI B segments.
        """
        if mapname not in HMI_B_SEGMENTS:
            raise ValueError(f"mapname: {mapname} must be one of {HMI_B_SEGMENTS}. Use loadmap method for others.")

        if mapname in self.sdomaps.keys():
            return self.sdomaps[mapname]

        loaded_map = Map(self.sdofitsfiles[mapname])
        fov_coords = self.corr_fov_coords(loaded_map, fov_coords)
        loaded_map = loaded_map.submap(fov_coords[0], top_right=fov_coords[1])
        # loaded_map = loaded_map.rotate(order=3)
        if mapname in ['azimuth']:
            if 'disambig' not in self.sdomaps.keys():
                self.sdomaps['disambig'] = Map(self.sdofitsfiles['disambig']).submap(fov_coords[0],
                                                                                     top_right=fov_coords[1])
            loaded_map = hmi_disambig(loaded_map, self.sdomaps['disambig'])

        self.sdomaps[mapname] = loaded_map
        return loaded_map

    def loadmap(self, mapname, fov_coords=None):
        """
        Loads a map from the available data.

        :param mapname: Name of the map to load.
        :type mapname: str
        :param fov_coords: Field of view coordinates (bottom left and top right) as SkyCoord objects, optional. Defaults to the entire FOV if not specified.
        :type fov_coords: list, optional
        :return: The requested map.
        :raises ValueError: If the specified map is not available.
        """
        if mapname not in self.avaliable_maps:
            raise ValueError(f"Map {mapname} is not available. mapname must be one of {self.avaliable_maps}")

        if mapname in self.sdomaps.keys():
            return self.sdomaps[mapname]

        if fov_coords is None:
            fov_coords = self.fov_coords

        if mapname in HMI_B_SEGMENTS:
            self._load_hmi_b_seg_maps(mapname, fov_coords)

        if mapname in HMI_B_PRODUCTS:
            for key in HMI_B_SEGMENTS:
                if key not in self.sdomaps.keys():
                    self.sdomaps[key] = self._load_hmi_b_seg_maps(key, fov_coords)
            map_bp, map_bt, map_br = hmi_b2ptr(self.sdomaps['field'], self.sdomaps['inclination'],
                                               self.sdomaps['azimuth'])
            self.sdomaps['bp'] = map_bp
            self.sdomaps['bt'] = map_bt
            self.sdomaps['br'] = map_br
            return self.sdomaps[mapname]

        # Load general maps
        loaded_map = Map(self.sdofitsfiles[mapname])
        fov_coords = self.corr_fov_coords(loaded_map, fov_coords)
        self.sdomaps[mapname] = loaded_map.submap(fov_coords[0], top_right=fov_coords[1])
        return self.sdomaps[mapname]

    def make_dummy_map(self, ref_coord):
        """
        Creates a dummy map for initialization purposes.

        :param ref_coord: Reference coordinate for the map.
        :type ref_coord: `~astropy.coordinates.SkyCoord`
        :return: The created dummy map.
        :rtype: sunpy.map.Map
        """
        instrument_data = np.nan * np.ones((50, 50))
        instrument_header = make_fitswcs_header(instrument_data,
                                                ref_coord,
                                                scale=u.Quantity([10, 10]) * u.arcsec / u.pix)
        return Map(instrument_data, instrument_header)

    def init_ui(self):
        """
        Initializes the user interface for the GxBox application.
        """
        self.setWindowTitle('GxBox Map Viewer')
        # self.setGeometry(100, 100, 800, 600)
        # Create a central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Layout
        main_layout = QVBoxLayout(central_widget)

        # Horizontal layout for dropdowns and labels
        dropdown_layout = QHBoxLayout()

        # Dropdown for bottom map selection
        self.map_bottom_selector = QComboBox()
        self.map_bottom_selector.addItems(list(self.avaliable_maps))
        self.map_bottom_selector.setCurrentIndex(self.avaliable_maps.index(self.init_map_bottom_name))
        self.map_bottom_selector_label = QLabel("Select Bottom Map:")
        dropdown_layout.addWidget(self.map_bottom_selector_label)
        dropdown_layout.addWidget(self.map_bottom_selector)

        # Dropdown for context map selection
        self.map_context_selector = QComboBox()
        self.map_context_selector.addItems(list(self.avaliable_maps))
        self.map_context_selector.setCurrentIndex(self.avaliable_maps.index(self.init_map_context_name))
        self.map_context_selector_label = QLabel("Select Context Map:")
        dropdown_layout.addWidget(self.map_context_selector_label)
        dropdown_layout.addWidget(self.map_context_selector)

        # Dropdown for 3D magnetic model selection
        self.b3d_model_selector = QComboBox()
        self.b3d_model_selector.addItems(self.box.b3dtype)
        self.b3d_model_selector.setCurrentIndex(0)
        self.b3d_model_selector_label = QLabel("Select 3D Mag. Model:")
        dropdown_layout.addWidget(self.b3d_model_selector_label)
        dropdown_layout.addWidget(self.b3d_model_selector)

        # Add the visualize button
        self.visualize_button = QPushButton("PyVista")
        self.visualize_button.clicked.connect(self.visualize_3d_magnetic_field)
        dropdown_layout.addWidget(self.visualize_button)

        main_layout.addLayout(dropdown_layout)

        # Connect dropdowns to their respective handlers
        self.map_bottom_selector.currentTextChanged.connect(self.update_bottom_map)
        self.map_context_selector.currentTextChanged.connect(self.update_context_map)

        if self.external_box is not None:
            if os.path.exists(self.external_box):
                self.load_gxbox(self.external_box)
        if self.box.b3d is {}:
            maglib_lff = mf_lfff()
            maglib_lff.set_field(self.map_bottom.data)
            self.box.b3d['lfff'] = maglib_lff.lfff_cube(self.box.dims_pix[-1].value)

        ## todo add external box import. using boundary map to update box input at pyampp..
        # import time
        #
        # start_time = time.time()
        # maglib_nlfff = MagFieldWrapper(nlfff_libpath)
        # dr = (self.box_res.to(u.m)).value
        # maglib_nlfff.load_cube_vars(self.box.b3d['lfff']['bx'], self.box.b3d['lfff']['by'],
        #                                                     self.box.b3d['lfff']['bz'], dr)
        # self.box.b3d['nlfff'] = maglib_nlfff.NLFFF()
        # end_time = time.time()
        #
        # print(f"The block of code took {end_time - start_time} seconds to run.")

        # from streamtracer import StreamTracer, VectorGrid
        #
        # bx = self.box.b3d_lfff['bx'].swapaxes(0, 1)
        # by = self.box.b3d_lfff['by'].swapaxes(0, 1)
        # bz = self.box.b3d_lfff['bz'].swapaxes(0, 1)
        # vector_field = np.stack([bx, by, bz], axis=-1).astype(np.float64)
        # grid = VectorGrid(vector_field, grid_coords=[self.box.grid_coords['x'].value,
        #                                              self.box.grid_coords['y'].value,
        #                                              self.box.grid_coords['z'].value])
        # step_size = 0.1  # grid space = 1
        # max_steps = 10000
        # tracer = StreamTracer(max_steps=max_steps, step_size=step_size)
        #
        # x = np.linspace(-50, 50, 10)
        # y = np.linspace(-30, 30, 10)
        # z = 696.5
        # xx, yy = np.meshgrid(x, y)
        #
        #
        # xx_final = np.hstack([xx.ravel()])
        # yy_final = np.hstack([yy.ravel()])
        # # Combine x, y, and z coordinates for seed points
        # seed_points = np.vstack([xx_final.ravel(), yy_final.ravel(), np.full_like(xx_final.ravel(), z)]).T
        #
        #
        # tracer.trace(seeds=seed_points, grid=grid, direction=0)
        # flines = tracer.xs

        # # Visual inspection of 3D field lines
        # fig_3d = plt.figure()
        # ax_3d = fig_3d.add_subplot(111, projection='3d')
        # for seed_point, fline in zip(seed_points, flines):
        #     color = next(ax_3d._get_lines.prop_cycler)['color']
        #     # ax_3d.plot(seed_point[0], seed_point[1], seed_point[2], 'o', color=color)
        #     ax_3d.plot(fline[:, 0], fline[:, 1], fline[:, 2], '-', color=color)
        # ax_3d.set_xlabel('X')
        # ax_3d.set_ylabel('Y')
        # ax_3d.set_zlabel('Z')
        # ax_3d.set_title('3D Field Lines')

        # Matplotlib Figure
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # Add Matplotlib Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)

        self.update_plot()

        map_context_aspect_ratio = (self.map_context.dimensions[1] / self.map_context.dimensions[0]).value
        window_width = 800
        window_height = int(window_width * map_context_aspect_ratio)

        # Adjust for padding, toolbar, and potential high DPI scaling
        window_width += 0  # Adjust based on your UI needs
        window_height += 150  # Includes space for toolbar and dropdowns

        self.setGeometry(100, 100, int(window_width), int(window_height))

    def visualize_3d_magnetic_field(self):
        """
        Launches the MagneticFieldVisualizer to visualize the 3D magnetic field data.
        """

        self.visualizer = MagneticFieldVisualizer(self.box)
        self.visualizer.show()

    def update_bottom_map(self, map_name):
        """
        Updates the bottom map displayed in the UI.

        :param map_name: Name of the map to be updated.
        :type map_name: str
        """
        map_bottom = self.sdomaps[map_name] if map_name in self.sdomaps.keys() else self.loadmap(map_name)
        self.map_bottom = map_bottom.reproject_to(self.bottom_wcs_header, algorithm="adaptive",
                                                  roundtrip_coords=False)
        self.update_plot()

    def update_context_map(self, map_name):
        """
        Updates the context map displayed in the UI.

        :param map_name: Name of the map to be updated.
        :type map_name: str
        """
        self.map_context = self.sdomaps[map_name] if map_name in self.sdomaps.keys() else self.loadmap(map_name)
        self.update_plot()

    def update_plot(self):
        """
        Updates the plot with the current data and settings.
        """
        self.fig.clear()
        self.axes = self.fig.add_subplot(projection=self.map_context)
        ax = self.axes
        self.map_context.plot(axes=ax)
        self.map_context.draw_grid(axes=ax, color='w', lw=0.5)
        self.map_context.draw_limb(axes=ax, color='w', lw=1.0)
        # for edge in self.simbox.bottom_edges:
        #     ax.plot_coord(edge, color='r', ls='-', marker='', lw=1.0)
        # for edge in self.simbox.non_bottom_edges:
        #     ax.plot_coord(edge, color='r', ls='--', marker='', lw=0.5)
        for edge in self.box.bottom_edges:
            ax.plot_coord(edge, color='tab:red', ls='--', marker='', lw=1.0)
        for edge in self.box.non_bottom_edges:
            ax.plot_coord(edge, color='tab:red', ls='-', marker='', lw=1.0)
        # self.map_context.draw_quadrangle(self.map_bottom.bottom_left_coord, axes=ax,
        #                                  width=self.map_bottom.top_right_coord.lon - self.map_bottom.bottom_left_coord.lon,
        #                                  height=self.map_bottom.top_right_coord.lat - self.map_bottom.bottom_left_coord.lat,
        #                                  edgecolor='tab:red', linestyle='--', linewidth=0.5)
        # ax.plot_coord(self.box_center, color='r', marker='+')
        # ax.plot_coord(self.box_origin, mec='r', mfc='none', marker='o')
        self.map_context.draw_quadrangle(
            self.box.bounds_coords,
            axes=ax,
            edgecolor="tab:blue",
            linestyle="--",
            linewidth=0.5,
        )
        self.map_bottom.plot(axes=ax, autoalign=True)
        ax.set_title(ax.get_title(), pad=45)
        self.fig.tight_layout()
        # Refresh canvas
        self.canvas.draw()

    def plot(self):
        """
        Plots the data in the UI.
        """
        self.update_plot()
        return self.fig

    def create_lines_of_sight(self):
        """
        Creates lines of sight for the visualization.
        """
        # The rest of the code for creating lines of sight goes here
        pass

    def visualize(self):
        """
        Visualizes the data in the UI.
        """
        # The rest of the code for visualization goes here
        pass


def main():
    """
    Main function to run the GxBox application.

    This function sets up the argument parser, processes the input arguments, and starts the GxBox application.

    Example
    -------
    To run the GxBox application from the command line, use the following command:

    .. code-block:: bash

        python pyAMPP/pyampp/gxboxox_factory.py --time 2014-11-01T16:40:00 --coords -632 -135 --hpc --box_dims 64 64 64 --box_res 1.400 --pad_frac 0.25
    """
    ## todo From Viktor: I advice you to switch from argparse to fire library. It can make it easy to create API for classes and functions and change it
    parser = argparse.ArgumentParser(description="Run GxBox with specified parameters.")
    parser.add_argument('--time', required=True, help='Observation time in ISO format, e.g., "2024-05-12T00:00:00"')
    parser.add_argument('--coords', nargs=2, type=float, required=True,
                        help='Center coordinates [x, y] in arcsec if HPC or deg if HGC or HGS')
    parser.add_argument('--hpc', action='store_true', help='Use Helioprojective coordinates (default)')
    parser.add_argument('--hgc', action='store_true', help='Use Heliographic Carrington coordinates')
    parser.add_argument('--hgs', action='store_true', help='Use Heliographic Stonyhurst coordinates')
    parser.add_argument('--box_dims', nargs=3, type=int, default=[64, 64, 64],
                        help='Box dimensions in pixels as three integers [dx, dy, dz]')
    parser.add_argument('--box_res', type=float, default=1.4, help='Box resolution in Mm per pixel')
    parser.add_argument('--observer', help='Observer location, default is Earth')
    parser.add_argument('--pad_frac', type=float, default=0.25,
                        help='Fractional padding applied to each side of the box, expressed as a decimal')
    parser.add_argument('--data_dir', default=DOWNLOAD_DIR, help='Directory for storing data')
    parser.add_argument('--gxmodel_dir', default=GXMODEL_DIR, help='Directory for storing model outputs')
    parser.add_argument('--external_box', default=os.path.abspath(os.getcwd()),
                        help='Path to external box file (optional)')
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive mode with access to memory and additional tools.')

    args = parser.parse_args()

    # Processing arguments
    time = Time(args.time)
    coords = args.coords
    box_dims = u.Quantity(args.box_dims, u.pix)
    box_res = args.box_res * u.Mm

    observer = get_earth(time) if not args.observer else SkyCoord.from_name(args.observer)

    if args.hpc:
        box_origin = SkyCoord(coords[0] * u.arcsec, coords[1] * u.arcsec, obstime=time, observer=observer,
                              rsun=696 * u.Mm, frame='helioprojective')
    elif args.hgc:
        box_origin = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time,
                              radius=696 * u.Mm,
                              frame='heliographic_carrington')
    elif args.hgs:
        box_origin = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time,
                              radius=696 * u.Mm,
                              frame='heliographic_stonyhurst')
    else:
        raise ValueError("Coordinate frame not specified or unknown.")

    # print(f"box_origin: {box_origin}")
    # print(f"box_origin.observer: {box_origin.observer}")
    # print(f"observer: {observer}")

    box_dimensions = box_dims / u.pix * box_res
    pad_frac = args.pad_frac
    data_dir = args.data_dir
    gxmodel_dir = args.gxmodel_dir
    external_box = args.external_box

    # Running the application
    app = QApplication([])
    gxbox = GxBox(time, observer, box_origin, box_dimensions, box_res, pad_frac=pad_frac, data_dir=data_dir,
                  gxmodel_dir=gxmodel_dir, external_box=external_box)
    gxbox.show()

    if args.interactive:
        # Start an interactive IPython session for more advanced debugging and exploration
        import IPython
        import matplotlib.pyplot as plt
        IPython.embed()
        # # Start the IPython interactive session in a separate thread to avoid blocking the Qt event loop
        # import threading
        # def interactive_shell(gxbox):
        #     IPython.embed(user_ns={'gxbox': gxbox})
        #
        # thread = threading.Thread(target=interactive_shell, args=(gxbox,))
        # thread.start()
    app.exec_()


if __name__ == '__main__':
    main()
    # import astropy.time
    # import sunpy.sun.constants
    # from astropy.coordinates import SkyCoord
    # from sunpy.coordinates import Heliocentric, Helioprojective, get_earth
    # import astropy.units as u
    # from pyampp.gxbox.gxbox_factory import GxBox
    #
    # # time = astropy.time.Time('2024-05-09T17:12:00')
    # # box_origin = SkyCoord(450 * u.arcsec, -256 * u.arcsec, distance,obstime=time, rsun = 696*u.Mm, observer="earth", frame='helioprojective')
    # # box_dimensions = u.Quantity([200, 200, 200]) * u.Mm
    #
    # time = astropy.time.Time('2014-11-01T16:40:00')
    # distance = sun.earth_distance(time)
    # box_origin = SkyCoord(lon=30 * u.deg, lat=20 * u.deg,
    #                       obstime=time,
    #                       radius=696 * u.Mm,
    #                       frame='heliographic_carrington')
    # ## dots source
    # # box_origin = SkyCoord(-475 * u.arcsec, -330 * u.arcsec, distance,obstime=time, rsun = 696*u.Mm, observer="earth", frame='helioprojective')
    # ## flare AR
    # box_origin = SkyCoord(-632 * u.arcsec, -135 * u.arcsec, obstime=time, rsun=696 * u.Mm, observer="earth",
    #                       frame='helioprojective')
    # box_dimensions = u.Quantity([150, 150, 100]) * u.Mm
    #
    # box_res = 0.6 * u.Mm
    # box_res = 1.4 * u.Mm
    # # box_dimensions = u.Quantity([128, 128, 128]) * u.Mm * 1.4
    # box_dimensions = u.Quantity([64, 64, 64]) * u.Mm * 1.4
    # observer = get_earth(time)
    #
    # app = QApplication(sys.argv)
    # gxbox = GxBox(time, observer, box_origin, box_dimensions, box_res)
    # gxbox.show()
    # sys.exit(app.exec_())
