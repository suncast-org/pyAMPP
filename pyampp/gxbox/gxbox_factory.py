import argparse
import itertools
import locale
# from pyampp.gxbox.magfield_viewer_dev import MagFieldViewer
import pickle
from pathlib import Path

import astropy.units as u
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QComboBox,QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, \
    QMainWindow, \
    QPushButton, QVBoxLayout, QWidget
from astropy.coordinates import SkyCoord
from astropy.time import Time
from matplotlib import colormaps as mplcmaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pyAMaFiL.mag_field_proc import MagFieldProcessor
from pyAMaFiL.mag_field_lin_fff import MagFieldLinFFF
from sunpy.coordinates import Heliocentric, HeliographicStonyhurst, Helioprojective, get_earth
from sunpy.map import Map, coordinate_is_on_solar_disk, make_fitswcs_header

import pyampp
from pyampp.data import downloader
from pyampp.gxbox.boxutils import hmi_b2ptr, hmi_disambig, read_b3d_h5
from pyampp.gxbox.magfield_viewer import MagFieldViewer
from pyampp.util.config import *


os.environ['OMP_NUM_THREADS'] = '16'  # number of parallel threads
locale.setlocale(locale.LC_ALL, "C");



## todo add chrom mask to the tool
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
        self.b3dtype = ['pot', 'nlfff']
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
        # bottom_cea_header['OBSRVTRY'] = str(origin.observer)
        bottom_cea_header['OBSRVTRY'] = 'None'
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

    @property
    def box_view_up(self):
        """
        Retrieves an edge from the bottom edges where the x and z coordinates of both points are the same.

        :return: The edge with the same x and z coordinates.
        :rtype: `~astropy.coordinates.SkyCoord` or None
        """
        for edge in self.bottom_edges:
            if np.allclose(edge.x[0], edge.x[1]) and np.allclose(edge.z[0], edge.z[1]):
                return edge
        return None

    @property
    def box_norm_direction(self):
        """
        Retrieves an edge from the bottom edges where the x and z coordinates of both points are the same.

        :return: The edge with the same x and z coordinates.
        :rtype: `~astropy.coordinates.SkyCoord` or None
        """
        for edge in self.non_bottom_edges:
            if np.allclose(edge.x[0], edge.x[1]) and np.allclose(edge.y[0], edge.y[1]):
                return edge
        return None


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
        print('observer:', self.box_origin)
        self.frame_hcc = Heliocentric(observer=self.box_origin, obstime=self.time)
        self.frame_obs = Helioprojective(observer=self.observer, obstime=self.time)
        self.frame_hgs = HeliographicStonyhurst(obstime=self.time)
        self.lines_of_sight = []
        self.edge_coords = []
        self.axes = None
        self.fig = None
        self.axes_world_coords = None
        self.axes_world_coords_init = None
        self.init_map_context_name = '171'
        self.init_map_bottom_name = 'field'
        self.external_box = external_box
        self.fieldlines_coords = []
        self.fieldlines_line_collection = []  # Initialize an empty list to store LineCollections
        self.fieldlines_show_status = True  # Initial status of the fieldlines visibility
        self.map_context_im = None
        self.map_bottom_im = None
        self.pot_res = None

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

        # print(self.bottom_wcs_header)
        self.map_bottom = self.sdomaps[self.init_map_bottom_name].reproject_to(self.bottom_wcs_header,
                                                                               algorithm="adaptive",
                                                                               roundtrip_coords=False)

        self.init_ui()

    def box_norm_direction(self):
        cartesian_coords = self.box_origin.transform_to(
            Heliocentric(observer=self.observer, obstime=self.time)).cartesian.xyz.value
        normal_vector = cartesian_coords / np.linalg.norm(cartesian_coords)
        return normal_vector

    # def box_norm_direction(self):
    #     cartesian_coords = np.diff(self.box.box_norm_direction.transform_to(Heliocentric(observer=self.observer, obstime=self.time)).cartesian.xyz).value
    #     normal_vector = np.squeeze(cartesian_coords / np.linalg.norm(cartesian_coords))
    #     normal_vector = normal_vector[1]/abs(normal_vector[1])*normal_vector
    #     return normal_vector

    def box_view_up(self):
        cartesian_coords = np.diff(self.box.box_view_up.transform_to(
            Heliocentric(observer=self.observer, obstime=self.time)).cartesian.xyz).value
        normal_vector = np.squeeze(cartesian_coords / np.linalg.norm(cartesian_coords))
        normal_vector = normal_vector[1] / abs(normal_vector[1]) * normal_vector
        return normal_vector

    def load_gxbox(self, boxfile):
        if os.path.basename(boxfile).endswith('.gxbox'):
            with open(boxfile, 'rb') as f:
                gxboxdata = pickle.load(f)
                for b3dtype in self.box.b3dtype:
                    self.box.b3d[b3dtype] = gxboxdata['b3d'][b3dtype] if b3dtype in gxboxdata['b3d'].keys() else None
        elif os.path.basename(boxfile).endswith('.h5'):
            self.box.b3d = read_b3d_h5(boxfile)

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

        print(f'fov_coords: {fov_coords}')
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
            if mapname in self.sdomaps.keys():
                return self.sdomaps[mapname]
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
        control_layout = QHBoxLayout()

        # Matplotlib Figure
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # Add Matplotlib Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)

        map_control_group = QGroupBox("Map Controls")
        # Horizontal layout for dropdowns and labels
        map_control_layout = QVBoxLayout()
        map_control_layout1 = QHBoxLayout()
        map_control_layout2 = QHBoxLayout()
        map_control_layout3 = QHBoxLayout()


        # Dropdown for bottom map selection
        self.map_bottom_selector = QComboBox()
        self.map_bottom_selector.addItems(list(self.avaliable_maps))
        self.map_bottom_selector.setCurrentIndex(self.avaliable_maps.index(self.init_map_bottom_name))
        self.map_bottom_selector.setMaximumWidth(100)
        self.map_bottom_selector_label = QLabel("Bottom Map:")
        map_control_layout1.addWidget(self.map_bottom_selector_label)
        map_control_layout1.addWidget(self.map_bottom_selector)

        # Dropdown for context map selection
        self.map_context_selector = QComboBox()
        self.map_context_selector.addItems(list(self.avaliable_maps))
        self.map_context_selector.setCurrentIndex(self.avaliable_maps.index(self.init_map_context_name))
        self.map_context_selector.setMaximumWidth(100)
        self.map_context_selector_label = QLabel("Context Map:")
        map_control_layout2.addWidget(self.map_context_selector_label)
        map_control_layout2.addWidget(self.map_context_selector)

        # Dropdown for 3D magnetic model selection
        self.b3d_model_selector = QComboBox()
        self.b3d_model_selector.addItems(self.box.b3dtype)
        self.b3d_model_selector.setCurrentIndex(0)
        self.b3d_model_selector_label = QLabel("3D Mag. Model:")
        map_control_layout3.addWidget(self.b3d_model_selector_label)
        map_control_layout3.addWidget(self.b3d_model_selector)

        map_control_layout.addLayout(map_control_layout1)
        map_control_layout.addLayout(map_control_layout2)
        map_control_layout.addLayout(map_control_layout3)
        map_control_group.setLayout(map_control_layout)
        control_layout.addWidget(map_control_group)
        map_control_group.setFixedHeight(160)
        # map_control_group.adjustSize()

        # Connect dropdowns to their respective handlers
        self.map_bottom_selector.currentTextChanged.connect(self.update_bottom_map)
        self.map_context_selector.currentTextChanged.connect(self.update_context_map)

        fieldline_control_group = QGroupBox("Field Line Controls")
        fieldline_control_layout = QVBoxLayout()
        fieldline_control_layout1 = QHBoxLayout()
        fieldline_control_layout2 = QHBoxLayout()
        fieldline_control_layout3 = QHBoxLayout()

        # Add the visualize button
        self.visualize_button = QPushButton("3D viewer")
        self.visualize_button.setToolTip("Visualize the 3D magnetic field.")
        self.visualize_button.clicked.connect(self.visualize_3d_magnetic_field)
        fieldline_control_layout1.addWidget(self.visualize_button)

        self.toggle_fieldlines_button = QPushButton("Hide")
        self.toggle_fieldlines_button.setToolTip("Toggle the visibility of the field lines.")
        self.toggle_fieldlines_button.clicked.connect(self.toggle_fieldlines_visibility)
        fieldline_control_layout1.addWidget(self.toggle_fieldlines_button)

        self.clear_fieldlines_button = QPushButton("Clear")
        self.clear_fieldlines_button.setToolTip("Clear the field lines.")
        self.clear_fieldlines_button.clicked.connect(self.clear_fieldlines)
        fieldline_control_layout1.addWidget(self.clear_fieldlines_button)

        self.save_fieldlines_button = QPushButton("Save")
        self.save_fieldlines_button.setToolTip("Save the field lines to a file.")
        self.save_fieldlines_button.clicked.connect(
            lambda: self.save_fieldlines(f'fieldlines_{self.time.to_datetime().strftime("%Y%m%dT%H%M%S")}.pkl'))
        fieldline_control_layout1.addWidget(self.save_fieldlines_button)

        self.bminmax_label = QLabel("Bmin/Bmax [G]:")
        self.bmin_input = QLineEdit(self)
        self.bmin_input.setText("0")  # Set default value
        self.bmin_clip_checkbox = QCheckBox("Clip")
        self.bmin_clip_checkbox.setChecked(False)
        self.bmax_input = QLineEdit(self)
        self.bmax_input.setText("1000")  # Set default value
        self.bmax_clip_checkbox = QCheckBox("Clip")
        self.bmax_clip_checkbox.setChecked(False)

        fieldline_control_layout2.addWidget(self.bminmax_label)
        fieldline_control_layout2.addWidget(self.bmin_input)
        fieldline_control_layout2.addWidget(self.bmin_clip_checkbox)
        fieldline_control_layout2.addWidget(self.bmax_input)
        fieldline_control_layout2.addWidget(self.bmax_clip_checkbox)

        self.cmap_selector = QComboBox(self)
        self.cmap_selector.addItems(sorted(list(mplcmaps)))  # List all available colormaps
        self.cmap_selector.setCurrentText("viridis")  # Set a default colormap
        self.cmap_selector.setMaximumWidth(200)
        self.discrete_cmap_bounds_input = QLineEdit(self)
        self.discrete_cmap_bounds_input.setPlaceholderText("Enter bounds (comma-separated)")


        # self.cmap_clip_checkbox = QCheckBox("Clip")
        # self.cmap_clip_checkbox.setChecked(False)
        # self.cmap_clip_checkbox.clicked.connect(self.toggle_cmap_clip)

        fieldline_control_layout3.addWidget(QLabel("cmap:"))
        fieldline_control_layout3.addWidget(self.cmap_selector)
        # fieldline_control_layout3.addWidget(self.cmap_clip_checkbox)
        fieldline_control_layout3.addWidget(self.discrete_cmap_bounds_input)
        fieldline_control_layout3.setStretch(0, 1)

        fieldline_control_layout.addLayout(fieldline_control_layout1)
        fieldline_control_layout.addLayout(fieldline_control_layout2)
        fieldline_control_layout.addLayout(fieldline_control_layout3)

        fieldline_control_group.setLayout(fieldline_control_layout)
        control_layout.addWidget(fieldline_control_group)
        fieldline_control_group.setFixedHeight(160)
        # fieldline_control_group.adjustSize()

        main_layout.addLayout(control_layout)

        if self.external_box is not None:
            if os.path.isfile(self.external_box):
                self.load_gxbox(self.external_box)

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
        box_norm_direction = self.box_norm_direction()
        box_view_up = self.box_view_up()
        b3dtype = self.b3d_model_selector.currentText()
        # print(f'type of self.box.b3d is {type(self.box.b3d)}')
        # print(f'value of self.box.b3d is {self.box.b3d}')
        # if b3dtype == 'pot':
        if b3dtype == 'nlfff' and self.box.b3d['nlfff'] is not None:
            pass
        else:
            if self.box.b3d['pot'] is None:
                if self.map_bottom_selector.currentText() != 'br':
                    self.map_bottom_selector.setCurrentIndex(self.avaliable_maps.index('br'))
                maglib_lff = MagFieldLinFFF()
                bnddata = self.map_bottom.data
                bnddata[np.isnan(bnddata)] = 0.0

                maglib_lff.set_field(bnddata)
                ## the axis order in res is y, x, z. so we need to swap the first two axes, so that the order becomes x, y, z.
                self.pot_res = maglib_lff.lfff_cube(nz = self.box.dims_pix[-1].value, alpha=0.0)
                self.box.b3d['pot'] = {}
                self.box.b3d['pot']['bx'] = self.pot_res['by'].swapaxes(0, 1)
                self.box.b3d['pot']['by'] = self.pot_res['bx'].swapaxes(0, 1)
                self.box.b3d['pot']['bz'] = self.pot_res['bz'].swapaxes(0, 1)

            if b3dtype == 'nlfff':
                self.box.b3d['nlfff'] = {}
                # if 'lfff' not in self.box.b3d.keys():
                #     self.box.b3d['lfff'] = {}
                bx_lff, by_lff, bz_lff = [self.box.b3d['pot'][k].swapaxes(0, 1) for k in ("by", "bx", "bz")]

                # replace bottom boundary of lff solution with initial boundary conditions
                bvect_bottom = {}
                bvect_bottom['bz'] = self.sdomaps['br'] if 'br' in self.sdomaps.keys() else self.loadmap('br')
                bvect_bottom['bx'] = -self.sdomaps['bt'] if 'bt' in self.sdomaps.keys() else -self.loadmap('bt')
                bvect_bottom['by'] = self.sdomaps['bp'] if 'bp' in self.sdomaps.keys() else self.loadmap('bp')

                self.bvect_bottom = {}
                for k in bvect_bottom.keys():
                    self.bvect_bottom[k] = bvect_bottom[k].reproject_to(self.bottom_wcs_header, algorithm="adaptive",
                                                                        roundtrip_coords=False)

                self.bvect_bottom_data = {}
                for k in bvect_bottom.keys():
                    self.bvect_bottom_data[k] = self.bvect_bottom[k].data
                    self.bvect_bottom_data[k][np.isnan(self.bvect_bottom_data[k])] = 0.0
                bx_lff[:, :, 0] = self.bvect_bottom_data['bx']
                by_lff[:, :, 0] = self.bvect_bottom_data['by']
                bz_lff[:, :, 0] = self.bvect_bottom_data['bz']

                import time
                t0 = time.time()
                print(f'Starting NLFFF computation...')
                maglib = MagFieldProcessor()
                maglib.load_cube_vars(self.pot_res)

                res_nlf = maglib.NLFFF()
                print(f'Time taken to compute NLFFF solution: {time.time() - t0} seconds')

                bx_nlff, by_nlff, bz_nlff = [res_nlf[k].swapaxes(0, 1) for k in ("by", "bx", "bz")]
                self.box.b3d['nlfff']['bx'] = bx_nlff
                self.box.b3d['nlfff']['by'] = by_nlff
                self.box.b3d['nlfff']['bz'] = bz_nlff


        self.visualizer = MagFieldViewer(self.box, parent=self, box_norm_direction=box_norm_direction,
                                         box_view_up=box_view_up, time=self.time, b3dtype=b3dtype)
        self.visualizer.show()



    def update_bottom_map(self, map_name):
        """
        Updates the bottom map displayed in the UI.

        :param map_name: Name of the map to be updated.
        :type map_name: str
        """
        if self.map_bottom_im is not None:
            self.map_bottom_im.remove()
        map_bottom = self.sdomaps[map_name] if map_name in self.sdomaps.keys() else self.loadmap(map_name)
        self.map_bottom = map_bottom.reproject_to(self.bottom_wcs_header, algorithm="adaptive",
                                                  roundtrip_coords=False)

        self.box._dims_pix[0] = self.map_bottom.data.shape[1]
        self.box._dims_pix[1] = self.map_bottom.data.shape[0]

        self.map_bottom_im = self.map_bottom.plot(axes=self.axes, autoalign=True)
        # self.update_plot()
        self.canvas.draw()

    def update_context_map(self, map_name):
        """
        Updates the context map displayed in the UI.

        :param map_name: Name of the map to be updated.
        :type map_name: str
        """
        if self.map_context_im is not None:
            self.map_context_im.remove()
        if map_name in HMI_B_SEGMENTS + HMI_B_PRODUCTS + ['magnetogram', 'continuum']:
            map_context_prev = self.map_context
        self.map_context = self.sdomaps[map_name] if map_name in self.sdomaps.keys() else self.loadmap(map_name)
        if map_name in HMI_B_SEGMENTS+HMI_B_PRODUCTS + ['magnetogram','continuum']:
            self.map_context = self.map_context.rotate(order=3)
            self.map_context = self.map_context.reproject_to(map_context_prev.wcs)
        # else:
        self.map_context_im = self.map_context.plot(axes=self.axes)
        self.canvas.draw()

    @property
    def get_axes_world_coords(self):
        # Get pixel bounds
        pixel_coords_x = self.axes.get_xlim()
        pixel_coords_y = self.axes.get_ylim()

        # Convert pixel bounds to world coordinates
        world_coords = self.map_context.wcs.pixel_to_world(pixel_coords_x, pixel_coords_y)

        return world_coords

    def get_axes_pixel_coords(self, coords_world=None):
        if coords_world is None:
            coords_world = self.get_axes_world_coords
        world_coords = SkyCoord(Tx=coords_world.Tx, Ty=coords_world.Ty, frame=self.map_context.coordinate_frame)
        pixel_coords_x, pixel_coords_y = self.map_context.wcs.world_to_pixel(world_coords)
        return pixel_coords_x, pixel_coords_y

    # def toggle_cmap_clip(self):
    #     """
    #     Toggles the clipping of the colormap.
    #     """
    #     if self.cmap_clip_checkbox.isChecked():
    #         self.bmax_clip_checkbox.setChecked(True)
    #         self.bmin_clip_checkbox.setChecked(True)
    #     else:
    #         self.bmax_clip_checkbox.setChecked(False)
    #         self.bmin_clip_checkbox.setChecked(False)

    def toggle_fieldlines_visibility(self):
        """
        Toggles the visibility of the fieldlines from the plot.
        """
        self.fieldlines_show_status = not self.fieldlines_show_status
        print("Toggling fieldlines visibility: ", self.fieldlines_show_status)

        if len(self.fieldlines_line_collection) == 0:
            return

        for lc in self.fieldlines_line_collection:
            lc.set_visible(self.fieldlines_show_status)

        if self.fieldlines_show_status:
            self.toggle_fieldlines_button.setText("Hide")
        else:
            self.toggle_fieldlines_button.setText("Show")

        self.canvas.draw()

    def clear_fieldlines(self):
        """
        Clears the fieldlines from the plot.
        """
        if len(self.fieldlines_line_collection) == 0:
            return

        while self.fieldlines_line_collection:
            lc = self.fieldlines_line_collection.pop()
            lc.remove()  # Remove the LineCollection from the axes
        self.canvas.draw()

    def update_plot(self, show_bound_box=True, show_box_outline=True):
        """
        Updates the plot with the current data and settings.
        """
        if self.axes is not None:
            self.axes_world_coords = self.get_axes_world_coords
        self.fig.clear()
        self.axes = self.fig.add_subplot(projection=self.map_context)
        ax = self.axes
        self.map_context_im = self.map_context.plot(axes=ax)
        self.map_context.draw_grid(axes=ax, color='w', lw=0.5)
        self.map_context.draw_limb(axes=ax, color='w', lw=1.0)

        # for edge in self.simbox.bottom_edges:
        #     ax.plot_coord(edge, color='r', ls='-', marker='', lw=1.0)
        # for edge in self.simbox.non_bottom_edges:
        #     ax.plot_coord(edge, color='r', ls='--', marker='', lw=0.5)
        if show_box_outline:
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
        if show_bound_box:
            self.map_context.draw_quadrangle(
                self.box.bounds_coords,
                axes=ax,
                edgecolor="tab:blue",
                linestyle="--",
                linewidth=0.5,
            )
        self.map_bottom_im = self.map_bottom.plot(axes=ax, autoalign=True)
        ax.set_title(ax.get_title(), pad=45)
        if self.axes_world_coords_init is None:
            self.axes_world_coords_init = self.get_axes_world_coords
        if self.axes_world_coords is not None:
            axes_pixel_coords = self.get_axes_pixel_coords()
            ax.set_xlim(axes_pixel_coords[0])
            ax.set_ylim(axes_pixel_coords[1])
        self.fig.tight_layout()
        # Refresh canvas
        self.canvas.draw()

    def extract_streamlines(self, streamlines):
        """
        Extracts individual streamlines from the streamlines data.

        :param streamlines: pyvista.PolyData
            The streamlines data.
        :return: list of numpy.ndarray
            A list of individual streamlines.
        """
        lines = []
        fields = []
        n_lines = streamlines.lines.shape[0]
        i = 0
        while i < n_lines:
            num_points = streamlines.lines[i]
            start_idx = streamlines.lines[i + 1]
            end_idx = start_idx + num_points
            line = streamlines.points[start_idx:end_idx]
            lines.append(line)
            bx = streamlines['bx'][start_idx:end_idx]
            by = streamlines['by'][start_idx:end_idx]
            bz = streamlines['bz'][start_idx:end_idx]
            magnitude = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)

            fields.append({'bx': bx, 'by': by, 'bz': bz, 'magnitude': magnitude})
            i += num_points + 1
        return lines, fields

    def plot_fieldlines(self, streamlines, z_base=0.0):
        """
        Plots the extracted fieldlines with colorization based on their magnitude.

        :param streamlines: pyvista.PolyData
            The streamlines data.
        """
        self.flines = {'coords_hcc': [], 'fields': [], 'frame_obs': self.frame_obs}

        from matplotlib.collections import LineCollection

        ax = self.axes

        # Fetch Bmin and Bmax values from input fields
        try:
            bmin = float(self.bmin_input.text())
            bmax = float(self.bmax_input.text())
        except ValueError:
            bmin = 0
            bmax = 1000



        # Normalize the magnitude values for colormap
        cmap = plt.get_cmap(self.cmap_selector.currentText())
        # Check if bounds input box is empty
        bounds_text = self.discrete_cmap_bounds_input.text()
        if bounds_text:
            bounds = list(map(float, bounds_text.split(',')))
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
        else:
            norm = mcolors.Normalize(vmin=bmin, vmax=bmax)

        for streamlines_subset in streamlines:
            coords_hcc = []

            coords, fields = self.extract_streamlines(streamlines_subset)
            for coord, field in zip(coords, fields):
                # Convert the streamline coordinates to the gxbox frame_obs
                coord_hcc = SkyCoord(x=coord[:, 0] * u.Mm, y=coord[:, 1] * u.Mm, z=(coord[:, 2] + z_base) * u.Mm,
                                     frame=self.frame_hcc)
                coords_hcc.append(coord_hcc)
                coord_hpc = coord_hcc.transform_to(self.frame_obs)
                # ax.plot_coord(coord_hpc, '-', c='tab:blue', lw=0.3, alpha=0.5)
                xpix, ypix = self.map_context.world_to_pixel(coord_hpc)
                x = xpix.value
                y = ypix.value
                magnitude = field['magnitude']
                segments = [((x[i], y[i]), (x[i + 1], y[i + 1])) for i in range(len(x) - 1)]
                colors = [cmap(norm(value)) for value in magnitude]  # Colormap for each segment
                if self.bmin_clip_checkbox.isChecked() or self.bmax_clip_checkbox.isChecked():
                    bmin=0.0
                    bmax=5e6 ## an unrealistic large B field value for solar corona
                    if self.bmin_clip_checkbox.isChecked():
                        bmin = float(self.bmin_input.text())
                    if self.bmax_clip_checkbox.isChecked():
                        bmax = float(self.bmax_input.text())
                    mask = np.logical_and(magnitude >= bmin, magnitude <= bmax)
                    colors = np.array(colors)[mask]
                    segments = np.array(segments)[mask[:-1]]
                # if self.cmap_clip_checkbox.isChecked():
                #     bmin = float(self.bmin_input.text())
                #     bmax = float(self.bmax_input.text())
                #     mask = np.logical_and(magnitude >= bmin, magnitude <= bmax)
                #     colors = np.array(colors)[mask]
                #     segments = np.array(segments)[mask[:-1]]
                lc = LineCollection(segments, colors=colors, linewidths=0.5)
                ax.add_collection(lc)
                self.fieldlines_line_collection.append(lc)
                if not self.fieldlines_show_status:
                    lc.set_visible(False)
            self.flines['coords_hcc'].append(coords_hcc)
            self.flines['fields'].append(fields)
        self.canvas.draw()

    def save_fieldlines(self, default_filename='fieldlines.pkl'):
        """
        Saves the fieldlines data to a file. Prompts the user to select a directory and input a filename.

        :param default_filename: str
            The default name of the file to save the fieldlines data.
        """
        # Open a file dialog to select directory and input filename
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save Field Lines", default_filename, "Pickle Files (*.pkl)",
                                                  options=options)

        # Save the fieldlines if a valid filename is provided
        if filename:
            with open(filename, 'wb') as f:
                pickle.dump(self.flines, f)
            print(f"Field lines saved to {filename}")

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
                        help='Box dimensions in pixels as three integers')
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
                              radius=696 * u.Mm, observer=observer,
                              frame='heliographic_carrington')
    elif args.hgs:
        box_origin = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time,
                              radius=696 * u.Mm, observer=observer,
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
