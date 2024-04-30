import itertools
import astropy.units as u
import astropy.time
import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
import sunpy.sun.constants
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Heliocentric, Helioprojective
from datetime import datetime, timedelta
import os

hcc_orientation = SkyCoord(lon=30 * u.deg, lat=20 * u.deg,
                           radius=sunpy.sun.constants.radius,
                           frame='heliographic_stonyhurst')
date = astropy.time.Time('2020-01-01')
observer = SkyCoord(lon=0 * u.deg, lat=10 * u.deg, radius=1 * u.AU, frame='heliographic_stonyhurst')
box_dimensions = u.Quantity([100, 100, 100]) * u.Mm


def download_sdo_image_FITS(time, uv=True, euv=True, hmi=True):
    """
    Downloads specified types of SDO solar images FITS for a given time using SunPy Fido from JSOC.

    Parameters:
    - time (Astropy.Time): The target time for image downloads.
    - uv (bool, optional): Download ultraviolet (UV) images. Defaults to True.
    - euv (bool, optional): Download extreme ultraviolet (EUV) images. Defaults to True.
    - hmi (bool, optional): Download Helioseismic and Magnetic Imager (HMI) images. Defaults to True.

    Returns:
    - dict: Keys are image types ('euv', 'uv', 'hmi_b', 'hmi_m', 'hmi_ic') with values as lists of paths to the downloaded FITS files.

    Saves files to 'gxbox_data/YYYYMMDD', creating the directory if necessary.
    """
    from sunpy.net import Fido, attrs as a
    notify = a.jsoc.Notify("suncasa-group@njit.edu")
    path = os.path.join('gxbox_data', time.datetime.strftime('%Y%m%d'))
    if not path.endswith('/'):
        path = path + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    def search_and_fetch(time, series, segments=None):
        if segments:
            result = Fido.search(a.Time(time, time),
                                 a.jsoc.Series(series),
                                 segments,
                                 notify)
        else:
            result = Fido.search(a.Time(time, time),
                                 a.jsoc.Series(series),
                                 notify)
        return Fido.fetch(result, path=path, overwrite=False)

    image_types = {}
    if euv:
        image_types['euv'] = {'series': 'aia.lev1_euv_12s', 'segments': a.jsoc.Segment('image')}
    if uv:
        image_types['uv'] = {'series': 'aia.lev1_uv_24s', 'segments': a.jsoc.Segment('image')}
    if hmi:
        image_types['hmi_b'] = {'series': 'hmi.B_720s',
                                'segments': a.jsoc.Segment('field') & a.jsoc.Segment('inclination') & a.jsoc.Segment(
                                    'azimuth') & a.jsoc.Segment('disambig')}
        image_types['hmi_m'] = {'series': 'hmi.M_720s', 'segments': a.jsoc.Segment('magnetogram')}
        image_types['hmi_ic'] = {'series': 'hmi.Ic_noLimbDark_720s', 'segments': a.jsoc.Segment('continuum')}

    all_files = {key: search_and_fetch(time, **value) for key, value in image_types.items()}
    return all_files


class Box:
    '''
    Represents a 3D box defined by its origin and dimensions. It calculates and stores the coordinates of the box's edges, distinguishing between bottom edges and other edges.
    '''

    def __init__(self, box_origin, box_dims):
        '''
        Initializes the Box instance with origin, dimensions, and computes the corners and edges.

        :param box_origin: SkyCoord, the origin point of the box in a given coordinate frame.
        :param box_dims: u.Quantity, the dimensions of the box (length, width, height) in specified units.
        '''
        self._box_origin = box_origin
        self._box_dims = box_dims
        # Generate corner points based on the dimensions
        self.corners = list(itertools.product(self._box_dims[0] / 2 * [-1, 1],
                                              self._box_dims[1] / 2 * [-1, 1],
                                              self._box_dims[2] / 2 * [-1, 1]))

        # Identify edges as pairs of corners differing by exactly one dimension
        self.edges = [edge for edge in itertools.combinations(self.corners, 2)
                      if np.count_nonzero(u.Quantity(edge[0]) - u.Quantity(edge[1])) == 1]
        # Initialize properties to store categorized edges
        self._bottom_edges = None
        self._non_bottom_edges = None
        self._calculate_edge_types()  # Categorize edges upon initialization

    def _get_edge_coords(self, edges, box_origin):
        '''
        Translates edge corner points to their corresponding SkyCoord based on the box's origin.

        :param edges: list of tuples, each tuple contains two corner points defining an edge.
        :return: list of SkyCoord, coordinates of edges in the box's frame.
        '''
        return [SkyCoord(x=box_origin.x + u.Quantity([edge[0][0], edge[1][0]]),
                         y=box_origin.y + u.Quantity([edge[0][1], edge[1][1]]),
                         z=box_origin.z + u.Quantity([edge[0][2], edge[1][2]]),
                         frame=box_origin.frame) for edge in edges]

    def _calculate_edge_types(self):
        '''
        Separates the box's edges into bottom edges and non-bottom edges. This is done in a single pass to improve efficiency.
        '''
        min_z = min(corner[2] for corner in self.corners)
        bottom_edges, non_bottom_edges = [], []
        for edge in self.edges:
            if edge[0][2] == min_z and edge[1][2] == min_z:
                bottom_edges.append(edge)
            else:
                non_bottom_edges.append(edge)
        self._bottom_edges = self._get_edge_coords(bottom_edges, self._box_origin)
        self._non_bottom_edges = self._get_edge_coords(non_bottom_edges, self._box_origin)

    @property
    def bottom_edges(self):
        '''
        Provides access to the box's bottom edge coordinates.

        :return: list of SkyCoord, coordinates of the box's bottom edges.
        '''
        return self._bottom_edges

    @property
    def non_bottom_edges(self):
        '''
        Provides access to the box's non-bottom edge coordinates.

        :return: list of SkyCoord, coordinates of the box's non-bottom edges.
        '''
        return self._non_bottom_edges

    @property
    def all_edges(self):
        '''
        Provides access to all the edge coordinates of the box, combining both bottom and non-bottom edges.

        :return: list of SkyCoord, coordinates of all the edges of the box.
        '''
        return self._bottom_edges + self._non_bottom_edges

    @property
    def box_origin(self):
        '''
        Provides read-only access to the box's origin coordinates.

        :return: SkyCoord, the origin of the box in the specified frame.
        '''
        return self._box_origin

    @property
    def box_dims(self):
        '''
        Provides read-only access to the box's dimensions.

        :return: u.Quantity, the dimensions of the box (length, width, height) in specified units.
        '''
        return self._box_dims


class GxBox:
    def __init__(self, time, observer, hcc_orientation, box_dimensions):
        self.time = time
        self.observer = observer
        self.box_dimensions = box_dimensions
        self.hcc_orientation = hcc_orientation
        self.frame_hcc = Heliocentric(observer=self.hcc_orientation, obstime=self.time)
        self.frame_hpc = Helioprojective(observer=self.observer, obstime=self.time)
        self.lines_of_sight = []
        self.box_origin = None
        self.edge_coords = []
        self.axes = None
        self.fig = None

        ## this is a dummy map. it should be replaced by a real map from inputs.
        self.instrument_map = self.make_dummy_map(self.hcc_orientation.transform_to(self.frame_hpc))

        box_origin = hcc_orientation.transform_to(self.frame_hcc)
        box_origin = SkyCoord(x=box_origin.x,
                              y=box_origin.y,
                              z=box_origin.z + box_dimensions[2] / 2,
                              frame=box_origin.frame)
        self.box_origin = box_origin

        self.define_simbox(box_origin, box_dimensions)

        sdofiles = {'euv':
                        ['gxbox_data/20200101/aia.lev1_euv_12s.2020-01-01T000001Z.94.image_lev1.fits',
                         'gxbox_data/20200101/aia.lev1_euv_12s.2020-01-01T000008Z.131.image_lev1.fits',
                         'gxbox_data/20200101/aia.lev1_euv_12s.2020-01-01T000010Z.171.image_lev1.fits',
                         'gxbox_data/20200101/aia.lev1_euv_12s.2020-01-01T000006Z.193.image_lev1.fits',
                         'gxbox_data/20200101/aia.lev1_euv_12s.2019-12-31T235959Z.211.image_lev1.fits',
                         'gxbox_data/20200101/aia.lev1_euv_12s.2020-01-01T000007Z.304.image_lev1.fits',
                         'gxbox_data/20200101/aia.lev1_euv_12s.2020-01-01T000002Z.335.image_lev1.fits'],
                    'uv':['gxbox_data/20200101/aia.lev1_uv_24s.2019-12-31T235952Z.1600.image_lev1.fits'],
                    'hmi_b':
                        ['gxbox_data/20200101/hmi.b_720s.20200101_000000_TAI.inclination.fits',
                         'gxbox_data/20200101/hmi.b_720s.20200101_000000_TAI.azimuth.fits',
                         'gxbox_data/20200101/hmi.b_720s.20200101_000000_TAI.disambig.fits',
                         'gxbox_data/20200101/hmi.b_720s.20200101_000000_TAI.field.fits'],
                    'hmi_m':
                        ['gxbox_data/20200101/hmi.m_720s.20200101_000000_TAI.3.magnetogram.fits'],
                    'hmi_ic':
                        ['gxbox_data/20200101/hmi.ic_nolimbdark_720s.20200101_000000_TAI.3.continuum.fits']}

        # if not fitsfiles_exist:
        #     download_sdo_image_FITS(time)

    ## todo use the bottom edges of the simbox as the CEA map, reproject to los map and get the cutout.


    def make_dummy_map(self, ref_coord):
        instrument_data = np.nan * np.ones((50, 50))
        instrument_header = sunpy.map.make_fitswcs_header(instrument_data,
                                                          ref_coord,
                                                          scale=u.Quantity([10, 10]) * u.arcsec / u.pix)
        return sunpy.map.Map(instrument_data, instrument_header)

    def define_simbox(self, box_origin, box_dimensions):
        # Finally, we can define the edge coordinates of the box by first creating a coordinate to represent the origin. This is easily computed from our point that defined the orientation since this is the point at which the box is tangent to the solar surface.

        # In[11]:

        # Using that origin, we can compute the coordinates of all edges.

        self.simbox = Box(box_origin, box_dimensions)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection=self.instrument_map)
        self.instrument_map.plot(axes=ax)
        self.instrument_map.draw_grid(axes=ax, grid_spacing=5 * u.deg, color='k')
        for edge in self.simbox.bottom_edges:
            ax.plot_coord(edge, color='r', ls='-', marker='', lw=1.0)
        for edge in self.simbox.non_bottom_edges:
            ax.plot_coord(edge, color='r', ls='--', marker='', lw=0.5)
        ax.plot_coord(self.box_origin, color='r', marker='+')
        ax.plot_coord(self.hcc_orientation, mec='r', mfc='none', marker='o')
        ax.set_xlim(-20, 70)
        ax.set_ylim(-20, 70)
        self.axes = ax
        self.fig = fig

    def create_lines_of_sight(self):
        # The rest of the code for creating lines of sight goes here
        pass

    def visualize(self):
        # The rest of the code for visualization goes here
        pass


gxbox = GxBox(date, observer, hcc_orientation, box_dimensions)

#
# # Now that we have our map representing our fake observation, we can get the coordinates, in pixel space, of the center of each pixel. These will be our lines of sight.
#
# # In[6]:
#
#
# map_indices = sunpy.map.all_pixel_indices_from_map(instrument_map).value.astype(int)
# map_indices = map_indices.reshape((2,map_indices.shape[1]*map_indices.shape[2]))
#
#
# # We can then use the WCS of the map to find the associate world coordinate for each pixel coordinate. Note that by default, the "z" or *distance* coordinate in the HPC frame is assumed to lie on the solar surface. As such, for each LOS, we will add a z coordinate that spans from 99% to 101% of the observer radius. This gives us a reasonable range of distances that will intersect our simulation box.
#
# # In[7]:
#
#
# lines_of_sight = []
# distance = np.linspace(0.99, 1.01, 10000)*observer.radius
# for indices in map_indices.T:
#     coord = instrument_map.wcs.pixel_to_world(*indices)
#     lines_of_sight.append(SkyCoord(Tx=coord.Tx, Ty=coord.Ty, distance=distance, frame=coord.frame))
#
#
# # We can do a simple visualization of all of our LOS on top of our synthetic map
#
# # In[8]:
#
#
# fig = plt.figure()
# ax = fig.add_subplot(projection=instrument_map)
# instrument_map.plot(axes=ax)
# instrument_map.draw_grid(axes=ax, color='k')
# for los in lines_of_sight:
#     ax.plot_coord(los[0], color='C0', marker='.', ls='', markersize=1,)
#
#
# # ## Defining the Simulation Box
#
#
#
#
# # Let's overlay the simulation box on top of our fake map we created earlier to see if things look right.
#
# # In[13]:
#
#
# fig = plt.figure()
# ax = fig.add_subplot(projection=instrument_map)
# instrument_map.plot(axes=ax)
# instrument_map.draw_grid(axes=ax, color='k')
# for edge in edge_coords:
#     ax.plot_coord(edge, color='k', ls='-', marker='')
# ax.plot_coord(box_origin, color='r', marker='x')
# ax.plot_coord(hcc_orientation, color='b', marker='x')
#
#
# # ## Visualizing the Simulation Box from the Synthetic Observer
#
# # Let's combine all of these pieces by plotting the lines of sight and the simulation box on a single plot. We'll also overlay the pixel grid of our fake image. Note that even though our box coordinates and LOS coordinates are in different coordinate frames, they are all automatically converted to the projected frame of our synthetic observer when plotting if we use the `plot_coord` command.
#
# # In[14]:
#
#
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(projection=instrument_map)
# instrument_map.plot(axes=ax)
# instrument_map.draw_grid(axes=ax,grid_spacing=5*u.deg, color='k')
# ax.set_xlim(-20,70)
# ax.set_ylim(-20,70)
#
# ax.plot_coord(box_origin, marker='x', color='r', ls='', label='Simulation Box Center')
# ax.plot_coord(hcc_orientation, marker='+', color='r',  ls='', label='Simulation Box Bottom')
# for i,edge in enumerate(edge_coords):
#     ax.plot_coord(edge, color='k', ls='-', label='Simulation Box' if i==0 else None)
# for i,los in enumerate(lines_of_sight):
#     ax.plot_coord(los[0], color='C0', marker='.', label='LOS' if i==0 else None, markersize=1)
# # This plots the pixel grid
# xpix_edges = np.array(range(int(instrument_map.dimensions.x.value)+1))-0.5
# ypix_edges = np.array(range(int(instrument_map.dimensions.y.value)+1))-0.5
# ax.vlines(x=xpix_edges,
#           ymin=ypix_edges[0],
#           ymax=ypix_edges[-1],
#           color='k', ls='--', lw=.5, label='pixel grid')
# ax.hlines(y=ypix_edges,
#           xmin=xpix_edges[0],
#           xmax=xpix_edges[-1],
#           color='k', ls='--', lw=.5,)
# ax.legend(loc=2, frameon=False)
#
#
# # ## Where do the LOS Coordinates Intersect the Box?
#
# # Finally, we have all of the pieces of information we need to understand whether a given LOS intersects the simulation box. First, we define a function that takes in the edge coordinates of our box and a LOS coordinate and returns to us a boolean mask of where that LOS coordinate falls in the box.
#
# # In[15]:
#
#
# def is_coord_in_box(box_edges, coord):
#     box_edges = SkyCoord(box_edges)
#     coord_hcc = coord.transform_to(box_edges.frame)
#     in_x = np.logical_and(coord_hcc.x<box_edges.x.max(), coord_hcc.x>box_edges.x.min())
#     in_y = np.logical_and(coord_hcc.y<box_edges.y.max(), coord_hcc.y>box_edges.y.min())
#     in_z = np.logical_and(coord_hcc.z<box_edges.z.max(), coord_hcc.z>box_edges.z.min())
#     return np.all([in_x, in_y, in_z], axis=0)
#
#
# # Next we define another map, using a different orientation from our observer, so we can look at the intersection of the box and the many LOS from a different viewing angle.
#
# # In[16]:
#
#
# new_obs = SkyCoord(lon=25*u.deg, lat=0*u.deg, radius=1*u.AU, frame='heliographic_stonyhurst')
# earth_map = make_dummy_map(
#     hcc_orientation.transform_to(Helioprojective(observer=new_obs, obstime=date))
# )
#
#
# # Finally, we'll create another visualization that combines the simulation box with the lines of sight and additional highlighting that shows *where* the LOS intersect the box.
#
# # In[18]:
#
#
# fig = plt.figure(figsize=(10,10), layout='constrained')
# ax = fig.add_subplot(projection=earth_map)
# earth_map.plot(axes=ax)
# earth_map.draw_grid(axes=ax,grid_spacing=5*u.deg, color='k')
# ax.set_xlim(-20,70)
# ax.set_ylim(-20,70)
#
# for los in lines_of_sight:
#     ax.plot_coord(los, color='C0', marker='', ls='-', alpha=0.15)
# for los in lines_of_sight:
#     inside_box = is_coord_in_box(edge_coords, los)
#     if inside_box.any():
#         ax.plot_coord(los[inside_box], color='C1', marker='', ls='-')
# for edge in edge_coords:
#     ax.plot_coord(edge, color='k', ls='-')
# ax.plot_coord(box_origin, marker='x', color='r')
# ax.plot_coord(hcc_orientation, color='b', marker='x', ls='')
#
#
# # In[ ]:
