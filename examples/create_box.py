#!/usr/bin/env python3

import astropy.time
import sunpy.sun.constants
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Heliocentric, Helioprojective,get_earth
import astropy.units as u

from PyQt5.QtWidgets import QApplication
app = QApplication([])
from pyampp.gxbox.gxbox_factory import GxBox

time = astropy.time.Time('2014-11-01T16:40:00')
# box_origin = SkyCoord(lon=30 * u.deg, lat=20 * u.deg,
#                       radius=sunpy.sun.constants.radius,
#                       frame='heliographic_carrington')
## dots source
# box_origin = SkyCoord(-475 * u.arcsec, -330 * u.arcsec, obstime=time, observer="earth", frame='helioprojective')
## flare AR
box_origin = SkyCoord(-632 * u.arcsec, -135 * u.arcsec, obstime=time, observer="earth", frame='helioprojective')
# box_origin = SkyCoord(-750 * u.arcsec, -400 * u.arcsec, obstime=time, observer="earth", frame='helioprojective')
observer = get_earth(time)
box_dimensions = u.Quantity([150, 150, 100]) * u.Mm
box_res = 1.4 * u.Mm


gxbox = GxBox(time, observer, box_origin, box_dimensions)
gxbox.show()
app.exec_()

# ## process hmi map
# from pyampp.util.config import HMI_B_SEGMENTS
# from sunpy.map import Map
# from pyampp.gxbox.boxutils import hmi_disambig, hmi_b2ptr
# map_azimuth = hmi_disambig(gxbox.sdofitsfiles['hmi_b']['azimuth'], gxbox.sdofitsfiles['hmi_b']['disambig'])
# map_inclination = Map(gxbox.sdofitsfiles['hmi_b']['inclination'])
# map_field = Map(gxbox.sdofitsfiles['hmi_b']['field'])
#
# map_bp, map_bt, map_br = hmi_b2ptr(map_field, map_inclination, map_azimuth)



# box_origin_hs = box_origin.transform_to('heliographic_carrington')
# new_observer = SkyCoord(box_origin_hs.lon, box_origin_hs.lat, box_origin.distance.value*u.AU, obstime=time,
#                         frame='heliographic_carrington')
#
# out_shape = aia_map.data.shape
#
# out_ref_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=new_observer.obstime,
#                          frame='helioprojective', observer=new_observer,
#                          rsun=aia_map.coordinate_frame.rsun)
# out_header = sunpy.map.make_fitswcs_header(
#     out_shape,
#     out_ref_coord,
#     scale=u.Quantity(aia_map.scale),
#     instrument=aia_map.instrument,
#     wavelength=aia_map.wavelength
# )
#
# with Helioprojective.assume_spherical_screen(gxbox.map_plt.observer_coordinate):
#     outmap_screen_all = gxbox.map_plt.reproject_to(out_header)
#
# fig = plt.figure()
# ax = fig.add_subplot(projection=outmap_screen_all)
# outmap_screen_all.plot(axes=ax)
# plt.show()
