#!/usr/bin/env python
# coding: utf-8

import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from astropy.time import Time
import sunpy.map

from pathlib import Path
import os, sys
import numpy as np
import datetime

import locale
from sunpy.map import all_coordinates_from_map
import h5py

from pyampp.util.lff import mf_lfff
# from .gx_chromo.combo_model import combo_model
from pyAMaFiL.mag_field_wrapper import MagFieldWrapper

maglib = MagFieldWrapper()

os.environ['OMP_NUM_THREADS'] = '16'  # number of parallel threads
locale.setlocale(locale.LC_ALL, "C");


def cutout2box(_map, center_x, center_y, dx_km, shape):
    hmi_wcs = _map.wcs
    center_crd = crd = SkyCoord(center_x, center_y, unit=u.arcsec, frame=_map.coordinate_frame) \
        .transform_to("heliographic_carrington")
    lon = center_crd.lon
    lat = center_crd.lat
    rad = center_crd.radius
    origin = SkyCoord(lon, lat, rad,
                      frame="heliographic_carrington",
                      observer="self",
                      obstime=_map.date)

    scale = np.arcsin(dx_km / origin.radius).to(u.deg) / u.pix
    scale = u.Quantity((scale, scale))
    box_header = sunpy.map.make_fitswcs_header(shape, origin,
                                               projection_code='CEA', scale=scale)

    outmap = _map.reproject_to(box_header, algorithm="adaptive", roundtrip_coords=False)
    return outmap


def hmi_b2ptr(map_field, map_inclination, map_azimuth):
    sz = map_field.data.shape
    ny, nx = sz

    field = map_field.data
    gamma = np.deg2rad(map_inclination.data)
    psi = np.deg2rad(map_azimuth.data)

    b_xi = -field * np.sin(gamma) * np.sin(psi)
    b_eta = field * np.sin(gamma) * np.cos(psi)
    b_zeta = field * np.cos(gamma)

    foo = all_coordinates_from_map(map_field).transform_to("heliographic_stonyhurst")
    phi = foo.lon
    lambda_ = foo.lat

    b = np.deg2rad(map_field.fits_header["crlt_obs"])
    p = np.deg2rad(-map_field.fits_header["crota2"])

    phi, lambda_ = np.deg2rad(phi), np.deg2rad(lambda_)

    sinb, cosb = np.sin(b), np.cos(b)
    sinp, cosp = np.sin(p), np.cos(p)
    sinphi, cosphi = np.sin(phi), np.cos(phi)  # nx*ny
    sinlam, coslam = np.sin(lambda_), np.cos(lambda_)  # nx*ny

    k11 = coslam * (sinb * sinp * cosphi + cosp * sinphi) - sinlam * cosb * sinp
    k12 = - coslam * (sinb * cosp * cosphi - sinp * sinphi) + sinlam * cosb * cosp
    k13 = coslam * cosb * cosphi + sinlam * sinb
    k21 = sinlam * (sinb * sinp * cosphi + cosp * sinphi) + coslam * cosb * sinp
    k22 = - sinlam * (sinb * cosp * cosphi - sinp * sinphi) - coslam * cosb * cosp
    k23 = sinlam * cosb * cosphi - coslam * sinb
    k31 = - sinb * sinp * sinphi + cosp * cosphi
    k32 = sinb * cosp * sinphi + sinp * cosphi
    k33 = - cosb * sinphi

    bptr = np.zeros((3, nx, ny))

    bptr[0, :, :] = k31 * b_xi + k32 * b_eta + k33 * b_zeta
    bptr[1, :, :] = k21 * b_xi + k22 * b_eta + k23 * b_zeta
    bptr[2, :, :] = k11 * b_xi + k12 * b_eta + k13 * b_zeta

    header = map_field.fits_header
    map_bp = sunpy.map.Map(bptr[0, :, :], header)
    map_bt = sunpy.map.Map(bptr[1, :, :], header)
    map_br = sunpy.map.Map(bptr[2, :, :], header)

    return map_bp, map_bt, map_br


def ampp_field(dl_path, out_model, x, y, dx, dy, dz, res):
    """Creates a model of coronal magnetic fields, including potential, nlfff and thermal corona model

    Args:
        dl_path (str): Path to the folder where downloaded files (HMI magnetograms) are stored
        out_model (str): Path to output HDF5 model file (.h5)
        x (float): X coordinate of active region center (arcsec)
        y (float): Y coordinate of active region center (arcsec)
        dx (int): number of voxels in X direction
        dy (int): number of voxels in Y direction
        dz (int): number of voxels in Z (vertical, to corona) direction
        res (dimensional astropy Quantity)

    Returns:
        None
    """

    input_path = Path(os.path.expanduser(dl_path)).resolve()

    if not input_path.exists():
        print("no input data")

    field_path = list(input_path.glob("*.field.fits"))[0]
    incli_path = list(input_path.glob("*.inclination.fits"))[0]
    azimu_path = list(input_path.glob("*.azimuth.fits"))[0]
    disam_path = list(input_path.glob("*.disambig.fits"))[0]
    conti_path = list(input_path.glob("*.continuum.fits"))[0]
    losma_path = list(input_path.glob("*.magnetogram.fits"))[0]
    # size_pix = f"[{dx}, {dy}, {dz}]"
    # centre = f"[{x}, {y}]"
    # wcs_rsun = 6.96e8
    res_km = res.to(u.km).value

    map_field = sunpy.map.Map(field_path)
    map_inclination = sunpy.map.Map(incli_path)
    map_azimuth = sunpy.map.Map(azimu_path)
    map_disambig = sunpy.map.Map(disam_path)

    map_conti = sunpy.map.Map(conti_path)
    map_losma = sunpy.map.Map(losma_path)

    dis = map_disambig.data
    map_azimuth.data[:, :] = map_azimuth.data + dis * 180.

    map_bp, map_bt, map_br = hmi_b2ptr(map_field, map_inclination, map_azimuth)
    box_bx = cutout2box(map_bp, x, y, res_km * u.km, [dy, dx])
    box_by = cutout2box(map_bt, x, y, res_km * u.km, [dy, dx])
    box_bz = cutout2box(map_br, x, y, res_km * u.km, [dy, dx])
    box_by.data[:, :] *= -1

    # earth_observer = SkyCoord(0 * u.deg, 0 * u.deg, 0 * u.km, frame=frames.GeocentricEarthEquatorial, observer="earth",
    #                           obstime=box_bx.date)

    print("opening file")
    out_file = h5py.File(out_model, "w")
    bottom = out_file.create_group("bottom_bounds")
    for name, array in zip(("bx", "by", "bz"), (box_bx.data, box_by.data, box_bz.data)):
        bottom.create_dataset(name, data=array)
    out_file.flush()

    maglib_lff = mf_lfff()
    maglib_lff.set_field(box_bz.data.T)

    res1 = maglib_lff.lfff_cube(dz)

    bx_lff, by_lff, bz_lff = [res1[k].transpose((2, 1, 0)) for k in ("bx", "by", "bz")]

    bx_lff[0, :, :] = box_bx.data  # replace bottom boundary of lff solution with initial boundary conditions
    by_lff[0, :, :] = box_by.data
    bz_lff[0, :, :] = box_bz.data

    potential = out_file.create_group("potential")
    for name, array in zip(("bx", "by", "bz"), (bx_lff, by_lff, bz_lff)):
        potential.create_dataset(name, data=array)
    out_file.flush()

    obs_dr = res.to(u.km) / (696000 * u.km)  # dimensionless
    potential.attrs["obs_dr"] = obs_dr.value
    potential.attrs["res_km"] = res_km

    maglib.load_cube_vars(bx_lff, by_lff, bz_lff, (obs_dr * sunpy.sun.constants.radius.to(u.cm)).value)
    box = maglib.NLFFF()
    energy_new = maglib.energy
    print('NLFFF energy:     ' + str(energy_new) + ' erg')

    nlfff = out_file.create_group("nlfff")
    for name, array in zip(("bx", "by", "bz"), (box["bx"], box["by"], box["bz"])):
        nlfff.create_dataset(name, data=array)
    nlfff.attrs["energy_erg"] = energy_new
    out_file.flush()

    print("Calculating field lines")
    # lines = maglib.lines(seeds=None)

    base_bz = cutout2box(map_losma, x, y, res_km * u.km, [dy, dx])
    base_ic = cutout2box(map_conti, x, y, res_km * u.km, [dy, dx])
    bottom.create_dataset("base_bz", data=base_bz.data)
    bottom.create_dataset("base_ic", data=base_ic.data)

    header_field = map_field.wcs.to_header()
    field_frame = box_bx.center.heliographic_carrington.frame
    lon, lat = field_frame.lon.value, field_frame.lat.value

    obs_time = Time(map_field.date)
    dsun_obs = header_field["DSUN_OBS"]

    header = {"lon": lon, "lat": lat, "dsun_obs": dsun_obs, "obs_time": str(obs_time.iso)}
    bottom.attrs.update(header)

    out_file.flush()

    # dr3 = [obs_dr.value, obs_dr.value, obs_dr.value]

    # chromo_box = combo_model(box, dr3, base_bz.data.T, base_ic.data.T)
    #
    # chromo_box["avfield"] = lines["av_field"].transpose((1, 2, 0))
    # chromo_box["physlength"] = lines["phys_length"].transpose((1, 2, 0)) * dr3[0]
    # chromo_box["status"] = lines["voxel_status"].transpose((1, 2, 0))
    # out_file.flush()
    #
    # chromo_ds = out_file.create_group("chromo")
    # for k in chromo_box.keys():
    #     chromo_ds.create_dataset(k, data=chromo_box[k])
    # out_file.flush()
    out_file.close()

