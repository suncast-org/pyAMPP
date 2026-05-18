#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure repo root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import sunpy.map
from sunpy.sun import constants as sun_consts
from sunpy.map import all_coordinates_from_map
import h5py

from pyAMaFiL.mag_field_proc import MagFieldProcessor
from pyAMaFiL.mag_field_lin_fff import MagFieldLinFFF

from pyampp.gx_chromo.combo_model import combo_model


def cutout2box(_map, center_crd: SkyCoord, dx_km, shape):
    center_crd = center_crd.transform_to("heliographic_carrington")
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

    outmap = _map.reproject_to(box_header, algorithm="exact")
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
    sinphi, cosphi = np.sin(phi), np.cos(phi)
    sinlam, coslam = np.sin(lambda_), np.cos(lambda_)

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


def write_b3d_h5(filename: str, box_b3d: dict):
    with h5py.File(filename, "w") as hdf_file:
        for model_type, components in box_b3d.items():
            if components is None:
                continue
            if model_type == "metadata":
                group = hdf_file.create_group("metadata")
                for key, value in components.items():
                    if isinstance(value, str):
                        group.create_dataset(key, data=np.string_(value))
                    else:
                        group.create_dataset(key, data=value)
                continue
            if model_type == "refmaps":
                group = hdf_file.create_group(model_type, track_order=True)
            else:
                group = hdf_file.create_group(model_type)
            refmap_idx = 0
            for component, data in components.items():
                if component == "attrs":
                    group.attrs.update(data)
                else:
                    if isinstance(data, dict):
                        if model_type == "refmaps":
                            sub = group.create_group(component, track_order=True)
                            sub.attrs["order_index"] = np.int64(refmap_idx)
                            refmap_idx += 1
                        else:
                            sub = group.create_group(component)
                        for sub_key, sub_val in data.items():
                            sub.create_dataset(sub_key, data=sub_val)
                    elif model_type == "chromo" and component == "voxel_status":
                        group.create_dataset(component, data=np.asarray(data, dtype=np.uint8))
                    else:
                        group.create_dataset(component, data=data)


def build_coord_tag(lon_deg: float, lat_deg: float) -> str:
    lon = (lon_deg + 180.0) % 360.0 - 180.0
    lon_dir = "W" if lon >= 0 else "E"
    lat_dir = "N" if lat_deg >= 0 else "S"
    lon_val = f"{abs(int(round(lon))):02d}"
    lat_val = f"{abs(int(round(lat_deg))):02d}"
    return f"{lon_dir}{lon_val}{lat_dir}{lat_val}CR"


def make_out_dir(gxmodel_dir: str, obs_time: Time) -> Path:
    date_str = obs_time.to_datetime().strftime("%Y-%m-%d")
    out_dir = Path(gxmodel_dir) / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def stage_filename(out_dir: Path, obs_time: Time, coord_tag: str, stage: str) -> Path:
    time_tag = obs_time.to_datetime().strftime("%Y%m%d_%H%M%S")
    base = f"hmi.M_720s.{time_tag}.{coord_tag}.CEA"
    return out_dir / f"{base}.{stage}.h5"


def load_hmi_maps(data_dir: Path) -> Dict[str, sunpy.map.Map]:
    field_path = list(data_dir.glob("*.field.fits"))[0]
    incli_path = list(data_dir.glob("*.inclination.fits"))[0]
    azimu_path = list(data_dir.glob("*.azimuth.fits"))[0]
    disam_path = list(data_dir.glob("*.disambig.fits"))[0]
    conti_path = list(data_dir.glob("*.continuum.fits"))[0]
    losma_path = list(data_dir.glob("*.magnetogram.fits"))[0]

    map_field = sunpy.map.Map(field_path)
    map_inclination = sunpy.map.Map(incli_path)
    map_azimuth = sunpy.map.Map(azimu_path)
    map_disambig = sunpy.map.Map(disam_path)
    map_conti = sunpy.map.Map(conti_path)
    map_losma = sunpy.map.Map(losma_path)

    dis = map_disambig.data
    map_azimuth.data[:, :] = map_azimuth.data + dis * 180.0

    return {
        "field": map_field,
        "inclination": map_inclination,
        "azimuth": map_azimuth,
        "disambig": map_disambig,
        "continuum": map_conti,
        "magnetogram": map_losma,
    }


def build_header(map_field: sunpy.map.Map) -> Dict[str, Any]:
    header_field = map_field.wcs.to_header()
    field_frame = map_field.center.heliographic_carrington.frame
    lon, lat = field_frame.lon.value, field_frame.lat.value
    obs_time = Time(map_field.date)
    dsun_obs = header_field["DSUN_OBS"]
    return {"lon": lon, "lat": lat, "dsun_obs": dsun_obs, "obs_time": str(obs_time.iso)}


def make_gen_chromo(chromo_box: dict) -> dict:
    gen_keys = [
        "dr",
        "bcube",
        "start_idx",
        "end_idx",
        "av_field",
        "phys_length",
        "voxel_status",
        "apex_idx",
        "codes",
        "seed_idx",
    ]
    gen = {k: chromo_box[k] for k in gen_keys if k in chromo_box}
    if "attrs" in chromo_box:
        gen["attrs"] = chromo_box["attrs"]
    return gen


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AMPP stages headlessly and save POT/NAS/GEN/CHR HDF5 files.")
    parser.add_argument("--data-dir", required=True, help="Directory with HMI fits (*.field.fits, etc.).")
    parser.add_argument("--gxmodel-dir", required=True, help="Output gx_models directory.")
    parser.add_argument("--time", required=True, help="Observation time ISO (used for naming).")
    parser.add_argument("--coords", nargs=2, type=float, required=True, metavar=("X", "Y"),
                        help="Center coords: arcsec if --hpc, degrees if --hgc.")
    parser.add_argument("--hpc", action="store_true", help="Interpret coords as helioprojective (arcsec).")
    parser.add_argument("--hgc", action="store_true", help="Interpret coords as heliographic carrington (deg).")
    parser.add_argument("--box-dims", nargs=3, type=int, required=True, metavar=("NX", "NY", "NZ"),
                        help="Box dimensions in pixels.")
    parser.add_argument("--box-res", type=float, required=True,
                        help="Box resolution in Mm per pixel.")
    parser.add_argument("--coord-tag", default=None,
                        help="Override coordinate tag in filename, e.g., E73S3CR. If not set, derive from map header.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    maps = load_hmi_maps(data_dir)

    obs_time = Time(args.time)
    out_dir = make_out_dir(args.gxmodel_dir, obs_time)

    if args.hpc == args.hgc:
        raise SystemExit("Specify exactly one of --hpc or --hgc.")

    map_bp, map_bt, map_br = hmi_b2ptr(maps["field"], maps["inclination"], maps["azimuth"])

    nx, ny, nz = args.box_dims
    res = args.box_res * u.Mm
    res_km = res.to(u.km).value

    # Cutouts
    if args.hpc:
        center_crd = SkyCoord(args.coords[0] * u.arcsec, args.coords[1] * u.arcsec,
                              frame=maps["field"].coordinate_frame)
    else:
        center_crd = SkyCoord(lon=args.coords[0] * u.deg, lat=args.coords[1] * u.deg,
                              radius=maps["field"].center.heliographic_carrington.radius,
                              obstime=maps["field"].date,
                              observer="earth",
                              frame="heliographic_carrington")

    box_bx = cutout2box(map_bp, center_crd, res_km * u.km, [ny, nx])
    box_by = cutout2box(map_bt, center_crd, res_km * u.km, [ny, nx])
    box_bz = cutout2box(map_br, center_crd, res_km * u.km, [ny, nx])
    box_by.data[:, :] *= -1.0

    # Potential field
    maglib_lff = MagFieldLinFFF()
    maglib_lff.set_field(box_bz.data.T)
    pot_res = maglib_lff.LFFF_cube(nz=nz)

    pot_bx = pot_res["by"].swapaxes(0, 1)
    pot_by = pot_res["bx"].swapaxes(0, 1)
    pot_bz = pot_res["bz"].swapaxes(0, 1)
    obs_dr = res.to(u.km) / sun_consts.radius.to(u.km)
    dr3 = [obs_dr.value, obs_dr.value, obs_dr.value]
    pot_box = {"corona": {"bx": pot_bx, "by": pot_by, "bz": pot_bz, "dr": np.array(dr3),
                          "attrs": {"model_type": "pot"}}}

    header = build_header(maps["field"])
    coord_tag = args.coord_tag or build_coord_tag(header["lon"], header["lat"])

    write_b3d_h5(str(stage_filename(out_dir, obs_time, coord_tag, "POT")), pot_box)

    # NLFFF: replace bottom boundary in potential
    bx_lff = pot_bx.copy()
    by_lff = pot_by.copy()
    bz_lff = pot_bz.copy()
    bx_lff[:, :, 0] = box_bx.data
    by_lff[:, :, 0] = box_by.data
    bz_lff[:, :, 0] = box_bz.data

    bnd_box = {
        "corona": {
            "bx": bx_lff,
            "by": by_lff,
            "bz": bz_lff,
            "dr": np.array(dr3),
            "attrs": {"model_type": "bnd"},
        }
    }
    write_b3d_h5(str(stage_filename(out_dir, obs_time, coord_tag, "BND")), bnd_box)

    # MagFieldProcessor expects the original LFFF cube layout/component naming.
    bnd_res = {
        "bx": by_lff.swapaxes(0, 1),
        "by": bx_lff.swapaxes(0, 1),
        "bz": bz_lff.swapaxes(0, 1),
    }

    maglib = MagFieldProcessor()
    maglib.load_cube_vars(bnd_res)
    res_nlf = maglib.NLFFF()

    nlfff_bx = res_nlf["by"].swapaxes(0, 1)
    nlfff_by = res_nlf["bx"].swapaxes(0, 1)
    nlfff_bz = res_nlf["bz"].swapaxes(0, 1)
    nlfff_box = {"corona": {"bx": nlfff_bx, "by": nlfff_by, "bz": nlfff_bz, "dr": np.array(dr3),
                            "attrs": {"model_type": "nlfff"}}}
    write_b3d_h5(str(stage_filename(out_dir, obs_time, coord_tag, "NAS")), nlfff_box)

    # Field lines + chromosphere
    lines = maglib.lines(seeds=None)

    base_bz = cutout2box(maps["magnetogram"], center_crd, res_km * u.km, [ny, nx])
    base_ic = cutout2box(maps["continuum"], center_crd, res_km * u.km, [ny, nx])

    obs_dr = res.to(u.km) / sun_consts.radius.to(u.km)
    dr3 = [obs_dr.value, obs_dr.value, obs_dr.value]

    chromo_box = combo_model({"bx": nlfff_bx, "by": nlfff_by, "bz": nlfff_bz}, dr3, base_bz.data.T, base_ic.data.T)
    for k in ["codes", "apex_idx", "start_idx", "end_idx", "seed_idx",
              "av_field", "phys_length", "voxel_status"]:
        chromo_box[k] = lines[k]
    chromo_box["phys_length"] *= dr3[0]
    chromo_box["attrs"] = header

    gen_box = {"chromo": make_gen_chromo(chromo_box)}
    write_b3d_h5(str(stage_filename(out_dir, obs_time, coord_tag, "NAS.GEN")), gen_box)

    chr_box = {"chromo": chromo_box}
    write_b3d_h5(str(stage_filename(out_dir, obs_time, coord_tag, "NAS.CHR")), chr_box)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
