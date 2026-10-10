#!/usr/bin/env python
"""Compare Python SFQ FOV crop/pos/rotate vs IDL prepare_basemaps,/sfq.

Requires:
  - HMI FITS under ``--data-dir`` (default ``/Users/gelu/jsoc_cache/2024-05-11``)
  - IDL dump SAV from ``dump_idl_sfq.pro`` (default ``/tmp/sfq_parity_20240512/idl_sfq_dump.sav``)

Exits non-zero if crop/pre-SFQ vector parity regresses; algorithm (post-SFQ)
agreement is reported but does not fail the script by default. Inputs are
sanitized like IDL (NaN/sentinel→0) before SFQ so NaN pollution is not
misreported as algorithm disagreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import readsav

from pyampp.gxbox import gx_fov2box
from pyampp.gxbox.boxutils import load_sunpy_map_compat
from pyampp.sfq import sfq_disambig


def _angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def _finite_maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a-b| on finite pairs; NaN-vs-finite counts as infinite disagreement."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    if not np.any(both):
        return float("nan")
    md = float(np.max(np.abs(a[both] - b[both])))
    if np.any(~both & (np.isfinite(a) | np.isfinite(b))):
        return float("inf")
    return md


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/gelu/jsoc_cache/2024-05-11"),
        help="Directory with HMI field/inclination/azimuth FITS",
    )
    p.add_argument(
        "--idl-dump",
        type=Path,
        default=Path("/tmp/sfq_parity_20240512/idl_sfq_dump.sav"),
        help="SAV written by dump_idl_sfq.pro",
    )
    p.add_argument(
        "--fail-on-algorithm",
        action="store_true",
        help="Also fail if post-SFQ transverse-sign agreement is below threshold",
    )
    p.add_argument("--algorithm-min-agree", type=float, default=0.90)
    args = p.parse_args(argv)

    if not args.idl_dump.is_file():
        print(f"Missing IDL dump: {args.idl_dump}", file=sys.stderr)
        return 2

    field_path = args.data_dir / "hmi.B_720s.20240512_000000_TAI.field.fits"
    incl_path = args.data_dir / "hmi.B_720s.20240512_000000_TAI.inclination.fits"
    az_path = args.data_dir / "hmi.B_720s.20240512_000000_TAI.azimuth.fits"
    for path in (field_path, incl_path, az_path):
        if not path.is_file():
            print(f"Missing FITS: {path}", file=sys.stderr)
            return 2

    idl = readsav(str(args.idl_dump), python_dict=True)
    x0, x1 = (int(v) for v in idl["xrange"])
    y0, y1 = (int(v) for v in idl["yrange"])
    pos_idl = np.asarray(idl["pos"], dtype=float)
    rsun_idl = float(idl["rsun_arcsec"])

    map_field = load_sunpy_map_compat(field_path)
    map_incl = load_sunpy_map_compat(incl_path)
    map_az = load_sunpy_map_compat(az_path)

    # Crop using IDL pixel bounds (numpy [y, x] == readsav field_s layout).
    field = gx_fov2box._sfq_sanitize_hmi_array(
        np.asarray(map_field.data[y0 : y1 + 1, x0 : x1 + 1], dtype=float), silent=True
    )
    incl = gx_fov2box._sfq_sanitize_hmi_array(
        np.asarray(map_incl.data[y0 : y1 + 1, x0 : x1 + 1], dtype=float), silent=True
    )
    az = gx_fov2box._sfq_sanitize_hmi_array(
        np.asarray(map_az.data[y0 : y1 + 1, x0 : x1 + 1], dtype=float), silent=True
    )
    field_idl = gx_fov2box._sfq_sanitize_hmi_array(
        np.asarray(idl["field_s"], dtype=float), silent=True
    )
    az_before_idl = gx_fov2box._sfq_sanitize_hmi_array(
        np.asarray(idl["az_before"], dtype=float), silent=True
    )

    crop_field_maxdiff = _finite_maxdiff(field, field_idl)
    crop_az_maxdiff = _finite_maxdiff(az, az_before_idl)

    inc_rad = np.deg2rad(incl)
    az_rad = np.deg2rad(az)
    bz = field * np.cos(inc_rad)
    bx = field * np.sin(inc_rad) * np.sin(az_rad)
    by = -field * np.sin(inc_rad) * np.cos(az_rad)
    bx = np.rot90(bx, 2)
    by = np.rot90(by, 2)
    bz = np.rot90(bz, 2)

    bx0 = np.asarray(idl["bx0"], dtype=float)
    by0 = np.asarray(idl["by0"], dtype=float)
    bz0 = np.asarray(idl["bz0"], dtype=float)
    pre_bx = _finite_maxdiff(bx, bx0)
    pre_by = _finite_maxdiff(by, by0)
    pre_bz = _finite_maxdiff(bz, bz0)

    rsun_py = gx_fov2box._sfq_rsun_arcsec(map_field)

    bx_o, by_o = sfq_disambig(
        bx.copy(), by.copy(), bz.copy(), pos_idl, rsun_idl, mode=True, silent=True
    )
    bx_idl_post = np.asarray(idl["bx"], dtype=float)
    by_idl_post = np.asarray(idl["by"], dtype=float)
    # IDL dump stores post-unrotate vectors; recover rotated-frame IDL result.
    bx_rot_idl = np.rot90(bx_idl_post, 2)
    by_rot_idl = -np.rot90(by_idl_post, 2)
    agree = (bx_o * bx_rot_idl + by_o * by_rot_idl) > 0
    strong = (field * np.sin(np.deg2rad(incl))) ** 2 > 100.0**2
    strong_rot = np.rot90(strong, 2)
    agree_all = float(np.mean(agree))
    agree_strong = float(np.mean(agree[strong_rot])) if np.any(strong_rot) else float("nan")

    by_u = -np.rot90(by_o, 2)
    bx_u = np.rot90(bx_o, 2)
    az_py = np.rad2deg(np.arctan2(bx_u, by_u))
    az_idl = np.asarray(idl["az_after"], dtype=float)
    d_az = _angle_diff_deg(az_py, az_idl)
    az_med = float(np.nanmedian(np.abs(d_az)))
    az_frac5 = float(np.mean(np.abs(d_az) < 5.0))

    report = {
        "case": "2024-05-12 HPC[0,0] CEA 64^3 dx=1400km",
        "idl_xrange": [x0, x1],
        "idl_yrange": [y0, y1],
        "idl_pos": pos_idl.tolist(),
        "idl_rsun_arcsec": rsun_idl,
        "python_rsun_arcsec": rsun_py,
        "crop_field_maxdiff": crop_field_maxdiff,
        "crop_azimuth_maxdiff": crop_az_maxdiff,
        "pre_sfq_bx_maxdiff": pre_bx,
        "pre_sfq_by_maxdiff": pre_by,
        "pre_sfq_bz_maxdiff": pre_bz,
        "post_sfq_sign_agree_all": agree_all,
        "post_sfq_sign_agree_strong": agree_strong,
        "post_sfq_az_absdiff_median_deg": az_med,
        "post_sfq_az_frac_absdiff_lt_5deg": az_frac5,
    }
    print(json.dumps(report, indent=2))

    ok = (
        crop_field_maxdiff == 0.0
        and crop_az_maxdiff == 0.0
        and pre_bx < 1e-10
        and pre_by < 1e-10
        and pre_bz < 1e-10
    )
    if args.fail_on_algorithm and not (agree_strong >= args.algorithm_min_agree):
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
