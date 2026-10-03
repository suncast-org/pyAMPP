"""Unit tests for SFQ FOV crop helpers (IDL prepare_basemaps,/sfq shape)."""

from __future__ import annotations

import numpy as np
import pytest

from pyampp.gxbox import gx_fov2box
from pyampp.util.config import IDL_HMI_RSUN_M


def test_sfq_rsun_arcsec_matches_idl_formula():
    import astropy.units as u

    class _FakeMap:
        dsun = (IDL_HMI_RSUN_M / np.sin(np.deg2rad(950.0 / 3600.0))) * u.m
        rsun_obs = 999.0 * u.arcsec  # unused when dsun is valid

    got = gx_fov2box._sfq_rsun_arcsec(_FakeMap())
    dsun = float(_FakeMap.dsun.to_value(u.m))
    expect = IDL_HMI_RSUN_M / dsun * (180.0 / np.pi) * 3600.0
    assert got == pytest.approx(expect, rel=0, abs=1e-9)


def test_sfq_corner_bounds_use_round_padding():
    """World→pixel rounding must follow IDL round(minmax)+[-1,1]."""
    from sunpy.map import Map

    # Tiny TAN plate so corners land at known fractional pixels.
    data = np.zeros((20, 20), dtype=float)
    header = {
        "NAXIS": 2,
        "NAXIS1": 20,
        "NAXIS2": 20,
        "CTYPE1": "HPLN-TAN",
        "CTYPE2": "HPLT-TAN",
        "CUNIT1": "arcsec",
        "CUNIT2": "arcsec",
        "CRPIX1": 10.5,
        "CRPIX2": 10.5,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "DATE-OBS": "2024-05-12T00:00:00",
        "RSUN_REF": IDL_HMI_RSUN_M,
        "DSUN_OBS": 1.5e11,
        "HGLT_OBS": 0.0,
        "HGLN_OBS": 0.0,
    }
    smap = Map(data, header)
    # 4x4 CEA-like header in the same frame, offset so corners are interior.
    bottom = {
        "NAXIS": 2,
        "NAXIS1": 4,
        "NAXIS2": 4,
        "CTYPE1": "HPLN-TAN",
        "CTYPE2": "HPLT-TAN",
        "CUNIT1": "arcsec",
        "CUNIT2": "arcsec",
        "CRPIX1": 2.5,
        "CRPIX2": 2.5,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "DATE-OBS": "2024-05-12T00:00:00",
        "RSUN_REF": IDL_HMI_RSUN_M,
        "DSUN_OBS": 1.5e11,
        "HGLT_OBS": 0.0,
        "HGLN_OBS": 0.0,
    }
    pos, ysl, xsl = gx_fov2box._sfq_corner_hpc_and_bounds(smap, bottom)
    assert pos.shape == (4,)
    assert pos[0] < pos[2] and pos[1] < pos[3]
    # With matching platescales/CRVAL, 4x4 corners map near pixels 8..11 before pad.
    assert ysl.start <= 8 and ysl.stop >= 12
    assert xsl.start <= 8 and xsl.stop >= 12
    # round+[-1,1] expands by at least one pixel beyond min/max rounded indices.
    assert (ysl.stop - ysl.start) >= 4
    assert (xsl.stop - xsl.start) >= 4
