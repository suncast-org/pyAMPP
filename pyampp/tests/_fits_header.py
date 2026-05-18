from __future__ import annotations

from astropy.io import fits


def canonical_base_index_header(
    *,
    crval1: float = 10.0,
    crval2: float = -5.0,
    rsun_ref: float = 695700000.0,
    date_obs: str = "2020-11-26T19:58:31",
) -> str:
    header = fits.Header()
    header["CRVAL1"] = float(crval1)
    header["CRVAL2"] = float(crval2)
    header["RSUN_REF"] = float(rsun_ref)
    header["DATE-OBS"] = date_obs
    return header.tostring(sep="\n", endcard=True)