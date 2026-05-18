#!/usr/bin/env python3
"""Export base-layer maps from a pyAMPP NONE H5 file to a FITS file readable in IDL.

Output: multi-extension FITS (MEF).
  HDU 0  – empty primary with run metadata
  HDU 1  – BX        (base/bx,  float64,  Gauss)
  HDU 2  – BY        (base/by,  float64,  Gauss)
  HDU 3  – BZ        (base/bz,  float64,  Gauss)
  HDU 4  – IC        (base/ic,  float64,  continuum intensity, normalised)
  HDU 5  – CHROMO_MASK  (base/chromo_mask, int32)

WCS is taken from the stored base/index header bytes when present;
otherwise a minimal fallback WCS is written from the geometry contract.

IDL reading:
  bx = mrdfits('file.fits', 1, hdr)
  by = mrdfits('file.fits', 2)
  bz = mrdfits('file.fits', 3)
  ic = mrdfits('file.fits', 4)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


_COMPONENTS = (
    ("bx", "BX", "Gauss", "float64"),
    ("by", "BY", "Gauss", "float64"),
    ("bz", "BZ", "Gauss", "float64"),
    ("ic", "IC", "normalized", "float64"),
    ("chromo_mask", "CHROMO_MASK", "", "int32"),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export pyAMPP NONE base maps to a FITS MEF readable in IDL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--h5-path", type=Path, required=True,
                   help="Path to the NONE stage HDF5 file")
    p.add_argument("--out-fits", type=Path, default=None,
                   help="Output FITS path. Default: <h5_stem>_base_maps.fits next to the H5 file")
    return p.parse_args()


def _decode_index_header(raw) -> fits.Header | None:
    """Try to parse the stored base/index blob as a FITS header."""
    try:
        blob = bytes(np.asarray(raw))
        hdr = fits.Header.fromstring(blob.decode("ascii", "replace"))
        return hdr
    except Exception:
        return None


def _fallback_wcs_from_contract(h5: h5py.File) -> fits.Header:
    hdr = fits.Header()
    try:
        gc = h5["metadata"]["geometry_contract"]
        naxis1 = int(h5["base"]["bx"].shape[1])
        naxis2 = int(h5["base"]["bx"].shape[0])

        def _scalar(name: str) -> float:
            v = gc[name][()]
            return float(np.asarray(v).flat[0])

        crpix1 = naxis1 / 2.0 + 0.5
        crpix2 = naxis2 / 2.0 + 0.5
        dx_as = _scalar("dx_arcsec") if "dx_arcsec" in gc else 1.0
        dy_as = _scalar("dy_arcsec") if "dy_arcsec" in gc else 1.0
        crval1 = _scalar("xc_arcsec") if "xc_arcsec" in gc else 0.0
        crval2 = _scalar("yc_arcsec") if "yc_arcsec" in gc else 0.0

        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = naxis1
        hdr["NAXIS2"] = naxis2
        hdr["CTYPE1"] = "HPLN-TAN"
        hdr["CTYPE2"] = "HPLT-TAN"
        hdr["CRPIX1"] = crpix1
        hdr["CRPIX2"] = crpix2
        hdr["CRVAL1"] = crval1
        hdr["CRVAL2"] = crval2
        hdr["CDELT1"] = dx_as
        hdr["CDELT2"] = dy_as
        hdr["CUNIT1"] = "arcsec"
        hdr["CUNIT2"] = "arcsec"
    except Exception:
        pass
    return hdr


def main() -> int:
    args = _parse_args()
    h5_path = args.h5_path.expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    out_fits = (
        args.out_fits.expanduser().resolve()
        if args.out_fits
        else h5_path.with_name(h5_path.stem + "_base_maps.fits")
    )
    out_fits.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        base = f["base"]

        # Try to recover WCS from stored FITS index header bytes
        wcs_hdr: fits.Header | None = None
        if "index" in base:
            wcs_hdr = _decode_index_header(base["index"][()])
        if wcs_hdr is None:
            wcs_hdr = _fallback_wcs_from_contract(f)

        # Primary HDU – no data, just provenance
        primary_hdr = fits.Header()
        primary_hdr["ORIGIN"] = ("pyAMPP", "exported by export_none_base_maps_to_fits.py")
        primary_hdr["H5SRC"] = (str(h5_path), "source HDF5 file")
        primary_hdr["STAGE"] = ("NONE", "pyAMPP pipeline stage")
        primary_hdr["COMMENT"] = "Extensions: 1=BX 2=BY 3=BZ 4=IC 5=CHROMO_MASK"
        primary_hdr["COMMENT"] = "IDL: bx = mrdfits(file, 1, hdr)"
        hdus = [fits.PrimaryHDU(header=primary_hdr)]

        for h5_key, extname, bunit, dtype in _COMPONENTS:
            if h5_key not in base:
                continue
            arr = np.asarray(base[h5_key], dtype=dtype)
            hdr = wcs_hdr.copy()
            hdr["EXTNAME"] = extname
            if bunit:
                hdr["BUNIT"] = bunit
            hdus.append(fits.ImageHDU(data=arr, header=hdr, name=extname))

    hdul = fits.HDUList(hdus)
    hdul.writeto(str(out_fits), overwrite=True)
    print(f"Wrote: {out_fits}")
    for i, hdu in enumerate(hdul[1:], start=1):
        print(f"  HDU {i}: {hdu.name:14s}  shape={hdu.data.shape}  dtype={hdu.data.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
