from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits
from astropy.time import Time

from pyampp.gxbox import gx_fov2box
from pyampp.gxbox.boxutils import load_sunpy_map_compat, read_b3d_h5, write_b3d_h5
from pyampp.io import load_model
from pyampp.util.config import IDL_HMI_RSUN_M
from pyampp.io._sav_convert import build_h5_from_sav


class _FakeMap:
    def __init__(self, date: str):
        self.date = Time(date)


class _FakeSourceMap(_FakeMap):
    def __init__(self, date: str):
        super().__init__(date)
        self.observer_coordinate = gx_fov2box.get_earth(self.date)
        self.dsun = self.observer_coordinate.radius.to(u.m)
        self.rsun_meters = 6.96e8 * u.m
        self.meta = {"telescop": "SDO", "instrume": "HMI"}


class _FakeWCS:
    def to_header(self):
        header = fits.Header()
        header["CTYPE1"] = "HPLN-TAN"
        header["CTYPE2"] = "HPLT-TAN"
        return header


class _FakeRefMap:
    def __init__(self, date: str):
        self.date = Time(date)
        self.wcs = _FakeWCS()
        self.rsun_obs = 972.3 * u.arcsec
        self.rsun_meters = 6.96e8 * u.m
        self.observer_coordinate = None


class _FakeDownloader:
    calls: list[dict[str, object]] = []

    def __init__(self, time, uv=True, euv=True, hmi=True, data_dir=None, backend="drms", force_download=False, poll_seconds=5):
        self.time = Time(time)
        self.uv = uv
        self.euv = euv
        self.hmi = hmi
        self.existence_report = {
            "hmi_b": {seg: True for seg in ("field", "inclination", "azimuth", "disambig")},
            "hmi_m": {"magnetogram": True},
            "hmi_ic": {"continuum": True},
            "euv": {pb: True for pb in gx_fov2box.AIA_EUV_PASSBANDS},
            "uv": {pb: True for pb in gx_fov2box.AIA_UV_PASSBANDS},
        }
        type(self).calls.append(
            {
                "time": self.time.isot,
                "uv": uv,
                "euv": euv,
                "hmi": hmi,
                "backend": backend,
                "force_download": force_download,
            }
        )

    def download_images(self):
        if self.hmi:
            return {
                "field": "field.fits",
                "inclination": "inclination.fits",
                "azimuth": "azimuth.fits",
                "disambig": "disambig.fits",
                "continuum": "continuum.fits",
                "magnetogram": "magnetogram.fits",
            }
        files = {}
        if self.euv:
            for pb in gx_fov2box.AIA_EUV_PASSBANDS:
                files[str(pb)] = f"aia_{pb}.fits"
        if self.uv:
            for pb in gx_fov2box.AIA_UV_PASSBANDS:
                files[str(pb)] = f"aia_{pb}.fits"
        return files


def _fake_map_loader(path):
    if path == "field.fits":
        return _FakeMap("2025-11-26T15:34:31.400")
    return _FakeMap("2025-11-26T15:34:31.400")


def test_load_hmi_maps_anchors_context_downloads_to_resolved_hmi_time():
    _FakeDownloader.calls.clear()
    requested = Time("2025-11-26T15:47:52")
    with patch.object(gx_fov2box, "SDOImageDownloader", _FakeDownloader), patch.object(
        gx_fov2box, "load_sunpy_map_compat", side_effect=_fake_map_loader
    ), patch.object(gx_fov2box, "hmi_disambig", side_effect=lambda azimuth, _disambig, method=2: azimuth):
        maps, info = gx_fov2box._load_hmi_maps_from_downloader(
            requested,
            Path("/tmp"),
            euv=True,
            uv=True,
            download_backend="drms",
            force_download=False,
        )

    assert len(_FakeDownloader.calls) == 2
    assert _FakeDownloader.calls[0]["time"] == requested.isot
    assert _FakeDownloader.calls[0]["hmi"] is True
    assert _FakeDownloader.calls[0]["euv"] is False
    assert _FakeDownloader.calls[0]["uv"] is False

    assert _FakeDownloader.calls[1]["time"] == "2025-11-26T15:34:31.400"
    assert _FakeDownloader.calls[1]["hmi"] is False
    assert _FakeDownloader.calls[1]["euv"] is True
    assert _FakeDownloader.calls[1]["uv"] is True

    assert info["requested_obs_time"] == requested.isot
    assert info["resolved_obs_time"] == "2025-11-26T15:34:31.400"
    assert maps["field"].date.isot == "2025-11-26T15:34:31.400"
    assert "AIA_94" in maps
    assert "AIA_1700" in maps


def test_refmap_wcs_header_preserves_date_obs():
    header_text = gx_fov2box._refmap_wcs_header(_FakeRefMap("2025-11-26T15:34:31.400"))
    header = fits.Header.fromstring(header_text, sep="\n")
    assert header["DATE-OBS"] == "2025-11-26T15:34:31.400"
    assert header["DATE_OBS"] == "2025-11-26T15:34:31.400"
    assert header["RSUN_OBS"] == 972.3
    assert header["RSUN_REF"] == 6.96e8


def test_build_index_header_converts_hgs_cea_reference_point_to_carrington():
    source_map = _FakeSourceMap("2025-11-26T15:34:31.400")
    bottom_header = fits.Header()
    bottom_header["SIMPLE"] = 1
    bottom_header["BITPIX"] = 8
    bottom_header["NAXIS"] = 2
    bottom_header["NAXIS1"] = 150
    bottom_header["NAXIS2"] = 100
    bottom_header["CTYPE1"] = "HGLN-CEA"
    bottom_header["CTYPE2"] = "HGLT-CEA"
    bottom_header["CUNIT1"] = "deg"
    bottom_header["CUNIT2"] = "deg"
    bottom_header["CRPIX1"] = 75.5
    bottom_header["CRPIX2"] = 50.5
    bottom_header["CDELT1"] = 0.115250131204
    bottom_header["CDELT2"] = 0.115250131204
    bottom_header["CRVAL1"] = -17.05992944118293
    bottom_header["CRVAL2"] = -12.247129818437514

    header_text = gx_fov2box._build_index_header(bottom_header, source_map)
    header = fits.Header.fromstring(header_text, sep="\n")

    expected = gx_fov2box.SkyCoord(
        lon=bottom_header["CRVAL1"] * u.deg,
        lat=bottom_header["CRVAL2"] * u.deg,
        radius=source_map.rsun_meters,
        frame=gx_fov2box.HeliographicStonyhurst(obstime=source_map.date),
    ).transform_to(
        gx_fov2box.HeliographicCarrington(observer=source_map.observer_coordinate, obstime=source_map.date)
    )

    assert header["CTYPE1"] == "CRLN-CEA"
    assert header["CTYPE2"] == "CRLT-CEA"
    assert np.isclose(header["CRVAL1"], expected.lon.to_value(u.deg))
    assert np.isclose(header["CRVAL2"], expected.lat.to_value(u.deg))
    assert not np.isclose(header["CRVAL1"], bottom_header["CRVAL1"])


def test_load_sunpy_map_compat_normalizes_rsun_ref_from_header():
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 2
    header["NAXIS2"] = 2
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    header["DATE-OBS"] = "2025-11-26T15:34:31.400"
    header["DSUN_OBS"] = 1.476e11
    header["HGLN_OBS"] = 0.0
    header["HGLT_OBS"] = 1.44
    header["RSUN_REF"] = 6.957e8

    smap = load_sunpy_map_compat(np.zeros((2, 2), dtype=float), header=header)

    assert float(smap.meta["rsun_ref"]) == float(IDL_HMI_RSUN_M)
    assert np.isclose(smap.rsun_meters.to_value(u.m), IDL_HMI_RSUN_M)


def _fake_sav_box_with_refmaps():
    base_and_index = _fake_sav_box_with_base_and_index()
    rec_dtype = [
        ("ID", object),
        ("DATA", object),
        ("XC", object),
        ("YC", object),
        ("DX", object),
        ("DY", object),
        ("TIME", object),
        ("XUNITS", object),
        ("YUNITS", object),
        ("RSUN", object),
        ("B0", object),
        ("L0", object),
    ]
    rec = np.empty(1, dtype=rec_dtype)
    rec["ID"][0] = b"AIA_171"
    rec["DATA"][0] = np.arange(6, dtype=np.float32).reshape(2, 3)
    rec["XC"][0] = 10.0
    rec["YC"][0] = -20.0
    rec["DX"][0] = 0.6
    rec["DY"][0] = 0.6
    rec["TIME"][0] = b"2025-11-26T15:34:33.350"
    rec["XUNITS"][0] = b"arcsec"
    rec["YUNITS"][0] = b"arcsec"
    rec["RSUN"][0] = 972.5
    rec["B0"][0] = 1.44
    rec["L0"][0] = 44.92

    ids = np.empty(1, dtype=object)
    ids[0] = b"slot0"
    ptrs = np.empty(1, dtype=object)
    ptrs[0] = rec

    pointer = np.empty(1, dtype=[("IDS", object), ("PTRS", object)])
    pointer["IDS"][0] = ids
    pointer["PTRS"][0] = ptrs

    omap = np.empty(1, dtype=[("POINTER", object)])
    omap["POINTER"][0] = pointer

    refmaps = np.empty(1, dtype=[("OMAP", object)])
    refmaps["OMAP"][0] = omap

    box = np.empty(1, dtype=[("BASE", object), ("INDEX", object), ("REFMAPS", object), ("ID", object), ("EXECUTE", object)])
    box["BASE"][0] = base_and_index["BASE"][0]
    box["INDEX"][0] = base_and_index["INDEX"][0]
    box["REFMAPS"][0] = refmaps
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NAS.CHR"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def _fake_sav_box_with_base_and_index():
    base = np.empty(
        1,
        dtype=[
            ("BX", np.float64, (2, 3)),
            ("BY", np.float64, (2, 3)),
            ("BZ", np.float64, (2, 3)),
            ("IC", np.float64, (2, 3)),
            ("CHROMO_MASK", np.int16, (2, 3)),
        ],
    )
    base["BX"][0] = np.arange(6, dtype=np.float64).reshape(2, 3)
    base["BY"][0] = np.arange(6, dtype=np.float64).reshape(2, 3) + 10.0
    base["BZ"][0] = np.arange(6, dtype=np.float64).reshape(2, 3) + 20.0
    base["IC"][0] = 1.0
    base["CHROMO_MASK"][0] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)

    index = np.empty(
        1,
        dtype=[
            ("SIMPLE", np.int16),
            ("BITPIX", np.int32),
            ("NAXIS", np.int32),
            ("NAXIS1", np.int32),
            ("NAXIS2", np.int32),
            ("WCSNAME", "S32"),
            ("CRPIX1", np.float64),
            ("CRVAL1", np.float64),
            ("CTYPE1", "S16"),
            ("CUNIT1", "S8"),
            ("CDELT1", np.float64),
            ("CRPIX2", np.float64),
            ("CRVAL2", np.float64),
            ("CTYPE2", "S16"),
            ("CUNIT2", "S8"),
            ("CDELT2", np.float64),
            ("CROTA2", np.float64),
            ("DATE_D$OBS", "S32"),
            ("DSUN_OBS", np.float64),
            ("SOLAR_B0", np.float64),
            ("HGLN_OBS", np.float64),
            ("HGLT_OBS", np.float64),
            ("CRLN_OBS", np.float64),
            ("CRLT_OBS", np.float64),
            ("DATE_OBS", "S32"),
            ("COMMENT", object),
            ("HISTORY", object),
        ],
    )
    index["SIMPLE"][0] = 1
    index["BITPIX"][0] = 8
    index["NAXIS"][0] = 2
    index["NAXIS1"][0] = 3
    index["NAXIS2"][0] = 2
    index["WCSNAME"][0] = b"Carrington-Heliographic"
    index["CRPIX1"][0] = 1.5
    index["CRVAL1"][0] = 27.8672448568
    index["CTYPE1"][0] = b"CRLN-CEA"
    index["CUNIT1"][0] = b"deg"
    index["CDELT1"][0] = 0.115250131204
    index["CRPIX2"][0] = 1.0
    index["CRVAL2"][0] = -12.2426138116
    index["CTYPE2"][0] = b"CRLT-CEA"
    index["CUNIT2"][0] = b"deg"
    index["CDELT2"][0] = 0.115250131204
    index["CROTA2"][0] = 0.0
    index["DATE_D$OBS"][0] = b"2025-11-26T15:34:31.400"
    index["DSUN_OBS"][0] = 147638514656.0
    index["SOLAR_B0"][0] = 1.44043069702
    index["HGLN_OBS"][0] = 0.0
    index["HGLT_OBS"][0] = 1.44043069702
    index["CRLN_OBS"][0] = 44.9246506781
    index["CRLT_OBS"][0] = 1.44043069702
    index["DATE_OBS"][0] = b"2025-11-26T15:34:31.400"
    index["COMMENT"][0] = np.array([b"FITSHEAD2STRUCT", b"", b"", b"", b""], dtype=object)
    index["HISTORY"][0] = np.array(
        [b"FITSHEAD2STRUCT run at: Tue Feb 10 18:27:01 2026", b"", b"", b"", b""],
        dtype=object,
    )

    box = np.empty(1, dtype=[("BASE", object), ("INDEX", object), ("ID", object), ("EXECUTE", object)])
    box["BASE"][0] = base
    box["INDEX"][0] = index
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NONE"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def _fake_sav_box_with_cubic_corona_and_chromo():
    index = _fake_sav_box_with_base_and_index()["INDEX"][0]
    base = np.empty(
        1,
        dtype=[
            ("BX", np.float64, (2, 2)),
            ("BY", np.float64, (2, 2)),
            ("BZ", np.float64, (2, 2)),
            ("IC", np.float64, (2, 2)),
        ],
    )
    base["BX"][0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    base["BY"][0] = np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float64)
    base["BZ"][0] = np.array([[21.0, 22.0], [23.0, 24.0]], dtype=np.float64)
    base["IC"][0] = np.ones((2, 2), dtype=np.float64)

    # IDL [x,y,z] restored by readsav as numpy [z,y,x].
    bx_zyx = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 100.0
    by_zyx = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 200.0
    bz_zyx = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 300.0
    chromo_bcube = np.stack([bx_zyx + 1000.0, by_zyx + 1000.0, bz_zyx + 1000.0], axis=0)

    box = np.empty(
        1,
        dtype=[
            ("BASE", object),
            ("BX", object),
            ("BY", object),
            ("BZ", object),
            ("DR", object),
            ("CORONA_BASE", np.int16),
            ("INDEX", object),
            ("CHROMO_IDX", object),
            ("CHROMO_T", object),
            ("CHROMO_N", object),
            ("CHROMO_BCUBE", object),
            ("CHROMO_LAYERS", np.int16),
            ("DZ", object),
            ("ID", object),
            ("EXECUTE", object),
        ],
    )
    box["BASE"][0] = base
    box["BX"][0] = bx_zyx
    box["BY"][0] = by_zyx
    box["BZ"][0] = bz_zyx
    box["DR"][0] = np.array([0.1, 0.1, 0.2], dtype=np.float64)
    box["CORONA_BASE"][0] = 1
    box["INDEX"][0] = index
    box["CHROMO_IDX"][0] = np.array([0, 1, 2, 3], dtype=np.int64)
    box["CHROMO_T"][0] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    box["CHROMO_N"][0] = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    box["CHROMO_BCUBE"][0] = chromo_bcube
    box["CHROMO_LAYERS"][0] = 1
    box["DZ"][0] = np.arange(8, dtype=np.float64).reshape(2, 2, 2) + 500.0
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NAS.CHR"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def _fake_sav_box_with_noncubic_corona_and_chromo():
    index = _fake_sav_box_with_base_and_index()["INDEX"][0]
    base = np.empty(
        1,
        dtype=[
            ("BX", np.float64, (3, 4)),
            ("BY", np.float64, (3, 4)),
            ("BZ", np.float64, (3, 4)),
            ("IC", np.float64, (3, 4)),
        ],
    )
    base["BX"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 1.0
    base["BY"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 101.0
    base["BZ"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 201.0
    base["IC"][0] = np.ones((3, 4), dtype=np.float64)

    # Real GX SAV BX/BY/BZ arrays restore through scipy.readsav as (z, y, x).
    bx_zyx = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 10.0
    by_zyx = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 110.0
    bz_zyx = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 210.0

    # Real GX SAV CHROMO_BCUBE restores as (component, z, y, x).
    chromo_bcube_czyx = np.stack(
        [
            np.arange(40, dtype=np.float32).reshape(2, 4, 5) + 1000.0,
            np.arange(40, dtype=np.float32).reshape(2, 4, 5) + 2000.0,
            np.arange(40, dtype=np.float32).reshape(2, 4, 5) + 3000.0,
        ],
        axis=0,
    )

    box = np.empty(
        1,
        dtype=[
            ("BASE", object),
            ("BX", object),
            ("BY", object),
            ("BZ", object),
            ("DR", object),
            ("CORONA_BASE", np.int16),
            ("INDEX", object),
            ("AVFIELD", object),
            ("PHYSLENGTH", object),
            ("STATUS", object),
            ("STARTIDX", object),
            ("ENDIDX", object),
            ("CHROMO_IDX", object),
            ("CHROMO_T", object),
            ("CHROMO_N", object),
            ("N_P", object),
            ("N_HI", object),
            ("N_HTOT", object),
            ("CHROMO_BCUBE", object),
            ("CHROMO_LAYERS", np.int16),
            ("DZ", object),
            ("ID", object),
            ("EXECUTE", object),
        ],
    )
    box["BASE"][0] = base
    box["BX"][0] = bx_zyx
    box["BY"][0] = by_zyx
    box["BZ"][0] = bz_zyx
    box["DR"][0] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    box["CORONA_BASE"][0] = 1
    box["INDEX"][0] = index
    box["AVFIELD"][0] = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 400.0
    box["PHYSLENGTH"][0] = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 500.0
    box["STATUS"][0] = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    box["STARTIDX"][0] = np.arange(24, dtype=np.int64).reshape(2, 3, 4) + 600
    box["ENDIDX"][0] = np.arange(24, dtype=np.int64).reshape(2, 3, 4) + 700
    box["CHROMO_IDX"][0] = np.array([0, 1, 4, 5, 6, 19, 20, 39], dtype=np.int64)
    box["CHROMO_T"][0] = np.arange(8, dtype=np.float32) + 1.0
    box["CHROMO_N"][0] = np.arange(8, dtype=np.float32) + 11.0
    box["N_P"][0] = np.arange(8, dtype=np.float32) + 21.0
    box["N_HI"][0] = np.arange(8, dtype=np.float32) + 31.0
    box["N_HTOT"][0] = np.arange(8, dtype=np.float32) + 41.0
    box["CHROMO_BCUBE"][0] = chromo_bcube_czyx
    box["CHROMO_LAYERS"][0] = 2
    box["DZ"][0] = np.arange(40, dtype=np.float64).reshape(2, 4, 5) + 500.0
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NAS.CHR"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def _fake_sav_box_with_noncubic_bcube_corona():
    index = _fake_sav_box_with_base_and_index()["INDEX"][0]
    base = np.empty(
        1,
        dtype=[
            ("BX", np.float64, (3, 4)),
            ("BY", np.float64, (3, 4)),
            ("BZ", np.float64, (3, 4)),
            ("IC", np.float64, (3, 4)),
        ],
    )
    base["BX"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 1.0
    base["BY"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 101.0
    base["BZ"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 201.0
    base["IC"][0] = np.ones((3, 4), dtype=np.float64)

    bcube_czyx = np.stack(
        [
            np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 10.0,
            np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 110.0,
            np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 210.0,
        ],
        axis=0,
    )

    box = np.empty(
        1,
        dtype=[
            ("BASE", object),
            ("BCUBE", object),
            ("DR", object),
            ("CORONA_BASE", np.int16),
            ("INDEX", object),
            ("ID", object),
            ("EXECUTE", object),
        ],
    )
    box["BASE"][0] = base
    box["BCUBE"][0] = bcube_czyx
    box["DR"][0] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    box["CORONA_BASE"][0] = 1
    box["INDEX"][0] = index
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NAS"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def test_load_entry_box_any_restores_sav_refmaps(tmp_path):
    fake_box = _fake_sav_box_with_refmaps()
    expected_h5 = tmp_path / "expected_refmaps.h5"
    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", expected_h5)
    expected_loaded = load_model(expected_h5)

    with patch.object(gx_fov2box, "load_model", return_value=expected_loaded) as mocked_loader:
        loaded = gx_fov2box._load_entry_box_any(tmp_path / "entry.sav")
    mocked_loader.assert_called_once()

    assert "refmaps" in loaded
    assert list(loaded["refmaps"]) == ["AIA_171"]
    payload = loaded["refmaps"]["AIA_171"]
    assert np.asarray(payload["data"]).shape == (2, 3)

    header = fits.Header.fromstring(payload["wcs_header"], sep="\n")
    assert header["DATE-OBS"] == "2025-11-26T15:34:33.350"
    assert header["DATE_OBS"] == "2025-11-26T15:34:33.350"
    assert header["RSUN_OBS"] == 972.5
    assert header["HGLT_OBS"] == 1.44
    assert header["HGLN_OBS"] == 44.92


def test_load_entry_box_any_serializes_sav_index_as_fits_header(tmp_path):
    fake_box = _fake_sav_box_with_base_and_index()
    expected_h5 = tmp_path / "expected_index.h5"
    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", expected_h5)
    expected_loaded = load_model(expected_h5)

    with patch.object(gx_fov2box, "load_model", return_value=expected_loaded) as mocked_loader:
        loaded = gx_fov2box._load_entry_box_any(tmp_path / "entry.sav")
    mocked_loader.assert_called_once()

    index_text = loaded["base"]["index"]
    if isinstance(index_text, (bytes, bytearray)):
        index_text = index_text.decode("utf-8", errors="ignore")
    assert not index_text.startswith("(")
    header = fits.Header.fromstring(index_text, sep="\n")
    assert header["CTYPE1"] == "CRLN-CEA"
    assert header["CTYPE2"] == "CRLT-CEA"
    assert header["CRVAL1"] == 27.8672448568
    assert header["CRVAL2"] == -12.2426138116
    assert header["CRLN_OBS"] == 44.9246506781
    assert header["DATE-OBS"] == "2025-11-26T15:34:31.400"
    assert header["DATE_OBS"] == "2025-11-26T15:34:31.400"

    out_h5 = tmp_path / "entry.h5"
    write_b3d_h5(str(out_h5), loaded)
    roundtrip = read_b3d_h5(str(out_h5))
    roundtrip_header = fits.Header.fromstring(roundtrip["base"]["index"], sep="\n")
    assert roundtrip_header["CTYPE1"] == "CRLN-CEA"
    assert roundtrip_header["CRLN_OBS"] == 44.9246506781


def test_sav_entry_box_roundtrip_preserves_cubic_3d_axis_order(tmp_path):
    fake_box = _fake_sav_box_with_cubic_corona_and_chromo()
    expected_h5 = tmp_path / "expected_cubic.h5"
    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", expected_h5)
    expected_loaded = load_model(expected_h5)

    with patch.object(gx_fov2box, "load_model", return_value=expected_loaded) as mocked_loader:
        loaded = gx_fov2box._load_entry_box_any(tmp_path / "entry.sav")
    mocked_loader.assert_called_once()

    out_h5 = tmp_path / "entry.h5"
    source_axis_order_3d = gx_fov2box._decode_id_text(loaded.get("metadata", {}).get("axis_order_3d", "zyx")).lower()
    normalized = gx_fov2box._normalize_stage_for_h5(loaded, source_axis_order_3d=source_axis_order_3d)
    write_b3d_h5(str(out_h5), normalized)
    roundtrip = read_b3d_h5(str(out_h5))

    assert np.array_equal(roundtrip["corona"]["bx"], read_b3d_h5(str(expected_h5))["corona"]["bx"])
    assert np.array_equal(roundtrip["corona"]["by"], read_b3d_h5(str(expected_h5))["corona"]["by"])
    assert np.array_equal(roundtrip["corona"]["bz"], read_b3d_h5(str(expected_h5))["corona"]["bz"])
    assert np.array_equal(roundtrip["chromo"]["bx"], read_b3d_h5(str(expected_h5))["chromo"]["bx"])
    assert np.array_equal(roundtrip["chromo"]["by"], read_b3d_h5(str(expected_h5))["chromo"]["by"])
    assert np.array_equal(roundtrip["chromo"]["bz"], read_b3d_h5(str(expected_h5))["chromo"]["bz"])
    assert np.array_equal(roundtrip["chromo"]["dz"], read_b3d_h5(str(expected_h5))["chromo"]["dz"])


def test_build_h5_from_sav_transposes_noncubic_corona_components_to_canonical_zyx(tmp_path):
    fake_box = _fake_sav_box_with_noncubic_corona_and_chromo()
    out_h5 = tmp_path / "noncubic_corona.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    expected_bx = np.asarray(fake_box["BX"][0], dtype=np.float32)
    expected_by = np.asarray(fake_box["BY"][0], dtype=np.float32)
    expected_bz = np.asarray(fake_box["BZ"][0], dtype=np.float32)

    with h5py.File(out_h5, "r") as f:
        assert f["corona/bx"].shape == (2, 3, 4)
        assert np.array_equal(np.asarray(f["corona/bx"]), expected_bx)
        assert np.array_equal(np.asarray(f["corona/by"]), expected_by)
        assert np.array_equal(np.asarray(f["corona/bz"]), expected_bz)


def test_build_h5_from_sav_preserves_noncubic_corona_bcube_component_identity(tmp_path):
    fake_box = _fake_sav_box_with_noncubic_bcube_corona()
    out_h5 = tmp_path / "noncubic_corona_bcube.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    bcube = np.asarray(fake_box["BCUBE"][0], dtype=np.float32)
    with h5py.File(out_h5, "r") as f:
        assert np.array_equal(np.asarray(f["corona/bx"]), bcube[0])
        assert np.array_equal(np.asarray(f["corona/by"]), bcube[1])
        assert np.array_equal(np.asarray(f["corona/bz"]), bcube[2])


def test_build_h5_from_sav_preserves_noncubic_chromo_component_identity(tmp_path):
    fake_box = _fake_sav_box_with_noncubic_corona_and_chromo()
    out_h5 = tmp_path / "noncubic_chromo.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    chromo_bcube = np.asarray(fake_box["CHROMO_BCUBE"][0], dtype=np.float32)

    with h5py.File(out_h5, "r") as f:
        assert f["chromo/bx"].shape == (2, 4, 5)
        assert np.array_equal(np.asarray(f["chromo/bx"]), chromo_bcube[0])
        assert np.array_equal(np.asarray(f["chromo/by"]), chromo_bcube[1])
        assert np.array_equal(np.asarray(f["chromo/bz"]), chromo_bcube[2])


def test_build_h5_from_sav_flattens_lines_in_canonical_zyx_c_order(tmp_path):
    fake_box = _fake_sav_box_with_noncubic_corona_and_chromo()
    out_h5 = tmp_path / "noncubic_lines.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    expected_av = np.asarray(fake_box["AVFIELD"][0], dtype=np.float64).reshape(-1, order="C")
    expected_phys = np.asarray(fake_box["PHYSLENGTH"][0], dtype=np.float64).reshape(-1, order="C")
    expected_status = np.asarray(fake_box["STATUS"][0], dtype=np.uint8).reshape(-1, order="C")
    expected_start = np.asarray(fake_box["STARTIDX"][0], dtype=np.int64).reshape(-1, order="C")
    expected_end = np.asarray(fake_box["ENDIDX"][0], dtype=np.int64).reshape(-1, order="C")

    with h5py.File(out_h5, "r") as f:
        av = np.asarray(f["lines/av_field"])
        phys = np.asarray(f["lines/phys_length"])
        status = np.asarray(f["lines/voxel_status"])
        start = np.asarray(f["lines/start_idx"])
        end = np.asarray(f["lines/end_idx"])

    assert np.array_equal(av, expected_av)
    assert np.array_equal(phys, expected_phys)
    assert np.array_equal(status, expected_status)
    assert np.array_equal(start, expected_start)
    assert np.array_equal(end, expected_end)

    assert np.array_equal(av.reshape((2, 3, 4), order="C"), np.asarray(fake_box["AVFIELD"][0], dtype=np.float64))
    assert np.array_equal(start.reshape((2, 3, 4), order="C"), np.asarray(fake_box["STARTIDX"][0], dtype=np.int64))
    assert start[0] == fake_box["STARTIDX"][0][0, 0, 0]
    assert start[1] == fake_box["STARTIDX"][0][0, 0, 1]
    assert start[4] == fake_box["STARTIDX"][0][0, 1, 0]
    assert start[12] == fake_box["STARTIDX"][0][1, 0, 0]


def test_build_h5_from_sav_preserves_chromo_sparse_index_mapping_in_canonical_zyx_c_order(tmp_path):
    fake_box = _fake_sav_box_with_noncubic_corona_and_chromo()
    out_h5 = tmp_path / "noncubic_chromo_sparse.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    expected_idx = np.asarray(fake_box["CHROMO_IDX"][0], dtype=np.int64)
    expected_n = np.asarray(fake_box["CHROMO_N"][0], dtype=np.float32)
    expected_t = np.asarray(fake_box["CHROMO_T"][0], dtype=np.float32)
    expected_n_p = np.asarray(fake_box["N_P"][0], dtype=np.float32)
    expected_n_hi = np.asarray(fake_box["N_HI"][0], dtype=np.float32)
    expected_n_htot = np.asarray(fake_box["N_HTOT"][0], dtype=np.float32)

    with h5py.File(out_h5, "r") as f:
        chromo_idx = np.asarray(f["chromo/chromo_idx"])
        chromo_n = np.asarray(f["chromo/chromo_n"])
        chromo_t = np.asarray(f["chromo/chromo_t"])
        n_p = np.asarray(f["chromo/n_p"])
        n_hi = np.asarray(f["chromo/n_hi"])
        n_htot = np.asarray(f["chromo/n_htot"])
        chromo_shape = np.asarray(f["chromo/bx"]).shape

    assert np.array_equal(chromo_idx, expected_idx)
    assert np.array_equal(chromo_n, expected_n)
    assert np.array_equal(chromo_t, expected_t)
    assert np.array_equal(n_p, expected_n_p)
    assert np.array_equal(n_hi, expected_n_hi)
    assert np.array_equal(n_htot, expected_n_htot)

    dense_n = np.zeros(chromo_shape, dtype=np.float32)
    dense_t = np.zeros(chromo_shape, dtype=np.float32)
    dense_n.flat[chromo_idx] = chromo_n
    dense_t.flat[chromo_idx] = chromo_t

    expected_dense_n = np.zeros(chromo_shape, dtype=np.float32)
    expected_dense_t = np.zeros(chromo_shape, dtype=np.float32)
    expected_dense_n.flat[expected_idx] = expected_n
    expected_dense_t.flat[expected_idx] = expected_t

    assert np.array_equal(dense_n, expected_dense_n)
    assert np.array_equal(dense_t, expected_dense_t)
    assert dense_n[0, 0, 1] == expected_n[1]
    assert dense_n[0, 1, 0] == expected_n[3]
    assert dense_n[1, 0, 0] == expected_n[6]
    assert dense_n[1, 3, 4] == expected_n[7]


def test_normalize_stage_for_h5_transposes_chr_2d_maps_from_xy_to_yx():
    stage_box = {
        "base": {
            "bx": np.arange(6, dtype=np.float32).reshape(2, 3),
            "by": np.arange(6, dtype=np.float32).reshape(2, 3),
            "bz": np.arange(6, dtype=np.float32).reshape(2, 3),
        },
        "chromo": {
            "tr": np.array([[10, 11], [12, 13], [14, 15]], dtype=np.int64),
            "tr_h": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            "chromo_mask": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16),
        },
    }

    normalized = gx_fov2box._normalize_stage_for_h5(
        stage_box,
        source_axis_order_3d="xyz",
        chromo_source_axis_order_2d="xy",
    )

    assert np.array_equal(normalized["chromo"]["tr"], stage_box["chromo"]["tr"].T)
    assert np.array_equal(normalized["chromo"]["tr_h"], stage_box["chromo"]["tr_h"].T)
    assert np.array_equal(normalized["chromo"]["chromo_mask"], stage_box["chromo"]["chromo_mask"].T)


def test_build_h5_from_sav_does_not_write_raw_sav_by_default(tmp_path):
    fake_box = _fake_sav_box_with_base_and_index()
    out_h5 = tmp_path / "entry.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    with h5py.File(out_h5, "r") as f:
        assert "raw_sav" not in f
