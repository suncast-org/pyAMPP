from __future__ import annotations

from pathlib import Path

from astropy.io import fits
from astropy.time import Time

from pyampp.data.downloader import SDOImageDownloader


def _write_map_fits(path: Path, date_obs: str) -> None:
    header = fits.Header()
    header["DATE-OBS"] = date_obs
    header["CTYPE1"] = "HPLN-TAN"
    fits.PrimaryHDU(data=[[0.0]], header=header).writeto(path, overwrite=True)


def test_find_nearest_cached_file_prefers_closer_hmi_bundle(tmp_path: Path) -> None:
    day_dir = tmp_path / "2026-04-03"
    day_dir.mkdir()
    early = day_dir / "hmi.B_720s.20260403_193600_TAI.field.fits"
    late = day_dir / "hmi.B_720s.20260403_194800_TAI.field.fits"
    _write_map_fits(early, "2026-04-03T19:34:37")
    _write_map_fits(late, "2026-04-03T19:48:00")

    downloader = SDOImageDownloader(
        Time("2026-04-03T19:46:37"),
        data_dir=str(tmp_path),
        backend="drms",
        hmi=True,
        euv=False,
        uv=False,
    )

    patterns = downloader._generate_filename_patterns(str(day_dir))["hmi_b"]["field"]
    nearest = downloader._find_nearest_cached_file(patterns, downloader.hmi_time_window)
    assert nearest == str(late)


def test_try_resolve_local_uses_index_json_without_network(tmp_path: Path) -> None:
    day_dir = tmp_path / "2026-04-03"
    day_dir.mkdir()
    fits_path = day_dir / "aia.lev1_euv_12s.2026-04-03T194623Z.image.131.fits"
    _write_map_fits(fits_path, "2026-04-03T19:46:23")

    downloader = SDOImageDownloader(
        Time("2026-04-03T19:46:37"),
        data_dir=str(tmp_path),
        backend="fido",
        hmi=False,
        euv=True,
        uv=False,
    )
    patterns = downloader._generate_filename_patterns(str(day_dir))["euv"]["131"]
    t1, t2 = downloader._make_query_bounds("aia.lev1_euv_12s", downloader.aia_time_window)
    query_key = downloader._make_query_key(t1, t2, "aia.lev1_euv_12s", "image", wave="131")
    downloader._cache_store(query_key, str(fits_path))

    resolved = downloader._try_resolve_local(
        "aia.lev1_euv_12s",
        "image",
        "131",
        downloader.aia_time_window,
        patterns,
    )
    assert resolved == str(fits_path)
