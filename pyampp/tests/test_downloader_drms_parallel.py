from __future__ import annotations

import concurrent.futures
from pathlib import Path

from astropy.time import Time

import pyampp.data.downloader as downloader_mod
from pyampp.data.downloader import SDOImageDownloader


def test_download_images_drms_schedules_all_missing_products_with_worker_cap(monkeypatch, tmp_path: Path) -> None:
    when = Time("2025-11-26T15:47:52")
    downloader = SDOImageDownloader(when, data_dir=str(tmp_path), backend="drms", force_download=False)

    hmi_keys = ["field", "inclination", "azimuth", "disambig", "magnetogram", "continuum"]
    expected_count = len(hmi_keys) + len(downloader_mod.AIA_EUV_PASSBANDS) + len(downloader_mod.AIA_UV_PASSBANDS)

    check_calls = {"count": 0}

    def fake_check_files_exist(_datadir, returnfilelist=False):
        if not returnfilelist:
            return {}
        check_calls["count"] += 1
        if check_calls["count"] == 1:
            return {k: None for k in hmi_keys + downloader_mod.AIA_EUV_PASSBANDS + downloader_mod.AIA_UV_PASSBANDS}
        return {k: f"/fake/{k}.fits" for k in hmi_keys + downloader_mod.AIA_EUV_PASSBANDS + downloader_mod.AIA_UV_PASSBANDS}

    drms_calls = []

    def fake_drms_get_fits(series, segment, wave=None, time_window=12):
        drms_calls.append((series, segment, wave, time_window))
        suffix = wave if wave is not None else segment
        return f"/fake/{series}.{suffix}.fits"

    captured_workers = {}

    class _ImmediateExecutor:
        def __init__(self, max_workers):
            captured_workers["value"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fut = concurrent.futures.Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as exc:  # pragma: no cover - defensive
                fut.set_exception(exc)
            return fut

    monkeypatch.setattr(downloader, "_check_files_exist", fake_check_files_exist)
    monkeypatch.setattr(downloader, "_drms_get_fits", fake_drms_get_fits)
    monkeypatch.setattr(downloader_mod, "ThreadPoolExecutor", _ImmediateExecutor)

    result = downloader._download_images_drms()

    assert len(drms_calls) == expected_count
    assert captured_workers["value"] == downloader.DRMS_MAX_WORKERS
    assert any(series == "aia.lev1_euv_12s" and segment == "image" and wave == "94" for series, segment, wave, _ in drms_calls)
    assert any(series == "hmi.B_720s" and segment == "field" and wave is None for series, segment, wave, _ in drms_calls)
    assert result["field"] == "/fake/field.fits"
    assert result["94"] == "/fake/94.fits"
