from __future__ import annotations

import concurrent.futures
from pathlib import Path
import time

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

    captured_workers = []

    class _ImmediateExecutor:
        def __init__(self, max_workers):
            captured_workers.append(max_workers)

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
    assert captured_workers[0] == downloader.DRMS_HMI_MAX_WORKERS
    assert captured_workers[1] == downloader.DRMS_MAX_WORKERS
    assert any(series == "aia.lev1_euv_12s" and segment == "image" and wave == "94" for series, segment, wave, _ in drms_calls)
    assert any(series == "hmi.B_720s" and segment == "field" and wave is None for series, segment, wave, _ in drms_calls)
    assert result["field"] == "/fake/field.fits"
    assert result["94"] == "/fake/94.fits"


def test_download_images_drms_keeps_returned_context_paths_when_final_scan_rejects_them(monkeypatch, tmp_path: Path) -> None:
    when = Time("2026-04-03T19:46:37.800")
    downloader = SDOImageDownloader(
        when,
        data_dir=str(tmp_path),
        backend="drms",
        hmi=False,
        euv=True,
        uv=False,
        force_download=False,
    )

    def fake_check_files_exist(_datadir, returnfilelist=False):
        if not returnfilelist:
            return {}
        return {pb: None for pb in downloader_mod.AIA_EUV_PASSBANDS}

    def fake_drms_get_fits(series, segment, wave=None, time_window=12):
        return f"/cache/aia.{wave}.fits"

    class _ImmediateExecutor:
        def __init__(self, max_workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fut = concurrent.futures.Future()
            fut.set_result(fn(*args, **kwargs))
            return fut

    monkeypatch.setattr(downloader, "_check_files_exist", fake_check_files_exist)
    monkeypatch.setattr(downloader, "_drms_get_fits", fake_drms_get_fits)
    monkeypatch.setattr(downloader_mod, "ThreadPoolExecutor", _ImmediateExecutor)

    result = downloader._download_images_drms()

    assert result["171"] == "/cache/aia.171.fits"


def test_cache_store_concurrent_updates_preserve_all_entries(monkeypatch, tmp_path: Path) -> None:
    downloader = SDOImageDownloader(Time("2025-11-26T15:47:52"), data_dir=str(tmp_path), backend="drms")
    original_load_cache_index = downloader._load_cache_index

    def delayed_load_cache_index():
        entries = original_load_cache_index()
        time.sleep(0.01)
        return entries

    monkeypatch.setattr(downloader, "_load_cache_index", delayed_load_cache_index)

    queries = [f"query-{idx}" for idx in range(20)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(downloader._cache_store, query, str(Path(downloader.path) / f"{query}.fits")) for query in queries
        ]
        for future in futures:
            future.result()

    entries = downloader._load_cache_index()
    assert set(entries) == set(queries)
    for query in queries:
        assert entries[query] == f"{query}.fits"


def test_drms_get_fits_retries_on_pending_export_limit(monkeypatch, tmp_path: Path) -> None:
    downloader = SDOImageDownloader(Time("2025-11-26T15:47:52"), data_dir=str(tmp_path), backend="drms")

    class _Urls:
        class _ILoc:
            def __getitem__(self, _idx):
                return {"url": "https://example.test/file.fits"}

        def __len__(self):
            return 1

        iloc = _ILoc()

    class _Request:
        def __init__(self):
            self.urls = _Urls()

        def wait(self, sleep=5):
            return None

    class _Client:
        def __init__(self):
            self.export_calls = 0

        def query(self, *_args, **_kwargs):
            return object(), object()

        def export(self, *_args, **_kwargs):
            self.export_calls += 1
            if self.export_calls == 1:
                raise RuntimeError("User foo has 3 pending export requests [status=7]")
            return _Request()

    client = _Client()
    sleep_calls = {"count": 0}

    monkeypatch.setattr(downloader, "_query_window_seconds", lambda *_args, **_kwargs: 12)
    monkeypatch.setattr(downloader, "_make_query_key", lambda *_args, **_kwargs: "q")
    monkeypatch.setattr(downloader, "_cache_lookup", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(downloader, "_get_drms_client", lambda: client)
    monkeypatch.setattr(downloader, "_make_query_recordset", lambda *_args, **_kwargs: "rec")
    monkeypatch.setattr(
        downloader,
        "_select_nearest_record",
        lambda *_args, **_kwargs: {"record": "hmi.B_720s[2020.11.26_20:00:00_TAI]", "t_rec": "2020.11.26_20:00:00_TAI"},
    )
    monkeypatch.setattr(downloader, "_make_local_filename", lambda *_args, **_kwargs: "out.fits")
    monkeypatch.setattr(downloader, "_download_from_url", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(downloader, "_normalize_drms_export", lambda *_args, **_kwargs: str(tmp_path / "out.fits"))
    monkeypatch.setattr(downloader, "_fits_has_map_metadata", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(downloader, "_cache_store", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(downloader_mod.time, "sleep", lambda _s: sleep_calls.__setitem__("count", sleep_calls["count"] + 1))

    result = downloader._drms_get_fits("hmi.B_720s", "field")

    assert result.endswith("out.fits")
    assert client.export_calls == 2
    assert sleep_calls["count"] == 1


def test_download_images_drms_degrades_context_to_sequential_on_throttle(monkeypatch, tmp_path: Path) -> None:
    when = Time("2025-11-26T15:47:52")
    downloader = SDOImageDownloader(when, data_dir=str(tmp_path), backend="drms", force_download=False)

    hmi_keys = ["field", "inclination", "azimuth", "disambig", "magnetogram", "continuum"]
    all_keys = hmi_keys + downloader_mod.AIA_EUV_PASSBANDS + downloader_mod.AIA_UV_PASSBANDS

    check_calls = {"count": 0}

    def fake_check_files_exist(_datadir, returnfilelist=False):
        if not returnfilelist:
            return {}
        check_calls["count"] += 1
        if check_calls["count"] == 1:
            return {k: None for k in all_keys}
        return {k: f"/fake/{k}.fits" for k in all_keys}

    attempts = {}

    def fake_drms_get_fits(series, segment, wave=None, time_window=12):
        key = f"{series}:{wave or segment}"
        attempts[key] = attempts.get(key, 0) + 1
        if str(series).startswith("aia.") and attempts[key] == 1:
            downloader._drms_throttle_seen = True
            return ""
        suffix = wave if wave is not None else segment
        return f"/fake/{series}.{suffix}.fits"

    captured_workers = []

    class _ImmediateExecutor:
        def __init__(self, max_workers):
            captured_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fut = concurrent.futures.Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as exc:  # pragma: no cover
                fut.set_exception(exc)
            return fut

    monkeypatch.setattr(downloader, "_check_files_exist", fake_check_files_exist)
    monkeypatch.setattr(downloader, "_drms_get_fits", fake_drms_get_fits)
    monkeypatch.setattr(downloader_mod, "ThreadPoolExecutor", _ImmediateExecutor)

    downloader._download_images_drms()

    # HMI pass, AIA parallel pass, then AIA sequential retry pass.
    assert captured_workers == [downloader.DRMS_HMI_MAX_WORKERS, downloader.DRMS_MAX_WORKERS, 1]


def test_download_images_drms_sequential_flag_forces_single_worker(monkeypatch, tmp_path: Path) -> None:
    when = Time("2025-11-26T15:47:52")
    downloader = SDOImageDownloader(
        when,
        data_dir=str(tmp_path),
        backend="drms",
        force_download=False,
        drms_sequential=True,
    )

    hmi_keys = ["field", "inclination", "azimuth", "disambig", "magnetogram", "continuum"]
    all_keys = hmi_keys + downloader_mod.AIA_EUV_PASSBANDS + downloader_mod.AIA_UV_PASSBANDS

    check_calls = {"count": 0}

    def fake_check_files_exist(_datadir, returnfilelist=False):
        if not returnfilelist:
            return {}
        check_calls["count"] += 1
        if check_calls["count"] == 1:
            return {k: None for k in all_keys}
        return {k: f"/fake/{k}.fits" for k in all_keys}

    def fake_drms_get_fits(series, segment, wave=None, time_window=12):
        suffix = wave if wave is not None else segment
        return f"/fake/{series}.{suffix}.fits"

    captured_workers = []

    class _ImmediateExecutor:
        def __init__(self, max_workers):
            captured_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fut = concurrent.futures.Future()
            fut.set_result(fn(*args, **kwargs))
            return fut

    monkeypatch.setattr(downloader, "_check_files_exist", fake_check_files_exist)
    monkeypatch.setattr(downloader, "_drms_get_fits", fake_drms_get_fits)
    monkeypatch.setattr(downloader_mod, "ThreadPoolExecutor", _ImmediateExecutor)

    downloader._download_images_drms()

    assert captured_workers == [1, 1]
