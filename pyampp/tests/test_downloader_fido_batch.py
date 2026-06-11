from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from astropy.time import Time

import pyampp.data.downloader as downloader_mod
from pyampp.data.downloader import SDOImageDownloader


def test_download_images_fido_batches_aia_wavelengths(monkeypatch, tmp_path: Path) -> None:
    when = Time("2026-04-03T19:46:37")
    downloader = SDOImageDownloader(
        when,
        data_dir=str(tmp_path),
        backend="fido",
        force_download=False,
        hmi=False,
        euv=True,
        uv=False,
    )

    search_calls = []
    fetch_calls = []

    def fake_search(*args, **kwargs):
        search_calls.append(args)
        result = MagicMock()
        result.__len__.return_value = 1
        return result

    def fake_fetch(*args, **kwargs):
        fetch_calls.append(args)
        return []

    monkeypatch.setattr(downloader_mod.Fido, "search", fake_search)
    monkeypatch.setattr(downloader_mod.Fido, "fetch", fake_fetch)
    monkeypatch.setattr(downloader, "_try_resolve_local", lambda *_args, **_kwargs: "")

    downloader._download_images_fido()

    assert len(search_calls) == 1
    assert len(fetch_calls) == 1


def test_download_images_fido_batches_hmi_segments(monkeypatch, tmp_path: Path) -> None:
    when = Time("2026-04-03T19:46:37")
    downloader = SDOImageDownloader(
        when,
        data_dir=str(tmp_path),
        backend="fido",
        force_download=False,
        hmi=True,
        euv=False,
        uv=False,
    )

    search_calls = []

    def fake_search(*args, **kwargs):
        search_calls.append(args)
        result = MagicMock()
        result.__len__.return_value = 1
        return result

    monkeypatch.setattr(downloader_mod.Fido, "search", fake_search)
    monkeypatch.setattr(downloader_mod.Fido, "fetch", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(downloader, "_try_resolve_local", lambda *_args, **_kwargs: "")

    downloader._download_images_fido()

    # hmi.B_720s (4 segments), hmi.M_720s, hmi.Ic_noLimbDark_720s
    assert len(search_calls) == 3
