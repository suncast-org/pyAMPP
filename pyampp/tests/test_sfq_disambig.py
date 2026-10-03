"""Unit smoke tests for vendored Sergey/vit1-irk SFQ (#42)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from pyampp.sfq import sfq_disambig


def _synthetic_magnetogram(nx: int = 64, ny: int = 64, seed: int = 42):
    """Potential-like bipolar AR with ~50% random azimuth flips (upstream example)."""
    np.random.seed(seed)
    x = np.linspace(-100, 100, nx)
    y = np.linspace(-100, 100, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    r1 = np.sqrt((xx + 30) ** 2 + (yy - 20) ** 2)
    r2 = np.sqrt((xx - 30) ** 2 + (yy + 20) ** 2)
    bz = 500 * np.exp(-(r1**2) / 800) - 400 * np.exp(-(r2**2) / 800)
    bz = bz + 30 * np.random.randn(ny, nx)

    bz_pad = np.zeros((2 * ny, 2 * nx), dtype=float)
    bz_pad[:ny, :nx] = bz
    kx = 2 * np.pi * np.fft.fftfreq(2 * nx)
    ky = 2 * np.pi * np.fft.fftfreq(2 * ny)
    kx, ky = np.meshgrid(kx, ky)
    q = np.sqrt(kx**2 + ky**2)
    q_safe = np.where(q > 1e-10, q, 1.0)
    bz_hat = np.fft.fft2(bz_pad)
    bx_hat = -1j * kx / q_safe * bz_hat
    by_hat = -1j * ky / q_safe * bz_hat
    bx_hat[0, 0] = 0
    by_hat[0, 0] = 0
    bx_true = np.fft.ifft2(bx_hat).real[:ny, :nx]
    by_true = np.fft.ifft2(by_hat).real[:ny, :nx]
    bx_true = gaussian_filter(bx_true, sigma=2)
    by_true = gaussian_filter(by_true, sigma=2)

    flip = np.random.rand(ny, nx) > 0.5
    bx_ambig = bx_true.copy()
    by_ambig = by_true.copy()
    bx_ambig[flip] = -bx_ambig[flip]
    by_ambig[flip] = -by_ambig[flip]
    bx_ambig = bx_ambig + 20 * np.random.randn(ny, nx)
    by_ambig = by_ambig + 20 * np.random.randn(ny, nx)

    pos = np.array([-100.0, -100.0, 100.0, 100.0])
    rsun = 960.0
    return bx_ambig, by_ambig, bz, pos, rsun, bx_true, by_true, flip


def test_sfq_disambig_recovers_transverse_sign():
    bx_ambig, by_ambig, bz, pos, rsun, bx_true, by_true, flip = _synthetic_magnetogram()

    # Ambiguous baseline should be near chance on strong-field pixels.
    mask = (bx_true**2 + by_true**2) > 100
    assert np.any(mask)
    ambig_dot = bx_true * bx_ambig + by_true * by_ambig
    ambig_agree = float(np.mean(ambig_dot[mask] > 0))
    assert ambig_agree < 0.6

    bx_out, by_out = sfq_disambig(
        bx_ambig.copy(),
        by_ambig.copy(),
        bz,
        pos,
        rsun,
        silent=True,
    )

    # Allow a global 180° flip of the recovered transverse field.
    err1 = np.sum((bx_out - bx_true) ** 2 + (by_out - by_true) ** 2)
    err2 = np.sum((bx_out + bx_true) ** 2 + (by_out + by_true) ** 2)
    if err2 < err1:
        bx_out = -bx_out
        by_out = -by_out

    recovered_dot = bx_true * bx_out + by_true * by_out
    sign_recovery = float(np.mean(recovered_dot[mask] > 0))
    assert sign_recovery > 0.90, f"SFQ transverse-sign recovery too low: {sign_recovery:.1%}"


def test_sfq_clean_filter_width_is_idl_size_not_radius():
    """IDL ``median/smooth(arr, s)`` uses neighborhood size ``s``, not ``2*s+1``."""
    from pyampp.sfq.utils import median_2d, smooth_2d

    arr = np.arange(25, dtype=float).reshape(5, 5)
    med3 = median_2d(arr, 3)
    # Size-3 median of center 3x3 block [6,7,8,11,12,13,16,17,18] = 12
    assert med3[2, 2] == 12.0
    sm = smooth_2d(arr, 3)
    assert sm.shape == arr.shape


def test_sfq_public_exports():
    from pyampp import sfq
    from pyampp.gxbox import boxutils

    assert callable(sfq.sfq_disambig)
    assert callable(boxutils.sfq_disambig)
    assert not hasattr(sfq, "pex_bl")


def test_load_hmi_maps_skips_disambig_when_sfq():
    """``--sfq`` must not apply HMI disambig bits before SFQ runs."""
    from pathlib import Path
    from unittest.mock import patch

    from astropy.time import Time

    from pyampp.gxbox import gx_fov2box
    from pyampp.tests.test_fov2box_time_anchor import _FakeDownloader, _fake_map_loader

    _FakeDownloader.calls.clear()
    called = {"hmi_disambig": 0}

    def _track_disambig(azimuth, _disambig, method=2):
        called["hmi_disambig"] += 1
        return azimuth

    requested = Time("2025-11-26T15:47:52")
    with patch.object(gx_fov2box, "SDOImageDownloader", _FakeDownloader), patch.object(
        gx_fov2box, "load_sunpy_map_compat", side_effect=_fake_map_loader
    ), patch.object(gx_fov2box, "hmi_disambig", side_effect=_track_disambig):
        gx_fov2box._load_hmi_maps_from_downloader(
            requested,
            Path("/tmp"),
            euv=False,
            uv=False,
            apply_hmi_disambig=False,
        )
        gx_fov2box._load_hmi_maps_from_downloader(
            requested,
            Path("/tmp"),
            euv=False,
            uv=False,
            apply_hmi_disambig=True,
        )

    assert called["hmi_disambig"] == 1
