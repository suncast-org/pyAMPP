"""Potential field solver for solar magnetic field extrapolation."""

import numpy as np


def pot_vmag(mag, simple=False):
    """Compute potential (current-free) magnetic field from observed Bz.

    Uses standard FFT-based potential field extrapolation:
    - Pad Bz with zeros
    - FFT to get Bz_hat(kx, ky)
    - Bx_hat = -i*kx*Bz_hat/q, By_hat = -i*ky*Bz_hat/q
    - IFFT to get spatial components

    Args:
        mag: Magnetogram structure from get_str_mag.
        simple: If True, use Neumann BC solver (ignored).

    Returns:
        mag structure with added potential field components.
    """
    bz = mag['t0'].astype(float)
    ny, nx = bz.shape

    # Zero-padding (sufficient for potential field with decay)
    pad_y, pad_x = ny // 2, nx // 2
    bz_pad = np.zeros((ny + 2 * pad_y, nx + 2 * pad_x), dtype=float)
    bz_pad[pad_y:pad_y + ny, pad_x:pad_x + nx] = bz

    p_ny, p_nx = bz_pad.shape

    # Wave numbers
    kx = 2 * np.pi * np.fft.fftfreq(p_nx)
    ky = 2 * np.pi * np.fft.fftfreq(p_ny)
    kx, ky = np.meshgrid(kx, ky)

    q = np.sqrt(kx ** 2 + ky ** 2)

    # FFT of padded Bz
    bz_hat = np.fft.fft2(bz_pad)

    # Potential field relations in Fourier space:
    # Bx_hat = -i*kx/q * Bz_hat, By_hat = -i*ky/q * Bz_hat
    q_safe = np.where(q > 1e-10, q, 1.0)

    bx_hat = -1j * kx / q_safe * bz_hat
    by_hat = -1j * ky / q_safe * bz_hat

    # Zero out k=0 mode (no net flux)
    bx_hat[0, 0] = 0
    by_hat[0, 0] = 0

    # Inverse FFT and extract original region
    bx_pot = np.fft.ifft2(bx_hat).real[pad_y:pad_y + ny, pad_x:pad_x + nx]
    by_pot = np.fft.ifft2(by_hat).real[pad_y:pad_y + ny, pad_x:pad_x + nx]
    bz_pot = np.fft.ifft2(bz_hat).real[pad_y:pad_y + ny, pad_x:pad_x + nx]

    result = dict(mag)
    result['t0'] = bz_pot
    result['t1'] = bx_pot
    result['t2'] = by_pot
    result['type'] = 'vecp ' + mag.get('type', '')
    return result
