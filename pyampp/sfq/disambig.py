"""SFQ disambiguation pipeline: step1, clean, frame, and main entry point."""

import numpy as np
import time

from .data import get_str_mag
from .field import pot_vmag
from .utils import gauss_smooth, gaussf, median_2d, smooth_2d


def sfq_step1(mag, pot, silent=False, acute=False):
    """Preliminary disambiguation step.

    Args:
        mag: Magnetogram structure with t1, t2 transverse components.
        pot: Potential field structure with t1, t2.
        silent: Suppress messages.
        acute: Use acute-angle method only.

    Returns:
        mag with disambiguated t1, t2.
    """
    t0 = time.time()
    if not silent:
        print("Starting preliminary SFQ disambiguation")

    # Flip pixels where t2 (originally By in LOS) is negative
    ind = mag['t2'] < 0
    if np.any(ind):
        mag['t1'][ind] = -mag['t1'][ind]
        mag['t2'][ind] = -mag['t2'][ind]

    if acute:
        au = mag['t1'] * pot['t1'] + mag['t2'] * pot['t2']
        ind = au < 0
        mag['t1'][ind] = -mag['t1'][ind]
        mag['t2'][ind] = -mag['t2'][ind]
        if not silent:
            print(f"Preliminary SFQ disambiguation complete in {time.time() - t0:.2f} seconds")
        return mag

    # Spatial gradient consistency method
    # Compare gradients of observed vs potential transverse fields
    pot_t1_shift_x = np.roll(pot['t1'], -1, axis=0)
    mag_t1_shift_x = np.roll(mag['t1'], -1, axis=0)
    au = (pot_t1_shift_x - pot['t1'] - (mag_t1_shift_x - mag['t1'])) ** 2
    au_ = (pot_t1_shift_x - pot['t1'] + (mag_t1_shift_x - mag['t1'])) ** 2

    pot_t1_shift_y = np.roll(pot['t1'], -1, axis=1)
    mag_t1_shift_y = np.roll(mag['t1'], -1, axis=1)
    au += (pot_t1_shift_y - pot['t1'] - (mag_t1_shift_y - mag['t1'])) ** 2
    au_ += (pot_t1_shift_y - pot['t1'] + (mag_t1_shift_y - mag['t1'])) ** 2

    pot_t2_shift_x = np.roll(pot['t2'], -1, axis=0)
    mag_t2_shift_x = np.roll(mag['t2'], -1, axis=0)
    au += (pot_t2_shift_x - pot['t2'] - (mag_t2_shift_x - mag['t2'])) ** 2
    au_ += (pot_t2_shift_x - pot['t2'] + (mag_t2_shift_x - mag['t2'])) ** 2

    pot_t2_shift_y = np.roll(pot['t2'], -1, axis=1)
    mag_t2_shift_y = np.roll(mag['t2'], -1, axis=1)
    au = np.sqrt(au + (pot_t2_shift_y - pot['t2'] - (mag_t2_shift_y - mag['t2'])) ** 2)
    au_ = np.sqrt(au_ + (pot_t2_shift_y - pot['t2'] + (mag_t2_shift_y - mag['t2'])) ** 2)

    ind = au > au_
    mag['t1'][ind] = -mag['t1'][ind]
    mag['t2'][ind] = -mag['t2'][ind]

    if not silent:
        print(f"Preliminary SFQ disambiguation complete in {time.time() - t0:.2f} seconds")
    return mag


def _clean_iterative(bx, by, s, use_gauss=False, use_median=False):
    """Iterative cleaning at a specific scale.

    Args:
        bx, by: Transverse field components (modified in place).
        s: IDL filter width / scale (same meaning as in ``sfq_clean.pro``).
        use_gauss: Use Gaussian smoothing.
        use_median: Use median filtering.
    """
    n = int(np.ceil(s * 3) * 2 + 1)
    ker = gaussf(n, s)
    gaussk = ker[n // 2] ** 2

    for _ in range(300):
        if use_gauss:
            mbx = gauss_smooth(bx, s) - bx * gaussk
            mby = gauss_smooth(by, s) - by * gaussk
        elif use_median:
            mbx = median_2d(bx, s)
            mby = median_2d(by, s)
        else:
            # IDL: smooth(bx,s,/edge_tr) - bx/float(s^2)
            mbx = smooth_2d(bx, s) - bx / float(s ** 2)
            mby = smooth_2d(by, s) - by / float(s ** 2)

        dot = mbx * bx + mby * by
        ind = dot < 0
        # Match IDL intent: stop when remaining flips are a tiny fraction.
        # IDL uses a boolean expression that effectively stops near ~0 flips for
        # typical FOVs; keep a small absolute floor for stability.
        threshold = max(bx.size * 0.0001, 5)
        if np.sum(ind) < threshold:
            break
        bx[ind] = -bx[ind]
        by[ind] = -by[ind]


def sfq_clean(bx, by, mode=False, silent=False):
    """Multi-scale noise cleaning of disambiguated transverse field.

    Args:
        bx, by: Transverse field components (modified in place).
        mode: If True, use SOLIS-optimized cleaning (skip large-scale gauss).
        silent: Suppress messages.
    """
    t0 = time.time()
    if not silent:
        print("Starting SFQ cleaning")

    # Make copies to avoid modifying originals directly in recursion
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)

    # Multi-scale cleaning cascade
    _clean_iterative(bx, by, 3, use_median=True)

    if not mode:
        ny, nx = bx.shape
        if min(nx, ny) > 150:
            _clean_iterative(bx, by, 19)
        if min(nx, ny) > 100:
            _clean_iterative(bx, by, 9)

    _clean_iterative(bx, by, 5)
    _clean_iterative(bx, by, 3, use_median=True)

    if not silent:
        print(f"SFQ cleaning complete in {time.time() - t0:.2f} seconds")
    return bx, by


def sfq_frame(mag, mode=False, silent=False, acute=False):
    """Full-frame potential field computation and disambiguation.

    Args:
        mag: Magnetogram structure.
        mode: If True, use SOLIS-optimized cleaning.
        silent: Suppress messages.
        acute: Use acute-angle disambiguation.

    Returns:
        mag with disambiguated t1, t2.
    """
    t0 = time.time()
    if not silent:
        print("Starting precise potential field calculating")

    pot = pot_vmag(mag, simple=True)
    if not silent:
        print(f"Potential field calculation complete in {time.time() - t0:.2f} seconds")

    mag = sfq_step1(mag, pot, silent=silent, acute=acute)

    # Add border padding to reduce edge artifacts
    by = mag['t1'].copy()
    bz = mag['t2'].copy()
    border = 10
    ny, nx = by.shape
    by_padded = np.zeros((ny + 2 * border, nx + 2 * border), dtype=float)
    bz_padded = np.zeros((ny + 2 * border, nx + 2 * border), dtype=float)
    by_padded[border:border + ny, border:border + nx] = by
    bz_padded[border:border + ny, border:border + nx] = bz

    by_padded, bz_padded = sfq_clean(by_padded, bz_padded, mode=mode, silent=silent)

    mag['t1'] = by_padded[border:border + ny, border:border + nx]
    mag['t2'] = bz_padded[border:border + ny, border:border + nx]
    return mag


def sfq_disambig(bx, by, bz, apos, rsun, mode=False, silent=False, acute=False):
    """Main entry point: resolves 180-degree azimuth ambiguity.

    Args:
        bx: Ambiguous Bx transverse component (2D array, modified in place).
        by: Ambiguous By transverse component (2D array, modified in place).
        bz: LOS component of magnetic field (2D array).
        apos: Position [x_min, y_min, x_max, y_max] in arcsec.
        rsun: Solar radius in arcsec.
        mode: If True, use SOLIS-optimized parameters.
        silent: Suppress status messages.
        acute: Use acute-angle disambiguation only.

    Returns:
        Tuple (bx_disambig, by_disambig) - disambiguated transverse components.
    """
    t0 = time.time()
    mag = get_str_mag(bx, by, bz, apos, rsun)

    n_pixels = bx.size
    field_extent = apos[2] - apos[0]

    # For small fields: full-frame processing
    if field_extent < 0.5 * rsun or n_pixels < 1024 * 1024:
        mag = sfq_frame(mag, mode=mode, silent=silent, acute=acute)
        if not silent:
            print(f"Full SFQ disambiguation complete in {time.time() - t0:.2f} seconds")
        return mag['t1'].copy(), mag['t2'].copy()

    # For large fields: use full-frame with reduced grid (simplified block processing)
    if not silent:
        print("Large field of view detected, using direct potential field computation")
    mag = sfq_frame(mag, mode=mode, silent=silent, acute=acute)
    if not silent:
        print(f"Full SFQ disambiguation complete in {time.time() - t0:.2f} seconds")
    return mag['t1'].copy(), mag['t2'].copy()
