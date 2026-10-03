"""Data structure building for SFQ."""

import numpy as np

from .utils import u_grid


def get_str_mag(bx, by, bz, apos, rsun):
    """Build magnetogram structure from raw data arrays.

    Args:
        bx: Ambiguous Bx transverse component (2D array).
        by: Ambiguous By transverse component (2D array).
        bz: LOS component of magnetic field (2D array).
        apos: Position [x_min, y_min, x_max, y_max] in arcsec.
        rsun: Solar radius in arcsec.

    Returns:
        dict with magnetic field structure.
    """
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    bz = np.asarray(bz, dtype=float)
    apos = np.asarray(apos, dtype=float)

    ny, nx = bx.shape
    gr = u_grid(apos[:2], apos[2:] - apos[:2], [nx, ny])

    alf = np.radians(rsun / 3600.0)
    dsun = 1.0 / np.sin(alf)
    tx = np.tan(np.radians(gr['x'] / 3600.0))
    ty = np.tan(np.radians(gr['y'] / 3600.0))

    a = dsun * (tx ** 2 + ty ** 2) / (tx ** 2 + ty ** 2 + 1)
    b = (dsun ** 2 * (tx ** 2 + ty ** 2) - 1) / (tx ** 2 + ty ** 2 + 1)
    det = a ** 2 - b

    indin = det >= 0
    indout = ~indin

    x = np.zeros_like(gr['x'])
    x[indin] = a[indin] + np.sqrt(det[indin])
    if np.any(indout):
        t = np.sqrt(tx[indout] ** 2 + ty[indout] ** 2)
        x[indout] = np.tan(alf) * dsun * t / (np.tan(alf) * t + 1)

    visible = np.ones_like(gr['x'], dtype=bool)
    qs = {'l': 0.0, 'b': 0.0, 'p': 0.0, 'r': float(rsun), 'type': 'arc'}

    return {
        'x0': x, 'x1': gr['x'], 'x2': gr['y'],
        't0': bz, 't1': bx, 't2': by,
        'pos': apos, 'qs': qs, 'ts': qs,
        'indin': indin, 'indout': indout,
        'visible': visible, 'unvisible': ~visible,
    }
