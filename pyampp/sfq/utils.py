"""Coordinate transformations, grid generation, B-spline interpolation, and utilities."""

import numpy as np
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter


def u_grid(r, ll, n, half=False):
    """Generate uniform 2D grid.

    Args:
        r: Array-like [x_start, y_start]
        ll: Array-like [x_length, y_length]
        n: Array-like [nx, ny]
        half: If True, shift grid by half-pixel at boundaries.

    Returns:
        dict with 'x' and 'y' 2D meshgrid arrays.
    """
    r = np.asarray(r, dtype=float)
    ll = np.asarray(ll, dtype=float)
    n = np.asarray(n, dtype=int)

    x = np.linspace(r[0], r[0] + ll[0], n[0])
    y = np.linspace(r[1], r[1] + ll[1], n[1])

    xx, yy = np.meshgrid(x, y, indexing='ij')
    return {'x': xx, 'y': yy}


def gaussf(n, sigma=3.0):
    """Generate 1D Gaussian kernel.

    Args:
        n: Number of points.
        sigma: Standard deviation.

    Returns:
        Normalized 1D Gaussian array.
    """
    x = np.arange(n) - 0.5 * (n - 1)
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def b_spline_2d(x, y, arr, pos=None):
    """2D cubic B-spline interpolation.

    Args:
        x, y: Query point coordinates (flattened arrays).
        arr: 2D input array (nx, ny).
        pos: [xmin, ymin, xmax, ymax]. Defaults to array bounds.

    Returns:
        Interpolated values at query points.
    """
    nx, ny = arr.shape
    if pos is None:
        pos = [0.0, 0.0, nx - 1.0, ny - 1.0]

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    xi = (x - pos[0]) / (pos[2] - pos[0]) * (nx - 1)
    yi = (y - pos[1]) / (pos[3] - pos[1]) * (ny - 1)

    ki = np.round(xi).astype(int)
    nu = np.round(yi).astype(int)

    z = xi - ki
    v = yi - nu

    zsqm = (z - 0.5) ** 2
    zsqp = (z + 0.5) ** 2
    zsq0 = -2 * z ** 2 + 1.5
    vsqm = (v - 0.5) ** 2
    vsqp = (v + 0.5) ** 2
    vsq0 = -2 * v ** 2 + 1.5

    h = _b_spl_k_2d(arr)

    def _idx(a, i, j):
        i = np.clip(i, 0, a.shape[0] - 1)
        j = np.clip(j, 0, a.shape[1] - 1)
        return a[i, j]

    result = (
        _idx(h, ki, nu) * zsqm * vsqm
        + _idx(h, ki, nu + 1) * zsqm * vsq0
        + _idx(h, ki, nu + 2) * zsqm * vsqp
        + _idx(h, ki + 1, nu) * zsq0 * vsqm
        + _idx(h, ki + 1, nu + 1) * zsq0 * vsq0
        + _idx(h, ki + 1, nu + 2) * zsq0 * vsqp
        + _idx(h, ki + 2, nu) * zsqp * vsqm
        + _idx(h, ki + 2, nu + 1) * zsqp * vsq0
        + _idx(h, ki + 2, nu + 2) * zsqp * vsqp
    )
    return result


def _b_spl_k_2d(a):
    """Compute 2D B-spline kernel coefficients.

    Expects array shape (nx, ny).
    """
    nx, ny = a.shape
    n = [nx - 1, ny - 1]
    f = np.zeros((n[0] + 5, n[1] + 5), dtype=float)
    f[2:2 + nx, 2:2 + ny] = a

    for dim, nn in enumerate(n):
        if dim == 0:
            ta = (f[4, :] - 2 * f[3, :] + f[2, :]) / 2
            tb = f[3, :] - f[2, :] - ta
            f[0, :] = 4 * ta - 2 * tb + f[2, :]
            f[1, :] = ta - tb + f[2, :]
            ta = (f[nn, :] - 2 * f[nn + 1, :] + f[nn + 2, :]) / 2
            tb = -f[nn + 1, :] + f[nn + 2, :] + ta
            f[nn + 4, :] = 4 * ta + 2 * tb + f[nn + 2, :]
            f[nn + 3, :] = ta + tb + f[nn + 2, :]
        else:
            ta = (f[:, 4] - 2 * f[:, 3] + f[:, 2]) / 2
            tb = f[:, 3] - f[:, 2] - ta
            f[:, 0] = 4 * ta - 2 * tb + f[:, 2]
            f[:, 1] = ta - tb + f[:, 2]
            ta = (f[:, nn] - 2 * f[:, nn + 1] + f[:, nn + 2]) / 2
            tb = -f[:, nn + 1] + f[:, nn + 2] + ta
            f[:, nn + 4] = 4 * ta + 2 * tb + f[:, nn + 2]
            f[:, nn + 3] = ta + tb + f[:, nn + 2]

    h = (
        f[0:n[0] + 3, 0:n[1] + 3]
        + f[2:n[0] + 5, 0:n[1] + 3]
        + f[0:n[0] + 3, 2:n[1] + 5]
        + f[2:n[0] + 5, 2:n[1] + 5]
        - 10 * (
            f[0:n[0] + 3, 1:n[1] + 4]
            + f[1:n[0] + 4, 0:n[1] + 3]
            + f[2:n[0] + 5, 1:n[1] + 4]
            + f[1:n[0] + 4, 2:n[1] + 5]
        )
        + 100 * f[1:n[0] + 4, 1:n[1] + 4]
    )
    return h / 256.0


def smooth_2d(arr, radius):
    """Box-car smoothing with edge truncation."""
    size = int(radius) * 2 + 1
    return uniform_filter(arr.astype(float), size=size, mode='constant', cval=0.0)


def median_2d(arr, radius):
    """Median filter."""
    size = int(radius) * 2 + 1
    return median_filter(arr.astype(float), size=size, mode='constant', cval=0.0)


def gauss_smooth(arr, sigma):
    """Gaussian smoothing."""
    return gaussian_filter(arr.astype(float), sigma=sigma, mode='constant', cval=0.0)


def stoem(th, ph, p):
    """Create orthonormal triad for solar coordinate transformations.

    Args:
        th: Latitude of solar center (B0) in radians.
        ph: Longitude of solar center (L0) in radians.
        p: Position angle (P0) in radians.

    Returns:
        (ex, ey, ez) each a 3-element unit vector.
    """
    cp = np.cos(p)
    sp = np.sin(p)
    cth = np.cos(np.pi / 2 - th)
    sph = np.sin(ph)
    cph = np.cos(ph)
    sth = np.sin(np.pi / 2 - th)

    ex = np.array([sth * cph, sth * sph, cth], dtype=float)
    zx = np.array([-ex[2] * ex[0], -ex[2] * ex[1], 1.0 - ex[2] ** 2], dtype=float)
    r = np.linalg.norm(zx)
    if r > 0:
        zx /= r
    if abs(th - np.pi / 2) <= 1e-6:
        zx = np.array([0.0, -1.0, 0.0])
    if abs(th + np.pi / 2) <= 1e-6:
        zx = np.array([0.0, 1.0, 0.0])

    zy = np.array([
        zx[1] * ex[2] - zx[2] * ex[1],
        zx[2] * ex[0] - zx[0] * ex[2],
        zx[0] * ex[1] - zx[1] * ex[0],
    ], dtype=float)

    ey = zx * sp + zy * cp
    ez = zx * cp - zy * sp

    ex /= np.linalg.norm(ex)
    ey /= np.linalg.norm(ey)
    ez /= np.linalg.norm(ez)
    return ex, ey, ez


def qs_crd(x0, x1, x2, l0, b0, p0, inversion=False, to_sph=False):
    """Quick solar coordinate transformation.

    Args:
        x0, x1, x2: Input coordinates.
        l0, b0, p0: Solar center longitude, latitude, position angle (degrees).
        inversion: If True, inverse transformation.
        to_sph: If True, return spherical coordinates (r, theta, phi).

    Returns:
        dict with x0, x1, x2 (and t0, t1, t2 if field vectors provided).
    """
    p0_rad = np.radians(p0)
    b0_rad = np.radians(b0)
    l0_rad = np.radians(l0)

    ex, ey, ez = stoem(b0_rad, l0_rad, -p0_rad)

    if inversion:
        xi0 = x0 * ex[0] + x1 * ey[0] + x2 * ez[0]
        xi1 = x0 * ex[1] + x1 * ey[1] + x2 * ez[1]
        xi2 = x0 * ex[2] + x1 * ey[2] + x2 * ez[2]
    else:
        xi0 = x0 * ex[0] + x1 * ex[1] + x2 * ex[2]
        xi1 = x0 * ey[0] + x1 * ey[1] + x2 * ey[2]
        xi2 = x0 * ez[0] + x1 * ez[1] + x2 * ez[2]

    if to_sph:
        r = np.sqrt(xi0 ** 2 + xi1 ** 2 + xi2 ** 2)
        theta = np.arccos(np.clip(xi2 / np.where(r > 0, r, 1), -1, 1))
        phi = np.arctan2(xi1, xi0)
        return {'x0': r, 'x1': theta, 'x2': phi}

    return {'x0': xi0, 'x1': xi1, 'x2': xi2}


def m0_to_arc(yi, zi, xi=None, radius=959.63, inv=False):
    """Convert between disk-center coordinates and 3D Cartesian on solar sphere.

    Args:
        yi, zi: Tangent-plane coordinates (arcsec) or Cartesian y, z.
        xi: Optional Cartesian x (for forward mode).
        radius: Solar radius in arcsec.
        inv: If True, inverse transform (arcsec -> Cartesian).

    Returns:
        dict with x0, x1, x2, indin, indout, visible, unvisible.
    """
    alf = np.radians(radius / 3600.0)
    dsun = 1.0 / np.sin(alf)

    if inv:
        xarcs = np.asarray(yi, dtype=float)
        yarcs = np.asarray(zi, dtype=float)
        tx = np.tan(np.radians(xarcs / 3600.0))
        ty = np.tan(np.radians(yarcs / 3600.0))
        t = tx ** 2 + ty ** 2
        a = dsun * t / (t + 1)
        b = (dsun ** 2 * t - 1) / (t + 1)
        det = a ** 2 - b

        x = np.zeros_like(xarcs)
        indin = det >= 0
        indout = ~indin
        x[indin] = a[indin] + np.sqrt(det[indin])
        if np.any(indout):
            t_out = np.sqrt(tx[indout] ** 2 + ty[indout] ** 2)
            x[indout] = np.tan(alf) * dsun * t_out / (np.tan(alf) * t_out + 1)

        y = tx * (dsun - x)
        z = ty * (dsun - x)

        betta = np.arccos(np.clip(x / np.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-30), -1, 1))
        visible = (t > 1 / (dsun ** 2 - 1)) | (betta <= np.pi / 2 - alf)
        unvisible = (t <= 1 / (dsun ** 2 - 1)) & (betta > np.pi / 2 - alf)

        return {
            'x0': x, 'x1': y, 'x2': z,
            'indin': indin, 'indout': indout,
            'visible': visible, 'unvisible': unvisible,
        }

    y = np.asarray(yi, dtype=float)
    z = np.asarray(zi, dtype=float)
    ro0 = np.cos(alf)
    ro1 = np.sqrt(y ** 2 + z ** 2)
    indin = (y ** 2 + z ** 2) <= ro0 ** 2
    indout = ~indin

    if xi is not None:
        x = np.asarray(xi, dtype=float)
    else:
        x = np.zeros_like(y)
        x[indin] = np.sqrt(np.maximum(0, 1 - ro1[indin] ** 2))
        if np.any(indout):
            x[indout] = ro1[indout] * np.tan(alf)

    xarcs = np.degrees(np.arctan2(y, dsun - x)) * 3600
    yarcs = np.degrees(np.arctan2(z, dsun - x)) * 3600

    t = np.tan(np.radians(xarcs / 3600.0)) ** 2 + np.tan(np.radians(yarcs / 3600.0)) ** 2
    betta = np.arccos(np.clip(x / np.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-30), -1, 1))
    visible = (t > 1 / (dsun ** 2 - 1)) | (betta <= np.pi / 2 - alf)
    unvisible = (t <= 1 / (dsun ** 2 - 1)) & (betta > np.pi / 2 - alf)

    return {
        'x0': x, 'x1': xarcs, 'x2': yarcs,
        'indin': indin, 'indout': indout,
        'visible': visible, 'unvisible': unvisible,
    }
