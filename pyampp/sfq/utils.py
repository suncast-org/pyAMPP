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

    # Image / scipy.io.readsav layout: axis0=y, axis1=x (IDL arrays arrive
    # transposed into NumPy). Matches FITS [row, col] used by sfq_disambig.
    xx, yy = np.meshgrid(x, y, indexing='xy')
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
    # Image layout (ny, nx): x along axis 1, y along axis 0.
    ny, nx = arr.shape
    if pos is None:
        pos = [0.0, 0.0, nx - 1.0, ny - 1.0]

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    xi = (x - pos[0]) / (pos[2] - pos[0]) * (nx - 1)
    yi = (y - pos[1]) / (pos[3] - pos[1]) * (ny - 1)

    # IDL long(xi+0.5) is floor(xi+0.5) for finite values.
    ki = np.floor(xi + 0.5).astype(int)
    nu = np.floor(yi + 0.5).astype(int)

    z = xi - ki
    v = yi - nu

    zsqm = (z - 0.5) ** 2
    zsqp = (z + 0.5) ** 2
    zsq0 = -2 * z ** 2 + 1.5
    vsqm = (v - 0.5) ** 2
    vsqp = (v + 0.5) ** 2
    vsq0 = -2 * v ** 2 + 1.5

    h = _b_spl_k_2d(arr)

    def _idx(a, iy, ix):
        iy = np.clip(iy, 0, a.shape[0] - 1)
        ix = np.clip(ix, 0, a.shape[1] - 1)
        return a[iy, ix]

    # h indices: first = y (nu), second = x (ki) — image layout.
    result = (
        _idx(h, nu, ki) * zsqm * vsqm
        + _idx(h, nu + 1, ki) * zsqm * vsq0
        + _idx(h, nu + 2, ki) * zsqm * vsqp
        + _idx(h, nu, ki + 1) * zsq0 * vsqm
        + _idx(h, nu + 1, ki + 1) * zsq0 * vsq0
        + _idx(h, nu + 2, ki + 1) * zsq0 * vsqp
        + _idx(h, nu, ki + 2) * zsqp * vsqm
        + _idx(h, nu + 1, ki + 2) * zsqp * vsq0
        + _idx(h, nu + 2, ki + 2) * zsqp * vsqp
    )
    return result


def _b_spl_k_2d(a):
    """Compute 2D B-spline kernel coefficients for image-layout ``(ny, nx)``."""
    # Work in IDL-native (nx, ny) for the kernel algebra, then return image layout.
    a_idl = np.asarray(a, dtype=float).T
    nx, ny = a_idl.shape
    n = [nx - 1, ny - 1]
    f = np.zeros((n[0] + 5, n[1] + 5), dtype=float)
    f[2:2 + nx, 2:2 + ny] = a_idl

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
    return (h / 256.0).T


def smooth_2d(arr, width):
    """Box-car smoothing matching IDL ``smooth(arr, width, /edge_truncate)``.

    ``width`` is the IDL neighborhood size (not a radius).
    """
    size = max(int(width), 1)
    # IDL /edge_truncate copies edge values; nearest is the closest ndimage mode.
    return uniform_filter(arr.astype(float), size=size, mode="nearest")


def median_2d(arr, width):
    """Median filter matching IDL ``median(arr, width, /even)``.

    ``width`` is the IDL neighborhood size (not a radius).
    """
    size = max(int(width), 1)
    return median_filter(arr.astype(float), size=size, mode="nearest")


def gauss_smooth(arr, sigma):
    """Gaussian smoothing."""
    return gaussian_filter(arr.astype(float), sigma=sigma, mode="nearest")



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


def _qs_type(qs):
    return str(qs.get('type', 'cdec')).lower()


def _qs_p(qs):
    return float(qs.get('p', 0.0))


def sol_crd(a, b=None, crd=False, fld=False):
    """Port of IDL ``SOL_crd`` for SFQ coordinate / field-frame transforms.

    Supports the frame types used by ``pot_vmag``: ``arc``, ``dec``, ``sbox``,
    and ``cdec``. Structures are plain dicts with keys matching the IDL tags.
    """
    if not fld and not crd:
        fld = True
        crd = True

    aqs = a.get('qs', {'l': 0.0, 'b': 0.0, 'p': 0.0, 'type': 'cdec'})
    if b is None:
        b = {
            'qs': {'l': 0.0, 'b': 0.0, 'p': 0.0, 'type': 'cdec'},
            'ts': {'l': 0.0, 'b': 0.0, 'p': 0.0, 'type': 'cdec'},
        }
    bqs = b.get('qs', {'l': 0.0, 'b': 0.0, 'p': 0.0, 'type': 'cdec'})

    # --- source coordinates → intermediate Cartesian (xi) ---
    aqs_type = _qs_type(aqs)
    if aqs_type == 'cdec':
        xi0 = np.asarray(a['x0'], dtype=float)
        xi1 = np.asarray(a['x1'], dtype=float)
        xi2 = np.asarray(a['x2'], dtype=float)
    else:
        e0, e1, e2 = stoem(
            np.radians(float(aqs['b'])),
            np.radians(float(aqs['l'])),
            -np.radians(_qs_p(aqs)),
        )
        if aqs_type == 'dec':
            x0 = np.asarray(a['x0'], dtype=float)
            x1 = np.asarray(a['x1'], dtype=float)
            x2 = np.asarray(a['x2'], dtype=float)
            xi0 = x0 * e0[0] + x1 * e1[0] + x2 * e2[0]
            xi1 = x0 * e0[1] + x1 * e1[1] + x2 * e2[1]
            xi2 = x0 * e0[2] + x1 * e1[2] + x2 * e2[2]
        elif aqs_type == 'sbox':
            ph = np.asarray(a['x0'], dtype=float)
            th = np.pi / 2 - np.asarray(a['x1'], dtype=float)
            r = np.asarray(a['x2'], dtype=float) + 1.0
            x0 = r * np.sin(th) * np.cos(ph)
            x1 = r * np.sin(th) * np.sin(ph)
            x2 = r * np.cos(th)
            xi0 = x0 * e0[0] + x1 * e1[0] + x2 * e2[0]
            xi1 = x0 * e0[1] + x1 * e1[1] + x2 * e2[1]
            xi2 = x0 * e0[2] + x1 * e1[2] + x2 * e2[2]
        elif aqs_type == 'arc':
            rad = float(aqs.get('r', 959.63))
            if 'x0' in a:
                s = m0_to_arc(a['x1'], a['x2'], a['x0'], radius=rad, inv=True)
            else:
                s = m0_to_arc(a['x1'], a['x2'], radius=rad, inv=True)
            xi0 = s['x0'] * e0[0] + s['x1'] * e1[0] + s['x2'] * e2[0]
            xi1 = s['x0'] * e0[1] + s['x1'] * e1[1] + s['x2'] * e2[1]
            xi2 = s['x0'] * e0[2] + s['x1'] * e1[2] + s['x2'] * e2[2]
        else:
            raise NotImplementedError(f"SOL_crd source qs type {aqs_type!r}")

    # --- source field → intermediate Cartesian (ti), if requested ---
    ti0 = ti1 = ti2 = None
    if fld:
        ats = a.get('ts', aqs)
        ats_type = _qs_type(ats)
        if ats_type == 'cdec':
            ti0 = np.asarray(a['t0'], dtype=float)
            ti1 = np.asarray(a['t1'], dtype=float)
            ti2 = np.asarray(a['t2'], dtype=float)
        else:
            e0, e1, e2 = stoem(
                np.radians(float(ats['b'])),
                np.radians(float(ats['l'])),
                -np.radians(_qs_p(ats)),
            )
            if ats_type in ('dec', 'arc'):
                t0 = np.asarray(a['t0'], dtype=float)
                t1 = np.asarray(a['t1'], dtype=float)
                t2 = np.asarray(a['t2'], dtype=float)
                ti0 = t0 * e0[0] + t1 * e1[0] + t2 * e2[0]
                ti1 = t0 * e0[1] + t1 * e1[1] + t2 * e2[1]
                ti2 = t0 * e0[2] + t1 * e1[2] + t2 * e2[2]
            elif ats_type in ('sbox', 'box', 'sph'):
                # Local spherical basis in the source frame, then to Cartesian.
                x0 = xi0 * e0[0] + xi1 * e0[1] + xi2 * e0[2]
                x1 = xi0 * e1[0] + xi1 * e1[1] + xi2 * e1[2]
                x2 = xi0 * e2[0] + xi1 * e2[1] + xi2 * e2[2]
                r = np.sqrt(x0 ** 2 + x1 ** 2 + x2 ** 2)
                r_safe = np.where(r > 0, r, 1.0)
                er0, er1, er2 = x0 / r_safe, x1 / r_safe, x2 / r_safe
                ep0, ep1 = -er1, er0
                et0 = ep1 * er2
                et1 = -ep0 * er2
                et2 = ep0 * er1 - ep1 * er0
                rp = np.sqrt(ep0 ** 2 + ep1 ** 2)
                rp_safe = np.where(rp > 0, rp, 1.0)
                ep0, ep1 = ep0 / rp_safe, ep1 / rp_safe
                rp = np.sqrt(et0 ** 2 + et1 ** 2 + et2 ** 2)
                rp_safe = np.where(rp > 0, rp, 1.0)
                et0, et1, et2 = et0 / rp_safe, et1 / rp_safe, et2 / rp_safe
                t0 = np.asarray(a['t0'], dtype=float)
                t1 = np.asarray(a['t1'], dtype=float)
                t2 = np.asarray(a['t2'], dtype=float)
                if ats_type in ('box', 'sbox'):
                    ti0_ = t2 * er0 - t1 * et0 + t0 * ep0
                    ti1_ = t2 * er1 - t1 * et1 + t0 * ep1
                    ti2_ = t2 * er2 - t1 * et2
                else:
                    ti0_ = t0 * er0 + t1 * et0 + t2 * ep0
                    ti1_ = t0 * er1 + t1 * et1 + t2 * ep1
                    ti2_ = t0 * er2 + t1 * et2
                ti0 = ti0_ * e0[0] + ti1_ * e1[0] + ti2_ * e2[0]
                ti1 = ti0_ * e0[1] + ti1_ * e1[1] + ti2_ * e2[1]
                ti2 = ti0_ * e0[2] + ti1_ * e1[2] + ti2_ * e2[2]
            else:
                raise NotImplementedError(f"SOL_crd source ts type {ats_type!r}")

    # --- intermediate Cartesian → destination coordinates ---
    out_extra = {}
    if crd:
        bqs_type = _qs_type(bqs)
        if bqs_type == 'cdec':
            x0, x1, x2 = xi0, xi1, xi2
        else:
            e0, e1, e2 = stoem(
                np.radians(float(bqs['b'])),
                np.radians(float(bqs['l'])),
                -np.radians(_qs_p(bqs)),
            )
            if bqs_type == 'sbox':
                xa0 = xi0 * e0[0] + xi1 * e0[1] + xi2 * e0[2]
                xa1 = xi0 * e1[0] + xi1 * e1[1] + xi2 * e1[2]
                xa2 = xi0 * e2[0] + xi1 * e2[1] + xi2 * e2[2]
                r = np.sqrt(xa0 ** 2 + xa1 ** 2 + xa2 ** 2)
                r_safe = np.where(r > 0, r, 1.0)
                t = np.arccos(np.clip(xa2 / r_safe, -1.0, 1.0))
                x0 = np.arctan2(xa1, xa0)
                x2 = r - 1.0
                x1 = np.pi / 2 - t
            elif bqs_type == 'dec':
                x0 = xi0 * e0[0] + xi1 * e0[1] + xi2 * e0[2]
                x1 = xi0 * e1[0] + xi1 * e1[1] + xi2 * e1[2]
                x2 = xi0 * e2[0] + xi1 * e2[1] + xi2 * e2[2]
            elif bqs_type == 'arc':
                xa0 = xi0 * e0[0] + xi1 * e0[1] + xi2 * e0[2]
                xa1 = xi0 * e1[0] + xi1 * e1[1] + xi2 * e1[2]
                xa2 = xi0 * e2[0] + xi1 * e2[1] + xi2 * e2[2]
                s = m0_to_arc(xa1, xa2, xa0, radius=float(bqs.get('r', 959.63)))
                x0, x1, x2 = s['x0'], s['x1'], s['x2']
                out_extra = {
                    'indin': s['indin'],
                    'indout': s['indout'],
                    'visible': s['visible'],
                    'unvisible': s['unvisible'],
                }
            else:
                raise NotImplementedError(f"SOL_crd dest qs type {bqs_type!r}")

    # --- intermediate field → destination field components ---
    if fld:
        bts = b.get('ts', bqs)
        bts_type = _qs_type(bts)
        if bts_type == 'cdec':
            t0, t1, t2 = ti0, ti1, ti2
        else:
            e0, e1, e2 = stoem(
                np.radians(float(bts['b'])),
                np.radians(float(bts['l'])),
                -np.radians(_qs_p(bts)),
            )
            if bts_type in ('sbox', 'box', 'sph'):
                xa0 = xi0 * e0[0] + xi1 * e0[1] + xi2 * e0[2]
                xa1 = xi0 * e1[0] + xi1 * e1[1] + xi2 * e1[2]
                xa2 = xi0 * e2[0] + xi1 * e2[1] + xi2 * e2[2]
                r = np.sqrt(xa0 ** 2 + xa1 ** 2 + xa2 ** 2)
                r_safe = np.where(r > 0, r, 1.0)
                er0, er1, er2 = xa0 / r_safe, xa1 / r_safe, xa2 / r_safe
                ep0, ep1 = -er1, er0
                et0 = ep1 * er2
                et1 = -ep0 * er2
                et2 = ep0 * er1 - ep1 * er0
                rp = np.sqrt(ep0 ** 2 + ep1 ** 2)
                rp_safe = np.where(rp > 0, rp, 1.0)
                ep0, ep1 = ep0 / rp_safe, ep1 / rp_safe
                rp = np.sqrt(et0 ** 2 + et1 ** 2 + et2 ** 2)
                rp_safe = np.where(rp > 0, rp, 1.0)
                et0, et1, et2 = et0 / rp_safe, et1 / rp_safe, et2 / rp_safe
                ti0_ = ti0 * e0[0] + ti1 * e0[1] + ti2 * e0[2]
                ti1_ = ti0 * e1[0] + ti1 * e1[1] + ti2 * e1[2]
                ti2_ = ti0 * e2[0] + ti1 * e2[1] + ti2 * e2[2]
                if bts_type in ('box', 'sbox'):
                    t2 = er0 * ti0_ + er1 * ti1_ + er2 * ti2_
                    t1 = -(et0 * ti0_ + et1 * ti1_ + et2 * ti2_)
                    t0 = ep0 * ti0_ + ep1 * ti1_
                else:
                    t0 = er0 * ti0_ + er1 * ti1_ + er2 * ti2_
                    t1 = et0 * ti0_ + et1 * ti1_ + et2 * ti2_
                    t2 = ep0 * ti0_ + ep1 * ti1_
            elif bts_type in ('dec', 'arc'):
                t0 = ti0 * e0[0] + ti1 * e0[1] + ti2 * e0[2]
                t1 = ti0 * e1[0] + ti1 * e1[1] + ti2 * e1[2]
                t2 = ti0 * e2[0] + ti1 * e2[1] + ti2 * e2[2]
            else:
                raise NotImplementedError(f"SOL_crd dest ts type {bts_type!r}")

    out = dict(a)
    out.update(out_extra)
    if crd and fld:
        out['x0'] = x0
        out['x1'] = x1
        out['x2'] = x2
        out['t0'] = t0
        out['t1'] = t1
        out['t2'] = t2
        out['qs'] = bqs
        out['ts'] = b.get('ts', bqs)
        return out
    if crd:
        out['x0'] = x0
        out['x1'] = x1
        out['x2'] = x2
        out['qs'] = bqs
        return out
    out['t0'] = t0
    out['t1'] = t1
    out['t2'] = t2
    out['ts'] = b.get('ts', bqs)
    return out


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
