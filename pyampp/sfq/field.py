"""Potential field solver for solar magnetic field extrapolation.

Geometry-aware ``pot_vmag`` ported from IDL SFQ (Rudenko / Anfinogentov):
``SOL_crd`` / ``a_field`` / ``get_fftplane`` / ``_Lfff_fft_n``.
"""

from __future__ import annotations

import numpy as np

from .utils import b_spline_2d, qs_crd, sol_crd, stoem, u_grid


def _as_flat_mask(indin, shape):
    """Normalize IDL WHERE indices or a boolean mask to a boolean array."""
    indin = np.asarray(indin)
    if indin.dtype == bool:
        return indin.reshape(shape)
    mask = np.zeros(shape, dtype=bool).reshape(-1, order="F")
    flat = np.asarray(indin, dtype=np.int64)
    valid = (flat >= 0) & (flat < mask.size)
    mask[flat[valid]] = True
    return mask.reshape(shape, order="F")


def _idl_fft_k(n, dtype=float):
    """IDL-style FFT mode indices (Nyquist positive for even ``n``)."""
    e = dtype.type(1) if isinstance(dtype, np.dtype) else dtype(1)
    n = int(n)
    if n % 2 == 0:
        km = e + np.arange((n - 2) // 2, dtype=dtype)
        return np.concatenate([np.array([0], dtype=dtype), km, np.array([n / 2.0], dtype=dtype), -km[::-1]])
    km = e + np.arange((n - 1) // 2, dtype=dtype)
    return np.concatenate([np.array([0], dtype=dtype), km, -km[::-1]])


class _LfffFftN:
    """Stateful port of IDL ``_Lfff_fft_n`` / ``Lfff_fft`` (simple=Neumann)."""

    def __init__(self):
        self.pq = None
        self.hx = self.hy = self.hz = None
        self.alfa = 0.0
        self.pos = None
        self.spos = None
        self.bpos = None

    def configure(self, bl, spos, bpos, alfa=0.0):
        """IDL ``Lfff_fft(set=..., /simple)`` setup."""
        bl = np.asarray(bl)
        self.spos = dict(spos)
        hs = np.asarray(bpos["hs"], dtype=float) * np.array([1.0, 1.0])
        self.bpos = {"l": float(bpos["l"]), "b": float(bpos["b"]), "hs": hs}
        self.alfa = float(alfa)

        pi_ = np.pi
        rl = float(spos["l"]) * pi_ / 180.0
        rb = float(spos["b"]) * pi_ / 180.0
        rp = float(spos["p"]) * pi_ / 180.0
        rbl = float(bpos["l"]) * pi_ / 180.0
        rbb = float(bpos["b"]) * pi_ / 180.0

        ex, ey, ez = stoem(rb, rl, -rp)
        exb, eyb, ezb = stoem(rbb, rbl, 0.0)
        # Unused ey, ez kept for IDL parity of the triad construction.
        _ = (ey, ez)
        hx = float(np.dot(ex, eyb))
        hy = float(np.dot(ex, ezb))
        hz = float(np.dot(ex, exb))
        pos = np.array([-hs[0], -hs[1], hs[0], hs[1]], dtype=float) * pi_ / 180.0
        self.pq = bl
        self.hx, self.hy, self.hz = hx, hy, hz
        self.pos = pos
        self.alfa = float(alfa)
        return np.array([hx, hy, hz], dtype=float)

    def evaluate(self, z=0.0):
        """IDL ``_Lfff_fft_n(z)`` / ``get_fftplane`` evaluation at height ``z``."""
        if self.pq is None:
            raise RuntimeError("LFFF FFT state not configured")
        pq = np.asarray(self.pq)
        dtype = pq.dtype if pq.dtype.kind == "f" else np.float64
        pi_ = np.pi
        alfa = dtype.type(self.alfa)
        hx, hy, hz = dtype.type(self.hx), dtype.type(self.hy), dtype.type(self.hz)
        pos = np.asarray(self.pos, dtype=dtype)
        hs = (pos[2:4] - pos[0:2]) / 2

        # Image layout (ny, nx): x/ph along axis 1, y/th along axis 0.
        n_y, n_x = pq.shape
        kn = _idl_fft_k(n_x, dtype)  # x / ph
        km = _idl_fft_k(n_y, dtype)  # y / th
        kx = (pi_ * kn)[None, :] / hs[0]
        ky = (pi_ * km)[:, None] / hs[1]

        if alfa == 0:
            q = np.sqrt(kx ** 2 + ky ** 2)
            bmnx = -1j * kx
            bmny = -1j * ky
        else:
            q = np.sqrt((kx ** 2 + ky ** 2 - alfa ** 2).astype(np.complex128))
            bmnx = -1j * kx + 1j * alfa * (alfa * q + kx * ky) / (q * kx + alfa * ky)
            bmny = -1j * ky + 1j * alfa * (alfa * q - kx * ky) / (q * ky - alfa * kx)
            bad = (kx ** 2 + ky ** 2 - alfa ** 2) <= 0
            bmnx = np.where(bad, 0, bmnx)
            bmny = np.where(bad, 0, bmny)

        ind = (kx ** 2 + ky ** 2 - alfa ** 2) > 0
        bmnh = bmnx * hx + bmny * hy + q * hz
        fq = np.fft.fft2(pq)
        cmn = np.zeros_like(fq, dtype=np.complex128)
        cmn[ind] = fq[ind] / bmnh[ind]

        z = dtype.type(z)
        ez = np.zeros_like(q, dtype=dtype)
        ez[ind] = np.exp(-np.asarray(q[ind], dtype=dtype) * z)

        bx = np.fft.ifft2(cmn * bmnx * ez).real.astype(dtype, copy=False)
        by = np.fft.ifft2(cmn * bmny * ez).real.astype(dtype, copy=False)
        bz = np.fft.ifft2(cmn * q * ez).real.astype(dtype, copy=False)
        bl = bx * hx + by * hy + bz * hz
        return {
            "t0": bx,
            "t1": by,
            "t2": bz,
            "bl": bl,
            "alf": alfa,
            "pos": pos,
            "z": z,
            "data": pq,
            "spos": self.spos,
            "bpos": self.bpos,
        }


def _b_spl_reshape(x, y, arr, pos):
    x = np.asarray(x)
    y = np.asarray(y)
    vals = b_spline_2d(x, y, arr, pos=pos)
    return vals.reshape(x.shape)


def a_field(x0, x1, x2, set_cfg):
    """Port of IDL ``a_field`` (B-spline remap + optional vector-frame rotate)."""
    s = set_cfg
    vec = int(s.get("vec", 1))
    nt = 1
    if "t1" in s["a"]:
        nt += 1
    if "t2" in s["a"]:
        nt += 1
    if nt != 3:
        vec = 0

    a = {"x0": x0, "x1": x1, "x2": x2, "qs": s["b"]["qs"]}
    b = {"qs": s["a"]["qs"]}
    u = sol_crd(a, b, crd=True)

    pos = np.asarray(s["a"]["pos"], dtype=float)
    t0 = np.zeros_like(np.asarray(x0, dtype=float))
    t1 = np.zeros_like(t0) if nt > 1 else None
    t2 = np.zeros_like(t0) if nt > 2 else None

    src_type = str(s["a"]["qs"]["type"]).lower()
    if pos.size == 4:
        if src_type == "sbox":
            ind = ((u["x0"] - pos[0]) * (u["x0"] - pos[2]) <= 0.0) & (
                (u["x1"] - pos[1]) * (u["x1"] - pos[3]) <= 0.0
            )
        else:
            ind = ((u["x1"] - pos[0]) * (u["x1"] - pos[2]) <= 0.0) & (
                (u["x2"] - pos[1]) * (u["x2"] - pos[3]) <= 0.0
            )
        if "visible" in u:
            vis = np.asarray(u["visible"], dtype=bool)
            if vis.shape == ind.shape:
                ind = ind & vis

        if not np.any(ind):
            return {"err": 1}

        if src_type in ("box", "sbox"):
            qx, qy = u["x0"][ind], u["x1"][ind]
        else:
            qx, qy = u["x1"][ind], u["x2"][ind]

        t0[ind] = _b_spl_reshape(qx, qy, np.asarray(s["a"]["t0"], dtype=float), pos).ravel()
        if nt > 1:
            t1[ind] = _b_spl_reshape(qx, qy, np.asarray(s["a"]["t1"], dtype=float), pos).ravel()
        if nt > 2:
            t2[ind] = _b_spl_reshape(qx, qy, np.asarray(s["a"]["t2"], dtype=float), pos).ravel()

        if vec:
            a1 = {
                "x0": np.asarray(x0)[ind],
                "x1": np.asarray(x1)[ind],
                "x2": np.asarray(x2)[ind],
                "t0": t0[ind],
                "t1": t1[ind],
                "t2": t2[ind],
                "qs": s["b"]["qs"],
                "ts": s["a"]["ts"],
            }
            b1 = {"qs": s["b"]["qs"], "ts": s["b"]["ts"]}
            u1 = sol_crd(a1, b1, fld=True)
            t0[ind] = u1["t0"]
            t1[ind] = u1["t1"]
            t2[ind] = u1["t2"]

        out = {"x0": x0, "x1": x1, "x2": x2, "t0": t0, "qs": s["b"]["qs"], "err": 0}
        if nt > 1:
            out["t1"] = t1
        if nt > 2:
            out["t2"] = t2
        if vec and nt == 3:
            out["ts"] = s["b"]["ts"]
        return out

    return {"err": 1}


def get_fftplane(pos=None, n=None, z=0.0, set_cfg=None, simple=True, solver=None):
    """Port of IDL ``get_fftplane`` (configure or sample the LFFF plane)."""
    if solver is None:
        raise ValueError("solver (_LfffFftN instance) is required")
    if set_cfg is not None:
        solver.configure(
            set_cfg["bl"],
            set_cfg["spos"],
            set_cfg["bpos"],
            alfa=set_cfg.get("alfa", 0.0),
        )
        return 1

    pos = np.asarray(pos, dtype=float)
    n = np.asarray(n, dtype=int)
    gr = u_grid(pos[0:2], pos[2:4] - pos[0:2], n)
    a = solver.evaluate(z)
    bx = _b_spl_reshape(gr["x"], gr["y"], a["t0"], a["pos"])
    by = _b_spl_reshape(gr["x"], gr["y"], a["t1"], a["pos"])
    bz = _b_spl_reshape(gr["x"], gr["y"], a["t2"], a["pos"])
    bl = _b_spl_reshape(gr["x"], gr["y"], a["bl"], a["pos"])
    bpos = dict(a["bpos"])
    bpos["hs"] = (pos[2:4] - pos[0:2]) / 2 * 180.0 / np.pi
    return {
        "x0": gr["x"],
        "x1": gr["y"],
        "t0": bx,
        "t1": by,
        "t2": bz,
        "bl": bl,
        "pos": pos,
        "z": a["z"],
        "spos": a["spos"],
        "bpos": bpos,
    }


def pot_vmag(mag, simple=True):
    """Compute potential field from observed LOS/Bz using IDL geometry-aware SFQ path.

    Args:
        mag: Magnetogram structure from ``get_str_mag``.
        simple: If True (SFQ default), use Neumann ``_Lfff_fft_n`` solver.

    Returns:
        Mag structure with potential ``t0/t1/t2`` components.
    """
    if not simple:
        # Full odd-extension ``_Lfff_fft`` is unused by SFQ ``/simple``.
        # Both flags use the Neumann path so callers stay IDL-SFQ compatible.
        pass

    l0 = b0 = p0 = 0.0
    rad = float(mag["qs"]["r"])
    alpha = 0.0

    mask = _as_flat_mask(mag["indin"], np.asarray(mag["x0"]).shape)
    ua = {
        "x0": mag["x0"][mask],
        "x1": mag["x1"][mask],
        "x2": mag["x2"][mask],
        "qs": {"l": l0, "b": b0, "p": p0, "r": rad, "type": "arc"},
    }
    ub = {"qs": {"l": l0, "b": b0, "p": p0, "r": rad, "type": "dec"}}
    u = sol_crd(ua, ub, crd=True)

    xc = float(np.mean(u["x0"]))
    yc = float(np.mean(u["x1"]))
    zc = float(np.mean(u["x2"]))
    ad = qs_crd(xc, yc, zc, l0, b0, p0, inversion=True, to_sph=True)
    lc = float(ad["x2"] * 180.0 / np.pi)
    bc = float(90.0 - ad["x1"] * 180.0 / np.pi)

    ad = qs_crd(u["x0"], u["x1"], u["x2"], l0, b0, p0, inversion=True, to_sph=False)
    as_ = qs_crd(ad["x0"], ad["x1"], ad["x2"], lc, bc, 0.0, to_sph=True)
    lon = as_["x2"] * 180.0 / np.pi
    lat = 90.0 - as_["x1"] * 180.0 / np.pi
    hs0 = np.array([np.max(np.abs(lon)), np.max(np.abs(lat))], dtype=float)

    rm = np.sqrt(u["x1"] ** 2 + u["x2"] ** 2)
    srt = np.argsort(rm)[:2]
    p0v = np.array([u["x0"][srt[0]], u["x1"][srt[0]], u["x2"][srt[0]]], dtype=float)
    p1v = np.array([u["x0"][srt[1]], u["x1"][srt[1]], u["x2"][srt[1]]], dtype=float)
    rob0 = np.linalg.norm(p0v)
    rob1 = np.linalg.norm(p1v)
    pixs = float(np.dot(p0v, p1v) / (rob0 * rob1))
    pixs = float(np.arccos(np.clip(pixs, -1.0, 1.0)))

    nn = np.fix(2 * hs0 * np.pi / 180.0 / pixs)
    nn = np.maximum(nn, 5)
    hs = (nn - 1) * pixs * 180.0 / np.pi / 2.0
    bpos = {"l": lc, "b": bc, "hs": hs, "n": np.concatenate([nn, [nn.min() / 2.0]])}
    n = np.asarray(bpos["n"][:2], dtype=int)
    hs = np.asarray(bpos["hs"], dtype=float) * np.array([1.0, 1.0])

    grb = u_grid([-hs[0], -hs[1]], 2 * hs, n)
    ph = grb["x"] * np.pi / 180.0
    th = grb["y"] * np.pi / 180.0
    r = np.zeros_like(ph)

    qs_arc = {"l": l0, "b": b0, "p": p0, "r": rad, "type": "arc"}
    qs_sbox = {"l": bpos["l"], "b": bpos["b"], "p": 0.0, "type": "sbox"}
    sa = {
        "a": {
            "x0": mag["x0"],
            "x1": mag["x1"],
            "x2": mag["x2"],
            "t0": mag["t0"],
            "qs": qs_arc,
            "pos": np.asarray(mag["pos"], dtype=float),
        },
        "b": {"qs": qs_sbox},
        "proc": "b_spl",
        "vec": 0,
    }
    q = a_field(ph, th, r, sa)
    if q.get("err", 0):
        return 0

    solver = _LfffFftN()
    set_cfg = {
        "bl": q["t0"],
        "spos": {"l": l0, "b": b0, "p": p0},
        "bpos": {"l": bpos["l"], "b": bpos["b"], "hs": hs},
        "alfa": alpha,
    }
    get_fftplane(set_cfg=set_cfg, simple=True, solver=solver)
    posp = np.array([-hs[0], -hs[1], hs[0], hs[1]], dtype=float) * np.pi / 180.0
    uu = get_fftplane(pos=posp, n=n, z=0.0, simple=True, solver=solver)

    sa2 = {
        "a": {
            "x0": uu["x0"],
            "x1": uu["x1"],
            "x2": np.zeros_like(uu["x0"]),
            "t0": uu["t0"],
            "t1": uu["t1"],
            "t2": uu["t2"],
            "qs": qs_sbox,
            "ts": qs_sbox,
            "pos": posp,
        },
        "b": {"qs": qs_arc, "ts": qs_arc},
        "proc": "b_spl",
        "vec": 1,
    }
    q1 = a_field(mag["x0"], mag["x1"], mag["x2"], sa2)
    if q1.get("err", 0):
        return 0

    result = dict(mag)
    result["t0"] = q1["t0"]
    result["t1"] = q1["t1"]
    result["t2"] = q1["t2"]
    result["type"] = "vecp " + str(mag.get("type", ""))
    return result
