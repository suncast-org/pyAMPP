# SFQ in pyAMPP

Python SFQ (Super Fast and Quality) azimuth disambiguation, vendored from
[Sergey-Anfinogentov/SFQ](https://github.com/Sergey-Anfinogentov/SFQ)
(Python port by vit1-irk, merged upstream 2026-06-11). This is the same
tree gx_simulator vendors as its `sfq` submodule.

## Citation

Rudenko, G. V. & Anfinogentov, S. A. (2014), Solar Physics, 289, 1499–1516.

## Entry point

```python
from pyampp.sfq import sfq_disambig

bx_out, by_out = sfq_disambig(bx, by, bz, pos, rsun, mode=True)  # mode=True ≈ IDL /hmi
```

`gx_fov2box --sfq` uses this path on an FOV crop of ambiguous HMI azimuth
(see issue [#42](https://github.com/suncast-org/pyAMPP/issues/42)).

## Potential field

``pot_vmag`` is the IDL geometry-aware path (local sbox remap + Neumann
LFFF FFT), not a planar FFT from Bz alone. Arrays use image layout
(axis0=y, axis1=x), matching FITS/`scipy.io.readsav` inputs.
