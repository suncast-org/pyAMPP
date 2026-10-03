# IDL Parity Notes (2024-05-12 run)

This note summarizes code changes and comparison results performed to align Python outputs with the IDL `gx_fov2box` pipeline.

## Scope
- Target: match IDL base maps (BX/BY/BZ/IC) and chromo mask distribution.
- Data: 2024-05-12 run
  - Python H5: `/Users/gelu/pyampp/gx_models/2024-05-12/hmi.M_720s.20240512_000000.E73S03CR.CEA.NONE.h5`
  - IDL SAV: `/Users/gelu/gx_models/2024-05-12/hmi.M_720s.20240511_235843.E73S3CR.CEA.NONE.sav`

## Code Changes
### Reprojection parity (IDL `/ssaa`)
- Switched `reproject_to` algorithm to `"exact"` (flux‑conserving) across:
  - `pyampp/gxbox/gx_fov2box.py`
  - `pyampp/util/compute.py`
- Removed `roundtrip_coords=False` where it was passed to `reproject_exact`.

### HMI disambiguation parity
- IDL non-SFQ path uses `hmi_disambig, ..., 0`.
- Python non-SFQ path uses `hmi_disambig(..., method=2)` (radial acute; historical pyAMPP default).
- Python `--sfq` skips HMI bits and runs Rudenko/Anfinogentov SFQ on the FOV crop
  (issue #42; toward 1.1.1).

### Component mapping parity
- IDL base convention: `BX=BP`, `BY=-BT`, `BZ=BR`.
- Applied in:
  - `pyampp/gxbox/gx_fov2box.py`

### RSUN parity
- IDL sets `WCS_RSUN=6.96e8` for coordinate transforms.
- Python now preserves `rsun_ref=6.96e8` in `hmi_b2ptr` output maps:
  - `pyampp/gxbox/boxutils.py`
  - `pyampp/util/compute.py`

### CEA origin parity
- IDL CEA uses Carrington lon/lat by default.
- `pyampp/gxbox/box.py`: bottom CEA header now uses `HeliographicCarrington` origin.

### Full‑map remap parity (no pre‑submap)
- IDL remaps from full input maps (not pre‑cropped).
- `pyampp/gxbox/gx_fov2box.py` updated to reproject base maps from full‑resolution maps.

### Decompose NaN handling
- Full‑map reprojection introduced NaNs in base maps.
- `pyampp/gx_chromo/decompose.py` now ignores NaNs when building histograms.

## Run/Command Notes
- Local run example (avoids JSOC downloads):
  ```bash
  env SUNPY_CONFIGDIR=/tmp/sunpy MPLCONFIGDIR=/tmp/mplconfig \
    /Users/gelu/miniforge3/envs/suncast/bin/python -m pyampp.gxbox.gx_fov2box \
    --time 2024-05-12T00:00:00 --coords 0 0 --hpc \
    --box-dims 64 64 64 --dx-km 1400.000 --pad-frac 0.10 \
    --data-dir /Users/gelu/pyampp/download --gxmodel-dir /Users/gelu/pyampp/gx_models \
    --euv --uv --save-empty-box --save-potential --save-bounds --local-only
  ```

## Comparison Results (Latest)
### Base Map Statistics (H5 vs IDL)
- **BX**
  - H5 min/max/mean: `-42.1516 / 45.6700 / -0.2907`
  - IDL min/max/mean: `-79.9754 / 80.9864 / 2.0924`
  - Diff (H5–IDL) absmax: `88.1449`
- **BY**
  - H5 min/max/mean: `-53.6737 / 57.2337 / -0.3542`
  - IDL min/max/mean: `-81.0593 / 67.6691 / 0.5290`
  - Diff (H5–IDL) absmax: `104.1693`
- **BZ**
  - H5 min/max/mean: `-118.6402 / 156.0102 / 0.5628`
  - IDL min/max/mean: `-200.1001 / 275.9383 / 0.5140`
  - Diff (H5–IDL) absmax: `119.9281`
- **IC**
  - H5 min/max/mean: `0.969339 / 1.039488 / 1.003992`
  - IDL min/max/mean: `0.949106 / 1.062015 / 1.003984`
  - Diff (H5–IDL) absmax: `0.02854`

### Chroma Mask Histogram (H5 vs IDL‑derived)
IDL `chromo_mask` is not stored in NONE `.sav`, so it was recomputed from IDL `BASE.BZ/IC` using the same `decompose()`:

```
Value  H5_count  IDL_count  diff
    1      3032       3007     25
    2       862        866     -4
    3       113        119     -6
    4        57         55      2
    5        32         49    -17
```

- Mismatched pixels: **628 / 4096**
- Diff map: `pyampp/tests/chromo_mask_diff.png`

## Open Issues / Next Steps
1. Base Bx/By/Bz still differ significantly vs IDL.
2. Full‑map reprojection produced NaNs at edges; decompose handles it but may still affect class assignments.
3. Need to verify whether the latest run used the updated full‑map remap code or older path (execute string in H5 does not encode code version).
4. Optional: compare mismatch spatial distribution and investigate if differences cluster by mask class or near edges.
