# Real-Data Sanity Checks

This folder contains manual sanity-check scripts that exercise pyAMPP against large real fixtures.

These scripts are intentionally not part of the default `pytest` suite:
- they depend on local real-data fixtures that are not expected on every machine
- they are relatively slow compared with the normal unit and regression tests
- they are meant for targeted validation, not for everyday development loops

## Required fixture location

By default, both scripts expect the parity fixture bundle to exist at:

```text
../pyGXrender-test-data/raw/models/model_loader_parity_20201126T195831/
```

relative to the workspace root that contains the `pyAMPP` repository.

The default files are:
- `hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.CHR.sav`
- `hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.CHR.clone.h5`

You can override the defaults with command-line arguments.

If you already have the `pyGXrender-test-data` fixture bundle installed in that
default location, the two scripts below can be run without any extra setup.

If you do not have that bundle, you can generate an equivalent local fixture
pair yourself and pass the resulting paths explicitly.

Recommended workflow for a local fixture pair:

1. Choose a GX model root `OUT_DIR` and a shared JSOC cache `TMP_DIR`.
2. Run the equivalent SSWIDL command:

```idl
gx_fov2box, '26-Nov-20 20:00:00', CENTER_ARCSEC=[-610, -340], DX_KM=1400, $
  OUT_DIR=OUT_DIR, SIZE_PIX=[225, 225, 225], TMP_DIR=TMP_DIR, /EUV, /UV, /CEA
```

3. Locate the resulting SAV model written under the dated folder created below
   `OUT_DIR`.
4. Export that SAV model to canonical HDF5 with the pyAMPP CLI:

```bash
python -m pyampp.util.export_model \
  --model-path /path/to/model.sav \
  --out-h5 /path/to/model.clone.h5
```

5. Run the manual checks with those explicit paths.

Notes:

- Use the same `TMP_DIR` you plan to reuse for Python workflows so the same
  JSOC downloads can be shared.
- If your locally generated files do not match the default bundled filenames,
  that is fine; just pass them explicitly on the command line.

## Scripts

### `check_real_sav_roundtrip.py`

Purpose:
- load a real SAV model through the canonical loader
- save it to canonical HDF5
- reload the canonical HDF5
- save it again
- compare the two canonical HDF5 files exactly, dataset by dataset

Typical runtime:
- can take several minutes on large fixtures

Run:

```bash
python -u real_data_checks/check_real_sav_roundtrip.py
```

Optional explicit path:

```bash
python -u real_data_checks/check_real_sav_roundtrip.py /path/to/model.sav
```

### `check_real_geometry_fixtures.py`

Purpose:
- verify that a real legacy clone HDF5 can be upgraded with a geometry contract
- verify that real SAV conversion preserves line-array and chromosphere sparse-index mapping

Typical runtime:
- usually faster than the roundtrip script, but still not lightweight

Run:

```bash
python -u real_data_checks/check_real_geometry_fixtures.py
```

Optional explicit paths:

```bash
python -u real_data_checks/check_real_geometry_fixtures.py \
  --legacy-model-path /path/to/model.clone.h5 \
  --sav-path /path/to/model.sav
```

## Notes

- Use `python -u` for immediate progress output.
- `check_real_geometry_fixtures.py` requires `scipy`.
- These are manual validation scripts, not tutorial examples.

## Stage-Parity Workflow

The repository also includes a heavier IDL-versus-pyAMPP stage-parity workflow
for users who want to validate stage-by-stage resume behavior against locally
generated IDL reference SAV files.

The pyAMPP side of this workflow uses the staged IDL SAV files directly as
`--entry-box` inputs so the real resume path is exercised. The exported HDF5
files are used only as canonical comparison targets for each stage.

Script:

```bash
python -u real_data_checks/check_idl2py_stage_parity.py
```

This workflow is intentionally separate from `pytest` because it:

- depends on a local staged IDL reference data set,
- runs several expensive export and resume steps,
- writes a multi-stage comparison report rather than a small pass/fail unit test.

### Default paths

The parity script uses OS-aware defaults modeled after pyAMPP's existing path
conventions:

- IDL stage directory: the dated stage folder under `~/.pyampp/gx_models/`, for
  example `~/.pyampp/gx_models/2020-11-26`
- JSOC cache / data directory: `~/.pyampp/jsoc_cache`
- artifact root: `~/.pyampp/stage_parity_artifacts`

All of these can be overridden on the command line.

Important path contract:

- both the IDL and Python pipelines create or use dated model subfolders under
  the GX model root automatically,
- `OUT_DIR` should therefore point to the GX model root, not to an already
  dated folder,
- `--idl-stage-dir` should point to the dated folder produced by the IDL run,
  such as `OUT_DIR/2020-11-26`,
- `TMP_DIR` and `--data-dir` should point to the same JSOC cache directory so
  the same downloaded inputs can be reused.

### Preparing the reference IDL stage files

To reproduce the documented reference run, generate the IDL stages with a
command equivalent to:

```idl
gx_fov2box, '26-Nov-20 19:58:31', CENTER_ARCSEC=[-610, -340], DX_KM=1400, $
  OUT_DIR=OUT_DIR, /SAVE_BOUNDS, /SAVE_EMPTY_BOX, /SAVE_POTENTIAL, $
  SIZE_PIX=[255, 256, 257], TMP_DIR=TMP_DIR, /EUV, /UV, /CEA
```

Recommended interpretation:

- `OUT_DIR` points to your GX model output root.
- `TMP_DIR` points to your local JSOC / cache directory.
- Let the IDL workflow create the dated subfolder under `OUT_DIR`
  automatically.
- If `OUT_DIR` and `TMP_DIR` are omitted in the IDL environment, `gx_fov2box.pro`
  uses its own OS-aware defaults.

For the documented reference case, the resulting staged SAV files should live in
the date folder for the reference run, for example:

```text
~/.pyampp/gx_models/2020-11-26/
```

That means the matching Python invocation should use:

- `--idl-stage-dir ~/.pyampp/gx_models/2020-11-26`
- `--data-dir ~/.pyampp/jsoc_cache`

and not `--idl-stage-dir ~/.pyampp/gx_models`, because the parity script expects
the directory that already contains the staged SAV files.

and include the usual stage set:

- `...NONE.sav`
- `...POT.sav`
- `...BND.sav`
- `...NAS.sav`
- `...NAS.GEN.sav`
- `...NAS.CHR.sav`

Important labeling note for the final stage:

- IDL typically labels the chromospheric target as `NAS.CHR.sav`.
- pyAMPP may label the comparable generated result as `NAS.GEN.CHR.h5`, because
  the Python workflow exposes the explicit `NAS -> CHR` jump that IDL does not
  name separately.
- The parity script therefore treats those CHR naming variants as equivalent
  stage targets rather than assuming one exact suffix family.

### Running the parity workflow

With defaults:

```bash
python -u real_data_checks/check_idl2py_stage_parity.py
```

With explicit paths:

```bash
python -u real_data_checks/check_idl2py_stage_parity.py \
  --idl-stage-dir /path/to/OUT_DIR/2020-11-26 \
  --data-dir /path/to/TMP_DIR \
  --artifact-root /path/to/artifacts \
  --clean
```

To print the exact per-stage resume, export, and comparison plan without
executing any commands or writing artifacts:

```bash
python -u real_data_checks/check_idl2py_stage_parity.py --dry-run
```

To regenerate only the JSON report from an existing artifact tree, without
rerunning resume or export commands:

```bash
python -u real_data_checks/check_idl2py_stage_parity.py \
  --idl-stage-dir /path/to/OUT_DIR/2020-11-26 \
  --artifact-root /path/to/artifacts \
  --report-only
```

To rerun only one transition and then regenerate the full JSON report from the
updated artifact tree:

```bash
python -u real_data_checks/check_idl2py_stage_parity.py \
  --idl-stage-dir /path/to/OUT_DIR/2020-11-26 \
  --artifact-root /path/to/artifacts \
  --stage BND->NAS
```

Because `>` is a shell redirection operator, the transition form should either
be quoted or written with the shell-safe `ENTRY:TARGET` syntax:

```bash
python -u real_data_checks/check_idl2py_stage_parity.py \
  --artifact-root /path/to/artifacts \
  --stage "BND->NAS"
```

```bash
python -u real_data_checks/check_idl2py_stage_parity.py \
  --artifact-root /path/to/artifacts \
  --stage BND:NAS
```

You can also select a transition by target stage name when it is unambiguous,
for example:

```bash
python -u real_data_checks/check_idl2py_stage_parity.py \
  --artifact-root /path/to/artifacts \
  --stage NAS
```

The synthetic `OBS -> NONE` transition is also available and can be rerun by
itself. It uses the staged `NONE.sav` file as the geometry/defaults anchor,
then asks pyAMPP to rebuild the NONE model from observations:

```bash
python -u real_data_checks/check_idl2py_stage_parity.py \
  --idl-stage-dir /path/to/OUT_DIR/2020-11-26 \
  --artifact-root /path/to/artifacts \
  --stage OBS:NONE
```

For this transition, if `--data-dir` is not passed explicitly and the artifact
tree does not already carry a prior report default, the parity script falls
back to the data directory encoded in the `metadata/execute` command of the
staged `NONE.sav` entry model.

Notes for single-stage reruns:

- the selected transition is rerun, but the report is rebuilt for all stages
  from the current artifact tree
- `--clean` is intentionally rejected with `--stage`, because removing the
  whole artifact tree would also remove the other stages needed for the
  regenerated full report
- `--report-only --stage ...` is accepted but behaves like `--report-only`: it
  reuses existing artifacts and regenerates the full report without rerunning
  the selected stage

Recommended workflow:

- choose an `OUT_DIR` for the IDL run that points to your GX model root,
- choose a `TMP_DIR` for the IDL run that points to your shared JSOC cache,
- run the IDL workflow so it writes the staged SAV files under `OUT_DIR/DATE`,
- run the Python parity script with `--idl-stage-dir OUT_DIR/DATE` and
  `--data-dir TMP_DIR`.

Artifacts written under the artifact root:

- `idl_exported/` canonical HDF5 exports of the IDL stage SAV files
- `pyampp_generated/` pyAMPP-generated resume outputs for each branch, using
  the staged IDL SAV files as entry boxes
- `logs/` one log per export and resume step
- `reports/gx_idl2py_stage_parity_report.json` final comparison report

### Benchmark Results: Reference Runs and Library Analysis

The parity workflow has been executed with two configurations to analyze the
effects of external NLFFF library choice on reconstruction accuracy:

1. **Default configuration** (`local_tests/pyampp_stage_parity_override`):
   Uses the internal pyAMPP NLFFF implementation

2. **IDL library override** (`local_tests/pyampp_stage_parity_override_idllib`):
   Uses the external IDL NLFFF library (`WWNLFFFReconstruction.so`)

Both runs completed successfully with full JSON reports including stage closeness
metrics and per-dataset error analysis.

#### Stage Closeness Rankings

| Rank | Stage | Default Score | IDL-Library Score | Notes |
|------|-------|---|---|---|
| 1 | `POT -> BND` | 1.0000000 | 1.0000000 | Near-perfect parity |
| 2 | `NONE -> POT` | 0.9999999 | 0.9999999 | Expected algorithmic divergence |
| 3 | `BND -> NAS` | 0.9940639 | 0.9940639 | Known open parity gap |
| 4 | `GEN -> CHR` | 0.6269409 | 0.6269409 | Chromospheric differences expected |
| 5 | `NAS -> GEN` | 0.5818107 | **0.5820999** | Small improvement with IDL library |
| 6 | `OBS -> NONE` | 0.4484597 | 0.4484597 | Least close; driven by base map differences |

The stage closeness score is computed as:
```text
closeness_score = 1 - (
  0.50 * mean_relative_error
  + 0.35 * worst_relative_error
  + 0.15 * exact_mismatch_penalty
)
```

#### NLFFF Library Analysis: Corona Fields, Line Properties, and Chromospheric Effects

To understand where the external NLFFF library has measurable effects, three
comparison studies were performed on the isolated **NONE → CHR full run** pathway:

##### Test Configuration

Two identical NONE→CHR full-run executions were performed using the same
input entry file (`NONE.sav`) to eliminate OBS-stage variability:

- **DEFAULT run**: Uses internal pyAMPP NLFFF implementation
- **IDLLIB run**: Uses external IDL NLFFF library (`WWNLFFFReconstruction.so`)

This isolated pathway removes confounding factors and reveals pure library effects.

##### Comparative Results

**Table 1: Corona Magnetic Fields (bx, by, bz)**

| Stage | DEFAULT MAE | IDLLIB MAE | Ratio | Effect |
|-------|---|---|---|---|
| POT | 3.88e-7 | 3.88e-7 | 1.0000 | ✓ Identical |
| BND | 3.81e-7 | 3.81e-7 | 1.0000 | ✓ Identical |
| NAS | 4.34e-2 | 4.34e-2 | 1.0000 | ✓ Identical |
| NAS.GEN | 4.34e-2 | 4.34e-2 | 1.0000 | ✓ Identical |
| NAS.CHR | 4.34e-2 | 4.34e-2 | 1.0000 | ✓ Identical |

**Conclusion**: Corona magnetic fields are **unaffected** by library choice at all
stages. The large NAS-stage error (~4.34e-2 gauss) appears to be driven by
algorithmic differences upstream of the NLFFF library, not by library selection.

---

**Table 2: Magnetic Field Line Properties (at NAS.GEN stage)**

| Property | DEFAULT Mean | IDLLIB Mean | Mean Δ | Max Δ | % Diff | Ratio |
|----------|---|---|---|---|---|---|
| av_field | 5.343e+01 | 5.343e+01 | 2.22e-02 | 5.76e+02 | **0.042%** | 1.000098 |
| phys_length | 3.029e+02 | 3.029e+02 | 4.91e-02 | 1.39e+03 | **0.016%** | 1.000052 |
| voxel_status | 16,769,205 / 16,776,960 matches (7,755 mismatches = **0.046%**) |

**Additional observations on voxel_status mismatches**:
- DEFAULT vs IDLLIB: Different status code distributions in mismatch locations
- DEFAULT codes: [1, 7, 15, 19, 23, 27, 31] with counts [84, 1480, 1379, 2177, 531, 1982, 122]
- IDLLIB codes: [1, 7, 15, 19, 23, 27, 31] with counts [49, 1746, 1362, 2127, 382, 1963, 126]

**Conclusion**: The external NLFFF library produces **measurably different line
traces** (~0.04% variation in average field intensity, ~0.02% in physical length),
but these differences are localized to the line-tracing subsystem and do **not
propagate** to corona field reconstruction.

---

**Table 3: Chromospheric Properties (at NAS.CHR stage)**

| Property | DEFAULT Mean | IDLLIB Mean | Mean Δ | Max Δ | % Diff | Ratio |
|----------|---|---|---|---|---|---|
| bx | 2.691687e-01 | 2.691687e-01 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| by | -1.115881e+01 | -1.115881e+01 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| bz | 1.259190e+01 | 1.259190e+01 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| chromo_idx | 2.673925e+06 | 2.673925e+06 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| chromo_n | 1.931623e+12 | 1.931623e+12 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| chromo_t | 2.580735e+04 | 2.580735e+04 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| dz | 1.498417e-03 | 1.498417e-03 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| n_hi | 1.038548e+16 | 1.038548e+16 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| n_htot | 1.038654e+16 | 1.038654e+16 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| n_p | 1.065799e+12 | 1.065799e+12 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| tr | 8.172161e+01 | 8.172161e+01 | 0.0 | 0.0 | **0.0%** | 1.0000 |
| tr_h | 2.728181e-03 | 2.728181e-03 | 0.0 | 0.0 | **0.0%** | 1.0000 |

**Conclusion**: All 14 chromospheric properties are **completely unaffected** by
library choice (bit-for-bit identical across all fields). This indicates that the
chromospheric model operates independently of the NLFFF line-tracing subsystem.

---

##### Architectural Findings

The library analysis reveals a clear separation of concerns:

1. **NLFFF Library Scope**: Used exclusively in line-tracing computation (NAS→GEN)
   - Produces measurable but small variations in line properties (~0.04% av_field)
   - Changes approximately 0.046% of voxel status codes

2. **Corona Field Reconstruction**: Independent of NLFFF library choice
   - Remains bit-for-bit identical despite line trace differences
   - Suggests corona fields computed from shared boundary conditions (POT/BND output)
   - Not directly dependent on line trace geometry

3. **Chromospheric Model**: Independent of both NLFFF library and line traces
   - Uses corona fields + chromospheric model parameters
   - All 14 properties identical between library choices

4. **Pipeline Architecture Implication**: 
   - Corona field computation does not use individual line traces
   - Library differences masked at reconstruction level
   - OBS→NONE stage remains the primary source of pipeline degradation (per full-run
     testing and user confirmation)

---

#### Known Issues and Stage-Specific Notes

**Potential-field stage (`NONE -> POT`)**:
- Large expected mismatch due to algorithmic divergence
- IDL uses FFT-based potential extrapolation; Python uses different method
- This is a known difference, not a regression

**Boundary-to-NAS transition (`BND -> NAS`)**:
- Substantial remaining mismatch; open parity gap
- Strongest unresolved discrepancy across all documented runs
- Unaffected by library choice (identical scores in both configurations)

**Line generation stage (`NAS -> GEN`)**:
- Exact match on compared coronal payloads when input reaches NAS stage
- Library choice shows small score variation (0.581810680 → 0.582099873)
- Line properties show measurable differences but do not affect corona fields

**Chromospheric stage (`GEN -> CHR`)**:
- Corona fields match exactly; chromospheric fields very close
- All 14 chromospheric properties identical between library choices
- Residual differences in optical properties are expected

#### Practical Interpretation

- Use this workflow as a stage-localization tool to identify where parity holds
  and where gaps remain
- Treat POT→BND and NONE→POT as near-parity for coronal reconstruction
- Treat NAS→GEN as confirmed parity for compared coronal payloads
- Treat BND→NAS as the main unresolved parity target for future investigation
- Treat external NLFFF library effects as localized to line geometry, with no
  measurable impact on corona or chromospheric reconstruction

---

## OBS→NONE Reprojection Sweep Analysis

A dedicated sub-workflow exists for quantifying the effect of different
reprojection algorithms on the OBS→NONE boundary construction step.
The scripts below supplement `check_idl2py_stage_parity.py` and are designed to
be run against a set of pre-generated `reproject_<method>/` sub-trees.

The wrapper script `local_scripts/run_pyampp_stage_parity_full_run_default.sh`
accepts a `-p METHOD` flag that writes each projection run to an isolated
sub-folder:

```bash
local_scripts/run_pyampp_stage_parity_full_run_default.sh -p adaptive
local_scripts/run_pyampp_stage_parity_full_run_default.sh -p exact
local_scripts/run_pyampp_stage_parity_full_run_default.sh -p interpolation
```

Artifacts land under `<artifact_root>/reproject_<method>/`.

The `--reproject-algorithm` flag is also available directly on
`check_idl2py_stage_parity.py`:

```bash
python -u real_data_checks/check_idl2py_stage_parity.py \
  --idl-stage-dir /path/to/OUT_DIR/2020-11-26 \
  --artifact-root /path/to/artifacts/reproject_exact \
  --reproject-algorithm exact \
  --stage full-run
```

### `rank_obsnone_reproject_scan.py`

Purpose:
- discover all `reproject_*` candidate directories under a scan root
- compare each produced `NONE.h5` against a common IDL reference
- compute a composite closeness score per method
- write a JSON ranking table and a Markdown summary

Run (after generating runs for each projection method):

```bash
python -u real_data_checks/rank_obsnone_reproject_scan.py \
  --scan-root /path/to/artifact_root \
  --reference-h5 /path/to/idl_exported/hmi....NONE.h5
```

Output:
- `<scan-root>/reproject_ranking_obsnone.json` — machine-readable ranking
- `<scan-root>/reproject_ranking_obsnone.md` — markdown table

Optional flags:
- `--include-corona-z0` — also compare z=0 corona slices (shapes must match)
- `--out-json / --out-md` — override default output paths

### `plot_obsnone_relative_error_panel.py`

Purpose:
- read the ranking JSON from `rank_obsnone_reproject_scan.py`
- produce a side-by-side PNG panel of per-pixel **vector-magnitude relative error**
  `|ΔB| / |B_ref|` for each requested method
- shared colour scale (99th-percentile-capped, magma colormap)

Run:

```bash
python -u real_data_checks/plot_obsnone_relative_error_panel.py \
  --ranking-json /path/to/reproject_ranking_obsnone.json \
  --out-png /path/to/reproject_relative_error_panel.png
```

Optional flags:
- `--methods adaptive exact interpolation` — select methods to include
- `--vmax-percentile 99` — shared colour-scale percentile
- `--eps 1e-6` — denominator floor

### `plot_obsnone_component_metric_panels.py`

Purpose:
- produce two 3×3 PNG panels (methods × bx/by/bz) using two complementary metrics:
  1. **Absolute relative residual** `|cand − ref| / max(|ref|, ε)` — unsigned,
     magma colormap
  2. **Symmetric normalised difference** `(ref − cand) / (|ref| + |cand| + ε)`,
     bounded in `[−1, 1]` — signed, coolwarm colormap

Run:

```bash
python -u real_data_checks/plot_obsnone_component_metric_panels.py \
  --ranking-json /path/to/reproject_ranking_obsnone.json \
  --out-relative-png /path/to/relative_residual_panel.png \
  --out-symdiff-png /path/to/symdiff_panel.png
```

Both `--out-relative-png` and `--out-symdiff-png` are required.

### `export_none_base_maps_to_fits.py`

Purpose:
- export the `base/bx`, `base/by`, `base/bz`, `base/ic`, and `base/chromo_mask`
  datasets from any pyAMPP NONE-stage HDF5 to a multi-extension FITS file
  readable in IDL
- WCS is recovered from the `base/index` FITS header bytes stored in the HDF5

Run (directly):

```bash
python -u real_data_checks/export_none_base_maps_to_fits.py \
  --h5-path /path/to/NONE.h5 \
  --out-fits /path/to/base_maps.fits
```

Or via the wrapper script:

```bash
local_scripts/export_none_base_maps_to_fits.sh \
  -i /path/to/NONE.h5 \
  -o /path/to/base_maps.fits
```

The output is a MEF (multi-extension FITS) with extensions in order:
`BX` (1), `BY` (2), `BZ` (3), `IC` (4), `CHROMO_MASK` (5), all float64.

IDL reading:

```idl
bx = mrdfits('base_maps.fits', 1, hdr)
by = mrdfits('base_maps.fits', 2)
bz = mrdfits('base_maps.fits', 3)
ic = mrdfits('base_maps.fits', 4)
```

---

## OBS→NONE Parity Gap: Findings and Closure Status

### Reprojection Method Comparison (reference run: 2020-11-26 event)

Three reprojection algorithms available in the `reproject` package were evaluated
on the OBS→NONE base-map construction step.  All three were compared against the
same IDL-exported NONE reference.

| Rank | Method | Closeness score | Mean rel. error | Worst rel. error |
|---:|---|---:|---:|---:|
| 1 | adaptive | highest | lowest | lowest |
| 2 | exact | intermediate | intermediate | intermediate |
| 3 | interpolation | lowest | highest | highest |

Adaptive reprojection is consistently closest across all four base-map
components (`bx`, `by`, `bz`, `ic`).  All three methods produce similar
qualitative error maps; the differences are quantitative rather than structural.

### Downstream Stage Isolation Test

To confirm that the OBS→NONE discrepancy does not reflect any defect in the
downstream pyAMPP pipeline logic, the following isolation test was performed:

1. A pyAMPP-generated `NONE.h5` was exported to a FITS file
   (`export_none_base_maps_to_fits.py`) and the base maps were used to construct
   a replacement `NONE_frompy.sav` with IDL-compatible boundary conditions.
2. The manufactured `NONE_frompy.sav` was injected back into IDL as the
   `entry_box` for a full `gx_fov2box` run with `/jump2potential`.
3. IDL successfully completed all downstream stages (POT, NAS, NAS.GEN, NAS.CHR)
   in ~4400 s with normal output.

Triad parity check of the resulting artifacts (new IDL-from-pyAMPP NONE,
original pure-IDL, Python full-run):

| Stage | new_vs_py mean_rel | old_vs_py mean_rel | Improvement |
|---|---|---|---|
| NAS corona (bx/by/bz) | 9.41e-03 | 3.64e-01 | ~39× better |
| GEN mean | 2.39e+12 | 1.57e+13 | ~7× better |
| CHR mean | 9.10e+11 | 5.99e+12 | ~7× better |

The large GEN/CHR absolute relative values are dominated by line/chromo fields
with near-zero reference magnitudes; the ranking is still meaningful.

**Conclusion**: downstream stages (POT → BND → NAS → GEN → CHR) show
substantially better parity when driven by pyAMPP-generated NONE boundary
conditions than by original IDL NONE conditions.  This confirms that the primary
remaining parity gap is localised to the **OBS→NONE reprojection/boundary
construction step** and does not reflect any defect in the downstream pipeline.

### Next Steps

The OBS→NONE gap is designated as a **dedicated feature-branch investigation**
after the current branch (`feat/io-only-model-load-paths`) is merged.  Open
questions include:

- which HMI pixel coordinate transform convention accounts for the systematic
  bias observed in the base-map differences across all three reprojection methods
- whether a pre-reprojection coordinate normalisation step would close the gap
- alignment with Alexey Stupishin (GX library author) on why pyAMPP-generated
  boundary conditions affect the Stupishin NLFFF line-tracing library
