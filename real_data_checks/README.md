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

### Reference results from the documented run

The documented run completed end to end and produced a full JSON report.

Primary conclusion:

- the parity workflow itself is operational and useful as a regression and
  diagnostic reference,
- the report should not be interpreted as "full stage parity achieved", because
  the remaining differences are stage-dependent and some of them are known,
  meaningful modeling differences rather than simple numerical noise.

Important note on the potential-field stage:

- a visible `NONE -> POT` difference is expected because the IDL and Python
  pipelines do not currently use the same potential extrapolation algorithm,
- the IDL workflow uses its established FFT-based potential-field path, while
  the Python workflow uses a different extrapolation implementation,
- as a result, mismatch at this stage should be interpreted in that algorithmic
  context and not treated by itself as evidence that the parity driver failed.

Key observations:

- `NONE -> POT`: large expected mismatch, consistent with the known difference
  between the IDL FFT-based POT implementation and the Python extrapolation path.
  This means the workflow is behaving as expected for that stage, but it also
  means this transition is not currently evidence of numerical parity.
- `POT -> BND`: partially close but not strictly identical; `bz` is effectively
  exact while `bx/by` still fail strict `allclose`. This suggests the Python
  resume path is preserving part of the boundary-state output very well, but the
  stage still falls short of full component-wise parity.
- `BND -> NAS`: substantial remaining mismatch; this remains an open parity gap
  and is the strongest unresolved discrepancy in the documented run.
- `NAS -> GEN`: exact match on the compared coronal payloads. This is the
  strongest positive parity result in the report and indicates that, once the
  input state reaches the NAS stage, the downstream line-generation step is
  reproduced exactly for the compared fields.
- `GEN -> CHR`: coronal fields match exactly; chromospheric `bx/by/bz` are
  extremely close but not strictly `allclose` under the current tolerances.
  In practice this means the chromospheric branch is very nearly reproduced,
  but the report still correctly classifies it as not strictly identical.

Practical interpretation:

- use this workflow primarily as a stage-localization tool: it tells you where
  parity holds, where it nearly holds, and where the remaining gaps are large,
- treat `NAS -> GEN` as confirmed parity for the compared coronal payloads,
- treat `GEN -> CHR` as near-parity with small residual chromospheric
  differences,
- treat `BND -> NAS` as the main unresolved parity target,
- treat `NONE -> POT` as a known algorithmic divergence rather than a simple
  regression signal.

These results are useful reference material even for users who do not want to
rerun the full workflow locally.
