pyAMPP GUI Workflow
===================

This page documents the current ``pyampp`` command-builder GUI behavior and how it maps to ``gx-fov2box``.

Purpose
-------

The GUI is a thin command constructor for reproducible CLI runs. It:

- captures model parameters,
- resolves entry-box resume/rebuild behavior,
- displays the exact command before execution,
- streams command output in the status panel.

Data Repositories
-----------------

The GUI has four repository/path fields:

- ``SDO Data Dir`` (mapped to ``--data-dir``)
- ``GX Models Dir`` (mapped to ``--gxmodel-dir``)
- ``Reference Map Path`` (mapped to ``--refmaps-path`` when set)
- ``Entry Box`` (mapped to ``--entry-box`` when set)

Current behavior:

- the GUI restores the last saved session state via ``QSettings`` (geometry, workflow toggles, entry path, and repository paths),
- If an entry box contains a valid ``metadata/execute`` path and that path exists on the current machine, the GUI pre-fills the corresponding directory field.
- If execute paths are invalid on this machine, the GUI falls back to default local directories and warns the user.
- ``Restore Last Saved`` reloads the persisted GUI state.
- ``Reset to Test Defaults`` restores the built-in deterministic test geometry/time preset.

Model Configuration
-------------------

Core inputs:

- observation time (UTC)
- center coordinates with one selected frame:
  - ``--hpc`` (helioprojective)
  - ``--hgc`` (Carrington)
  - ``--hgs`` (Stonyhurst)
- projection:
  - ``--cea`` (default)
  - ``--top``
- box dimensions:
  - ``--box-dims NX NY NZ``
- spatial scale:
  - ``--dx-km``
- padding:
  - ``--pad-frac`` (GUI percent value is converted to fraction)
- disambiguation:
  - ``HMI`` (default, no extra flag)
  - ``SFQ`` (adds ``--sfq`` when applicable)

Downloader Controls
-------------------

The time row includes downloader controls that map directly to the CLI:

- ``Downloader: DRMS``:
  - default backend
  - emits no extra backend flag
- ``Downloader: Fido``:
  - emits ``--use-fido``
- ``Use cache``:
  - checked by default
  - when unchecked, emits ``--force-download``

Current downloader policy:

- ``DRMS`` is the default because it now follows the IDL-style JSOC workflow more closely:
  - resolve the nearest record explicitly
  - download the raw export
  - normalize it into a reusable local FITS cache file
- ``Fido`` remains available as the legacy fallback path.

Entry-Box Modes
---------------

When ``Entry Box`` is provided, the GUI detects the stage/type and offers four execution modes:

- ``Continue``:
  - continue from detected entry stage,
  - command is kept minimal (entry + paths + workflow flags).
- ``Rebuild from NONE``:
  - keep entry as source metadata/context but restart stage computation from NONE (``--rebuild-from-none``).
- ``Rebuild from OBS``:
  - keep CLI entry-driven (``--entry-box ... --rebuild``),
  - ``gx-fov2box`` restores entry timing/geometry internally.
- ``Modify``:
  - GUI-only helper mode (no dedicated CLI flag),
  - unlock populated fields from the selected entry box,
  - build a fresh command from current GUI fields (no ``--entry-box`` argument).

Pipeline Options and Stage Stops
--------------------------------

Save toggles:

- ``--save-empty-box``
- ``--save-potential``
- ``--save-bounds``
- ``--save-nas``
- ``--save-gen``
- ``--save-chr`` is emitted automatically when the run is allowed to reach CHR.

Stop rules:

- ``Stop after data download`` -> ``--stop-after dl``
- ``NONE only`` -> ``--stop-after none``
- ``POT only`` -> ``--stop-after pot``
- ``NAS only`` -> ``--stop-after nas``
- ``GEN only`` -> ``--stop-after gen``
- default full-path completion -> ``--stop-after chr``

CHR behavior:

- the GUI keeps the CHR checkbox non-editable, but the generated command is explicit:
  - reaching CHR adds ``--save-chr --stop-after chr``
- selecting an earlier stop stage prevents CHR from running/saving
- when continuing from a ``GEN`` or ``CHR`` entry box on the default final-CHR path, the GUI adds ``--jump2chromo`` so existing line metadata is reused instead of recomputing GEN

Other stage controls:

- ``Skip NLFFF`` -> ``--use-potential``
- ``Skip line computation`` -> ``--skip-lines``

Context-map toggles:

- EUV checkbox -> ``--euv``
- UV checkbox -> ``--uv``
- Reference Map Path -> ``--refmaps-path``. The path may point to a FITS file
  or a directory of FITS files. These maps are imported through the shared
  ``pyampp.io.refmaps`` policy and are available to ``gxbox-view2d`` after
  download-stage stops.

Command Display and Execution
-----------------------------

- The command preview updates live as inputs change.
- ``Run`` launches ``gx-fov2box`` as a subprocess.
- stdout/stderr are streamed into the status pane.
- A successful run reports generated model paths and stage timings in the log.
- explicit stop-stage completions auto-launch ``gxbox-view2d`` on the produced box and then adopt that box back into ``Entry Box`` for the next ``Continue`` step.
- ``gxbox-view2d`` opens the 2D FOV / box selector directly from the GUI on demand.
- The bottom command row includes a command-export dropdown:
  - ``Copy Command``
  - ``Save Script As...``

Operational Notes
-----------------

- If you modify package code, reinstall with ``pip install -e .`` before re-launching the GUI.
- Cached file reuse depends on ``--data-dir`` and timestamp/series compatibility with downloader filename matching.
- For clean downloader benchmarks:
  - GUI: uncheck ``Use cache``
  - CLI: add ``--force-download``
