Changelog
=========

Unreleased
----------

SFQ / issue ``#42`` (toward ``1.1.1``):

- Replace the incomplete in-tree ``pyampp/sfq`` stub with the Sergey/vit1-irk
  Python SFQ package (same source gx_simulator vendors).
- Wire ``gx_fov2box --sfq`` to skip HMI disambiguation bits, run SFQ on the
  FOV crop (IDL ``prepare_basemaps, /sfq`` shape), then continue
  ``hmi_b2ptr`` / remap.
- Crop SFQ using model-base (CEA/TOP) WCS corners projected into the HMI
  frame (not the padded helioprojective context FOV), matching IDL.
- Correct CLI/``--info`` wording that previously claimed SFQ was HMI bit
  method ``0``.
- Add synthetic SFQ unit smoke tests and a real-data IDL crop/rotate parity
  checker (``real_data_checks/check_sfq_idl_parity.py``).
- Fix vendored SFQ ``sfq_clean`` filter width: IDL ``median/smooth(arr, s)``
  uses neighborhood size ``s``, not radius ``2*s+1`` (was the dominant
  post-SFQ disagreement vs IDL after matched inputs).
- Port IDL geometry-aware ``pot_vmag`` (``SOL_crd`` / ``a_field`` /
  ``get_fftplane`` / ``_Lfff_fft_n``) into ``pyampp/sfq``, and align
  ``u_grid``/B-spline with image/``readsav`` axis layout. Matched-input
  full-pipeline transverse-sign agree vs IDL rises to ~0.998.
- Harden ``--sfq`` wiring after independent review: NaN/sentinel→0 before
  SFQ (IDL), HMI disambig bits outside the SFQ crop (Vert_current/FOV),
  and CEA edge-midpoint sampling so curved bases are less often remapped
  from ambiguous azimuth.

1.1.0
-----

Release focus:

- STEREO / reference-map display and embed workflow from PR ``#51``:
  native-time spatial crops, context ``DATE-OBS`` display anchoring,
  observer-LOS context drawing, and FOV persistence across observers.

Highlights:

- Anchor ``gxbox-view2d`` display time to the selected context refmap
  ``DATE-OBS`` (fallback: model time). The figure banner shows that
  observation time and Δt versus the model. The SDO display-LOS option is
  removed; Earth covers SDO/AIA/HMI.
- Embed reference maps, including STEREO, with a spatial crop
  (``pyampp.io.refmap_crop``) that keeps each map's native ``DATE-OBS`` and
  observer metadata (``PYALIGN=False``).
- Record the source FITS on an embedded refmap as ``SRC_PATH``. When that
  file sits under the JSOC cache (``--data-dir``) or the model directory
  (the model file's folder or ``--gxmodel-dir``), also record
  ``SRC_RELPATH`` and ``SRC_ROOT`` (``data-dir`` or ``model-dir``).
  Filesystem mode prefers an existing recorded ``map_files`` hit (live
  JSOC cache or ``--refmaps-path``), then ``SRC_RELPATH`` under the current
  root / ``SRC_PATH``, then the embedded crop. A context map that exists
  only as that crop selects Map Source Embedded. Older embeds without
  ``SRC_PATH`` keep the crop until Add Reference Maps or
  ``--refmaps-path`` re-embeds them on save.
- ``target_fov`` and ``target_template`` on ``build_fits_refmaps_for_model``
  and ``build_refmap_payload_for_model`` now raise. Callers must use
  ``box_corners_world``. ``crop_refmap`` on ``add_fits_refmaps_to_h5`` is
  ignored and warns; embeds crop from model box corners.
- Selector fallback box corners place the geometric center half a model
  height above the surface origin, matching the viewer box.
- ``remove_refmaps_from_h5(telescope=...)`` rejects an empty telescope token
  instead of deleting every embedded refmap.
- ``remove_refmaps_from_h5(..., missing_ok=False)`` validates all requested
  ids before deleting any group, so a strict miss leaves the HDF5 unchanged.
- ``pyampp.io`` loads ``refmap_crop`` lazily so importing ``gx_fov2box`` (or
  ``from pyampp.io import load_model``) no longer hits a circular import.
- ``SRC_PATH`` and ``SRC_ROOT`` are ordinary 8-character FITS keys.
  ``SRC_RELPATH`` is stored as a HIERARCH card.
- The combined embed AIA + STEREO, reload H5, display-at-context-anchor
  check remains a manual GUI exercise. Unit tests cover the pieces.
- Reproject cross-observer helioprojective maps through the masked display
  pipeline at that time anchor.
- Keep the session FOV fixed when the display observer changes. Save / Apply
  asks whether to recompute the FOV for the current observer, save without an
  FOV, or cancel.
- Draw every selector map and the model boxes in the display observer's
  helioprojective line of sight. An empty context canvas uses that same
  observer frame. The Carrington ``base/index`` header remains model geometry
  and is not the context axes.
- Keep model-grid base overlays on their native WCS (Carrington CEA) and draw
  them with ``autoalign`` on the observer LOS context. Treating that
  heliographic plate scale as sky arcsec built a 32-pixel canvas and tore the
  overlay.

Known limitations:

- ``--sfq`` / GUI SFQ still only selects HMI disambiguation bit method ``0``
  (potential acute). It does **not** run Rudenko/Anfinogentov SFQ
  (issue ``#42``). Planned for ``1.1.1``.

Packaging/versioning:

- Bumped package version to ``1.1.0`` in packaging metadata.

1.0.6
-----

Release focus:

- revert the temporary ``pyAMaFiL`` git pin now that PyPI ships AMaFiL
  ``4.4.26.601``.

Highlights:

- Depend on ``pyAMaFiL>=1.2.0`` from PyPI instead of the GitHub commit pin
  introduced in ``1.0.3``. PyPI ``1.2.0`` bundles the AMaFiL core
  ``4.4.26.601`` that was previously required from git.

Packaging/versioning:

- Bumped package version to ``1.0.6`` in packaging metadata.

1.0.5
-----

Release focus:

- address Copilot PR review follow-ups from releases 1.0.3 and 1.0.4.

Highlights:

- Align local cache tolerance with JSOC query bounds by using half the
  configured query window for nearest-file matching (consistent with
  ``_make_query_bounds()``).
- Restore batched Fido ``search`` / ``fetch`` for AIA wavelengths and HMI
  segment groups instead of one network round-trip per product.
- Reorganize changelog: move previously shipped notes out of ``Unreleased``
  into their release sections.

Packaging/versioning:

- Bumped package version to ``1.0.5`` in packaging metadata.

1.0.4
-----

Release focus:

- align SDO cache/download behavior with IDL ``gx_box_jsoc_get_fits`` / ``gx_fov2box``.

Highlights:

- Resolve cached HMI/AIA FITS by **nearest** timestamp within the search window
  (not first sorted glob match), matching IDL nearest-record selection.
- Use ``index.json`` query-key cache for both DRMS and Fido backends.
- Unify DRMS and Fido download orchestration through a shared local-resolve path.
- Anchor AIA context downloads to HMI **continuum** ``DATE-OBS`` (IDL ``gx_fov2box``),
  not the field-map timestamp.
- Added ``--hmi-time-window`` and ``--aia-time-window`` CLI options (IDL
  ``HMI_time_window`` / ``AIA_time_window``).

Packaging/versioning:

- Bumped package version to ``1.0.4`` in packaging metadata.

1.0.3
-----

Release focus:

- pin ``pyAMaFiL`` to the June 2026 AMaFiL core until PyPI catches up.

Highlights:

- Pinned ``pyAMaFiL`` to Alexey Stupishin's GitHub repository at commit
  ``3b3d141`` (AMaFiL ``4.4.26.601``). PyPI ``1.1.5`` still ships the older
  WWNLFFF core ``4.2.25.326``, which ``pip install -U`` does not refresh when
  the wrapper version is unchanged.
- Retired the legacy ``gxbox`` GUI entrypoint from the public package surface.
- Retired the compatibility aliases ``gxbox-view`` and ``gxbox-select`` from the
  public package surface.
- Repositioned ``pyampp`` as the main GUI application and clarified the distinct
  roles of ``gxbox-view2d``, ``gxbox-view3d``, and ``gxrefmap-view``.
- ``h5tree`` now prints ``metadata/*`` values by default; replaced ``--show-metadata``
  with ``--no-metadata`` and added ``--meta`` for metadata-only output.
- Simplified GUI launcher commands: removed ``gxampp`` alias; use ``pyampp`` as
  the single launcher.
- Added a DRMS downloader backend and made it the default backend; added
  ``--use-fido`` and ``--force-download`` CLI options.
- Improved DRMS downloader throughput by scheduling independent HMI/AIA requests
  concurrently.
- Added GUI downloader controls (``Downloader`` selector and ``Use cache`` checkbox)
  and GUI command-export actions (copy command, save shell script).
- Implemented DRMS normalization of raw JSOC exports into reusable local FITS
  files and fixed DRMS nearest-record selection for HMI products.
- Reworked ``Vert_current`` generation to use an IDL-style remapped-input path
  plus a vectorized NumPy kernel.
- Fixed 2D viewer loading of embedded-only maps such as ``Vert_current`` when
  ``Map Source`` is set to ``Filesystem``.
- Added redundant derived ``observer/pb0r`` metadata alongside canonical
  ``observer/ephemeris`` for SSW-style ``B0 / L0 / Rsun`` interoperability.

Packaging/versioning:

- Bumped package version to ``1.0.3`` in packaging metadata.

1.0.2
-----

Release focus:

- shared FITS reference-map import for ``gx-fov2box`` and viewer tools,
- external AIA/EOVSA context maps via ``--refmaps-path``,
- DRMS downloader and runtime stage-normalization fixes.

Highlights:

- Added ``pyampp.io.refmaps`` for FITS discovery, model-time alignment from
  ``base/index``, and HDF5 ``refmaps/`` embedding (AIA, EOVSA, and generic
  user-supplied maps).
- Wired ``--refmaps-path`` through ``gx-fov2box``, the GUI (directory picker),
  and ``gxbox-view2d`` (file or directory for interactive selector context).
- JSOC cache scans import only recognized context products (``generic=False``);
  explicit ``--refmaps-path`` directories use generic fallback ids.
- ``Vert_current`` reference maps now use ``build_refmap_payload_for_model`` for
  WCS serialization and model-FOV alignment like other Earth line-of-sight maps.
- Fixed DRMS downloads to retain returned AIA context FITS paths when the final
  cache verification pass does not list them yet.
- Fixed runtime stage normalization to preserve internal 3D axis order while
  still injecting ``geometry_contract`` metadata.

Packaging/versioning:

- Bumped package version to ``1.0.2`` in packaging metadata.

1.0.1
-----

Release focus:

- SAV/HDF5 import parity fixes for CHR entry boxes,
- corrected CHR magnetic-cube handling in the Python ``combo_model`` path,
- documentation updates clarifying expected IDL-vs-Python POT-stage differences,
- observer geometry API delegation and WCS header time normalization.

Highlights:

- Fixed CHR import from legacy SAV entry boxes so chromospheric 2D/3D payloads
  preserve the intended axis ordering during SAV -> HDF5 -> pyAMPP round-trips.
- Fixed Python CHR ``BCUBE`` generation to match the intended ``combo_model``
  magnetic-cube contract, eliminating large differences caused by incorrect
  axis ordering during the interpolation path.
- Documented that small coronal/chromospheric magnetic-cube differences between
  IDL and pyAMPP may still occur by design because the POT stage uses different
  implementations:
  - IDL uses an FFT-based method
  - pyAMPP uses the Python extrapolation-library path
- Removed raw SAV payload dumping from the normalized HDF5 conversion path by
  default.
- Delegated observer/geometry resolution to the public ``pyampp.geometry`` API
  in ``make_observer_wcs_header``; ``obs_time`` is now the authoritative time
  source, normalized via ``Time(...).isot`` for consistent ``DATE-OBS`` /
  ``DATE_OBS`` serialization (PR #44).
- Added regression test coverage for observer WCS header time consistency.

Packaging/versioning:

- Bumped package version to ``1.0.1`` in packaging metadata.

1.0.0
-----

Release focus:

- downloader compatibility restoration and cache reuse reliability,
- GUI workflow hardening for iterative model-production sessions,
- updated documentation for HDF5 stage format and GUI functionality.

Highlights:

- Restored downloader behavior while preserving IDL-style date folder layout (``YYYY-MM-DD``).
- Improved cache matching for existing HMI/AIA products across filename variants, reducing unnecessary re-downloads.
- Fixed missing-HMI edge cases during resume/rebuild paths when files were already present in cache.
- Made GUI repository path persistence robust for both:
  - ``--data-dir``
  - ``--gxmodel-dir``
- Default local data cache path uses ``~/pyampp/jsoc_cache``.
- Added/updated documentation:
  - ``docs/model_hdf5_format.rst``
  - ``docs/gui_workflow.rst``
  - ``docs/viewers.rst``

Packaging/versioning:

- Bumped package version to ``1.0.0`` in packaging metadata.
