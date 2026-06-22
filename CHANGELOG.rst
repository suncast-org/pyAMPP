Changelog
=========

Unreleased
----------

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
