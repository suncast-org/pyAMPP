Changelog
=========

Unreleased
----------

- Retired the legacy ``gxbox`` GUI entrypoint from the public package surface.
- Retired the compatibility aliases ``gxbox-view`` and ``gxbox-select`` from the public package surface.
- Repositioned ``pyampp`` as the main GUI application and clarified the distinct roles of:
  - ``gxbox-view2d``
  - ``gxbox-view3d``
  - ``gxrefmap-view``
- ``h5tree`` now prints ``metadata/*`` values by default.
- Replaced ``--show-metadata`` with ``--no-metadata`` to explicitly suppress metadata output.
- Added ``--meta`` mode to print only ``metadata/*`` values (without tree output).
- Simplified GUI launcher commands: removed ``gxampp`` alias; use ``pyampp`` as the single launcher.
- Added a DRMS downloader backend and made it the default backend.
- Added ``--use-fido`` to explicitly select the legacy SunPy/Fido downloader.
- Added ``--force-download`` to bypass cache hits for clean downloader benchmarking.
- Improved DRMS downloader throughput by scheduling independent HMI/AIA requests concurrently.
- Added GUI downloader controls:
  - ``Downloader`` selector (`DRMS` / `Fido`)
  - ``Use cache`` checkbox
- Added GUI command-export actions for the current ``gx-fov2box`` script:
  - copy command
  - save shell script
- Implemented DRMS normalization of raw JSOC exports into reusable local FITS files.
- Fixed DRMS nearest-record selection for HMI products so cadence choice now matches the intended IDL-style behavior.
- Reworked ``Vert_current`` generation to use an IDL-style remapped-input path plus a vectorized NumPy kernel.
- Fixed 2D viewer loading of embedded-only maps such as ``Vert_current`` when ``Map Source`` is set to ``Filesystem``.
- Added redundant derived ``observer/pb0r`` metadata alongside canonical
  ``observer/ephemeris`` for SSW-style ``B0 / L0 / Rsun`` interoperability.

1.0.1
-----

Release focus:

- SAV/HDF5 import parity fixes for CHR entry boxes,
- corrected CHR magnetic-cube handling in the Python ``combo_model`` path,
- documentation updates clarifying expected IDL-vs-Python POT-stage differences.

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
