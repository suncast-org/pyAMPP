Observer/Geometry Delegation Update
===================================

Scope
-----

This update introduces two targeted API-level changes to support strict
downstream delegation of observer and geometry handling to the public
``pyampp.geometry`` surface.


Files Changed
-------------

1. ``pyampp/gxbox/observer_restore.py``

- Updated ``normalize_observer_key`` aliases:

  - Added ``terra`` -> ``earth``
  - Added ``stereo ahead`` -> ``stereo-a``
  - Added ``stereo behind`` -> ``stereo-b``

- Changed fallback normalization behavior from:

  - unknown key -> ``earth``

  to:

  - unknown key -> preserved normalized key

2. ``pyampp/geometry/core.py``

- Added public helper ``make_observer_wcs_header`` to generate an
  observer-aware SunPy-compatible WCS header with canonical observer cards.

3. ``pyampp/geometry/__init__.py``

- Exported ``make_observer_wcs_header`` in the public geometry import surface.


Purpose
-------

These changes are intended to remove downstream geometry duplication and make
``pyampp.geometry`` sufficient as the single integration surface for:

- observer identity normalization/resolution,
- observer-aware FOV and projection utilities,
- observer-aware WCS header generation.


Behavior and Compatibility Notes
--------------------------------

1. Canonical model I/O normalization remains unchanged:

- Missing observer identity in restored legacy models still defaults to Earth
  via the model normalization pipeline.

2. The observer key normalization change only affects explicit observer-key
   handling in the observer helper path:

- Known aliases still map to canonical keys.
- Unknown keys are no longer collapsed to Earth at normalization time.


Downstream Migration Intent
--------------------------

Downstream code (for example gximagecomputing workflows) can use these APIs to
avoid local reimplementation of observer/WCS routines:

- ``pyampp.geometry.normalize_observer_key``
- ``pyampp.geometry.resolve_named_observer``
- ``pyampp.geometry.resolve_observer_with_info``
- ``pyampp.geometry.compute_inscribing_fov_from_world``
- ``pyampp.geometry.make_observer_wcs_header``


PR Summary Template
-------------------

Use this summary in the PR description:

- Added observer alias coverage and preserved unknown normalized observer keys
  in ``observer_restore.normalize_observer_key``.
- Added public ``make_observer_wcs_header`` helper in
  ``pyampp.geometry.core`` and exported it through
  ``pyampp.geometry.__init__``.
- Purpose: enable strict downstream delegation of observer/geometry/WCS
  handling to pyAMPP public API and reduce duplicated geometry code in
  consumers.
