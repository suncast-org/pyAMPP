pyAMPP HDF5 Model Format
========================

This page documents the current stage-file contract written by ``gx-fov2box`` and consumed by:

- ``gxbox-view3d`` (3D viewer),
- ``gxbox-view2d`` (2D FOV / box viewer),
- ``gxrefmap-view`` (base/refmap browser),
- resume/rebuild workflows via ``--entry-box``.

File Naming
-----------

Typical output naming pattern:

- ``hmi.M_720s.YYYYMMDD_HHMMSS.<region>.CEA.NONE.h5``
- same stem with stage suffixes such as ``POT``, ``BND``, ``NAS``, ``NAS.GEN``, ``NAS.CHR``, ``NAS.GEN.CHR``

Common Groups (All Stages)
--------------------------

``base``
~~~~~~~~

2D boundary maps:

- ``base/bx``, ``base/by``, ``base/bz``
- ``base/ic``
- ``base/chromo_mask``
- ``base/index`` (IDL-compatible serialized header payload)

``base/index`` compatibility notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``base/index`` is intended to preserve the scientific meaning of the saved base
map geometry and to remain consumable by IDL/GX-oriented tools, but it is not
guaranteed to be numerically identical to the internal WCS produced by the IDL
``prepare_basemaps.pro`` round-trip.

Current design intent:

- pyAMPP treats the imported SunPy/HMI map WCS and observer metadata as the
  primary source of truth for newly generated base maps.
- pyAMPP writes an IDL-compatible serialized header payload whose Carrington /
  observer tags are semantically correct for the saved base grid.
- pyAMPP does **not** intentionally reproduce every SSW internal normalization
  step (for example the IDL ``wcs2map -> map2wcs`` round-trip) unless that
  behavior is independently justified.

Practical implications:

- Small differences in cards such as ``CRVAL*``, ``CRLN_OBS``, ``DSUN_OBS``, or
  ``SOLAR_B0`` may remain when comparing a fresh pyAMPP model against an IDL
  model built from the same source FITS files.
- Those differences do not automatically imply a scientific error in the pyAMPP
  base maps; they may reflect different WCS normalization conventions between
  SunPy and SSW.
- Users who require strict IDL basemap parity may start from an IDL-produced
  ``NONE`` model via ``--entry-box`` and continue the workflow from there.
- For downstream numerical equivalency beyond the basemap stage, IDL and pyAMPP
  should also use the same NLFFF library/version; otherwise later model stages
  can diverge even when the starting ``NONE`` model is identical.

Compatibility requirement for imported legacy models:

- SAV/HDF5 import paths should produce a ``base/index`` payload that is
  structurally compatible with the current pyAMPP contract, even when the
  original IDL header representation is not preserved byte-for-byte.

``metadata``
~~~~~~~~~~~~

Core provenance:

- ``metadata/id``: model identifier
- ``metadata/execute``: full command string used to create the model
- ``metadata/projection``: ``CEA`` or ``TOP``
- ``metadata/disambiguation``: e.g. ``HMI`` or ``SFQ``
- ``metadata/axis_order_2d`` and ``metadata/axis_order_3d`` when present
- ``metadata/vector_layout`` when present

Geometry contract metadata (when present):

- ``metadata/geometry_contract/nx``
- ``metadata/geometry_contract/ny``
- ``metadata/geometry_contract/nz``
- ``metadata/geometry_contract/dr_x``
- ``metadata/geometry_contract/dr_y``
- ``metadata/geometry_contract/dr_z``
- ``metadata/geometry_contract/rsun_m``
- ``metadata/geometry_contract/anchor_lon_deg``
- ``metadata/geometry_contract/anchor_lat_deg``
- ``metadata/geometry_contract/anchor_radius_rsun``
- ``metadata/geometry_contract/frame``
- ``metadata/geometry_contract/obstime``
- ``metadata/geometry_contract/inferred_from``

Contract semantics:

- Tier 1 (intrinsic box) and Tier 2 (world embedding/time) fields are expected
  to be complete for new pyAMPP outputs.
- Legacy models may be missing this group on disk; canonical loaders complete
  and attach these fields at load time.
- Saving through ``pyampp.io.save_model`` persists the completed contract
  to this location.

``observer``
~~~~~~~~~~~~

Optional observer / FOV metadata used by 2D resume and viewer flows:

- ``observer/name``
- ``observer/label`` as the user-facing observer identifier shown in the 2D selector
- ``observer/source`` when the observer record comes from an uploaded reference file
- ``observer/fov/frame``
- ``observer/fov/xc_arcsec``, ``observer/fov/yc_arcsec``
- ``observer/fov/xsize_arcsec``, ``observer/fov/ysize_arcsec``
- ``observer/fov/square``
- ``observer/ephemeris/*`` as the canonical observer-state block
- ``observer/pb0r/*`` as an optional redundant derived block for SSW-style
  ``B0 / L0 / Rsun`` interoperability

Viewer UI semantics:

- ``MODEL-DATE`` is the saved model time.
- ``OBS-DATE`` is the true observer-record time.
- For built-in observers those typically match; uploaded/manual custom observers may intentionally differ.

The current derived ``observer/pb0r`` keys are:

- ``observer/pb0r/obs_date``
- ``observer/pb0r/b0_deg``
- ``observer/pb0r/l0_deg``
- ``observer/pb0r/p_deg`` when available
- ``observer/pb0r/rsun_arcsec``

``refmaps``
~~~~~~~~~~~

Optional context maps, each in its own subgroup:

- ``refmaps/<map_id>/data``
- ``refmaps/<map_id>/wcs_header``

``grid``
~~~~~~~~

Grid geometry and identity:

- ``grid/dx``, ``grid/dy``, ``grid/dz``
- ``grid/voxel_id`` (uint32)

Stage-Specific Groups
---------------------

NONE
~~~~

- Minimal model shell for downstream continuation.
- No solved extrapolation required.
- ``corona`` may exist as placeholder fields with ``attrs/model_type = "none"``.

POT
~~~

Potential-field stage in ``corona``:

- ``corona/bx``, ``corona/by``, ``corona/bz``
- ``corona/dr``
- ``corona/attrs/model_type = "pot"``
- ``corona/corona_base`` when available

BND
~~~

Boundary-conditioned stage in ``corona``:

- ``corona/bx``, ``corona/by``, ``corona/bz``
- ``corona/dr``
- ``corona/corona_base`` when available

NAS
~~~

NLFFF stage in ``corona``:

- ``corona/bx``, ``corona/by``, ``corona/bz``
- ``corona/dr``
- ``corona/attrs/model_type = "nlfff"`` (or equivalent)

NAS.GEN
~~~~~~~

Adds line-tracing products (stored in ``lines`` and/or stage-compatible chromo metadata):

- line indexing arrays (start/end/apex/seed)
- status/code arrays
- aggregate line properties (average field, physical length)

NAS.CHR
~~~~~~~

Adds chromospheric payload in ``chromo`` and finalized grid metadata:

- chromospheric thermodynamic variables (where generated)
- chromosphere-resolved field cubes (when present)
- final ``grid/voxel_id`` and geometry values used by viewers/renderers

Reference-map IDs
-----------------

Typical AIA-derived IDs used by viewers:

- ``AIA_94``, ``AIA_131``, ``AIA_171``, ``AIA_193``, ``AIA_211``, ``AIA_304``, ``AIA_335``
- ``AIA_1600``, ``AIA_1700`` (if UV requested)

Axis and Layout Notes
---------------------

- 2D base/refmap arrays are map-shaped with explicit WCS headers retained.
- 3D field arrays are stored in HDF5 canonical ``(z, y, x)`` order when written by current writers.
- Viewers and adapters handle any required transpose/reindex for rendering and legacy compatibility.

Resume/Entry Compatibility
--------------------------

For robust resume behavior, entry files should include:

- ``metadata/execute`` (for reproducible command provenance and path recovery),
- valid ``metadata/id`` stage naming,
- required stage groups for the selected jump/rebuild mode.

When ``--entry-box`` is supplied, ``gx-fov2box`` detects stage availability and enforces required groups for continuation/rebuild paths.

Canonical Read Path
-------------------

For application and downstream usage, model files should be restored via
``pyampp.io`` rather than low-level HDF5 readers. The centralized loader path
ensures geometry contract completion and observer normalization before models
are consumed by geometry or rendering code.
