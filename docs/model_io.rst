Model I/O And Contract Enforcement
=================================

This page documents the canonical model loading/saving architecture introduced for
geometry contract enforcement.

Design Intent
-------------

All model restore paths must go through the centralized ``pyampp.io`` surface so
that loaded models are normalized into one metadata state before geometry is
consumed.

Policy:

- New models: Tier 1 + Tier 2 geometry metadata must be complete.
- Legacy models (SAV or incomplete HDF5): missing Tier 1 + Tier 2 fields are
  inferred from available fallbacks (for example: ``base/index``,
  coronal cube shape, ``corona/dr``).
- On save, completed contract metadata is persisted to HDF5 so next load does
  not need to recompute those fields.
- If a loaded model is not saved, recomputation on future loads is expected and
  acceptable.

Public API
----------

Use these functions from ``pyampp.io``:

- ``load_model(path, strict=False, keep_temp_h5=False)``
- ``save_model(model_dict, path)``
- ``load_model_metadata(path, strict=False)``
- ``save_thin_model(thin_model, path)``
- ``export_thin_model(source_model, output_h5=None, strict=False)``
- ``add_fits_refmaps_to_h5(h5_path, fits_paths, ...)``
- ``add_fits_refmaps_from_dir_to_h5(h5_path, fits_dir, ...)``
- ``build_fits_refmaps_for_model(paths, model_obstime=..., target_fov=..., ...)``
- ``discover_fits_refmap_map_ids(paths, ...)``

The package-level import surface is:

.. code-block:: python

   from pyampp import io, geometry

  model = io.load_model("/path/to/model.h5")
   contract = model["metadata"]["geometry_contract"]

   red_world = geometry.world_corners_from_geometry_contract(contract)

Contract Behavior At Load Time
------------------------------

``load_model`` performs the following in order:

1. Read model payload.
2. Reuse persisted ``metadata/geometry_contract`` if present.
3. Otherwise, complete contract fields via
   ``pyampp.geometry.complete_geometry_contract``.
4. Normalize observer metadata.
5. Return the normalized model dictionary.

Because this happens centrally, downstream geometry code should not branch on
legacy metadata patterns.

Contract Behavior At Save Time
------------------------------

``save_model`` writes the model and persists geometry contract metadata
when it exists in ``model_dict["metadata"]["geometry_contract"]``.

This ensures that once a legacy model is loaded and completed, its next saved
HDF5 form contains the completed contract and can load without recomputation.

Migration Guidance
------------------

Recommended for application and downstream code:

- Do import from ``pyampp.io`` for model load/save.
- Do import from ``pyampp.geometry`` for geometry operations.
- Prefer ``pyampp.io`` as the canonical app-level interface.
- ``pyampp.gxbox.boxutils.read_b3d_h5`` currently delegates to
  ``pyampp.io.load_model`` and therefore does perform contract
  completion and observer normalization, but it remains a legacy compatibility
  surface with gxbox-shaped return conventions.

This separation removes duplicated fallback logic in downstream consumers and
keeps one authoritative contract-completion path in pyAMPP.

Reference-Map Import API
------------------------

External FITS context maps should be imported through ``pyampp.io.refmaps``.
This keeps instrument identification, model-time alignment, and HDF5 layout in
one public API used by both ``gx-fov2box`` and the viewer tools.

Typical in-place HDF5 import:

.. code-block:: python

   from pyampp.io import add_fits_refmaps_from_dir_to_h5

   added = add_fits_refmaps_from_dir_to_h5(
       "/path/to/model.h5",
       "/path/to/refmap_dir",
       overwrite=True,
   )

Policy:

- AIA maps are identified from FITS telescope/instrument metadata plus
  ``WAVELNTH`` and are stored as ``AIA_<wavelength>``.
- EOVSA maps are identified from EOVSA telescope/instrument metadata plus
  ``CRVAL3``/``CUNIT3='Hz'`` and are stored as frequency-labelled maps such as
  ``EOVSA_f1.418GHz``.
- Known-map scans of the JSOC cache use ``generic=False`` so unrelated HMI
  source products are not imported as context maps.
- Explicit user paths use ``generic=True`` by default, so unknown FITS files
  receive sanitized filename-based ids.
- The model time is inferred from ``base/index`` via
  ``model_obstime_from_base_index``.
- Earth/SDO line-of-sight maps are time-aligned and reprojected to the selected
  model reference-map footprint. Non-Earth maps are stored with their native
  WCS footprint.

Runtime Enforcement Policy
--------------------------

To keep model semantics provenance-agnostic and prevent axis/metadata drift,
runtime code should follow these rules:

- All ``.h5`` and ``.sav`` model loads must go through ``pyampp.io``.
- ``.sav`` must be treated as an input format only; conversion and load policy
  are owned by ``pyampp.io.load_model``.
- Runtime modules (viewers, selector flows, pipeline entry-box handlers) must
  not call ``scipy.io.readsav`` for model reconstruction.
- Runtime modules must not call SAV->H5 conversion helpers directly; they
  should consume the canonical ``pyampp.io`` surface.

Operationally, this means that after load, in-memory model payloads should
follow the same pyAMPP-native contract regardless of source provenance.

Thin Metadata CLI
-----------------

Thin IO APIs:

- ``load_model_metadata(path, strict=False)``
  restores a model through the canonical loader and returns a thin model with
  full ``metadata`` and optional ``observer`` when a geometry contract is
  available after restore.
- ``save_thin_model(thin_model, path)``
  writes a lightweight HDF5 containing only ``metadata`` and optional
  ``observer`` sections.
- ``export_thin_model(source_model, output_h5=None, strict=False)``
  is the public convenience helper to generate a metadata-only artifact
  directly from any supported full model input.

For quick metadata inspection through the canonical loader, use:

.. code-block:: bash

  show-model-metadata /path/to/model.h5

JSON output for scripts:

.. code-block:: bash

  show-model-metadata /path/to/model.h5 --json

To fail when contract completion cannot be restored from the source model:

.. code-block:: bash

  show-model-metadata /path/to/model.h5 --strict

To export a portable metadata-only artifact from any supported model file:

.. code-block:: bash

  export-model-metadata /path/to/full_model.h5

Optional explicit destination:

.. code-block:: bash

  export-model-metadata /path/to/full_model.h5 --output /path/to/model_metadata.h5

Exit code behavior follows the Typer command result: successful metadata
inspection returns ``0``; ``--strict`` propagates model-restore failures.
