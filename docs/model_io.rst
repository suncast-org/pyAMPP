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
  ``metadata/execute``, coronal cube shape, ``corona/dr``).
- On save, completed contract metadata is persisted to HDF5 so next load does
  not need to recompute those fields.
- If a loaded model is not saved, recomputation on future loads is expected and
  acceptable.

Public API
----------

Use these functions from ``pyampp.io``:

- ``load_model_from_h5(path, strict=False)``
- ``load_model_from_sav(path, strict=False, keep_temp_h5=False)``
- ``save_model_to_h5(model_dict, path)``
- ``complete_and_persist_contract_in_h5(path, strict=False)``

The package-level import surface is:

.. code-block:: python

   from pyampp import io, geometry

   model = io.load_model_from_h5("/path/to/model.h5")
   contract = model["metadata"]["geometry_contract"]

   red_world = geometry.world_corners_from_geometry_contract(contract)

Contract Behavior At Load Time
------------------------------

``load_model_from_h5`` and ``load_model_from_sav`` perform the following in order:

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

``save_model_to_h5`` writes the model and persists geometry contract metadata
when it exists in ``model_dict["metadata"]["geometry_contract"]``.

This ensures that once a legacy model is loaded and completed, its next saved
HDF5 form contains the completed contract and can load without recomputation.

Migration Guidance
------------------

Recommended for application and downstream code:

- Do import from ``pyampp.io`` for model load/save.
- Do import from ``pyampp.geometry`` for geometry operations.
- Do not use ``pyampp.gxbox.boxutils.read_b3d_h5`` as an app-level load path.
  It is a low-level reader and does not enforce model normalization.

This separation removes duplicated fallback logic in downstream consumers and
keeps one authoritative contract-completion path in pyAMPP.

Thin Metadata CLI
-----------------

Thin IO APIs:

- ``load_geometry_contract_and_observer_from_h5(path)``
  returns a thin model with full ``metadata`` and optional ``observer``
  only when ``metadata/geometry_contract`` exists (otherwise ``None``).
- ``save_thin_model_to_h5(thin_model, path)``
  writes a lightweight HDF5 containing only ``metadata`` and optional
  ``observer`` sections.
- ``export_thin_model_from_h5(source_h5, output_h5=None, strict=False)``
  is the public convenience helper to generate a metadata-only artifact
  directly from a full model HDF5.

For quick validation/testing without loading full model payloads, use:

.. code-block:: bash

  h5thin /path/to/model.h5

JSON output for scripts:

.. code-block:: bash

  h5thin /path/to/model.h5 --json

To fail CI/preflight when contract metadata is missing:

.. code-block:: bash

  h5thin /path/to/model.h5 --require-contract

Exit code behavior:

- ``0``: command succeeded (contract present, or not required)
- ``2``: contract missing and ``--require-contract`` was set
