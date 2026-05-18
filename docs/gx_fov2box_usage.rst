gx-fov2box Usage Guide
======================

``gx-fov2box`` is the headless model-generation CLI. It is the Python-side
equivalent of the IDL ``gx_fov2box`` workflow and is the command executed by the
main ``pyampp`` GUI.

Minimum Inputs
--------------

Fresh runs need:

- ``--time`` observation time in ISO format
- ``--coords X Y`` box center
- exactly one coordinate frame flag:
  - ``--hpc``
  - ``--hgc``
  - ``--hgs``
- ``--box-dims NX NY NZ``
- ``--dx-km``

Typical repository paths:

- ``--data-dir`` for downloaded SDO source data
- ``--gxmodel-dir`` for generated model stages

Projection choice:

- ``--cea`` for Carrington Equal Area basemaps
- ``--top`` for top-view basemaps

If neither is given, the current workflow normally expects ``--cea`` explicitly.

Stage Model
-----------

``gx-fov2box`` can save these stages:

- ``NONE`` via ``--save-empty-box``
- ``POT`` via ``--save-potential``
- ``BND`` via ``--save-bounds``
- ``NAS`` via ``--save-nas``
- ``NAS.GEN`` via ``--save-gen``
- ``NAS.GEN.CHR`` or ``NAS.CHR`` via ``--save-chr``

Execution stop point is controlled by:

- ``--stop-after dl|none|pot|bnd|nas|gen|chr``

Legacy convenience flags still exist:

- ``--empty-box-only``
- ``--potential-only``
- ``--nlfff-only``
- ``--generic-only``

Recommended practice is to prefer the explicit ``--stop-after`` form.

Canonical Examples
------------------

Fresh build to ``NONE`` only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   gx-fov2box \
     --time 2025-11-26T15:47:52 \
     --coords -280 -230 \
     --hpc \
     --cea \
     --box-dims 150 100 100 \
     --dx-km 1400 \
     --data-dir /path/to/jsoc_cache \
     --gxmodel-dir /path/to/gx_models \
     --save-empty-box \
     --stop-after none

Full fresh build to ``CHR``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   gx-fov2box \
     --time 2025-11-26T15:47:52 \
     --coords -280 -230 \
     --hpc \
     --cea \
     --box-dims 150 100 100 \
     --dx-km 1400 \
     --data-dir /path/to/jsoc_cache \
     --gxmodel-dir /path/to/gx_models \
     --euv \
     --uv \
     --save-potential \
     --save-bounds \
     --save-nas \
     --save-gen \
     --save-chr \
     --stop-after chr

Continue from an existing entry box
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   gx-fov2box \
     --entry-box /path/to/model.h5 \
     --save-gen \
     --save-chr \
     --stop-after chr

In this mode, timing/geometry/stage defaults are restored from the entry box
metadata when available.

Rebuild from ``NONE`` using an entry box as source context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   gx-fov2box \
     --entry-box /path/to/model.sav \
     --data-dir /path/to/jsoc_cache \
     --gxmodel-dir /path/to/gx_models \
     --save-empty-box \
     --save-potential \
     --save-bounds \
     --save-nas \
     --save-gen \
     --save-chr \
     --stop-after chr \
     --rebuild-from-none

This is the most useful path when you want:

- IDL-derived geometry/context as the starting point
- a fresh forward run through the Python stages

Clone or normalize an entry box without recomputation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pyampp-export-model \
     --model-path /path/to/model.sav \
     --out-h5 /path/to/output_dir/model.h5

This is the simplest way to convert a legacy SAV entry box into normalized HDF5.

Importing IDL-Generated Entry Boxes
-----------------------------------

``gx-fov2box`` can use IDL-generated ``.sav`` entry boxes directly.

This is mainly an expert workflow, but it is the right starting point when you
want:

- parity checks against an existing IDL model,
- to preserve an IDL-derived ``NONE`` basemap/context state,
- to continue later Python stages from an IDL-produced entry box,
- to normalize a legacy SAV box into HDF5 before downstream inspection or rendering.

Recommended patterns:

- Convert an IDL SAV box to normalized HDF5 without recomputing:

  .. code-block:: bash

     pyampp-export-model \
       --model-path /path/to/model.sav \
       --out-h5 /path/to/output_dir/model.h5

  Use this when you want the closest HDF5 equivalent of the original SAV entry.
  This path does not rerun ``POT``, ``NAS``, ``GEN``, or ``CHR``.

- Start from an IDL ``NONE``-equivalent entry and recompute later stages in Python:

  .. code-block:: bash

     gx-fov2box \
       --entry-box /path/to/model.sav \
       --data-dir /path/to/jsoc_cache \
       --gxmodel-dir /path/to/gx_models \
       --save-empty-box \
       --save-potential \
       --save-bounds \
       --save-nas \
       --save-gen \
       --save-chr \
       --stop-after chr \
       --rebuild-from-none

- Resume from an existing entry stage and recompute only the later stages:

  .. code-block:: bash

     gx-fov2box \
       --entry-box /path/to/model.sav \
       --save-chr \
       --stop-after chr \
       --jump2chromo

  This is not a pure clone. For a ``NAS.CHR`` entry it will rebuild the CHR
  products from the stored corona/base inputs and write a new HDF5 result.

Practical parity note:

- If you want strict downstream numerical parity beyond the imported ``NONE``
  stage, IDL and pyAMPP should use the same NLFFF library/version.
- Small WCS/header differences between fresh SunPy-native pyAMPP basemaps and
  IDL basemaps do not by themselves imply a scientific error.
- Small coronal/chromospheric magnetic-cube differences may also occur even
  when both workflows start from the same imported IDL entry box. This is an
  expected implementation difference, not a hidden import bug:
  - the IDL POT stage uses an FFT-based method
  - the Python POT stage uses the extrapolation-library solver path
- ``pyAMPP`` can reuse an existing source-data repository that was already
  populated by the IDL downloader. Point ``--data-dir`` at that shared cache
  when running parity checks to avoid unnecessary redownloads and to keep both
  workflows anchored to the same downloaded source products.
- The inverse is not fully symmetric today: files downloaded later by
  ``pyAMPP`` are not added to the legacy IDL ``index.sav`` repository index.
  An IDL session therefore may not discover Python-downloaded products through
  its normal repository scan unless the IDL-side index is rebuilt or updated.
- When the purpose is a controlled IDL-vs-Python comparison, starting from an
  IDL-generated entry box is the cleanest workflow.

Inspect resolved defaults before running
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   gx-fov2box \
     --entry-box /path/to/model.h5 \
     --info

Downloader Behavior
-------------------

Default behavior uses the DRMS-backed downloader/cache workflow.

Useful flags:

- ``--use-fido`` to switch to the legacy SunPy/Fido path
- ``--force-download`` to bypass cache reuse
- ``--euv`` to include AIA EUV context maps
- ``--uv`` to include AIA UV context maps

The download stage can be exercised by itself with:

.. code-block:: bash

   gx-fov2box \
     --time 2025-11-26T15:47:52 \
     --coords -280 -230 \
     --hpc \
     --cea \
     --box-dims 150 100 100 \
     --dx-km 1400 \
     --data-dir /path/to/jsoc_cache \
     --gxmodel-dir /path/to/gx_models \
     --euv \
     --uv \
     --stop-after dl

Entry-Box Workflow Flags
------------------------

``--entry-box`` accepts ``.h5`` or ``.sav`` inputs.

Important control flags:

- ``--rebuild``:
  recompute from ``NONE`` using parameters restored from the entry box
- ``--rebuild-from-none``:
  treat the entry as the authoritative ``NONE``-equivalent source and run
  forward from there
- ``--jump2potential``
- ``--jump2bounds``
- ``--jump2nlfff``
- ``--jump2lines``
- ``--jump2chromo``

The jump flags are mainly for controlled resume/testing workflows. For ordinary
use, start from the detected entry stage or use one of the explicit rebuild
modes.

Supported expert-only jump behavior:

- The GUI does not expose the jump flags as general controls. They remain
  CLI-level expert options for controlled resume/testing workflows.
- Some allowed jump requests expand through implicit intermediate stages:
  - ``NONE -> BND`` expands through ``POT``
  - ``POT -> NAS`` expands through ``BND``
- Direct expert shortcuts currently supported by the CLI include:
  - ``--jump2lines`` from a ``POT`` entry, producing ``POT.GEN``
  - ``--jump2chromo`` from a ``NAS`` entry, producing a fresh chromo
    augmentation without requiring existing GEN metadata
- Unsupported jump requests are rejected explicitly rather than guessed.

Observer and Display Metadata
-----------------------------

These options store observer/display metadata into the output model:

- ``--observer-name``
- ``--fov-xc``
- ``--fov-yc``
- ``--fov-xsize``
- ``--fov-ysize``
- ``--square-fov``

These are primarily used by the 2D viewer/resume workflow and do not replace
the core model box geometry inputs.

Related Tools
-------------

- Use ``pyampp`` for the GUI command builder.
- Use ``gxbox-view2d`` to inspect or edit saved observer/FOV metadata.
- Use ``gxbox-view3d`` to inspect a saved 3D model.
- Use ``gxrefmap-view`` to inspect saved base maps and reference maps.
