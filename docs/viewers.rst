Model Viewers
=============

gxbox-view3d
------------

``gxbox-view3d`` is the interactive 3D inspection tool for an existing model HDF5 file.
It does not recompute the model.

Expected inputs:

- A model file containing at least one of:
  - ``corona`` (preferred),
  - ``nlfff`` / ``pot`` (accepted and normalized), or
  - ``chromo`` with sufficient magnetic cube fields.

Usage:

.. code-block:: bash

   gxbox-view3d /path/to/model.h5

Optional file picker mode:

.. code-block:: bash

   gxbox-view3d --pick

gxbox-view2d
------------

``gxbox-view2d`` is the interactive 2D geometry and context-map tool for an existing
``.h5`` or ``.sav`` box file.

Behavior:

- loads geometry, stage metadata, and available embedded/base/refmap context from the selected box
- when a box contains ``metadata/execute`` paths, discovers matching filesystem
  context maps from ``--data-dir`` and additional FITS maps from
  ``--refmaps-path``
- persists updated observer/FOV metadata back to ``.h5`` boxes when the dialog is accepted
- supports built-in, manual custom, and uploaded-reference observer workflows in the observer panel
- exposes ``MODEL-DATE`` separately from the true observer ``OBS-DATE`` when the observer record comes from an external reference
- is used by the main ``pyampp`` GUI after explicit stop-stage completions (POT, NAS, GEN, CHR)

Usage:

.. code-block:: bash

   gxbox-view2d /path/to/model.h5

Optional file picker mode:

.. code-block:: bash

   gxbox-view2d --pick

Additional filesystem reference-map paths:

.. code-block:: bash

   gxbox-view2d /path/to/model.h5 --refmaps-path /path/to/refmaps

``--refmaps-path`` may be passed more than once. It accepts a FITS file or a
directory of FITS files. ``gx-fov2box`` and the GUI are intended for
**directories** of external FITS maps; ``gxbox-view2d`` also accepts a single
FITS file for interactive context. The shared ``pyampp.io.refmaps`` discovery
policy applies here, so recognized AIA maps appear with their wavelength labels
and EOVSA maps appear with frequency labels such as ``EOVSA_f1.418GHz``.

gxrefmap-view
-------------

``gxrefmap-view`` is a lightweight 2D browser for base maps and refmaps stored in model HDF5 files.

Expected layout:

- Base maps:
  - ``base/bx``, ``base/by``, ``base/bz``, ``base/ic``, ``base/chromo_mask``
  - base header in ``base/index`` (fallbacks: ``base/index_header``, ``base/wcs_header``)
- Refmaps:
  - ``refmaps/<map_id>/data``
  - ``refmaps/<map_id>/wcs_header``

Usage:

.. code-block:: bash

   gxrefmap-view /path/to/model.h5

List available maps only:

.. code-block:: bash

   gxrefmap-view /path/to/model.h5 --list

Start from a specific map:

.. code-block:: bash

   gxrefmap-view /path/to/model.h5 --start AIA_94

h5tree
------

``h5tree`` prints a compact tree of HDF5 groups/datasets and their shapes/dtypes.
It accepts either canonical ``.h5`` input or legacy ``.sav`` input.
For ``.sav`` input, the file is normalized through ``load_model()`` first and
the tree is printed from that canonical in-memory structure, so the reported
tree matches what a ``load_model()`` consumer receives without paying for an
extra default HDF5 rewrite. Use ``--save-normalized`` when you also want the
canonical HDF5 written to disk.

Metadata behavior:

- Metadata values (``metadata/*``) are shown by default.
- Observer summary values (``observer/name``, optional ``observer/label`` / ``observer/source``, and ``observer/pb0r/*``) are also shown by default when present.
- Use ``--no-metadata`` to suppress metadata value lines.
- Use ``--meta`` to print only metadata values (no tree output).

Usage:

.. code-block:: bash

   h5tree /path/to/model.h5

Inspect the canonical normalized tree for a legacy SAV:

.. code-block:: bash

   h5tree /path/to/model.sav

Persist that normalized canonical HDF5 while also printing its tree:

.. code-block:: bash

   h5tree /path/to/model.sav --save-normalized /path/to/model.normalized.h5

Suppress metadata values:

.. code-block:: bash

   h5tree /path/to/model.h5 --no-metadata

Metadata-only mode:

.. code-block:: bash

   h5tree /path/to/model.h5 --meta

Role Summary
------------

- Use ``pyampp`` to configure and launch model-production runs.
- Use ``gxbox-view2d`` to inspect or edit FOV / box geometry on existing model files.
- Use ``gxbox-view3d`` to inspect the saved 3D magnetic model.
- Use ``gxrefmap-view`` when you only need to browse stored 2D maps.
