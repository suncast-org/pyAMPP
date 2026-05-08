pyAMPP: Python Automatic Model Production Pipeline
==================================================

**pyAMPP** is a Python implementation of the Automatic Model Production Pipeline (AMPP) for solar coronal modeling.  
It streamlines the process of generating realistic 3D solar atmosphere models with minimal user input.


Documentation
-------------

Full documentation is available at:
https://pyampp.readthedocs.io/en/latest/

Format and viewer references:

- Canonical model I/O and contract enforcement:
  ``docs/model_io.rst``
- Upgraded HDF5 stage format (NONE/BND/POT/NAS/NAS.GEN/NAS.CHR):
  ``docs/model_hdf5_format.rst``
- Public geometry API and migration status:
  ``docs/geometry_api.rst``
- Practical ``gx-fov2box`` CLI guide:
  ``docs/gx_fov2box_usage.rst``
- Main GUI workflow and command mapping:
  ``docs/gui_workflow.rst``
- Viewer guide (2D selector, 3D viewer, and refmap browser):
  ``docs/viewers.rst``
- Release notes:
  ``CHANGELOG.rst``

Overview
--------

**AMPP** automates the production of 3D solar models by:

- Downloading vector magnetic field data from the Helioseismic and Magnetic Imager (HMI) onboard the Solar Dynamics Observatory (SDO)
- Optionally downloading contextual Atmospheric Imaging Assembly (AIA) data
- Performing magnetic field extrapolations (Potential and/or Nonlinear Force-Free Field)
- Generating synthetic plasma emission models assuming either steady-state or impulsive heating
- Producing non-LTE chromospheric models constrained by photospheric measurements
- Enabling interactive inspection and customization through focused GUIs


Installation
------------

pyAMPP can be installed directly from PyPI.

0.  **Setting up a Python 3.10 Environment (Recommended)**

We **strongly recommend** running **pyAMPP** in a dedicated Python 3.10 environment.
You may use any of the following tools to install Python and create an isolated environment:

- Miniforge: https://github.com/conda-forge/miniforge
- Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install
- Anaconda: https://www.anaconda.com/docs/getting-started/anaconda/install
- Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
- pyenv: https://github.com/pyenv/pyenv?tab=readme-ov-file#installation

Please refer to the official installation instructions for your chosen tool. Their documentation is comprehensive and up to date, so we will not repeat it here.

*You can skip this environment setup section if you already have a suitable Python environment.*

The instructions below use Conda as an example:

1. **Install Conda**
   If you don’t have Conda yet, follow the instructions for your platform here:
   https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

2. **Create and activate your environment**
   Open an Anaconda Prompt or Command Prompt and run:

   .. code-block:: bash

      conda create -n suncast python=3.10
      conda activate suncast

3. **Upgrade pip and install pyampp**

   .. code-block:: bash

      pip install -U pip setuptools wheel
      pip install -U pyampp

4. **(Optional) Install additional dependencies**

   For a full scientific Python stack (e.g., SunPy and related tools):

   .. code-block:: bash

      pip install -U sunpy[all]

Development And Test Setup
--------------------------

If you are developing ``pyAMPP`` from a local checkout, install it in editable mode from the repository root:

.. code-block:: bash

  pip install -e .

This installs the package and its core runtime dependencies, but it does **not** install optional dependency groups such as the test tools.

The optional dependency groups are declared in ``pyproject.toml`` under ``[project.optional-dependencies]``.
Use the extras that match your task:

.. code-block:: bash

  pip install -e ".[tests]"
  pip install -e ".[docs]"
  pip install -e ".[tests,docs]"

For local test runs, the recommended setup is:

.. code-block:: bash

  pip install -U pip setuptools wheel
  pip install -e ".[tests]"
  python -m pytest -q

Using ``python -m pytest`` is recommended because it guarantees that ``pytest`` runs from the same environment as the interpreter where ``pyAMPP`` and its test extras were installed.
If ``pytest`` reports an unknown option such as ``--doctest-rst``, the active shell is usually picking up a different environment that does not have ``pytest-doctestplus`` installed.

We currently keep test and documentation dependencies in ``pyproject.toml`` extras rather than separate requirements files.
That keeps the install metadata in one authoritative place and lets users opt into ``tests`` and ``docs`` explicitly.

Main Applications
-----------------

pyAMPP exposes one main GUI and three focused command-line tools:

1. **pyampp** – The main desktop application. It is the front-end to ``gx-fov2box``:
   - captures model parameters,
   - builds the exact reproducible CLI command,
   - runs the command,
   - opens the follow-up viewers when needed.
2. **gx-fov2box** – Headless model generator (IDL ``gx_fov2box`` equivalent).
3. **gxbox-view2d** – 2D FOV / box selector for existing ``.h5`` or ``.sav`` models.
4. **gxbox-view3d** – 3D viewer for existing model HDF5 files.
5. **gxrefmap-view** – 2D browser for base maps and reference maps stored in model HDF5 files.

Canonical Model I/O And Geometry Contract
-----------------------------------------

pyAMPP now provides a centralized model I/O surface via ``pyampp.io``.
This is the recommended path for application-level model loading and saving.

Key behavior:

- all model restores (H5 and SAV) are normalized through one loader path,
- missing Tier 1 + Tier 2 geometry contract fields in legacy models are
  inferred at load time,
- completed contract metadata is persisted on save,
- geometry consumers can assume one normalized metadata state.

Use these APIs:

.. code-block:: python

    from pyampp import io, geometry

    model = io.load_model_from_h5("/path/to/model.h5")
    contract = model["metadata"]["geometry_contract"]
    corners = geometry.world_corners_from_geometry_contract(contract)

Low-level readers under ``pyampp.gxbox.boxutils`` remain available for internal
and compatibility use, but they are not the canonical application-level load
entrypoint for contract-enforced workflows.

Usage Examples
--------------

**1. Launch the time/coord selector (pyampp)**

.. code-block:: bash

    pyampp

.. image:: docs/images/pyampp_gui.png
    :alt: pyampp GUI screenshot
    :align: center
    :width: 600px

**2. Generate a model via CLI (gx-fov2box)**

.. code-block:: bash

    gx-fov2box \
      --time "2022-03-30T17:22:37" \
      --coords 34.44988566346035 14.26110705696788 \
      --hgs \
      --box-dims 360 180 200 \
      --dx-km 1400 \
      --pad-frac 0.25 \
      --data-dir /path/to/download_dir \
      --gxmodel-dir /path/to/gx_models_dir \
      --save-potential \
      --save-bounds

**3. Open the 2D FOV / box selector for an existing model**

.. code-block:: bash

    gxbox-view2d /path/to/model.h5

**4. Open the 3D viewer for an existing model**

.. code-block:: bash

    gxbox-view3d /path/to/model.h5

Notes:

- `--coords` takes two floats, separated by space (no brackets or commas).
- One of `--hpc`, `--hgc`, or `--hgs` must be specified to define the coordinate system.
- Remaining parameters are optional and have default values.
- The default downloader backend is ``DRMS``.
- Use ``--use-fido`` to switch to the legacy SunPy/Fido downloader.
- Use ``--force-download`` to bypass cache reuse and benchmark the downloader path directly.
- The GUI mirrors these controls with a ``Downloader`` selector and a ``Use cache`` checkbox.
- The GUI bottom command toolbar includes command-export actions:
  - copy the current ``gx-fov2box`` command
  - save the current command as a shell script
- Set ``PYAMPP_JSOC_NOTIFY_EMAIL`` to override the JSOC export notification email (default: ``suncasa-group@njit.edu``).

Entry-Box Resume / Jump Rules
-----------------------------

``gx-fov2box`` supports ``--entry-box`` with ``.h5`` or ``.sav`` input for stage-aware resume/recompute.

- ``--rebuild``: ignore stage payload and recompute from ``NONE`` using resolved entry parameters.
- ``--clone-only``: convert/copy an entry box to normalized HDF5 without recomputation (useful as ``convert-from-sav``).
- ``--clone-only`` is the exact normalization path. By contrast, ``--jump2chromo`` resumes at CHR and recomputes that stage from the stored entry payload.
- Without ``--jump2*``, the pipeline starts from the detected entry stage.
- When ``--entry-box`` contains ``metadata/execute``, ``--data-dir`` and ``--gxmodel-dir`` default from that execute string.
- Explicit CLI values for ``--data-dir`` / ``--gxmodel-dir`` always override execute-derived defaults.
- Jump validation rules are enforced:
  - backward jumps are allowed,
  - forward jumps are allowed by one stage,
  - ``POT -> NAS`` is explicitly allowed (implicit ``POT -> BND -> NAS``),
  - ``POT -> GEN`` is explicitly allowed (skip both ``BND`` and ``NAS/NLFFF``; keep true POT vectors).
- Save requests for stages before the selected start stage are ignored with a warning.
- In ``--clone-only`` mode, only no-jump or jump-to-self is allowed.

Command-Line Tools
------------------

After installation, the following commands become available:

- ``pyampp``: Launch the command-builder GUI.
- ``gx-fov2box``: Run the model-generation pipeline headlessly.
- ``gxbox-view2d``: Open the 2D FOV / box selector on an existing model or entry box.
- ``gxbox-view3d``: Open an existing HDF5 model in the 3D viewer.
- ``gxrefmap-view``: Open base/refmaps from an HDF5 model in a 2D map browser.
- ``h5tree``: Print an HDF5 tree (metadata shown by default, plus ``observer/name``, optional ``observer/label`` / ``observer/source``, and ``observer/pb0r/*`` when present; ``--no-metadata`` hides these value lines, ``--meta`` prints them only).
- ``gx-idl2fov2box``: Translate IDL ``gx_fov2box`` execute strings (or SAV ``EXECUTE``) into Python ``gx-fov2box`` commands.
- ``gx-fov2box2idl``: Translate Python ``gx-fov2box`` commands (or HDF5 ``metadata/execute``) into simple IDL ``gx_fov2box`` calls.
- ``h5thin``: Inspect only ``metadata/geometry_contract`` and optional ``observer`` metadata from an HDF5 model.

License
-------

Copyright (c) 2024, `SUNCAST <https://github.com/suncast-org/>`_ team. Released under the 3-clause BSD license.
