Public Geometry API
===================

The ``pyampp.geometry`` package is the public, headless geometry surface for
observer-aware projection and field-of-view calculations. It promotes the
tested gxbox geometry kernel into an import path that downstream code can rely
on without depending on the GUI or viewer entrypoints.

Canonical Model Loading Requirement
-----------------------------------

Geometry consumers should use ``pyampp.io`` for model restore so contract
completion and observer normalization happen before geometry APIs are called.

Use:

- ``pyampp.io.load_model_from_h5``
- ``pyampp.io.load_model_from_sav``

Avoid direct application-level restores through low-level readers such as
``pyampp.gxbox.boxutils.read_b3d_h5``, which do not enforce metadata
normalization by themselves.

Scope of This Update
--------------------

This update cycle establishes the public Python geometry contract in pyAMPP.
The immediate goals are:

- expose the stable import surface through ``pyampp.geometry``
- preserve existing gxbox behavior by routing current methods through wrappers
- keep all transforms vectorized for arbitrary point sets and box corners
- defer downstream adoption in gximagecomputing until this API is reviewed and merged upstream

Public Import Surface
---------------------

Downstream consumers should import from ``pyampp.geometry`` rather than from
``pyampp.gxbox`` internals.

Current public exports include:

- ``local_cartesian_to_world``
- ``project_world_to_observer_hpc``
- ``project_world_to_observer_hcc``
- ``compute_inscribing_fov_from_hpc``
- ``compute_inscribing_fov_from_world``
- ``compute_inscribing_fov_box_from_world``
- ``observer_fov_box_to_world_corners``
- ``observer_rectangle_to_hpc_corners``
- ``project_coordinate_edges_to_observer_hpc``
- ``project_box_front_face_to_observer_hpc``
- ``project_world_to_pixel``
- observer-resolution helpers re-exported from ``pyampp.geometry.observer``

Compatibility wrappers are still exposed through ``Box`` and
``BoxGeometryMixin``, but these names are transitional compatibility surfaces.
New downstream code should prefer the function-level API where possible.

Current Migration Status
------------------------

Already migrated in this branch:

- box local-to-world conversion
- observer HPC/HCC projection for box corners
- inscribing FOV and observer-aligned FOV-box computation
- saved ``fov_box`` world-corner reconstruction
- field-line projection in the 2D viewer
- FOV-box projected edge and front-face overlays in the 2D viewer

Still intentionally wrapper-backed:

- existing gxbox box methods remain available and delegate into the public core
- public observer helpers currently forward to the tested gxbox observer module

Deferred until after pyAMPP merge:

- replacing gximagecomputing's local Python geometry implementation with imports
  from ``pyampp.geometry``

Function Contracts
------------------

``local_cartesian_to_world(local_points_mm, frame, z_base_mm=0.0)``

- Input: an ``(N, 3)`` array of local Cartesian points in Mm
- Output: a vectorized ``SkyCoord`` in the supplied world frame
- Returns ``None`` when inputs are missing, invalid, or contain non-finite rows

``project_world_to_observer_hpc(world, observer=None, obstime=None, frame_obs=None)``

- Input: any vectorized coordinates with a meaningful frame
- Output: helioprojective coordinates for the resolved observer frame
- Returns ``None`` when the observer context cannot be resolved

``compute_inscribing_fov_from_world(world, ...)``

- Input: arbitrary 3D world-space point set
- Output: smallest axis-aligned observer HPC rectangle covering the projected points
- Returns a dictionary with center, extent, bounds, and projected coordinates

``compute_inscribing_fov_box_from_world(world, ...)``

- Input: arbitrary 3D world-space point set
- Output: observer-aligned 3D ``fov_box`` metadata with 2D footprint and Z extent
- Z padding remains explicit through ``pad_z_frac``

``observer_fov_box_to_world_corners(...)``

- Input: saved observer ``fov_box`` metadata and target frame
- Output: 8 reconstructed world-space corners suitable for reprojection or parity checks

``observer_rectangle_to_hpc_corners(...)``

- Input: observer-centered 2D rectangle definition in arcsec
- Output: 4 helioprojective rectangle corners in the specified observer frame

``project_coordinate_edges_to_observer_hpc(coords, edge_pairs, ...)``

- Input: vectorized coordinates and an edge-pair list
- Output: list of projected 2-point ``SkyCoord`` edges in observer HPC
- Intended for box overlays and arbitrary polyline edge sets

``project_box_front_face_to_observer_hpc(world_corners, ...)``

- Input: 8 world-space box corners
- Output: the observer-nearest projected face as a closed 5-point ``SkyCoord`` polygon

``project_world_to_pixel(world, smap)``

- Input: vectorized world coordinates and a SunPy map
- Output: two NumPy arrays ``(x, y)`` in pixel coordinates

Examples
--------

Load a model through the canonical I/O path, then call geometry:

.. code-block:: python

   from pyampp import io, geometry

   model = io.load_model_from_h5("/path/to/model.h5")
   contract = model["metadata"]["geometry_contract"]
   red_world = geometry.world_corners_from_geometry_contract(contract)

Convert local model points to world coordinates:

.. code-block:: python

   world = local_cartesian_to_world(local_points_mm, frame=box_center.frame, z_base_mm=grid_z_base)

Project arbitrary 3D points into an observer image plane:

.. code-block:: python

   coords_hpc = project_world_to_observer_hpc(world, frame_obs=frame_obs)
   xpix, ypix = project_world_to_pixel(coords_hpc, smap)

Compute an inscribing 2D FOV or 3D FOV-box from one vectorized point set:

.. code-block:: python

   fov = compute_inscribing_fov_from_world(world, frame_obs=frame_obs)
   fov_box = compute_inscribing_fov_box_from_world(world, frame_obs=frame_obs)

Reconstruct and reproject a saved observer ``fov_box``:

.. code-block:: python

   corners_world = observer_fov_box_to_world_corners(
       xc_arcsec=meta["xc_arcsec"],
       yc_arcsec=meta["yc_arcsec"],
       xsize_arcsec=meta["xsize_arcsec"],
       ysize_arcsec=meta["ysize_arcsec"],
       zmin_mm=meta["zmin_mm"],
       zmax_mm=meta["zmax_mm"],
       observer=observer,
       obstime=obs_time,
       target_frame=box_center.frame,
   )
   face = project_box_front_face_to_observer_hpc(corners_world, frame_obs=frame_obs)

Reviewer Notes
--------------

The intended review standard for this PR is behavioral parity with the current
gxbox workflows, not a broad geometry rewrite. The public API is designed to be
adopted downstream first, then become the single shared geometry path in later
repo updates.