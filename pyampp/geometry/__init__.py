"""Public geometry import surface for pyAMPP.

This package promotes the existing gxbox geometry and observer helpers into a
headless, reusable module that other pyAMPP components and external consumers
can import without depending on the gxbox viewer entrypoints.
"""

from .core import (
    build_fov_box_from_red_box_world,
    build_fov_box_from_user_hpc_and_red_box_world,
    compute_inscribing_fov_box_from_world,
    compute_inscribing_fov_from_hpc,
    compute_inscribing_fov_from_world,
    local_cartesian_to_world,
    make_observer_wcs_header,
    observer_fov_box_to_world_corners,
    observer_rectangle_to_hpc_corners,
    project_box_front_face_to_observer_hpc,
    project_coordinate_edges_to_observer_hpc,
    project_world_to_observer_hcc,
    project_world_to_observer_hpc,
    project_world_to_pixel,
    world_to_local_cartesian_mm,
)
from .contract import (
    GeometryContract,
    RSUN_HMI_METERS,
    complete_geometry_contract,
    infer_box_dims,
    infer_voxel_resolution,
    infer_world_anchor_from_index,
    infer_obstime,
    world_corners_from_geometry_contract,
)
from .observer import (
    build_ephemeris_from_pb0r,
    build_pb0r_metadata_from_ephemeris,
    normalize_observer_key,
    resolve_named_observer,
    resolve_observer_from_metadata,
    resolve_observer_with_info,
    resolve_sdo_observer_from_b3d,
)

__all__ = [
    "compute_inscribing_fov_box_from_world",
    "compute_inscribing_fov_from_hpc",
    "compute_inscribing_fov_from_world",
    "build_fov_box_from_red_box_world",
    "build_fov_box_from_user_hpc_and_red_box_world",
    "local_cartesian_to_world",
    "make_observer_wcs_header",
    "observer_fov_box_to_world_corners",
    "observer_rectangle_to_hpc_corners",
    "project_box_front_face_to_observer_hpc",
    "project_coordinate_edges_to_observer_hpc",
    "project_world_to_observer_hcc",
    "project_world_to_observer_hpc",
    "project_world_to_pixel",
    "world_to_local_cartesian_mm",
    "GeometryContract",
    "RSUN_HMI_METERS",
    "complete_geometry_contract",
    "infer_box_dims",
    "infer_voxel_resolution",
    "infer_world_anchor_from_index",
    "infer_obstime",
    "world_corners_from_geometry_contract",
    "build_ephemeris_from_pb0r",
    "build_pb0r_metadata_from_ephemeris",
    "normalize_observer_key",
    "resolve_named_observer",
    "resolve_observer_from_metadata",
    "resolve_observer_with_info",
    "resolve_sdo_observer_from_b3d",
    "Box",
    "BoxGeometryMixin",
]


def __getattr__(name: str):
    if name in {"Box", "BoxGeometryMixin"}:
        from .core import Box, BoxGeometryMixin

        exports = {
            "Box": Box,
            "BoxGeometryMixin": BoxGeometryMixin,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")