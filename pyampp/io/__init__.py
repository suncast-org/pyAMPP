"""I/O module for model loading and saving with contract enforcement."""

from __future__ import annotations

from typing import Any

from .model import (
    export_thin_model,
    load_model,
    load_model_metadata,
    save_model,
    save_thin_model,
)
from .refmaps import (
    AddedRefmap,
    RemovedRefmap,
    add_fits_refmaps_from_dir_to_h5,
    add_fits_refmaps_to_h5,
    box_corners_world_from_model,
    build_fits_refmaps_for_model,
    build_refmap_payload_for_model,
    discover_fits_refmap_map_ids,
    discover_fits_refmap_paths,
    infer_fits_refmap_id,
    list_embedded_refmap_ids,
    model_obstime_from_base_index,
    remove_refmaps_from_h5,
)

# refmap_crop is loaded lazily: it imports helpers from gx_fov2box, while
# gx_fov2box imports this package for load_model / refmaps. Eagerly importing
# crop symbols here recreates that cycle for the gx-fov2box entry point.

_REFMAP_CROP_EXPORTS = {
    "RefmapCropResult": "RefmapCropResult",
    "crop_refmap_spatial": "crop_refmap_spatial",
    "crop_refmap_to_model_box": "crop_refmap_to_model_box",
    "crop_refmap_to_model_box_after_pangle_rotation": (
        "crop_refmap_to_model_box_after_pangle_rotation"
    ),
    "crop_fov_xy_from_inscribing_box": "crop_fov_xy_from_inscribing_box",
    "compute_crop_fov_for_observer": "compute_crop_fov_for_observer",
    "compute_crop_fov_box_for_observer": "compute_crop_fov_box_for_observer",
    "compute_inscribing_fov_box_for_observer": "compute_inscribing_fov_box_for_observer",
    "infer_model_obstime_from_box_corners": "infer_model_obstime_from_box_corners",
    "make_empty_observer_fov_map": "make_empty_observer_fov_map",
    "mask_pixels_not_visible_from_source": "mask_pixels_not_visible_from_source",
    "plot_inscribing_fov_box_on_axes": "plot_inscribing_fov_box_on_axes",
    "project_fov_box_edges_to_observer_hpc": "project_fov_box_edges_to_observer_hpc",
    "reproject_map_to_target_observer_fov": "reproject_map_to_target_observer_fov",
    "reproject_refmap_to_observer": "reproject_refmap_to_observer",
    "rotate_refmap_for_display": "rotate_refmap_for_display",
}

__all__ = [
    "AddedRefmap",
    "RemovedRefmap",
    "RefmapCropResult",
    "add_fits_refmaps_from_dir_to_h5",
    "add_fits_refmaps_to_h5",
    "box_corners_world_from_model",
    "build_fits_refmaps_for_model",
    "build_refmap_payload_for_model",
    "crop_refmap_spatial",
    "compute_crop_fov_for_observer",
    "compute_crop_fov_box_for_observer",
    "compute_inscribing_fov_box_for_observer",
    "crop_fov_xy_from_inscribing_box",
    "crop_refmap_to_model_box",
    "crop_refmap_to_model_box_after_pangle_rotation",
    "discover_fits_refmap_map_ids",
    "discover_fits_refmap_paths",
    "infer_fits_refmap_id",
    "infer_model_obstime_from_box_corners",
    "list_embedded_refmap_ids",
    "make_empty_observer_fov_map",
    "mask_pixels_not_visible_from_source",
    "plot_inscribing_fov_box_on_axes",
    "project_fov_box_edges_to_observer_hpc",
    "reproject_map_to_target_observer_fov",
    "reproject_refmap_to_observer",
    "rotate_refmap_for_display",
    "remove_refmaps_from_h5",
    "export_thin_model",
    "load_model",
    "load_model_metadata",
    "model_obstime_from_base_index",
    "save_model",
    "save_thin_model",
]


def __getattr__(name: str) -> Any:
    attr = _REFMAP_CROP_EXPORTS.get(name)
    if attr is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import refmap_crop

    value = getattr(refmap_crop, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
