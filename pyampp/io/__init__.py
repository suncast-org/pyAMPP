"""I/O module for model loading and saving with contract enforcement."""

from .model import (
    export_thin_model,
    load_model,
    load_model_metadata,
    save_model,
    save_thin_model,
)
from .refmaps import (
    AddedRefmap,
    add_fits_refmaps_from_dir_to_h5,
    add_fits_refmaps_to_h5,
    build_fits_refmaps_for_model,
    build_refmap_payload_for_model,
    discover_fits_refmap_map_ids,
    discover_fits_refmap_paths,
    infer_fits_refmap_id,
    model_obstime_from_base_index,
)

__all__ = [
    "AddedRefmap",
    "add_fits_refmaps_from_dir_to_h5",
    "add_fits_refmaps_to_h5",
    "build_fits_refmaps_for_model",
    "build_refmap_payload_for_model",
    "discover_fits_refmap_map_ids",
    "discover_fits_refmap_paths",
    "infer_fits_refmap_id",
    "export_thin_model",
    "load_model",
    "load_model_metadata",
    "model_obstime_from_base_index",
    "save_model",
    "save_thin_model",
]
