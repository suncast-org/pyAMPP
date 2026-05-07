"""Promoted observer-resolution helpers for geometry workflows.

These functions remain implemented in the tested gxbox observer restore module
for now. This public wrapper provides the stable import path that future
consumers, including gximagecomputing, should target.
"""

from pyampp.gxbox.observer_restore import (
    build_ephemeris_from_pb0r,
    build_pb0r_metadata_from_ephemeris,
    normalize_observer_key,
    resolve_named_observer,
    resolve_observer_from_metadata,
    resolve_observer_with_info,
    resolve_sdo_observer_from_b3d,
)

__all__ = [
    "build_ephemeris_from_pb0r",
    "build_pb0r_metadata_from_ephemeris",
    "normalize_observer_key",
    "resolve_named_observer",
    "resolve_observer_from_metadata",
    "resolve_observer_with_info",
    "resolve_sdo_observer_from_b3d",
]