from __future__ import annotations

"""Reusable API contracts for focused map/box selection GUIs.

This module intentionally contains no Qt code. It defines the data contracts
that a purpose-built visualization GUI (for example, a post-download
FOV/box-selector) can use to communicate geometry changes back to ``pyAMPP``.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class CoordMode(str, Enum):
    """Coordinate representation used by the GUI center fields."""

    HPC = "hpc"
    HGC = "hgc"
    HGS = "hgs"


@dataclass(slots=True)
class BoxGeometrySelection:
    """Workflow-relevant box geometry values in the same shape as pyAMPP GUI fields.

    These map directly to the existing ``PyAmppGUI`` input widgets:
    - ``coord_x`` / ``coord_y`` -> ``coord_x_edit`` / ``coord_y_edit``
    - ``grid_x`` / ``grid_y`` / ``grid_z`` -> ``grid_*_edit``
    - ``dx_km`` -> ``res_edit`` (CLI ``--dx-km``)
    """

    coord_mode: CoordMode
    coord_x: float
    coord_y: float
    grid_x: int
    grid_y: int
    grid_z: int
    dx_km: float

    def as_gui_text_fields(self) -> dict[str, str]:
        """Return values formatted for direct assignment to QLineEdit widgets."""
        return {
            "coord_x_edit": f"{self.coord_x:.2f}",
            "coord_y_edit": f"{self.coord_y:.2f}",
            "grid_x_edit": str(int(self.grid_x)),
            "grid_y_edit": str(int(self.grid_y)),
            "grid_z_edit": str(int(self.grid_z)),
            "res_edit": f"{float(self.dx_km):.2f}",
        }


@dataclass(slots=True)
class SelectorSessionInput:
    """Inputs needed to launch a future standalone map/box selector GUI.

    ``map_ids`` and ``initial_map_id`` are UI-only state for visualization and
    are not part of the geometry return contract.
    """

    time_iso: str
    data_dir: str
    geometry: BoxGeometrySelection
    fov: "DisplayFovSelection | None" = None
    fov_box: "DisplayFovBoxSelection | None" = None
    square_fov: bool = False
    allow_geometry_edit: bool = True
    map_ids: tuple[str, ...] = ()
    map_files: dict[str, str] | None = None
    external_ref_map_paths: tuple[str, ...] = ()
    refmaps: dict[str, dict[str, Any]] | None = None
    base_maps: dict[str, Any] | None = None
    base_wcs_header: str | None = None
    base_geometry: "BoxGeometrySelection | None" = None
    map_source_mode: str = "auto"
    display_observer_key: str = "earth"
    custom_observer_ephemeris: dict[str, Any] | None = None
    custom_observer_label: str | None = None
    custom_observer_source: str | None = None
    available_observer_keys: tuple[str, ...] | None = None
    observer_availability_notice: str | None = None
    initial_map_id: str | None = None
    pad_frac: float | None = None


@dataclass(slots=True)
class DisplayFovSelection:
    """Observer-plane image FOV used for synthetic image framing and canvas centering.

    This is intentionally separate from the model box geometry. Values are expressed
    in helioprojective (observer-plane) arcsec, regardless of the red-box input
    coordinate mode.
    """

    center_x_arcsec: float
    center_y_arcsec: float
    width_arcsec: float
    height_arcsec: float

    def as_gui_text_fields(self) -> dict[str, str]:
        return {
            "fov_x_edit": f"{self.center_x_arcsec:.2f}",
            "fov_y_edit": f"{self.center_y_arcsec:.2f}",
            "fov_w_edit": f"{self.width_arcsec:.2f}",
            "fov_h_edit": f"{self.height_arcsec:.2f}",
        }


@dataclass(slots=True)
class DisplayFovBoxSelection:
    """Observer-aligned 3D FOV box.

    The x/y footprint matches the image-plane FOV rectangle in helioprojective
    arcsec. The z extent is stored in observer-centric heliocentric coordinates
    (Mm) along the line of sight, where larger values are closer to the
    observer.
    """

    center_x_arcsec: float
    center_y_arcsec: float
    width_arcsec: float
    height_arcsec: float
    z_min_mm: float
    z_max_mm: float
    corners_local_mm: tuple[tuple[float, float, float], ...] | None = None
    observer_key: str = "earth"

    @classmethod
    def from_display_fov(
        cls,
        fov: "DisplayFovSelection",
        z_min_mm: float,
        z_max_mm: float,
        *,
        observer_key: str = "earth",
    ) -> "DisplayFovBoxSelection":
        return cls(
            center_x_arcsec=float(fov.center_x_arcsec),
            center_y_arcsec=float(fov.center_y_arcsec),
            width_arcsec=float(fov.width_arcsec),
            height_arcsec=float(fov.height_arcsec),
            z_min_mm=float(z_min_mm),
            z_max_mm=float(z_max_mm),
            observer_key=str(observer_key or "earth"),
        )

    def as_observer_metadata(self, *, square: bool = False) -> dict[str, float | str | bool]:
        meta: dict[str, Any] = {
            "frame": "observer_heliocentric",
            "xc_arcsec": float(self.center_x_arcsec),
            "yc_arcsec": float(self.center_y_arcsec),
            "xsize_arcsec": float(self.width_arcsec),
            "ysize_arcsec": float(self.height_arcsec),
            "zmin_mm": float(self.z_min_mm),
            "zmax_mm": float(self.z_max_mm),
            "square": bool(square),
            "observer_key": str(self.observer_key or "earth"),
        }
        return meta


@dataclass(slots=True)
class SelectorDialogResult:
    """Accepted result returned by the standalone selector dialog."""

    geometry: BoxGeometrySelection
    fov: "DisplayFovSelection | None" = None
    square_fov: bool = False


class GeometrySelectionConsumer(Protocol):
    """Callback protocol for receiving accepted geometry selections."""

    def apply_geometry_selection(self, selection: BoxGeometrySelection) -> None:
        """Apply an accepted geometry selection to the host GUI/application."""


__all__ = [
    "CoordMode",
    "BoxGeometrySelection",
    "DisplayFovSelection",
    "DisplayFovBoxSelection",
    "SelectorDialogResult",
    "SelectorSessionInput",
    "GeometrySelectionConsumer",
]
