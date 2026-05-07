import itertools
from contextlib import nullcontext

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import (
    HeliographicCarrington,
    Helioprojective,
    HeliographicStonyhurst,
    Heliocentric,
)
from sunpy.map import make_fitswcs_header
from pyampp.geometry.core import (
    compute_inscribing_fov_box_from_world,
    compute_inscribing_fov_from_world,
    local_cartesian_to_world,
    observer_fov_box_to_world_corners,
    project_world_to_observer_hcc,
    project_world_to_observer_hpc,
)
from pyampp.gxbox.observer_restore import resolve_observer_from_metadata, resolve_named_observer, resolve_observer_with_info
from pyampp.util.config import IDL_HMI_RSUN_M
try:
    from sunpy.coordinates.screens import SphericalScreen
except Exception:  # pragma: no cover
    SphericalScreen = None


class BoxGeometryMixin:
    """Shared geometric helpers for model and observer-aligned FOV boxes."""

    @staticmethod
    def _corners_from_bounds(xmin, xmax, ymin, ymax, zmin, zmax):
        return np.asarray(
            [
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmin, ymax, zmin],
                [xmax, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmin, ymax, zmax],
                [xmax, ymax, zmax],
            ],
            dtype=float,
        )

    def grid_zbase_mm(self) -> float:
        grid = getattr(self, "grid_coords", {})
        z = grid.get("z")
        if z is None:
            return 0.0
        try:
            return float(np.min(z.to_value(u.Mm)))
        except Exception:
            try:
                return float(np.min(z))
            except Exception:
                return 0.0

    def model_box_corners_local_mm(self) -> np.ndarray:
        grid = self.grid_coords
        xmin = float(np.min(grid["x"].to_value(u.Mm)))
        xmax = float(np.max(grid["x"].to_value(u.Mm)))
        ymin = float(np.min(grid["y"].to_value(u.Mm)))
        ymax = float(np.max(grid["y"].to_value(u.Mm)))
        zmin = float(np.min(grid["z"].to_value(u.Mm)))
        zmax = float(np.max(grid["z"].to_value(u.Mm)))
        corners = self._corners_from_bounds(xmin, xmax, ymin, ymax, zmin, zmax)
        corners[:, 2] -= zmin
        return corners

    def model_box_corners_world(self) -> SkyCoord | None:
        box_frame = getattr(getattr(self, "_center", None), "frame", None)
        if box_frame is None:
            return None
        local = self.model_box_corners_local_mm()
        return local_cartesian_to_world(local, frame=box_frame, z_base_mm=self.grid_zbase_mm())

    def model_box_corners_projected(self, observer=None, obstime=None) -> SkyCoord | None:
        """
        Project the physical red-box corners into a target observer's sky plane.

        Parameters
        ----------
        observer : optional
            Observer accepted by SunPy's ``Helioprojective`` frame, typically a
            ``SkyCoord`` in a solar frame. If omitted, the current box observer
            is used.
        obstime : optional
            Observation time for the target frame. If omitted, the current box
            frame time is used.

        Returns
        -------
        `astropy.coordinates.SkyCoord` or None
            The 8 model-box corners transformed into the target
            ``Helioprojective`` frame.
        """
        world = self.model_box_corners_world()
        if world is None:
            return None
        box_frame = getattr(world, "frame", None)
        frame_obs = getattr(self, "_frame_obs", None)
        if observer is None and frame_obs is not None:
            observer = getattr(frame_obs, "observer", None)
        if observer is None and box_frame is not None:
            observer = getattr(box_frame, "observer", None)
        if obstime is None and frame_obs is not None:
            obstime = getattr(frame_obs, "obstime", None)
        if obstime is None and box_frame is not None:
            obstime = getattr(box_frame, "obstime", None)
        if observer is None:
            return None
        return project_world_to_observer_hpc(world, observer=observer, obstime=obstime, frame_obs=frame_obs)

    def model_box_corners_observer_heliocentric(self, observer=None, obstime=None) -> SkyCoord | None:
        """
        Transform the physical red-box corners into an observer-centric
        heliocentric frame where ``z`` is aligned with the line of sight.
        """
        world = self.model_box_corners_world()
        if world is None:
            return None
        box_frame = getattr(world, "frame", None)
        frame_obs = getattr(self, "_frame_obs", None)
        if observer is None and frame_obs is not None:
            observer = getattr(frame_obs, "observer", None)
        if observer is None and box_frame is not None:
            observer = getattr(box_frame, "observer", None)
        if obstime is None and frame_obs is not None:
            obstime = getattr(frame_obs, "obstime", None)
        if obstime is None and box_frame is not None:
            obstime = getattr(box_frame, "obstime", None)
        if observer is None:
            return None
        return project_world_to_observer_hcc(world, observer=observer, obstime=obstime, frame_obs=frame_obs)

    def model_box_inscribing_fov(self, observer=None, obstime=None, pad_arcsec: float = 0.0) -> dict | None:
        """
        Compute the inscribing sky-plane rectangle for the physical red box.

        The returned rectangle is the smallest axis-aligned HPC rectangle that
        contains the projected 8 red-box corners for the given observer.
        """
        world = self.model_box_corners_world()
        return compute_inscribing_fov_from_world(
            world,
            observer=observer,
            obstime=obstime,
            frame_obs=getattr(self, "_frame_obs", None),
            pad_arcsec=pad_arcsec,
        )

    def model_box_inscribing_fov_box(
        self,
        observer=None,
        obstime=None,
        pad_xy_arcsec: float = 0.0,
        pad_z_frac: float = 0.10,
    ) -> dict | None:
        """
        Compute the full observer-aligned 3D FOV box that inscribes the red box.

        The x/y footprint is the smallest axis-aligned rectangle on the target
        observer's sky plane that contains the projected red-box corners. The z
        extent is derived from the red-box corners in the observer-centric
        heliocentric frame, with an optional fractional padding.
        """
        world = self.model_box_corners_world()
        return compute_inscribing_fov_box_from_world(
            world,
            observer=observer,
            obstime=obstime,
            frame_obs=getattr(self, "_frame_obs", None),
            pad_xy_arcsec=pad_xy_arcsec,
            pad_z_frac=pad_z_frac,
        )

    def fov_box_corners_world(self, fov_box_meta: dict | None = None) -> SkyCoord | None:
        observer_meta = self.b3d.get("observer", {}) if isinstance(getattr(self, "b3d", None), dict) else {}
        if not isinstance(observer_meta, dict):
            observer_meta = {}
        fov_box = fov_box_meta if isinstance(fov_box_meta, dict) else observer_meta.get("fov_box")
        if not isinstance(fov_box, dict):
            return None

        def _meta_float(key):
            try:
                return float(fov_box[key])
            except Exception:
                return None

        xc = _meta_float("xc_arcsec")
        yc = _meta_float("yc_arcsec")
        xsize = _meta_float("xsize_arcsec")
        ysize = _meta_float("ysize_arcsec")
        zmin = _meta_float("zmin_mm")
        zmax = _meta_float("zmax_mm")
        if any(v is None for v in (xc, yc, xsize, ysize, zmin, zmax)):
            return None

        frame_obs = getattr(self, "_frame_obs", None)
        box_frame = getattr(getattr(self, "_center", None), "frame", None)
        observer = getattr(frame_obs, "observer", None)
        obstime = getattr(frame_obs, "obstime", None)
        if observer is None and box_frame is not None:
            observer = getattr(box_frame, "observer", None)
        if obstime is None and box_frame is not None:
            obstime = getattr(box_frame, "obstime", None)
        # If metadata specifies an observer frame for the FOV box, resolve and use it.
        observer_key = fov_box.get("observer_key")
        resolved_observer = None
        if observer_key:
            resolved_observer, _warning, _used_key = resolve_observer_with_info(
                getattr(self, "b3d", None) if isinstance(getattr(self, "b3d", None), dict) else {},
                observer_key,
                obstime,
            )
        if resolved_observer is not None:
            observer = resolved_observer
        if observer is None or box_frame is None:
            return None

        return observer_fov_box_to_world_corners(
            xc_arcsec=xc,
            yc_arcsec=yc,
            xsize_arcsec=xsize,
            ysize_arcsec=ysize,
            zmin_mm=zmin,
            zmax_mm=zmax,
            observer=observer,
            obstime=obstime,
            target_frame=box_frame,
        )

    def fov_box_corners_local_mm(self, fov_box_meta: dict | None = None) -> np.ndarray | None:
        world = self.fov_box_corners_world(fov_box_meta=fov_box_meta)
        if world is None:
            return None
        zbase = self.grid_zbase_mm()
        try:
            return np.column_stack(
                [
                    world.x.to_value(u.Mm),
                    world.y.to_value(u.Mm),
                    world.z.to_value(u.Mm) - zbase,
                ]
            ).astype(float)
        except Exception:
            return None


class Box(BoxGeometryMixin):
    """
    Represents a 3D box in solar or observer coordinates defined by its origin, center, dimensions, and resolution.

    This class calculates and stores the coordinates of the box's edges, differentiating between bottom edges and other
    edges. It is designed to integrate with solar physics data analysis frameworks such as SunPy and Astropy.
    """

    def __init__(self, frame_obs, box_origin, box_center, box_dims, box_res):
        self._frame_obs = frame_obs
        if hasattr(Helioprojective, "assume_spherical_screen"):
            screen_ctx = Helioprojective.assume_spherical_screen(frame_obs.observer)
        elif SphericalScreen is not None:
            screen_ctx = SphericalScreen(frame_obs.observer)
        else:
            screen_ctx = nullcontext()
        with (screen_ctx or nullcontext()):
            self._origin = box_origin
            self._center = box_center
        self._dims = box_dims / u.pix * box_res
        self._res = box_res
        self._dims_pix = np.int_(box_dims.value)
        self.corners = list(itertools.product(self._dims[0] / 2 * [-1, 1],
                                              self._dims[1] / 2 * [-1, 1],
                                              self._dims[2] / 2 * [-1, 1]))
        self.edges = [edge for edge in itertools.combinations(self.corners, 2)
                      if np.count_nonzero(u.Quantity(edge[0]) - u.Quantity(edge[1])) == 1]
        self._bottom_edges = None
        self._non_bottom_edges = None
        self._calculate_edge_types()
        self.b3dtype = ['pot', 'nlfff']
        self.b3d = {"corona": None, "chromo": None}
        self.corona_models = {}
        self.corona_type = None

    @property
    def dims_pix(self):
        return self._dims_pix

    @property
    def grid_coords(self):
        return self._get_grid_coords(self._center)

    def _get_grid_coords(self, grid_center):
        grid_coords = {}
        grid_coords['x'] = np.linspace(grid_center.x.to(self._dims.unit) - self._dims[0] / 2,
                                       grid_center.x.to(self._dims.unit) + self._dims[0] / 2, self._dims_pix[0])
        grid_coords['y'] = np.linspace(grid_center.y.to(self._dims.unit) - self._dims[1] / 2,
                                       grid_center.y.to(self._dims.unit) + self._dims[1] / 2, self._dims_pix[1])
        grid_coords['z'] = np.linspace(grid_center.z.to(self._dims.unit) - self._dims[2] / 2,
                                       grid_center.z.to(self._dims.unit) + self._dims[2] / 2, self._dims_pix[2])
        grid_coords['frame'] = self._frame_obs
        return grid_coords

    def _get_edge_coords(self, edges, box_center):
        return [SkyCoord(x=box_center.x + u.Quantity([edge[0][0], edge[1][0]]),
                         y=box_center.y + u.Quantity([edge[0][1], edge[1][1]]),
                         z=box_center.z + u.Quantity([edge[0][2], edge[1][2]]),
                         frame=box_center.frame) for edge in edges]

    def _get_bottom_cea_header(self):
        origin = self._origin.transform_to(
            HeliographicCarrington(observer="earth", obstime=getattr(self._origin, "obstime", None))
        )
        shape = self._dims[:-1][::-1] / self._res.to(self._dims.unit)
        shape = list(shape.value)
        shape = [int(np.ceil(s)) for s in shape]
        rsun = (IDL_HMI_RSUN_M * u.m).to(self._res.unit)
        scale = np.arcsin(self._res / rsun).to(u.deg) / u.pix
        scale = u.Quantity((scale, scale))
        bottom_cea_header = make_fitswcs_header(shape, origin,
                                                scale=scale, projection_code='CEA')
        bottom_cea_header['OBSRVTRY'] = 'None'
        return bottom_cea_header

    def _get_bottom_top_header(self, dsun_obs=None):
        origin = self._origin.transform_to(self._frame_obs)
        shape = self._dims[:-1][::-1] / self._res.to(self._dims.unit)
        shape = list(shape.value)
        shape = [int(np.ceil(s)) for s in shape]
        if dsun_obs is None:
            dsun_obs = origin.observer.radius.to(self._res.unit)
        else:
            dsun_obs = u.Quantity(dsun_obs).to(self._res.unit)
        rsun = (IDL_HMI_RSUN_M * u.m).to(self._res.unit)
        # Match IDL TOP scaling: dx_arcsec ~ dx_km / (DSUN_OBS - RSUN).
        scale = ((self._res / (dsun_obs - rsun)).to(u.dimensionless_unscaled) * u.rad).to(u.arcsec) / u.pix
        scale = u.Quantity((scale, scale))
        bottom_top_header = make_fitswcs_header(shape, origin, scale=scale, projection_code='TAN')
        bottom_top_header['OBSRVTRY'] = 'None'
        return bottom_top_header

    def _calculate_edge_types(self):
        min_z = min(corner[2] for corner in self.corners)
        bottom_edges, non_bottom_edges = [], []
        for edge in self.edges:
            if edge[0][2] == min_z and edge[1][2] == min_z:
                bottom_edges.append(edge)
            else:
                non_bottom_edges.append(edge)
        self._bottom_edges = self._get_edge_coords(bottom_edges, self._center)
        self._non_bottom_edges = self._get_edge_coords(non_bottom_edges, self._center)

    def _get_bounds_coords(self, edges, bltr=False, pad_frac=0.0):
        xx = []
        yy = []
        for edge in edges:
            xx.append(edge.transform_to(self._frame_obs).Tx)
            yy.append(edge.transform_to(self._frame_obs).Ty)
        unit = xx[0][0].unit
        min_x = np.min(xx)
        max_x = np.max(xx)
        min_y = np.min(yy)
        max_y = np.max(yy)
        if pad_frac > 0:
            _pad = pad_frac * np.max([max_x - min_x, max_y - min_y, 20])
            min_x -= _pad
            max_x += _pad
            min_y -= _pad
            max_y += _pad
        if bltr:
            bottom_left = SkyCoord(min_x * unit, min_y * unit, frame=self._frame_obs)
            top_right = SkyCoord(max_x * unit, max_y * unit, frame=self._frame_obs)
            return [bottom_left, top_right]
        else:
            coords = SkyCoord(Tx=[min_x, max_x] * unit, Ty=[min_y, max_y] * unit,
                              frame=self._frame_obs)
            return coords

    def bounds_coords_bl_tr(self, pad_frac=0.0):
        return self._get_bounds_coords(self.all_edges, bltr=True, pad_frac=pad_frac)

    @property
    def bounds_coords(self):
        return self._get_bounds_coords(self.all_edges)

    @property
    def bottom_bounds_coords(self):
        return self._get_bounds_coords(self.bottom_edges)

    @property
    def bottom_cea_header(self):
        return self._get_bottom_cea_header()

    def bottom_top_header(self, dsun_obs=None):
        return self._get_bottom_top_header(dsun_obs=dsun_obs)

    @property
    def bottom_edges(self):
        return self._bottom_edges

    @property
    def non_bottom_edges(self):
        return self._non_bottom_edges

    @property
    def all_edges(self):
        return self.bottom_edges + self.non_bottom_edges

    @property
    def box_view_up(self):
        """
        Return two nearby points in the observer frame that define the local +Y (screen up) direction.
        The viewer converts these to Heliocentric and takes their vector difference.
        """
        origin_obs = self._origin.transform_to(self._frame_obs)
        delta = 10 * u.arcsec
        return SkyCoord(
            Tx=[origin_obs.Tx, origin_obs.Tx],
            Ty=[origin_obs.Ty - delta, origin_obs.Ty + delta],
            frame=self._frame_obs,
        )
    @staticmethod
    def _normalize_observer_key(observer_key: str | None) -> str:
        raw = observer_key
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "ignore")
        if isinstance(raw, np.ndarray) and raw.shape == ():
            raw = raw.item()
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", "ignore")
        key = str(raw or "earth").strip().lower()
        aliases = {
            "earth": "earth",
            "solo": "solar orbiter",
            "solar-orbiter": "solar orbiter",
            "solarorbiter": "solar orbiter",
            "solar orbiter": "solar orbiter",
            "stereo a": "stereo-a",
            "stereo-a": "stereo-a",
            "stereoa": "stereo-a",
            "stereo b": "stereo-b",
            "stereo-b": "stereo-b",
            "stereob": "stereo-b",
        }
        return aliases.get(key, "earth")

    @classmethod
    def _resolve_observer_from_key(cls, observer_key: str | None, obstime):
        return resolve_named_observer(cls._normalize_observer_key(observer_key), obstime)
