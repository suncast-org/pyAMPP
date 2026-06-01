from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits
from astropy.time import Time

from pyampp.gxbox import gx_fov2box
from pyampp.gxbox.boxutils import load_sunpy_map_compat, read_b3d_h5, write_b3d_h5
from pyampp.io import load_model
from pyampp.util.config import IDL_HMI_RSUN_M
from pyampp.io._sav_convert import build_h5_from_sav


class _FakeMap:
    def __init__(self, date: str):
        self.date = Time(date)


class _FakeSourceMap(_FakeMap):
    def __init__(self, date: str):
        super().__init__(date)
        self.observer_coordinate = gx_fov2box.get_earth(self.date)
        self.dsun = self.observer_coordinate.radius.to(u.m)
        self.rsun_meters = 6.96e8 * u.m
        self.meta = {"telescop": "SDO", "instrume": "HMI"}


class _FakeWCS:
    def to_header(self):
        header = fits.Header()
        header["CTYPE1"] = "HPLN-TAN"
        header["CTYPE2"] = "HPLT-TAN"
        return header


class _FakeRefMap:
    def __init__(self, date: str):
        self.date = Time(date)
        self.wcs = _FakeWCS()
        self.rsun_obs = 972.3 * u.arcsec
        self.rsun_meters = 6.96e8 * u.m
        self.observer_coordinate = None


class _FakeDownloader:
    calls: list[dict[str, object]] = []

    def __init__(self, time, uv=True, euv=True, hmi=True, data_dir=None, backend="drms", force_download=False, poll_seconds=5):
        self.time = Time(time)
        self.uv = uv
        self.euv = euv
        self.hmi = hmi
        self.existence_report = {
            "hmi_b": {seg: True for seg in ("field", "inclination", "azimuth", "disambig")},
            "hmi_m": {"magnetogram": True},
            "hmi_ic": {"continuum": True},
            "euv": {pb: True for pb in gx_fov2box.AIA_EUV_PASSBANDS},
            "uv": {pb: True for pb in gx_fov2box.AIA_UV_PASSBANDS},
        }
        type(self).calls.append(
            {
                "time": self.time.isot,
                "uv": uv,
                "euv": euv,
                "hmi": hmi,
                "backend": backend,
                "force_download": force_download,
            }
        )

    def download_images(self):
        if self.hmi:
            return {
                "field": "field.fits",
                "inclination": "inclination.fits",
                "azimuth": "azimuth.fits",
                "disambig": "disambig.fits",
                "continuum": "continuum.fits",
                "magnetogram": "magnetogram.fits",
            }
        files = {}
        if self.euv:
            for pb in gx_fov2box.AIA_EUV_PASSBANDS:
                files[str(pb)] = f"aia_{pb}.fits"
        if self.uv:
            for pb in gx_fov2box.AIA_UV_PASSBANDS:
                files[str(pb)] = f"aia_{pb}.fits"
        return files


def _fake_map_loader(path):
    if path == "field.fits":
        return _FakeMap("2025-11-26T15:34:31.400")
    return _FakeMap("2025-11-26T15:34:31.400")


def test_load_hmi_maps_anchors_context_downloads_to_resolved_hmi_time():
    _FakeDownloader.calls.clear()
    requested = Time("2025-11-26T15:47:52")
    with patch.object(gx_fov2box, "SDOImageDownloader", _FakeDownloader), patch.object(
        gx_fov2box, "load_sunpy_map_compat", side_effect=_fake_map_loader
    ), patch.object(gx_fov2box, "hmi_disambig", side_effect=lambda azimuth, _disambig, method=2: azimuth):
        maps, info = gx_fov2box._load_hmi_maps_from_downloader(
            requested,
            Path("/tmp"),
            euv=True,
            uv=True,
            download_backend="drms",
            force_download=False,
        )

    assert len(_FakeDownloader.calls) == 2
    assert _FakeDownloader.calls[0]["time"] == requested.isot
    assert _FakeDownloader.calls[0]["hmi"] is True
    assert _FakeDownloader.calls[0]["euv"] is False
    assert _FakeDownloader.calls[0]["uv"] is False

    assert _FakeDownloader.calls[1]["time"] == "2025-11-26T15:34:31.400"
    assert _FakeDownloader.calls[1]["hmi"] is False
    assert _FakeDownloader.calls[1]["euv"] is True
    assert _FakeDownloader.calls[1]["uv"] is True

    assert info["requested_obs_time"] == requested.isot
    assert info["resolved_obs_time"] == "2025-11-26T15:34:31.400"
    assert maps["field"].date.isot == "2025-11-26T15:34:31.400"
    assert "AIA_94" in maps
    assert "AIA_1700" in maps


def test_refmap_wcs_header_preserves_date_obs():
    header_text = gx_fov2box._refmap_wcs_header(_FakeRefMap("2025-11-26T15:34:31.400"))
    header = fits.Header.fromstring(header_text, sep="\n")
    assert header["DATE-OBS"] == "2025-11-26T15:34:31.400"
    assert header["DATE_OBS"] == "2025-11-26T15:34:31.400"
    assert header["RSUN_OBS"] == 972.3
    assert header["RSUN_REF"] == 6.96e8


def test_prepare_resume_jump_boxes_none_to_potential_recomputes() -> None:
    entry_corona = {
        "bx": np.zeros((2, 2, 2), dtype=float),
        "by": np.zeros((2, 2, 2), dtype=float),
        "bz": np.zeros((2, 2, 2), dtype=float),
        "attrs": {"model_type": "none"},
    }

    pot_box, bnd_box, nlfff_box = gx_fov2box._prepare_resume_jump_boxes(
        "potential",
        "NONE",
        entry_corona,
    )

    assert pot_box is None
    assert bnd_box is None
    assert nlfff_box is None


def test_prepare_resume_jump_boxes_pot_to_bounds_uses_pot_then_computes_bnd() -> None:
    entry_corona = {
        "bx": np.ones((2, 2, 2), dtype=float),
        "by": np.ones((2, 2, 2), dtype=float),
        "bz": np.ones((2, 2, 2), dtype=float),
        "attrs": {"model_type": "pot"},
    }

    pot_box, bnd_box, nlfff_box = gx_fov2box._prepare_resume_jump_boxes(
        "bounds",
        "POT",
        entry_corona,
    )

    assert pot_box is not None
    assert pot_box is not entry_corona
    assert pot_box["attrs"]["model_type"] == "pot"
    assert bnd_box is None
    assert nlfff_box is None


def test_jump_allowed_preserves_supported_expert_shortcuts() -> None:
    assert gx_fov2box._jump_allowed("NONE", "BND") is True
    assert gx_fov2box._jump_allowed("POT", "NAS") is True
    assert gx_fov2box._jump_allowed("POT", "GEN") is True
    assert gx_fov2box._jump_allowed("POT", "CHR") is True
    assert gx_fov2box._jump_allowed("NAS", "CHR") is True


def test_jump_allowed_rejects_unsupported_skips() -> None:
    assert gx_fov2box._jump_allowed("NONE", "NAS") is False
    assert gx_fov2box._jump_allowed("BND", "GEN") is False
    assert gx_fov2box._jump_allowed("BND", "CHR") is False


def test_jump_chain_records_implicit_intermediate_stages() -> None:
    assert gx_fov2box._jump_chain("NONE", "BND") == ("NONE", "POT", "BND")
    assert gx_fov2box._jump_chain("POT", "NAS") == ("POT", "BND", "NAS")
    assert gx_fov2box._jump_chain("POT", "GEN") == ("POT", "GEN")
    assert gx_fov2box._jump_chain("NAS", "CHR") == ("NAS", "CHR")


def _make_transition_cfg(**overrides):
    params = dict(
        time=None,
        coords=None,
        hpc=True,
        hgc=False,
        hgs=False,
        cea=True,
        top=False,
        box_dims=None,
        dx_km=1400.0,
        pad_frac=0.1,
        data_dir="/tmp",
        gxmodel_dir="/tmp",
        nlfff_lib=None,
        download_backend="drms",
        drms_sequential=False,
        force_download=False,
        entry_box="/tmp/model.h5",
        save_empty_box=False,
        save_potential=False,
        save_bounds=False,
        save_nas=False,
        save_gen=False,
        save_chr=False,
        stop_after=None,
        empty_box_only=False,
        potential_only=False,
        nlfff_only=False,
        generic_only=False,
        use_potential=False,
        skip_lines=False,
        center_vox=False,
        reduce_passed=None,
        euv=False,
        uv=False,
        sfq=False,
        observer_name="earth",
        fov_xc=None,
        fov_yc=None,
        fov_xsize=None,
        fov_ysize=None,
        square_fov=False,
        jump2potential=False,
        jump2bounds=False,
        jump2nlfff=False,
        jump2lines=False,
        jump2chromo=False,
        rebuild=False,
        rebuild_from_none=False,
        info=False,
        reproject_algorithm="adaptive",
        reproject_scan=None,
    )
    params.update(overrides)
    return gx_fov2box.Fov2BoxConfig(**params)


def test_plan_transition_tracks_resume_jump_mode() -> None:
    cfg = _make_transition_cfg(jump2lines=True)

    plan = gx_fov2box._plan_transition(cfg, "POT")

    assert plan.target_stage == "GEN"
    assert plan.jump_chain == ("POT", "GEN")
    assert plan.active_jump == "lines"
    assert plan.goto_lines is True
    assert plan.goto_chromo is False


def test_plan_transition_rejects_unsupported_resume_skip() -> None:
    cfg = _make_transition_cfg(jump2lines=True)

    try:
        gx_fov2box._plan_transition(cfg, "BND")
    except ValueError as exc:
        assert "cannot jump to GEN" in str(exc)
    else:
        raise AssertionError("Expected unsupported BND->GEN resume jump to be rejected")


def test_prepare_resume_state_backfills_base_and_infers_metadata() -> None:
    entry_loaded = {
        "base": {
            "bx": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "by": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
            "bz": np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
        },
        "refmaps": {"Bz_reference": {"data": np.ones((2, 2), dtype=float)}},
        "corona": {
            "dr": np.array([0.1, 0.2, 0.3], dtype=float),
        },
        "metadata": {
            "projection": "top",
            "id": "20251126_153431.UNKNOWN.TOP.POT",
        },
    }
    cfg = _make_transition_cfg(entry_box="/tmp/model.h5", dx_km=1400.0)

    prepared = gx_fov2box._prepare_resume_state(entry_loaded, cfg, Time("2025-11-26T15:34:31"))

    assert np.array_equal(prepared.base_group["ic"], entry_loaded["base"]["bz"])
    assert "chromo_mask" in prepared.base_group
    assert prepared.refmaps == entry_loaded["refmaps"]
    assert np.array_equal(prepared.base_bz_arr, entry_loaded["base"]["bz"])
    assert np.array_equal(prepared.base_ic_arr, entry_loaded["base"]["bz"])
    assert np.array_equal(prepared.bottom_bz_data, entry_loaded["base"]["bz"])
    assert prepared.projection_tag == "TOP"
    assert prepared.base == "20251126_153431.UNKNOWN.TOP"
    assert np.array_equal(prepared.dr3, np.array([0.1, 0.2, 0.3], dtype=float))


def test_prepare_observation_state_builds_expected_prepared_payload() -> None:
    class _ScaleAxis:
        def __init__(self, value):
            self._value = value

        def to_value(self, unit):
            return self._value

    class _Scale:
        axis1 = _ScaleAxis(1.0)
        axis2 = _ScaleAxis(1.0)

    class _WCSW:
        crpix = [1.0, 1.0]

    class _WCSWrap:
        wcs = _WCSW()

    class _MiniMap:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=float)
            self.meta = {"dummy": True}
            self.rsun_obs = 972.3 * u.arcsec
            self.scale = _Scale()
            self.wcs = _WCSWrap()
            self.dsun = 1.0 * u.m

        def reproject_to(self, header, algorithm="exact"):
            return self

    class _FakeBox:
        def __init__(self, *args, **kwargs):
            header = fits.Header()
            header["CTYPE1"] = "HPLN-TAN"
            header["CTYPE2"] = "HPLT-TAN"
            self.bottom_cea_header = header

        def bottom_top_header(self, dsun_obs=None):
            return self.bottom_cea_header

        def bounds_coords_bl_tr(self, pad_frac=0.1):
            return (None, None)

    maps = {
        "field": _MiniMap([[1, 2], [3, 4]]),
        "inclination": _MiniMap([[0, 0], [0, 0]]),
        "azimuth": _MiniMap([[0, 0], [0, 0]]),
        "continuum": _MiniMap([[10, 11], [12, 13]]),
        "magnetogram": _MiniMap([[20, 21], [22, 23]]),
    }
    cfg = _make_transition_cfg(entry_box=None, coords=(10.0, 20.0), box_dims=(2, 2, 2), dx_km=1400.0)

    with patch.object(gx_fov2box, "_load_hmi_maps_from_downloader", return_value=(maps, {"resolved_obs_time": None})), patch.object(
        gx_fov2box, "_resolve_cli_observer", return_value=gx_fov2box.get_earth(Time("2025-11-26T15:34:31"))
    ), patch.object(gx_fov2box, "Box", _FakeBox), patch.object(
        gx_fov2box, "hmi_b2ptr", return_value=(maps["field"], maps["field"], maps["field"])
    ), patch.object(
        gx_fov2box, "_submap_with_fov_safe", side_effect=lambda smap, bl, tr: smap
    ), patch.object(
        gx_fov2box, "map_from_data_header_compat", side_effect=lambda data, meta: _MiniMap(data)
    ), patch.object(
        gx_fov2box, "_build_index_header", return_value="INDEXHDR"
    ), patch.object(
        gx_fov2box, "_refmap_wcs_header", return_value="REFHDR"
    ), patch.object(
        gx_fov2box, "remap_vertical_current_inputs", side_effect=lambda a, b, c: (a, b, c)
    ), patch.object(
        gx_fov2box, "compute_vertical_current", return_value=np.ones((2, 2), dtype=float)
    ), patch.object(
        gx_fov2box, "_format_coord_tag", return_value="TAG"
    ), patch.object(
        gx_fov2box, "_observer_metadata_from_source_map", return_value={"observer": "earth"}
    ):
        prepared = gx_fov2box._prepare_observation_state(
            cfg,
            Time("2025-11-26T15:34:31"),
            (2, 2, 2),
            lambda label, func: func(),
            0.0,
        )

    assert prepared is not None
    assert prepared.maps is maps
    assert prepared.base_group["index"] == "INDEXHDR"
    assert "Bz_reference" in prepared.refmaps
    assert "Ic_reference" in prepared.refmaps
    assert "Vert_current" in prepared.refmaps
    assert prepared.projection_tag == "CEA"
    assert prepared.base.endswith(".TAG.CEA")
    assert prepared.observer_metadata == {"observer": "earth"}
    assert prepared.vert_current_error is None


def test_prepare_run_state_unifies_resume_and_lineage_metadata() -> None:
    entry_loaded = {
        "base": {
            "bx": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float),
            "by": np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=float),
            "bz": np.array([[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]], dtype=float),
        },
        "refmaps": {},
        "corona": {
            "bx": np.arange(24, dtype=float).reshape(4, 3, 2),
            "by": np.arange(24, dtype=float).reshape(4, 3, 2) + 100.0,
            "bz": np.arange(24, dtype=float).reshape(4, 3, 2) + 200.0,
            "dr": np.array([0.1, 0.2, 0.3], dtype=float),
            "attrs": {"model_type": "pot"},
        },
        "chromo": {"legacy_lines": np.array([1, 2, 3], dtype=int)},
        "metadata": {
            "projection": "cea",
            "id": "20251126_153431.UNKNOWN.CEA.POT",
            "axis_order_3d": "zyx",
        },
    }
    cfg = _make_transition_cfg(entry_box="/tmp/model.h5")

    prepared = gx_fov2box._prepare_run_state(
        cfg,
        True,
        entry_loaded,
        "POT",
        "POT",
        Time("2025-11-26T15:34:31"),
        (2, 3, 4),
        {"observer": "entry"},
        lambda label, func: func(),
        0.0,
    )

    assert prepared is not None
    assert prepared.maps is None
    assert prepared.base == "20251126_153431.UNKNOWN.CEA"
    assert prepared.lineage_root == "ENTRY.POT"
    assert prepared.lineage_marker == "ENTRY.POT"
    assert prepared.entry_stage_for_marker == "POT"
    assert prepared.observer_metadata == {"observer": "entry"}
    assert prepared.entry_model == "pot"
    assert prepared.entry_lines == entry_loaded["chromo"]
    assert prepared.entry_corona is not None
    assert prepared.entry_corona["bx"].shape == (2, 3, 4)
    assert np.array_equal(
        prepared.entry_corona["bx"],
        np.asarray(entry_loaded["corona"]["bx"]).transpose((2, 1, 0)),
    )


def test_prepare_run_state_unifies_observation_preparation() -> None:
    obs_prepared = gx_fov2box.ObservationPreparedState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps={"field": object()},
        base_group={"bz": np.ones((2, 2), dtype=float)},
        refmaps={"Bz_reference": {"data": np.ones((2, 2), dtype=float)}},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.1, 0.1], dtype=float),
        observer_metadata={"observer": "earth"},
    )
    cfg = _make_transition_cfg(entry_box=None)

    with patch.object(gx_fov2box, "_prepare_observation_state", return_value=obs_prepared):
        prepared = gx_fov2box._prepare_run_state(
            cfg,
            False,
            None,
            None,
            "NONE",
            Time("2025-11-26T15:34:31"),
            (2, 2, 2),
            None,
            lambda label, func: func(),
            0.0,
        )

    assert prepared is not None
    assert prepared.maps == obs_prepared.maps
    assert prepared.lineage_root == "OBS"
    assert prepared.lineage_marker == ""
    assert prepared.entry_stage_for_marker == ""
    assert prepared.observer_metadata == {"observer": "earth"}


def test_prepare_transition_stage_inputs_handles_lines_jump_from_entry() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps=None,
        base_group={"bz": np.ones((2, 2), dtype=float)},
        refmaps={},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.2, 0.3], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
        entry_corona={
            "bx": np.ones((2, 2, 2), dtype=float),
            "by": np.ones((2, 2, 2), dtype=float),
            "bz": np.ones((2, 2, 2), dtype=float),
            "dr": np.array([0.1, 0.2, 0.3], dtype=float),
            "attrs": {"model_type": "nlfff"},
        },
        entry_lines={"codes": np.array([1], dtype=int)},
    )
    transition_plan = gx_fov2box.TransitionPlan(
        target_stage="GEN",
        jump_chain=("NAS", "GEN"),
        active_jump="lines",
        goto_lines=True,
        goto_chromo=False,
    )

    prepared = gx_fov2box._prepare_transition_stage_inputs(
        prepared_run,
        transition_plan,
        entry_stage="NAS",
        box_dims_resolved=(2, 2, 2),
    )

    assert prepared.goto_lines is True
    assert prepared.nlfff_box == prepared_run.entry_corona
    assert prepared.entry_lines == prepared_run.entry_lines
    assert prepared.pot_box is None
    assert prepared.bnd_box is None


def test_prepare_transition_stage_inputs_materializes_bnd_for_pot_to_nlfff_jump() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps=None,
        base_group={
            "bx": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "by": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
            "bz": np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
        },
        refmaps={},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.2, 0.3], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
        entry_corona={
            "bx": np.ones((2, 2, 2), dtype=float),
            "by": np.ones((2, 2, 2), dtype=float) * 2.0,
            "bz": np.ones((2, 2, 2), dtype=float) * 3.0,
            "dr": np.array([0.1, 0.2, 0.3], dtype=float),
            "attrs": {"model_type": "pot"},
        },
    )
    transition_plan = gx_fov2box.TransitionPlan(
        target_stage="NAS",
        jump_chain=("POT", "BND", "NAS"),
        active_jump="nlfff",
        goto_lines=False,
        goto_chromo=False,
    )

    prepared = gx_fov2box._prepare_transition_stage_inputs(
        prepared_run,
        transition_plan,
        entry_stage="POT",
        box_dims_resolved=(2, 2, 2),
    )

    assert prepared.active_jump == "nlfff"
    assert prepared.pot_box is not None
    assert prepared.bnd_box is not None
    assert prepared.nlfff_box is None
    assert np.array_equal(prepared.bnd_box["bx"][:, :, 0], prepared_run.base_group["bx"].T)


def test_run_nas_stage_uses_prepared_spacing_for_potential_path() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps=None,
        base_group={"bz": np.ones((2, 2), dtype=float)},
        refmaps={},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.2, 0.3], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
    )
    cfg = _make_transition_cfg(use_potential=True, nlfff_only=False, stop_after=None)
    pot_box = {
        "bx": np.ones((2, 2, 2), dtype=float),
        "by": np.ones((2, 2, 2), dtype=float) * 2.0,
        "bz": np.ones((2, 2, 2), dtype=float) * 3.0,
        "attrs": {"model_type": "pot"},
    }
    saved = []

    class _FakeProgress:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def finish(self):
            return 0.0

    result = gx_fov2box._run_nas_stage(
        cfg,
        prepared_run,
        pot_box,
        None,
        None,
        (2, 2, 2),
        lambda stage_tag, stage_box, **kwargs: saved.append((stage_tag, stage_box)),
        lambda: (_ for _ in ()).throw(AssertionError("finalize should not be called")),
        {},
        _FakeProgress,
    )

    assert result.finalized is False
    assert saved == []
    assert result.nlfff_box["attrs"]["model_type"] == "pot"
    assert np.array_equal(result.nlfff_box["dr"], prepared_run.dr3)


def test_compute_none_stage_box_returns_canonical_xyz_payload() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps=None,
        base_group={
            "bx": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "by": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
            "bz": np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
        },
        refmaps={},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.2, 0.3], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
    )

    stage_box = gx_fov2box._compute_none_stage_box(prepared_run, (2, 2, 2))

    assert stage_box["corona"]["attrs"]["model_type"] == "none"
    assert stage_box["corona"]["bx"].shape == (2, 2, 2)
    assert np.array_equal(stage_box["corona"]["bx"][:, :, 0], prepared_run.base_group["bx"].T)
    assert np.array_equal(stage_box["corona"]["by"][:, :, 0], prepared_run.base_group["by"].T)
    assert np.array_equal(stage_box["corona"]["bz"][:, :, 0], prepared_run.base_group["bz"].T)


def test_compute_none_stage_box_uses_internal_xyz_contract_non_cubic() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps=None,
        base_group={
            "bx": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
            "by": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=float),
            "bz": np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]], dtype=float),
        },
        refmaps={},
        base_bz_arr=np.ones((2, 3), dtype=float),
        base_ic_arr=np.ones((2, 3), dtype=float),
        bottom_bz_data=np.ones((2, 3), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.2, 0.3], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
    )

    stage_box = gx_fov2box._compute_none_stage_box(prepared_run, (3, 2, 4))
    bx = stage_box["corona"]["bx"]
    by = stage_box["corona"]["by"]
    bz = stage_box["corona"]["bz"]

    assert bx.shape == (3, 2, 4)
    assert np.array_equal(bx[:, :, 0], prepared_run.base_group["bx"].T)
    assert np.array_equal(by[:, :, 0], prepared_run.base_group["by"].T)
    assert np.array_equal(bz[:, :, 0], prepared_run.base_group["bz"].T)
    assert np.array_equal(bx[:, :, 1:], np.zeros((3, 2, 3), dtype=float))
    assert np.array_equal(by[:, :, 1:], np.zeros((3, 2, 3), dtype=float))
    assert np.array_equal(bz[:, :, 1:], np.zeros((3, 2, 3), dtype=float))


def test_normalize_runtime_stage_box_for_pipeline_uses_private_io_normalizer() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps=None,
        base_group={
            "bx": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "by": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
            "bz": np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
            "ic": np.array([[13.0, 14.0], [15.0, 16.0]], dtype=float),
            "index": "SIMPLE  =                    T",
        },
        refmaps={},
        base_bz_arr=np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
        base_ic_arr=np.array([[13.0, 14.0], [15.0, 16.0]], dtype=float),
        bottom_bz_data=np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.2, 0.3], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
    )
    stage_box = {
        "corona": {
            "bx": np.arange(24, dtype=float).reshape(2, 3, 4),
            "by": np.arange(24, dtype=float).reshape(2, 3, 4) + 100.0,
            "bz": np.arange(24, dtype=float).reshape(2, 3, 4) + 200.0,
            "dr": prepared_run.dr3,
            "attrs": {"model_type": "none"},
        }
    }

    captured = {}
    contract = object()
    expected_loaded = {
        "corona": {"bx": np.ones((4, 3, 2), dtype=float)},
        "metadata": {"axis_order_3d": "zyx", "geometry_contract": contract},
    }

    with patch.object(
        gx_fov2box,
        "_normalize_loaded_model_dict",
        return_value=expected_loaded,
    ) as mocked_normalize:
        normalized = gx_fov2box._normalize_runtime_stage_box_for_pipeline(
            stage_box,
            prepared_run=prepared_run,
            stage_tag="NONE",
        )

    assert normalized is not expected_loaded
    assert normalized["corona"]["bx"].shape == (2, 3, 4)
    assert np.array_equal(normalized["corona"]["bx"], stage_box["corona"]["bx"])
    assert normalized["metadata"]["geometry_contract"] is contract
    mocked_normalize.assert_called_once()
    payload = mocked_normalize.call_args.args[0]
    assert mocked_normalize.call_args.kwargs["source_kind"] == "h5"
    assert mocked_normalize.call_args.kwargs["strict"] is False
    assert mocked_normalize.call_args.kwargs["stored_contract"] is None
    assert mocked_normalize.call_args.kwargs["source_path"].name == "BASE.CEA.NONE.h5"
    assert payload["metadata"]["axis_order_3d"] == "zyx"
    assert payload["metadata"]["vector_layout"] == "split_components"
    assert payload["metadata"]["projection"] == "CEA"
    assert payload["corona"]["bx"].shape == (4, 3, 2)
    assert np.array_equal(payload["corona"]["bx"], stage_box["corona"]["bx"].transpose((2, 1, 0)))
    assert "base" in payload
    assert np.array_equal(payload["base"]["bx"], prepared_run.base_group["bx"])


def test_compute_bnd_stage_box_overwrites_bottom_boundary_from_base() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps=None,
        base_group={
            "bx": np.array([[101.0, 102.0], [103.0, 104.0]], dtype=float),
            "by": np.array([[201.0, 202.0], [203.0, 204.0]], dtype=float),
            "bz": np.array([[301.0, 302.0], [303.0, 304.0]], dtype=float),
        },
        refmaps={},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.2, 0.3], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
    )
    pot_box = {
        "corona": {
            "bx": np.zeros((2, 2, 2), dtype=float),
            "by": np.zeros((2, 2, 2), dtype=float),
            "bz": np.zeros((2, 2, 2), dtype=float),
            "dr": prepared_run.dr3,
            "attrs": {"model_type": "pot"},
        }
    }

    stage_box = gx_fov2box._compute_bnd_stage_box(prepared_run, pot_box, (2, 2, 2))

    assert stage_box["corona"]["attrs"]["model_type"] == "bnd"
    assert np.array_equal(stage_box["corona"]["bx"][:, :, 0], prepared_run.base_group["bx"].T)
    assert np.array_equal(stage_box["corona"]["by"][:, :, 0], prepared_run.base_group["by"].T)
    assert np.array_equal(stage_box["corona"]["bz"][:, :, 0], prepared_run.base_group["bz"].T)


def test_run_gen_chr_stages_stops_after_gen_and_uses_pot_prefix() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps={"field": object()},
        base_group={"bz": np.ones((2, 2), dtype=float), "ic": np.ones((2, 2), dtype=float)},
        refmaps={},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.1, 0.1], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
    )
    cfg = _make_transition_cfg(generic_only=True, skip_lines=False, center_vox=False, reduce_passed=2)
    nlfff_box = {
        "bx": np.ones((2, 2, 2), dtype=float),
        "by": np.ones((2, 2, 2), dtype=float),
        "bz": np.ones((2, 2, 2), dtype=float),
        "attrs": {"model_type": "pot"},
    }
    saved = []
    finalized = []
    stage_times = {}

    class _FakeProgress:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def finish(self):
            return 1.25

    class _FakeMaglib:
        pass

    fake_lines = {
        "codes": np.array([1, 2], dtype=int),
        "apex_idx": np.array([0, 1], dtype=int),
        "start_idx": np.array([0, 1], dtype=int),
        "end_idx": np.array([1, 2], dtype=int),
        "seed_idx": np.array([0, 1], dtype=int),
        "av_field": np.array([1.0, 2.0], dtype=float),
        "phys_length": np.array([3.0, 4.0], dtype=float),
        "voxel_status": np.array([0, 1], dtype=int),
    }

    with patch.object(gx_fov2box, "MagFieldProcessor", return_value=_FakeMaglib()), patch.object(
        gx_fov2box,
        "_load_maglib_idl_cube",
        side_effect=lambda maglib, box, dr: None,
    ), patch.object(
        gx_fov2box,
        "_lines_fast",
        return_value=fake_lines,
    ), patch.object(
        gx_fov2box,
        "_make_header",
        return_value={"header": "ok"},
    ), patch.object(
        gx_fov2box,
        "combo_model",
        return_value={"phys_length": np.array([3.0, 4.0], dtype=float)},
    ), patch.object(
        gx_fov2box,
        "_make_lines_group",
        side_effect=lambda lines, dr: {"lines": lines, "dr": dr},
    ):
        result = gx_fov2box._run_gen_chr_stages(
            cfg,
            prepared_run,
            nlfff_box,
            resume_mode=False,
            entry_stage=None,
            target_stage="GEN",
            goto_chromo=False,
            entry_lines=None,
            save_stage=lambda stage_tag, stage_box, **kwargs: saved.append((stage_tag, stage_box, kwargs)),
            finalize=lambda: finalized.append(True),
            stage_times=stage_times,
            stage_progress_cls=_FakeProgress,
        )

    assert result.finalized is True
    assert finalized == [True]
    assert len(saved) == 1
    assert saved[0][0] == "POT.GEN"
    assert "lines" in saved[0][1]
    assert np.isclose(stage_times["POT.GEN"], 1.25)


def test_run_gen_chr_stages_preserves_legacy_gen_payload_on_jump2chromo() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps={"field": object()},
        base_group={"bz": np.ones((2, 2), dtype=float), "ic": np.ones((2, 2), dtype=float)},
        refmaps={},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.1, 0.1], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
    )
    cfg = _make_transition_cfg(generic_only=False, skip_lines=False, center_vox=False, reduce_passed=None)
    nlfff_box = {
        "bx": np.ones((2, 2, 2), dtype=float),
        "by": np.ones((2, 2, 2), dtype=float),
        "bz": np.ones((2, 2, 2), dtype=float),
        "attrs": {"model_type": "nlfff"},
    }
    saved = []
    stage_times = {}

    class _FakeProgress:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def finish(self):
            return 2.5

    legacy_lines = {
        "start_idx": np.array([0, 1], dtype=int),
        "end_idx": np.array([1, 2], dtype=int),
        "av_field": np.array([1.0, 2.0], dtype=float),
        "phys_length": np.array([3.0, 4.0], dtype=float),
        "voxel_status": np.array([0, 1], dtype=np.uint8),
    }

    with patch.object(gx_fov2box, "_make_header", return_value={"header": "ok"}), patch.object(
        gx_fov2box,
        "combo_model",
        return_value={"phys_length": np.array([3.0, 4.0], dtype=float)},
    ):
        result = gx_fov2box._run_gen_chr_stages(
            cfg,
            prepared_run,
            nlfff_box,
            resume_mode=True,
            entry_stage="GEN",
            target_stage="CHR",
            goto_chromo=True,
            entry_lines=legacy_lines,
            save_stage=lambda stage_tag, stage_box, **kwargs: saved.append((stage_tag, stage_box, kwargs)),
            finalize=lambda: None,
            stage_times=stage_times,
            stage_progress_cls=_FakeProgress,
        )

    assert result.finalized is False
    assert len(saved) == 1
    assert saved[0][0] == "NAS.GEN.CHR"
    assert "lines" in saved[0][1]
    assert np.isclose(stage_times["NAS.GEN.CHR"], 2.5)


def test_run_gen_stage_stops_after_gen_and_uses_pot_prefix() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps={"field": object()},
        base_group={"bz": np.ones((2, 2), dtype=float), "ic": np.ones((2, 2), dtype=float)},
        refmaps={},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.1, 0.1], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
    )
    cfg = _make_transition_cfg(generic_only=True, skip_lines=False, center_vox=False, reduce_passed=2)
    nlfff_box = {
        "bx": np.ones((2, 2, 2), dtype=float),
        "by": np.ones((2, 2, 2), dtype=float),
        "bz": np.ones((2, 2, 2), dtype=float),
        "attrs": {"model_type": "pot"},
    }
    saved = []
    finalized = []
    stage_times = {}

    class _FakeProgress:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def finish(self):
            return 1.25

    class _FakeMaglib:
        pass

    fake_lines = {
        "codes": np.array([1, 2], dtype=int),
        "apex_idx": np.array([0, 1], dtype=int),
        "start_idx": np.array([0, 1], dtype=int),
        "end_idx": np.array([1, 2], dtype=int),
        "seed_idx": np.array([0, 1], dtype=int),
        "av_field": np.array([1.0, 2.0], dtype=float),
        "phys_length": np.array([3.0, 4.0], dtype=float),
        "voxel_status": np.array([0, 1], dtype=int),
    }

    with patch.object(gx_fov2box, "MagFieldProcessor", return_value=_FakeMaglib()), patch.object(
        gx_fov2box,
        "_load_maglib_idl_cube",
        side_effect=lambda maglib, box, dr: None,
    ), patch.object(
        gx_fov2box,
        "_lines_fast",
        return_value=fake_lines,
    ):
        result = gx_fov2box._run_gen_stage(
            cfg,
            prepared_run,
            nlfff_box,
            resume_mode=False,
            entry_stage=None,
            target_stage="GEN",
            goto_chromo=False,
            entry_lines=None,
            save_stage=lambda stage_tag, stage_box, **kwargs: saved.append((stage_tag, stage_box, kwargs)),
            finalize=lambda: finalized.append(True),
            stage_times=stage_times,
            stage_progress_cls=_FakeProgress,
        )

    assert result.finalized is True
    assert result.stage_prefix == "POT"
    assert result.lines is not None
    assert finalized == [True]
    assert len(saved) == 1
    assert saved[0][0] == "POT.GEN"
    assert np.isclose(stage_times["POT.GEN"], 1.25)


def test_run_chr_stage_writes_chr_payload_with_lines() -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps={"field": object()},
        base_group={
            "bz": np.ones((2, 2), dtype=float),
            "ic": np.ones((2, 2), dtype=float),
            "chromo_mask": np.ones((2, 2), dtype=float),
        },
        refmaps={},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.1, 0.1], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
    )
    nlfff_box = {
        "bx": np.ones((2, 2, 2), dtype=float),
        "by": np.ones((2, 2, 2), dtype=float),
        "bz": np.ones((2, 2, 2), dtype=float),
        "attrs": {"model_type": "nlfff"},
    }
    lines = {
        "codes": np.array([1, 2], dtype=int),
        "apex_idx": np.array([0, 1], dtype=int),
        "start_idx": np.array([0, 1], dtype=int),
        "end_idx": np.array([1, 2], dtype=int),
        "seed_idx": np.array([0, 1], dtype=int),
        "av_field": np.array([1.0, 2.0], dtype=float),
        "phys_length": np.array([3.0, 4.0], dtype=float),
        "voxel_status": np.array([0, 1], dtype=int),
    }
    saved = []
    stage_times = {}

    class _FakeProgress:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def finish(self):
            return 2.5

    with patch.object(gx_fov2box, "_make_header", return_value={"header": "ok"}), patch.object(
        gx_fov2box,
        "combo_model",
        return_value={
            "chromo_bcube": np.ones((2, 2, 2, 3), dtype=float),
            "phys_length": np.array([3.0, 4.0], dtype=float),
        },
    ):
        gx_fov2box._run_chr_stage(
            prepared_run,
            nlfff_box,
            stage_prefix="NAS",
            lines=lines,
            save_stage=lambda stage_tag, stage_box, **kwargs: saved.append((stage_tag, stage_box, kwargs)),
            stage_times=stage_times,
            stage_progress_cls=_FakeProgress,
        )

    assert len(saved) == 1
    assert saved[0][0] == "NAS.GEN.CHR"
    assert "chromo" in saved[0][1]
    assert "lines" in saved[0][1]
    assert saved[0][2]["chromo_source_axis_order_2d"] == "xy"
    assert np.isclose(stage_times["NAS.GEN.CHR"], 2.5)


def test_save_stage_passthrough_returns_original_payload(tmp_path) -> None:
    prepared_run = gx_fov2box.PreparedRunState(
        obs_time=Time("2025-11-26T15:34:31"),
        maps=None,
        base_group={
            "bx": np.ones((2, 2), dtype=float),
            "by": np.ones((2, 2), dtype=float),
            "bz": np.ones((2, 2), dtype=float),
        },
        refmaps={},
        base_bz_arr=np.ones((2, 2), dtype=float),
        base_ic_arr=np.ones((2, 2), dtype=float),
        bottom_bz_data=np.ones((2, 2), dtype=float),
        vert_current_error=None,
        projection_tag="CEA",
        base="BASE.CEA",
        dr3=np.array([0.1, 0.1, 0.1], dtype=float),
        observer_metadata=None,
        lineage_root="OBS",
        lineage_marker="",
        entry_stage_for_marker="",
    )
    cfg = _make_transition_cfg(save_potential=True)
    produced = []
    context = gx_fov2box.StageSaveContext(
        cfg=cfg,
        prepared_run=prepared_run,
        execute_cmd="gx-fov2box",
        out_dir=tmp_path,
        default_grid={"voxel_id": np.zeros((2, 2, 2), dtype=np.int32)},
        empty_grid=np.zeros((2, 2, 2), dtype=float),
        produced=produced,
    )
    stage_box = {
        "corona": {
            "bx": np.ones((2, 2, 2), dtype=float),
            "by": np.ones((2, 2, 2), dtype=float) * 2.0,
            "bz": np.ones((2, 2, 2), dtype=float) * 3.0,
            "dr": prepared_run.dr3,
            "attrs": {"model_type": "pot"},
        }
    }
    original_bx = stage_box["corona"]["bx"].copy()

    with patch.object(gx_fov2box, "gx_box2id", return_value=(np.zeros((2, 2, 2), dtype=np.int32), None)), patch.object(
        gx_fov2box,
        "write_b3d_h5",
    ) as mocked_write:
        returned = gx_fov2box._save_stage_passthrough("POT", stage_box, context=context)

    assert returned is stage_box
    assert np.array_equal(stage_box["corona"]["bx"], original_bx)
    mocked_write.assert_called_once()
    assert len(produced) == 1


def test_solve_nlfff_from_bnd_uses_idl_loader_and_preserves_solver_component_names() -> None:
    bnd_box = {
        "bx": np.arange(24, dtype=float).reshape(2, 3, 4),
        "by": np.arange(24, dtype=float).reshape(2, 3, 4) + 10.0,
        "bz": np.arange(24, dtype=float).reshape(2, 3, 4) + 20.0,
        "dr": np.array([0.1, 0.2, 0.3], dtype=float),
        "attrs": {"model_type": "bnd"},
    }

    class _FakeMagLib:
        def NLFFF(self):
            return {
                "bx": np.arange(24, dtype=float).reshape(2, 3, 4) + 1.0,
                "by": np.arange(24, dtype=float).reshape(2, 3, 4) + 101.0,
                "bz": np.arange(24, dtype=float).reshape(2, 3, 4) + 201.0,
            }

    fake_maglib = _FakeMagLib()
    loaded = []

    def _fake_load_maglib_idl_cube(maglib, box, dr):
        loaded.append((maglib, box, np.asarray(dr, dtype=float)))

    with patch.object(gx_fov2box, "MagFieldProcessor", return_value=fake_maglib), patch.object(
        gx_fov2box,
        "_load_maglib_idl_cube",
        side_effect=_fake_load_maglib_idl_cube,
    ):
        nlfff_box = gx_fov2box._solve_nlfff_from_bnd(bnd_box, np.array([0.1, 0.2, 0.3], dtype=float))

    assert len(loaded) == 1
    assert loaded[0][0] is fake_maglib
    assert loaded[0][1] is bnd_box
    assert np.array_equal(loaded[0][2], np.array([0.1, 0.2, 0.3], dtype=float))
    assert nlfff_box["attrs"]["model_type"] == "nlfff"
    assert np.array_equal(nlfff_box["dr"], np.array([0.1, 0.2, 0.3], dtype=float))
    assert np.array_equal(
        nlfff_box["bx"],
        np.arange(24, dtype=float).reshape(2, 3, 4) + 1.0,
    )
    assert np.array_equal(
        nlfff_box["by"],
        np.arange(24, dtype=float).reshape(2, 3, 4) + 101.0,
    )
    assert np.array_equal(
        nlfff_box["bz"],
        np.arange(24, dtype=float).reshape(2, 3, 4) + 201.0,
    )


def test_build_index_header_converts_hgs_cea_reference_point_to_carrington():
    source_map = _FakeSourceMap("2025-11-26T15:34:31.400")
    bottom_header = fits.Header()
    bottom_header["SIMPLE"] = 1
    bottom_header["BITPIX"] = 8
    bottom_header["NAXIS"] = 2
    bottom_header["NAXIS1"] = 150
    bottom_header["NAXIS2"] = 100
    bottom_header["CTYPE1"] = "HGLN-CEA"
    bottom_header["CTYPE2"] = "HGLT-CEA"
    bottom_header["CUNIT1"] = "deg"
    bottom_header["CUNIT2"] = "deg"
    bottom_header["CRPIX1"] = 75.5
    bottom_header["CRPIX2"] = 50.5
    bottom_header["CDELT1"] = 0.115250131204
    bottom_header["CDELT2"] = 0.115250131204
    bottom_header["CRVAL1"] = -17.05992944118293
    bottom_header["CRVAL2"] = -12.247129818437514

    header_text = gx_fov2box._build_index_header(bottom_header, source_map)
    header = fits.Header.fromstring(header_text, sep="\n")

    expected = gx_fov2box.SkyCoord(
        lon=bottom_header["CRVAL1"] * u.deg,
        lat=bottom_header["CRVAL2"] * u.deg,
        radius=source_map.rsun_meters,
        frame=gx_fov2box.HeliographicStonyhurst(obstime=source_map.date),
    ).transform_to(
        gx_fov2box.HeliographicCarrington(observer=source_map.observer_coordinate, obstime=source_map.date)
    )

    assert header["CTYPE1"] == "CRLN-CEA"
    assert header["CTYPE2"] == "CRLT-CEA"
    assert np.isclose(header["CRVAL1"], expected.lon.to_value(u.deg))
    assert np.isclose(header["CRVAL2"], expected.lat.to_value(u.deg))
    assert not np.isclose(header["CRVAL1"], bottom_header["CRVAL1"])


def test_load_sunpy_map_compat_normalizes_rsun_ref_from_header():
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 2
    header["NAXIS2"] = 2
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    header["DATE-OBS"] = "2025-11-26T15:34:31.400"
    header["DSUN_OBS"] = 1.476e11
    header["HGLN_OBS"] = 0.0
    header["HGLT_OBS"] = 1.44
    header["RSUN_REF"] = 6.957e8

    smap = load_sunpy_map_compat(np.zeros((2, 2), dtype=float), header=header)

    assert float(smap.meta["rsun_ref"]) == float(IDL_HMI_RSUN_M)
    assert np.isclose(smap.rsun_meters.to_value(u.m), IDL_HMI_RSUN_M)


def _fake_sav_box_with_refmaps():
    base_and_index = _fake_sav_box_with_base_and_index()
    rec_dtype = [
        ("ID", object),
        ("DATA", object),
        ("XC", object),
        ("YC", object),
        ("DX", object),
        ("DY", object),
        ("TIME", object),
        ("XUNITS", object),
        ("YUNITS", object),
        ("RSUN", object),
        ("B0", object),
        ("L0", object),
    ]
    rec = np.empty(1, dtype=rec_dtype)
    rec["ID"][0] = b"AIA_171"
    rec["DATA"][0] = np.arange(6, dtype=np.float32).reshape(2, 3)
    rec["XC"][0] = 10.0
    rec["YC"][0] = -20.0
    rec["DX"][0] = 0.6
    rec["DY"][0] = 0.6
    rec["TIME"][0] = b"2025-11-26T15:34:33.350"
    rec["XUNITS"][0] = b"arcsec"
    rec["YUNITS"][0] = b"arcsec"
    rec["RSUN"][0] = 972.5
    rec["B0"][0] = 1.44
    rec["L0"][0] = 44.92

    ids = np.empty(1, dtype=object)
    ids[0] = b"slot0"
    ptrs = np.empty(1, dtype=object)
    ptrs[0] = rec

    pointer = np.empty(1, dtype=[("IDS", object), ("PTRS", object)])
    pointer["IDS"][0] = ids
    pointer["PTRS"][0] = ptrs

    omap = np.empty(1, dtype=[("POINTER", object)])
    omap["POINTER"][0] = pointer

    refmaps = np.empty(1, dtype=[("OMAP", object)])
    refmaps["OMAP"][0] = omap

    box = np.empty(1, dtype=[("BASE", object), ("INDEX", object), ("REFMAPS", object), ("ID", object), ("EXECUTE", object)])
    box["BASE"][0] = base_and_index["BASE"][0]
    box["INDEX"][0] = base_and_index["INDEX"][0]
    box["REFMAPS"][0] = refmaps
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NAS.CHR"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def _fake_sav_box_with_base_and_index():
    base = np.empty(
        1,
        dtype=[
            ("BX", np.float64, (2, 3)),
            ("BY", np.float64, (2, 3)),
            ("BZ", np.float64, (2, 3)),
            ("IC", np.float64, (2, 3)),
            ("CHROMO_MASK", np.int16, (2, 3)),
        ],
    )
    base["BX"][0] = np.arange(6, dtype=np.float64).reshape(2, 3)
    base["BY"][0] = np.arange(6, dtype=np.float64).reshape(2, 3) + 10.0
    base["BZ"][0] = np.arange(6, dtype=np.float64).reshape(2, 3) + 20.0
    base["IC"][0] = 1.0
    base["CHROMO_MASK"][0] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)

    index = np.empty(
        1,
        dtype=[
            ("SIMPLE", np.int16),
            ("BITPIX", np.int32),
            ("NAXIS", np.int32),
            ("NAXIS1", np.int32),
            ("NAXIS2", np.int32),
            ("WCSNAME", "S32"),
            ("CRPIX1", np.float64),
            ("CRVAL1", np.float64),
            ("CTYPE1", "S16"),
            ("CUNIT1", "S8"),
            ("CDELT1", np.float64),
            ("CRPIX2", np.float64),
            ("CRVAL2", np.float64),
            ("CTYPE2", "S16"),
            ("CUNIT2", "S8"),
            ("CDELT2", np.float64),
            ("CROTA2", np.float64),
            ("DATE_D$OBS", "S32"),
            ("DSUN_OBS", np.float64),
            ("SOLAR_B0", np.float64),
            ("HGLN_OBS", np.float64),
            ("HGLT_OBS", np.float64),
            ("CRLN_OBS", np.float64),
            ("CRLT_OBS", np.float64),
            ("DATE_OBS", "S32"),
            ("COMMENT", object),
            ("HISTORY", object),
        ],
    )
    index["SIMPLE"][0] = 1
    index["BITPIX"][0] = 8
    index["NAXIS"][0] = 2
    index["NAXIS1"][0] = 3
    index["NAXIS2"][0] = 2
    index["WCSNAME"][0] = b"Carrington-Heliographic"
    index["CRPIX1"][0] = 1.5
    index["CRVAL1"][0] = 27.8672448568
    index["CTYPE1"][0] = b"CRLN-CEA"
    index["CUNIT1"][0] = b"deg"
    index["CDELT1"][0] = 0.115250131204
    index["CRPIX2"][0] = 1.0
    index["CRVAL2"][0] = -12.2426138116
    index["CTYPE2"][0] = b"CRLT-CEA"
    index["CUNIT2"][0] = b"deg"
    index["CDELT2"][0] = 0.115250131204
    index["CROTA2"][0] = 0.0
    index["DATE_D$OBS"][0] = b"2025-11-26T15:34:31.400"
    index["DSUN_OBS"][0] = 147638514656.0
    index["SOLAR_B0"][0] = 1.44043069702
    index["HGLN_OBS"][0] = 0.0
    index["HGLT_OBS"][0] = 1.44043069702
    index["CRLN_OBS"][0] = 44.9246506781
    index["CRLT_OBS"][0] = 1.44043069702
    index["DATE_OBS"][0] = b"2025-11-26T15:34:31.400"
    index["COMMENT"][0] = np.array([b"FITSHEAD2STRUCT", b"", b"", b"", b""], dtype=object)
    index["HISTORY"][0] = np.array(
        [b"FITSHEAD2STRUCT run at: Tue Feb 10 18:27:01 2026", b"", b"", b"", b""],
        dtype=object,
    )

    box = np.empty(1, dtype=[("BASE", object), ("INDEX", object), ("ID", object), ("EXECUTE", object)])
    box["BASE"][0] = base
    box["INDEX"][0] = index
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NONE"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def _fake_sav_box_with_cubic_corona_and_chromo():
    index = _fake_sav_box_with_base_and_index()["INDEX"][0]
    base = np.empty(
        1,
        dtype=[
            ("BX", np.float64, (2, 2)),
            ("BY", np.float64, (2, 2)),
            ("BZ", np.float64, (2, 2)),
            ("IC", np.float64, (2, 2)),
        ],
    )
    base["BX"][0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    base["BY"][0] = np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float64)
    base["BZ"][0] = np.array([[21.0, 22.0], [23.0, 24.0]], dtype=np.float64)
    base["IC"][0] = np.ones((2, 2), dtype=np.float64)

    # IDL [x,y,z] restored by readsav as numpy [z,y,x].
    bx_zyx = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 100.0
    by_zyx = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 200.0
    bz_zyx = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 300.0
    chromo_bcube = np.stack([bx_zyx + 1000.0, by_zyx + 1000.0, bz_zyx + 1000.0], axis=0)

    box = np.empty(
        1,
        dtype=[
            ("BASE", object),
            ("BX", object),
            ("BY", object),
            ("BZ", object),
            ("DR", object),
            ("CORONA_BASE", np.int16),
            ("INDEX", object),
            ("CHROMO_IDX", object),
            ("CHROMO_T", object),
            ("CHROMO_N", object),
            ("CHROMO_BCUBE", object),
            ("CHROMO_LAYERS", np.int16),
            ("DZ", object),
            ("ID", object),
            ("EXECUTE", object),
        ],
    )
    box["BASE"][0] = base
    box["BX"][0] = bx_zyx
    box["BY"][0] = by_zyx
    box["BZ"][0] = bz_zyx
    box["DR"][0] = np.array([0.1, 0.1, 0.2], dtype=np.float64)
    box["CORONA_BASE"][0] = 1
    box["INDEX"][0] = index
    box["CHROMO_IDX"][0] = np.array([0, 1, 2, 3], dtype=np.int64)
    box["CHROMO_T"][0] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    box["CHROMO_N"][0] = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    box["CHROMO_BCUBE"][0] = chromo_bcube
    box["CHROMO_LAYERS"][0] = 1
    box["DZ"][0] = np.arange(8, dtype=np.float64).reshape(2, 2, 2) + 500.0
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NAS.CHR"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def _fake_sav_box_with_noncubic_corona_and_chromo():
    index = _fake_sav_box_with_base_and_index()["INDEX"][0]
    base = np.empty(
        1,
        dtype=[
            ("BX", np.float64, (3, 4)),
            ("BY", np.float64, (3, 4)),
            ("BZ", np.float64, (3, 4)),
            ("IC", np.float64, (3, 4)),
        ],
    )
    base["BX"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 1.0
    base["BY"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 101.0
    base["BZ"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 201.0
    base["IC"][0] = np.ones((3, 4), dtype=np.float64)

    # Real GX SAV BX/BY/BZ arrays restore through scipy.readsav as (z, y, x).
    bx_zyx = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 10.0
    by_zyx = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 110.0
    bz_zyx = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 210.0

    # Real GX SAV CHROMO_BCUBE restores as (component, z, y, x).
    chromo_bcube_czyx = np.stack(
        [
            np.arange(40, dtype=np.float32).reshape(2, 4, 5) + 1000.0,
            np.arange(40, dtype=np.float32).reshape(2, 4, 5) + 2000.0,
            np.arange(40, dtype=np.float32).reshape(2, 4, 5) + 3000.0,
        ],
        axis=0,
    )

    box = np.empty(
        1,
        dtype=[
            ("BASE", object),
            ("BX", object),
            ("BY", object),
            ("BZ", object),
            ("DR", object),
            ("CORONA_BASE", np.int16),
            ("INDEX", object),
            ("AVFIELD", object),
            ("PHYSLENGTH", object),
            ("STATUS", object),
            ("STARTIDX", object),
            ("ENDIDX", object),
            ("CHROMO_IDX", object),
            ("CHROMO_T", object),
            ("CHROMO_N", object),
            ("N_P", object),
            ("N_HI", object),
            ("N_HTOT", object),
            ("CHROMO_BCUBE", object),
            ("CHROMO_LAYERS", np.int16),
            ("DZ", object),
            ("ID", object),
            ("EXECUTE", object),
        ],
    )
    box["BASE"][0] = base
    box["BX"][0] = bx_zyx
    box["BY"][0] = by_zyx
    box["BZ"][0] = bz_zyx
    box["DR"][0] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    box["CORONA_BASE"][0] = 1
    box["INDEX"][0] = index
    box["AVFIELD"][0] = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 400.0
    box["PHYSLENGTH"][0] = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 500.0
    box["STATUS"][0] = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    box["STARTIDX"][0] = np.arange(24, dtype=np.int64).reshape(2, 3, 4) + 600
    box["ENDIDX"][0] = np.arange(24, dtype=np.int64).reshape(2, 3, 4) + 700
    box["CHROMO_IDX"][0] = np.array([0, 1, 4, 5, 6, 19, 20, 39], dtype=np.int64)
    box["CHROMO_T"][0] = np.arange(8, dtype=np.float32) + 1.0
    box["CHROMO_N"][0] = np.arange(8, dtype=np.float32) + 11.0
    box["N_P"][0] = np.arange(8, dtype=np.float32) + 21.0
    box["N_HI"][0] = np.arange(8, dtype=np.float32) + 31.0
    box["N_HTOT"][0] = np.arange(8, dtype=np.float32) + 41.0
    box["CHROMO_BCUBE"][0] = chromo_bcube_czyx
    box["CHROMO_LAYERS"][0] = 2
    box["DZ"][0] = np.arange(40, dtype=np.float64).reshape(2, 4, 5) + 500.0
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NAS.CHR"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def _fake_sav_box_with_noncubic_bcube_corona():
    index = _fake_sav_box_with_base_and_index()["INDEX"][0]
    base = np.empty(
        1,
        dtype=[
            ("BX", np.float64, (3, 4)),
            ("BY", np.float64, (3, 4)),
            ("BZ", np.float64, (3, 4)),
            ("IC", np.float64, (3, 4)),
        ],
    )
    base["BX"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 1.0
    base["BY"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 101.0
    base["BZ"][0] = np.arange(12, dtype=np.float64).reshape(3, 4) + 201.0
    base["IC"][0] = np.ones((3, 4), dtype=np.float64)

    bcube_czyx = np.stack(
        [
            np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 10.0,
            np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 110.0,
            np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 210.0,
        ],
        axis=0,
    )

    box = np.empty(
        1,
        dtype=[
            ("BASE", object),
            ("BCUBE", object),
            ("DR", object),
            ("CORONA_BASE", np.int16),
            ("INDEX", object),
            ("ID", object),
            ("EXECUTE", object),
        ],
    )
    box["BASE"][0] = base
    box["BCUBE"][0] = bcube_czyx
    box["DR"][0] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    box["CORONA_BASE"][0] = 1
    box["INDEX"][0] = index
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NAS"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def test_load_entry_box_any_restores_sav_refmaps(tmp_path):
    fake_box = _fake_sav_box_with_refmaps()
    expected_h5 = tmp_path / "expected_refmaps.h5"
    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", expected_h5)
    expected_loaded = load_model(expected_h5)

    with patch.object(gx_fov2box, "load_model", return_value=expected_loaded) as mocked_loader:
        loaded = gx_fov2box._load_entry_box_any(tmp_path / "entry.sav")
    mocked_loader.assert_called_once()

    assert "refmaps" in loaded
    assert list(loaded["refmaps"]) == ["AIA_171"]
    payload = loaded["refmaps"]["AIA_171"]
    assert np.asarray(payload["data"]).shape == (2, 3)

    header = fits.Header.fromstring(payload["wcs_header"], sep="\n")
    assert header["DATE-OBS"] == "2025-11-26T15:34:33.350"
    assert header["DATE_OBS"] == "2025-11-26T15:34:33.350"
    assert header["RSUN_OBS"] == 972.5
    assert header["HGLT_OBS"] == 1.44
    assert header["HGLN_OBS"] == 44.92


def test_load_entry_box_any_serializes_sav_index_as_fits_header(tmp_path):
    fake_box = _fake_sav_box_with_base_and_index()
    expected_h5 = tmp_path / "expected_index.h5"
    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", expected_h5)
    expected_loaded = load_model(expected_h5)

    with patch.object(gx_fov2box, "load_model", return_value=expected_loaded) as mocked_loader:
        loaded = gx_fov2box._load_entry_box_any(tmp_path / "entry.sav")
    mocked_loader.assert_called_once()

    index_text = loaded["base"]["index"]
    if isinstance(index_text, (bytes, bytearray)):
        index_text = index_text.decode("utf-8", errors="ignore")
    assert not index_text.startswith("(")
    header = fits.Header.fromstring(index_text, sep="\n")
    assert header["CTYPE1"] == "CRLN-CEA"
    assert header["CTYPE2"] == "CRLT-CEA"
    assert header["CRVAL1"] == 27.8672448568
    assert header["CRVAL2"] == -12.2426138116
    assert header["CRLN_OBS"] == 44.9246506781
    assert header["DATE-OBS"] == "2025-11-26T15:34:31.400"
    assert header["DATE_OBS"] == "2025-11-26T15:34:31.400"

    out_h5 = tmp_path / "entry.h5"
    write_b3d_h5(str(out_h5), loaded)
    roundtrip = read_b3d_h5(str(out_h5))
    roundtrip_header = fits.Header.fromstring(roundtrip["base"]["index"], sep="\n")
    assert roundtrip_header["CTYPE1"] == "CRLN-CEA"
    assert roundtrip_header["CRLN_OBS"] == 44.9246506781


def test_sav_entry_box_roundtrip_preserves_cubic_3d_axis_order(tmp_path):
    fake_box = _fake_sav_box_with_cubic_corona_and_chromo()
    expected_h5 = tmp_path / "expected_cubic.h5"
    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", expected_h5)
    expected_loaded = load_model(expected_h5)

    with patch.object(gx_fov2box, "load_model", return_value=expected_loaded) as mocked_loader:
        loaded = gx_fov2box._load_entry_box_any(tmp_path / "entry.sav")
    mocked_loader.assert_called_once()

    out_h5 = tmp_path / "entry.h5"
    source_axis_order_3d = gx_fov2box._decode_id_text(loaded.get("metadata", {}).get("axis_order_3d", "zyx")).lower()
    normalized = gx_fov2box._normalize_stage_for_h5(loaded, source_axis_order_3d=source_axis_order_3d)
    write_b3d_h5(str(out_h5), normalized)
    roundtrip = read_b3d_h5(str(out_h5))

    assert np.array_equal(roundtrip["corona"]["bx"], read_b3d_h5(str(expected_h5))["corona"]["bx"])
    assert np.array_equal(roundtrip["corona"]["by"], read_b3d_h5(str(expected_h5))["corona"]["by"])
    assert np.array_equal(roundtrip["corona"]["bz"], read_b3d_h5(str(expected_h5))["corona"]["bz"])
    assert np.array_equal(roundtrip["chromo"]["bx"], read_b3d_h5(str(expected_h5))["chromo"]["bx"])
    assert np.array_equal(roundtrip["chromo"]["by"], read_b3d_h5(str(expected_h5))["chromo"]["by"])
    assert np.array_equal(roundtrip["chromo"]["bz"], read_b3d_h5(str(expected_h5))["chromo"]["bz"])
    assert np.array_equal(roundtrip["chromo"]["dz"], read_b3d_h5(str(expected_h5))["chromo"]["dz"])


def test_build_h5_from_sav_transposes_noncubic_corona_components_to_canonical_zyx(tmp_path):
    fake_box = _fake_sav_box_with_noncubic_corona_and_chromo()
    out_h5 = tmp_path / "noncubic_corona.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    expected_bx = np.asarray(fake_box["BX"][0], dtype=np.float32)
    expected_by = np.asarray(fake_box["BY"][0], dtype=np.float32)
    expected_bz = np.asarray(fake_box["BZ"][0], dtype=np.float32)

    with h5py.File(out_h5, "r") as f:
        assert f["corona/bx"].shape == (2, 3, 4)
        assert np.array_equal(np.asarray(f["corona/bx"]), expected_bx)
        assert np.array_equal(np.asarray(f["corona/by"]), expected_by)
        assert np.array_equal(np.asarray(f["corona/bz"]), expected_bz)


def test_build_h5_from_sav_preserves_noncubic_corona_bcube_component_identity(tmp_path):
    fake_box = _fake_sav_box_with_noncubic_bcube_corona()
    out_h5 = tmp_path / "noncubic_corona_bcube.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    bcube = np.asarray(fake_box["BCUBE"][0], dtype=np.float32)
    with h5py.File(out_h5, "r") as f:
        assert np.array_equal(np.asarray(f["corona/bx"]), bcube[0])
        assert np.array_equal(np.asarray(f["corona/by"]), bcube[1])
        assert np.array_equal(np.asarray(f["corona/bz"]), bcube[2])


def test_build_h5_from_sav_preserves_noncubic_chromo_component_identity(tmp_path):
    fake_box = _fake_sav_box_with_noncubic_corona_and_chromo()
    out_h5 = tmp_path / "noncubic_chromo.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    chromo_bcube = np.asarray(fake_box["CHROMO_BCUBE"][0], dtype=np.float32)

    with h5py.File(out_h5, "r") as f:
        assert f["chromo/bx"].shape == (2, 4, 5)
        assert np.array_equal(np.asarray(f["chromo/bx"]), chromo_bcube[0])
        assert np.array_equal(np.asarray(f["chromo/by"]), chromo_bcube[1])
        assert np.array_equal(np.asarray(f["chromo/bz"]), chromo_bcube[2])


def test_build_h5_from_sav_flattens_lines_in_canonical_zyx_c_order(tmp_path):
    fake_box = _fake_sav_box_with_noncubic_corona_and_chromo()
    out_h5 = tmp_path / "noncubic_lines.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    expected_av = np.asarray(fake_box["AVFIELD"][0], dtype=np.float64).reshape(-1, order="C")
    expected_phys = np.asarray(fake_box["PHYSLENGTH"][0], dtype=np.float64).reshape(-1, order="C")
    expected_status = np.asarray(fake_box["STATUS"][0], dtype=np.uint8).reshape(-1, order="C")
    expected_start = np.asarray(fake_box["STARTIDX"][0], dtype=np.int64).reshape(-1, order="C")
    expected_end = np.asarray(fake_box["ENDIDX"][0], dtype=np.int64).reshape(-1, order="C")

    with h5py.File(out_h5, "r") as f:
        av = np.asarray(f["lines/av_field"])
        phys = np.asarray(f["lines/phys_length"])
        status = np.asarray(f["lines/voxel_status"])
        start = np.asarray(f["lines/start_idx"])
        end = np.asarray(f["lines/end_idx"])

    assert np.array_equal(av, expected_av)
    assert np.array_equal(phys, expected_phys)
    assert np.array_equal(status, expected_status)
    assert np.array_equal(start, expected_start)
    assert np.array_equal(end, expected_end)

    assert np.array_equal(av.reshape((2, 3, 4), order="C"), np.asarray(fake_box["AVFIELD"][0], dtype=np.float64))
    assert np.array_equal(start.reshape((2, 3, 4), order="C"), np.asarray(fake_box["STARTIDX"][0], dtype=np.int64))
    assert start[0] == fake_box["STARTIDX"][0][0, 0, 0]
    assert start[1] == fake_box["STARTIDX"][0][0, 0, 1]
    assert start[4] == fake_box["STARTIDX"][0][0, 1, 0]
    assert start[12] == fake_box["STARTIDX"][0][1, 0, 0]


def test_build_h5_from_sav_preserves_chromo_sparse_index_mapping_in_canonical_zyx_c_order(tmp_path):
    fake_box = _fake_sav_box_with_noncubic_corona_and_chromo()
    out_h5 = tmp_path / "noncubic_chromo_sparse.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    expected_idx = np.asarray(fake_box["CHROMO_IDX"][0], dtype=np.int64)
    expected_n = np.asarray(fake_box["CHROMO_N"][0], dtype=np.float32)
    expected_t = np.asarray(fake_box["CHROMO_T"][0], dtype=np.float32)
    expected_n_p = np.asarray(fake_box["N_P"][0], dtype=np.float32)
    expected_n_hi = np.asarray(fake_box["N_HI"][0], dtype=np.float32)
    expected_n_htot = np.asarray(fake_box["N_HTOT"][0], dtype=np.float32)

    with h5py.File(out_h5, "r") as f:
        chromo_idx = np.asarray(f["chromo/chromo_idx"])
        chromo_n = np.asarray(f["chromo/chromo_n"])
        chromo_t = np.asarray(f["chromo/chromo_t"])
        n_p = np.asarray(f["chromo/n_p"])
        n_hi = np.asarray(f["chromo/n_hi"])
        n_htot = np.asarray(f["chromo/n_htot"])
        chromo_shape = np.asarray(f["chromo/bx"]).shape

    assert np.array_equal(chromo_idx, expected_idx)
    assert np.array_equal(chromo_n, expected_n)
    assert np.array_equal(chromo_t, expected_t)
    assert np.array_equal(n_p, expected_n_p)
    assert np.array_equal(n_hi, expected_n_hi)
    assert np.array_equal(n_htot, expected_n_htot)

    dense_n = np.zeros(chromo_shape, dtype=np.float32)
    dense_t = np.zeros(chromo_shape, dtype=np.float32)
    dense_n.flat[chromo_idx] = chromo_n
    dense_t.flat[chromo_idx] = chromo_t

    expected_dense_n = np.zeros(chromo_shape, dtype=np.float32)
    expected_dense_t = np.zeros(chromo_shape, dtype=np.float32)
    expected_dense_n.flat[expected_idx] = expected_n
    expected_dense_t.flat[expected_idx] = expected_t

    assert np.array_equal(dense_n, expected_dense_n)
    assert np.array_equal(dense_t, expected_dense_t)
    assert dense_n[0, 0, 1] == expected_n[1]
    assert dense_n[0, 1, 0] == expected_n[3]
    assert dense_n[1, 0, 0] == expected_n[6]
    assert dense_n[1, 3, 4] == expected_n[7]


def test_normalize_stage_for_h5_transposes_chr_2d_maps_from_xy_to_yx():
    stage_box = {
        "base": {
            "bx": np.arange(6, dtype=np.float32).reshape(2, 3),
            "by": np.arange(6, dtype=np.float32).reshape(2, 3),
            "bz": np.arange(6, dtype=np.float32).reshape(2, 3),
        },
        "chromo": {
            "tr": np.array([[10, 11], [12, 13], [14, 15]], dtype=np.int64),
            "tr_h": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            "chromo_mask": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16),
        },
    }

    normalized = gx_fov2box._normalize_stage_for_h5(
        stage_box,
        source_axis_order_3d="xyz",
        chromo_source_axis_order_2d="xy",
    )

    assert np.array_equal(normalized["chromo"]["tr"], stage_box["chromo"]["tr"].T)
    assert np.array_equal(normalized["chromo"]["tr_h"], stage_box["chromo"]["tr_h"].T)
    assert np.array_equal(normalized["chromo"]["chromo_mask"], stage_box["chromo"]["chromo_mask"].T)


def test_runtime_stage_normalization_preserves_internal_3d_order():
    stage_box = {
        "corona": {
            "bx": np.zeros((4, 3, 2), dtype=float),
            "by": np.zeros((4, 3, 2), dtype=float),
            "bz": np.zeros((4, 3, 2), dtype=float),
            "dr": np.ones(3, dtype=float),
        }
    }
    prepared = SimpleNamespace(
        base_group={
            "bx": np.zeros((3, 4), dtype=float),
            "by": np.zeros((3, 4), dtype=float),
            "bz": np.zeros((3, 4), dtype=float),
            "ic": np.zeros((3, 4), dtype=float),
        },
        base_ic_arr=np.zeros((3, 4), dtype=float),
        refmaps={},
        observer_metadata=None,
        base="test",
        projection_tag="CEA",
    )
    contract = object()

    with patch.object(
        gx_fov2box,
        "_normalize_loaded_model_dict",
        return_value={"metadata": {"geometry_contract": contract}},
    ):
        normalized = gx_fov2box._normalize_runtime_stage_box_for_pipeline(
            stage_box,
            prepared_run=prepared,
            stage_tag="NONE",
            source_axis_order_3d="xyz",
        )

    assert normalized["corona"]["bx"].shape == (4, 3, 2)
    assert normalized["metadata"]["geometry_contract"] is contract


def test_build_h5_from_sav_does_not_write_raw_sav_by_default(tmp_path):
    fake_box = _fake_sav_box_with_base_and_index()
    out_h5 = tmp_path / "entry.h5"

    with patch("pyampp.io._sav_convert.readsav", return_value={"box": fake_box}):
        build_h5_from_sav(tmp_path / "entry.sav", out_h5)

    with h5py.File(out_h5, "r") as f:
        assert "raw_sav" not in f
