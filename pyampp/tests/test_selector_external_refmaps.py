import numpy as np
from astropy.io import fits

from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
from pyampp.gxbox.selector_api import DisplayFovSelection
from pyampp.gxbox.gxbox_selector_view import (
    _available_map_ids_from_sources,
    _build_session_input,
    _discover_external_ref_map_files,
    _parse_execute_refmap_paths,
)
from pyampp.tests._fits_header import canonical_base_index_header
from unittest.mock import patch
from types import SimpleNamespace


class _FakeMap:
    def __init__(self, data):
        self.data = data
        self.meta = {}
        self.plot_settings = {}


def test_available_map_ids_includes_external_refmaps():
    refmaps = {
        "Bz_reference": {"data": np.zeros((2, 2))},
        "AIA_171": {"data": np.zeros((2, 2))},
        "EOVSA_f1.418GHz": {"data": np.zeros((2, 2))},
    }

    map_ids = _available_map_ids_from_sources({}, refmaps, {})

    assert "Bz" in map_ids
    assert "171" in map_ids
    assert "EOVSA_f1.418GHz" in map_ids


def test_available_map_ids_includes_external_filesystem_maps():
    map_ids = _available_map_ids_from_sources({"EOVSA_f1.418GHz": "/tmp/eovsa.fits"}, {}, {})

    assert "EOVSA_f1.418GHz" in map_ids


def test_embedded_refmap_key_passes_through_external_refmap_ids():
    assert MapBoxDisplayWidget._embedded_refmap_key("171") == "AIA_171"
    assert MapBoxDisplayWidget._embedded_refmap_key("EOVSA_f1.418GHz") == "EOVSA_f1.418GHz"


def test_eovsa_refmaps_use_hot_temperature_colormap():
    smap = _FakeMap(np.arange(100, dtype=float).reshape(10, 10))

    MapBoxDisplayWidget._apply_display_scaling(smap, "EOVSA_f1.418GHz")

    assert smap.plot_settings["cmap"] == "hot"
    assert smap.plot_settings["norm"].vmin > 0.0
    assert smap.plot_settings["norm"].vmax < 99.0


def test_embedded_aia_maps_keep_sdoaia_colormap_after_reproject():
    from sunpy.visualization import colormaps as sunpy_colormaps

    smap = _FakeMap(np.linspace(10.0, 1000.0, 100, dtype=float).reshape(10, 10))
    smap.meta = {}

    MapBoxDisplayWidget._apply_display_scaling(smap, "171")

    expected = sunpy_colormaps.cm.cmlist.get("sdoaia171")
    assert smap.plot_settings["cmap"] is expected
    assert smap.plot_settings["norm"].vmax > smap.plot_settings["norm"].vmin


def test_eovsa_context_allows_bottom_overlay():
    assert MapBoxDisplayWidget._should_plot_bottom_overlay("EOVSA_f1.418GHz", "bz")


def test_discover_external_ref_map_files_infers_eovsa_ids(tmp_path):
    header = fits.Header()
    header["CRVAL3"] = 1.418334960938e9
    header["CUNIT3"] = "Hz"
    header["TELESCOP"] = "EOVSA"
    path = tmp_path / "eovsa_20260403_200000_f1.418GHz.fits"
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32), header=header).writeto(path)

    discovered = _discover_external_ref_map_files([str(tmp_path)])

    assert discovered == {"EOVSA_f1.418GHz": str(path)}


def test_discover_external_ref_map_files_can_ignore_generic_fits(tmp_path):
    generic = tmp_path / "not_a_known_context.fits"
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(generic)
    aia_header = fits.Header()
    aia_header["TELESCOP"] = "SDO/AIA"
    aia_header["WAVELNTH"] = 171
    aia = tmp_path / "aia171.fits"
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32), header=aia_header).writeto(aia)

    discovered = _discover_external_ref_map_files([str(tmp_path)], generic=False)

    assert discovered == {"171": str(aia)}


def test_parse_execute_refmap_paths_handles_repeated_and_equals_forms():
    execute = (
        "gx-fov2box --refmaps-path '/tmp/eovsa maps' "
        "--refmap-path=/tmp/extra.fits --refmaps-path /tmp/second"
    )

    assert _parse_execute_refmap_paths(execute) == (
        "/tmp/eovsa maps",
        "/tmp/extra.fits",
        "/tmp/second",
    )


def test_build_session_input_uses_refmap_paths_from_execute_metadata(tmp_path):
    entry_path = tmp_path / "model.NONE.h5"
    execute_refmaps = tmp_path / "execute_refmaps"
    explicit_refmaps = tmp_path / "explicit_refmaps"
    execute_refmaps.mkdir()
    explicit_refmaps.mkdir()
    entry = {
        "metadata": {
            "execute": (
                "gx-fov2box --time 2026-04-03T19:46:37 --coords -70 160 --hpc "
                "--box-dims 4 3 2 --dx-km 1400 --data-dir /tmp/jsoc "
                f"--refmaps-path {execute_refmaps}"
            )
        },
        "corona": {
            "bx": np.zeros((4, 3, 2), dtype=float),
            "by": np.zeros((4, 3, 2), dtype=float),
            "bz": np.zeros((4, 3, 2), dtype=float),
        },
    }

    with patch("pyampp.gxbox.gxbox_selector_view._load_entry_box_any", return_value=entry), patch(
        "pyampp.gxbox.gxbox_selector_view._discover_filesystem_maps",
        return_value={"171": "/tmp/aia171.fits"},
    ), patch(
        "pyampp.gxbox.gxbox_selector_view._discover_external_ref_map_files",
        return_value={"EOVSA_f1.418GHz": "/tmp/eovsa.fits"},
    ) as discover_external:
        session = _build_session_input(entry_path, ref_map_paths=[str(explicit_refmaps)])

    discover_external.assert_called_once_with((str(execute_refmaps), str(explicit_refmaps)))
    assert session.map_files["171"] == "/tmp/aia171.fits"
    assert session.map_files["EOVSA_f1.418GHz"] == "/tmp/eovsa.fits"
    assert "EOVSA_f1.418GHz" in session.map_ids


@patch("pyampp.gxbox.gxbox_selector_view.build_fits_refmaps_for_model")
def test_embed_external_refmaps_into_box_data_uses_explicit_paths_only(mock_build, tmp_path):
    from pyampp.gxbox.gxbox_selector_view import _embed_external_refmaps_into_box_data
    from pyampp.gxbox.selector_api import BoxGeometrySelection, CoordMode

    stereo = tmp_path / "stereo304.fits"
    stereo.write_text("placeholder")
    mock_build.return_value = {"stereo304": {"data": np.ones((4, 4)), "wcs_header": "SIMPLE  = T"}}
    box_data = {
        "base": {"index": canonical_base_index_header(date_obs="2026-04-03T19:46:37.800")},
        "refmaps": {},
    }

    embedded, skipped = _embed_external_refmaps_into_box_data(
        box_data,
        [str(stereo)],
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        overwrite=True,
    )

    assert embedded == ["stereo304"]
    assert skipped == []
    assert "stereo304" in box_data["refmaps"]
    mock_build.assert_called_once()


@patch("pyampp.gxbox.gxbox_selector_view.build_fits_refmaps_for_model")
def test_persist_selector_result_embeds_external_refmaps(mock_build, tmp_path):
    from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry
    from pyampp.gxbox.selector_api import DisplayFovSelection, SelectorDialogResult, BoxGeometrySelection, CoordMode

    entry = tmp_path / "model.h5"
    out = tmp_path / "saved.h5"
    box_data = {
        "base": {"index": canonical_base_index_header(date_obs="2026-04-03T19:46:37.800")},
        "observer": {"name": "earth", "fov": {"xc_arcsec": 0.0, "yc_arcsec": 0.0, "xsize_arcsec": 100.0, "ysize_arcsec": 100.0}},
        "refmaps": {},
    }
    mock_build.return_value = {"stereo304": {"data": np.ones((4, 4)), "wcs_header": "SIMPLE  = T"}}
    result = SelectorDialogResult(
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        fov=DisplayFovSelection(0.0, 0.0, 100.0, 100.0),
        square_fov=True,
    )

    with patch("pyampp.gxbox.gxbox_selector_view.load_model", return_value=box_data), patch(
        "pyampp.gxbox.gxbox_selector_view.save_model"
    ) as save_model:
        ok = _persist_selector_result_to_entry(
            entry,
            result,
            output_path=out,
            external_ref_map_paths=("/tmp/stereo",),
        )

    assert ok is True
    mock_build.assert_called_once()
    save_model.assert_called_once()
    saved = save_model.call_args[0][0]
    assert "stereo304" in saved["refmaps"]
    from pathlib import Path
    assert Path(mock_build.call_args.kwargs["model_dir"]) == out.resolve().parent


def test_reproject_without_fov_override_uses_full_disk_not_roi():
    from types import SimpleNamespace
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
    from pyampp.gxbox.selector_api import DisplayFovSelection

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
        fov=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
        fov_definition_observer_key="stereo-a",
    )
    widget._view_mode = "box_fov"
    widget._record_prepare_event = lambda _msg: None
    smap = SimpleNamespace(
        data=np.ones((8, 8)),
        meta={},
        observer_coordinate=SimpleNamespace(
            lon=SimpleNamespace(to_value=lambda _u: 1.0),
            lat=SimpleNamespace(to_value=lambda _u: 0.0),
        ),
        scale=SimpleNamespace(
            axis1=SimpleNamespace(to_value=lambda _u: 1.0),
            axis2=SimpleNamespace(to_value=lambda _u: 1.0),
        ),
        rsun_meters=SimpleNamespace(to_value=lambda _u: 6.96e8),
        dsun=SimpleNamespace(to_value=lambda _u: 1.496e11),
        wcs=SimpleNamespace(
            pixel_to_world=lambda _x, _y: SimpleNamespace(
                transform_to=lambda _frame: SimpleNamespace()
            )
        ),
        reproject_to=lambda *_a, **_k: SimpleNamespace(data=np.zeros((8, 8))),
    )

    with patch(
        "pyampp.io.refmap_crop.reproject_refmap_to_observer",
        return_value=SimpleNamespace(data=np.zeros((8, 8))),
    ) as full_mock, patch.object(
        MapBoxDisplayWidget,
        "_current_display_prepare_fov",
        return_value=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
    ) as implicit_fov_mock, patch.object(
        MapBoxDisplayWidget,
        "_resolve_display_observer_coord",
        return_value="stereo-observer",
    ):
        widget._reproject_map_for_display_observer(smap, fov_override=None)

    full_mock.assert_called_once()
    assert full_mock.call_args.kwargs.get("mask_off_limb") is True
    implicit_fov_mock.assert_not_called()


def test_stereo_map_skips_display_observer_reprojection(tmp_path):
    from types import SimpleNamespace
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
    from pyampp.gxbox.boxutils import load_sunpy_map_compat

    header = fits.Header()
    header["TELESCOP"] = "STEREO"
    header["INSTRUME"] = "SECCHI"
    header["WAVELNTH"] = 284
    header["NAXIS"] = 2
    header["NAXIS1"] = 8
    header["NAXIS2"] = 8
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 4.5
    header["CRPIX2"] = 4.5
    header["CRVAL1"] = -1020.0
    header["CRVAL2"] = 110.0
    header["CDELT1"] = 12.0
    header["CDELT2"] = 12.0
    header["DATE-OBS"] = "2026-04-03T19:58:30.006"
    header["HGLN_OBS"] = 54.3710463167
    header["HGLT_OBS"] = -0.875478602937
    header["DSUN_OBS"] = 1.496e11
    header["RSUN_OBS"] = 959.63
    path = tmp_path / "stereo284.fits"
    fits.PrimaryHDU(data=np.arange(64, dtype=np.uint16).reshape(8, 8), header=header).writeto(path)
    smap = load_sunpy_map_compat(path)

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
    )
    widget._observer_coord_cache = {}
    widget._observer_metadata_cache = {}
    widget._observer_warning_cache = set()
    widget._last_status_text = ""
    widget._status_callback = None
    widget._observer_source_b3d = lambda: {}
    widget._prep_trace_counts = {}
    widget._record_prepare_event = lambda _msg: None

    native_key = widget._infer_native_display_observer_key_from_map(smap)
    assert native_key == "stereo-a"
    rotated = MapBoxDisplayWidget._rotate_map_for_display(smap)
    assert rotated is not smap
    out, coverage = widget._reproject_map_for_display_observer(smap)
    assert coverage is None
    assert out is not smap
    assert np.array_equal(np.asarray(out.data), np.asarray(rotated.data))


def test_native_spacecraft_payload_skips_observer_reprojection_even_with_earth_observer_cards():
    from types import SimpleNamespace
    from astropy.io import fits
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
    from pyampp.gxbox.boxutils import map_from_data_header_compat

    header = fits.Header()
    header["TELESCOP"] = "STEREO"
    header["INSTRUME"] = "SECCHI"
    header["PYALIGN"] = False
    header["NAXIS"] = 2
    header["NAXIS1"] = 8
    header["NAXIS2"] = 8
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 4.5
    header["CRPIX2"] = 4.5
    header["CRVAL1"] = -1020.0
    header["CRVAL2"] = 110.0
    header["CDELT1"] = 12.0
    header["CDELT2"] = 12.0
    header["DATE-OBS"] = "2026-04-03T19:58:30.006"
    header["HGLN_OBS"] = 0.0
    header["HGLT_OBS"] = 0.0
    smap = map_from_data_header_compat(np.ones((8, 8), dtype=np.float32), header)

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
    )
    widget._observer_coord_cache = {}
    widget._observer_metadata_cache = {}
    widget._observer_warning_cache = set()
    widget._record_prepare_event = lambda _msg: None
    widget._observer_source_b3d = lambda: {}

    assert MapBoxDisplayWidget._is_native_spacecraft_payload(smap)
    out, coverage = widget._reproject_map_for_display_observer(smap)
    assert coverage is None
    assert out is not smap


def test_native_stereo_map_reprojects_for_earth_display_observer(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch
    from astropy.io import fits
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
    from pyampp.gxbox.boxutils import load_sunpy_map_compat, map_from_data_header_compat

    header = fits.Header()
    header["TELESCOP"] = "STEREO"
    header["INSTRUME"] = "SECCHI"
    header["WAVELNTH"] = 284
    header["NAXIS"] = 2
    header["NAXIS1"] = 8
    header["NAXIS2"] = 8
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 4.5
    header["CRPIX2"] = 4.5
    header["CRVAL1"] = -1020.0
    header["CRVAL2"] = 110.0
    header["CDELT1"] = 12.0
    header["CDELT2"] = 12.0
    header["DATE-OBS"] = "2026-04-03T19:58:30.006"
    header["HGLN_OBS"] = 54.3710463167
    header["HGLT_OBS"] = -0.875478602937
    header["DSUN_OBS"] = 1.496e11
    header["RSUN_OBS"] = 959.63
    path = tmp_path / "stereo284_earth.fits"
    fits.PrimaryHDU(data=np.arange(64, dtype=np.uint16).reshape(8, 8), header=header).writeto(path)
    smap = load_sunpy_map_compat(path)

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="earth",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
    )
    widget._observer_coord_cache = {}
    widget._observer_metadata_cache = {}
    widget._observer_warning_cache = set()
    widget._last_status_text = ""
    widget._status_callback = None
    events = []
    widget._record_prepare_event = events.append
    widget._observer_source_b3d = lambda: {}

    reprojected = map_from_data_header_compat(np.zeros((8, 8), dtype=np.float32), header.copy())
    with patch(
        "pyampp.io.refmap_crop.reproject_refmap_to_observer",
        return_value=reprojected,
    ) as reproj_mock:
        out, coverage = widget._reproject_map_for_display_observer(smap)

    reproj_mock.assert_called_once()
    assert reproj_mock.call_args.kwargs.get("mask_off_limb") is True
    assert out is reprojected
    assert coverage is None
    assert any("observer reproj:" in msg and "-> earth" in msg for msg in events)
    assert not any("display rotate: native spacecraft solar north" in msg for msg in events)


@patch("pyampp.gxbox.gxbox_selector_view.build_fits_refmaps_for_model")
def test_embed_external_refmaps_into_session_uses_base_index_time(mock_build, tmp_path):
    from pyampp.gxbox.gxbox_selector_view import _embed_external_refmaps_into_session
    from pyampp.gxbox.selector_api import SelectorSessionInput, BoxGeometrySelection, CoordMode
    from pyampp.tests._fits_header import canonical_base_index_header

    stereo = tmp_path / "stereo304.fits"
    stereo.write_text("placeholder")
    mock_build.return_value = {"stereo304": {"data": np.ones((4, 4)), "wcs_header": "SIMPLE  = T"}}
    model_time = "2026-04-03T19:46:37.800"
    session = SelectorSessionInput(
        time_iso=model_time,
        data_dir="",
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        refmaps={},
        base_wcs_header=canonical_base_index_header(date_obs=model_time),
    )

    embedded, skipped = _embed_external_refmaps_into_session(session, [str(stereo)])

    assert embedded == ["stereo304"]
    assert skipped == []
    mock_build.assert_called_once()
    assert mock_build.call_args.kwargs["model_obstime"] == model_time


@patch("pyampp.gxbox.gxbox_selector_view.build_fits_refmaps_for_model")
def test_embed_external_refmaps_into_session_updates_session_refmaps(mock_build, tmp_path):
    from pyampp.gxbox.gxbox_selector_view import _embed_external_refmaps_into_session
    from pyampp.gxbox.selector_api import SelectorSessionInput, BoxGeometrySelection, CoordMode

    stereo = tmp_path / "stereo304.fits"
    stereo.write_text("placeholder")
    mock_build.return_value = {"stereo304": {"data": np.ones((4, 4)), "wcs_header": "SIMPLE  = T"}}
    session = SelectorSessionInput(
        time_iso="2026-04-03T19:46:37.800",
        data_dir="",
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        refmaps={},
    )

    embedded, skipped = _embed_external_refmaps_into_session(session, [str(stereo)])

    assert embedded == ["stereo304"]
    assert skipped == []
    assert "stereo304" in session.refmaps
    mock_build.assert_called_once()


@patch("pyampp.gxbox.gxbox_selector_view.build_fits_refmaps_for_model")
def test_persist_selector_result_merges_session_refmaps(mock_build, tmp_path):
    from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry
    from pyampp.gxbox.selector_api import DisplayFovSelection, SelectorDialogResult, BoxGeometrySelection, CoordMode

    entry = tmp_path / "model.h5"
    out = tmp_path / "saved.h5"
    box_data = {
        "base": {"index": np.bytes_(b"DATE-OBS = '2026-04-03T19:46:37.800'\n")},
        "observer": {"name": "earth", "fov": {"xc_arcsec": 0.0, "yc_arcsec": 0.0, "xsize_arcsec": 100.0, "ysize_arcsec": 100.0}},
        "refmaps": {"legacy_map": {"data": np.zeros((2, 2)), "wcs_header": "SIMPLE  = T"}},
    }
    mock_build.return_value = {}
    result = SelectorDialogResult(
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        fov=DisplayFovSelection(0.0, 0.0, 100.0, 100.0),
        square_fov=True,
    )
    session_refmaps = {
        "stereo304": {"data": np.ones((4, 4)), "wcs_header": "SIMPLE  = T"},
    }

    with patch("pyampp.gxbox.gxbox_selector_view.load_model", return_value=box_data), patch(
        "pyampp.gxbox.gxbox_selector_view.save_model"
    ) as save_model:
        ok = _persist_selector_result_to_entry(
            entry,
            result,
            output_path=out,
            session_refmaps=session_refmaps,
        )

    assert ok is True
    save_model.assert_called_once()
    saved = save_model.call_args[0][0]
    assert "legacy_map" in saved["refmaps"]
    assert "stereo304" in saved["refmaps"]


def test_prepare_context_map_full_disk_variant_rotates_without_crop():
    from types import SimpleNamespace
    from astropy.io import fits
    from pyampp.gxbox.box_view2d import (
        MapBoxDisplayWidget,
        _CONTEXT_PREPARE_VARIANT_FULL_DISK,
        _EMBEDDED_REFMAP_FLAG,
    )
    from pyampp.gxbox.boxutils import map_from_data_header_compat
    from pyampp.gxbox.selector_api import DisplayFovSelection

    header = fits.Header()
    header["TELESCOP"] = "STEREO"
    header["INSTRUME"] = "SECCHI"
    header["PYALIGN"] = False
    header["NAXIS"] = 2
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 32.5
    header["CRPIX2"] = 32.5
    header["CRVAL1"] = -1020.0
    header["CRVAL2"] = 110.0
    header["CDELT1"] = 12.0
    header["CDELT2"] = 12.0
    header["DATE-OBS"] = "2026-04-03T19:58:30.006"
    header["HGLN_OBS"] = 54.3710463167
    header["HGLT_OBS"] = -0.875478602937
    header["DSUN_OBS"] = 1.496e11
    smap = map_from_data_header_compat(np.ones((64, 64), dtype=np.float32), header)
    smap.meta[_EMBEDDED_REFMAP_FLAG] = True

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
        fov=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
    )
    widget._refmap_display_notices = []
    widget._observer_warning_cache = set()
    widget._observer_coord_cache = {}
    widget._observer_metadata_cache = {}
    widget._observer_source_b3d = lambda: {}
    widget._prep_trace_counts = {}
    widget._prep_trace_order = []
    widget._last_status_base_text = ""
    widget._last_status_text = ""
    widget._status_callback = None
    widget._record_prepare_event = lambda _msg: None
    widget._refresh_status_text = lambda: None

    rotated = SimpleNamespace(data=np.zeros((12, 12)))
    with patch.object(MapBoxDisplayWidget, "_build_native_crop", return_value=(smap, None)) as native_crop_mock, patch.object(
        MapBoxDisplayWidget, "_reproject_map_for_display_observer", return_value=(rotated, None)
    ) as reproj_mock, patch.object(MapBoxDisplayWidget, "_apply_display_scaling", side_effect=lambda m, _k: m):
        out, coverage = widget._prepare_context_map(
            "stereo171",
            smap,
            prepare_variant=_CONTEXT_PREPARE_VARIANT_FULL_DISK,
        )

    assert out is rotated
    reproj_mock.assert_called_once()
    native_crop_mock.assert_not_called()
    assert coverage is None


def test_prepare_context_map_fov_crop_variant_crops_after_rotate():
    from types import SimpleNamespace
    from astropy.io import fits
    from pyampp.gxbox.box_view2d import (
        MapBoxDisplayWidget,
        _CONTEXT_PREPARE_VARIANT_FOV_CROP,
        _EMBEDDED_REFMAP_FLAG,
    )
    from pyampp.gxbox.boxutils import map_from_data_header_compat
    from pyampp.gxbox.selector_api import DisplayFovSelection

    header = fits.Header()
    header["TELESCOP"] = "STEREO"
    header["INSTRUME"] = "SECCHI"
    header["PYALIGN"] = False
    header["NAXIS"] = 2
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 32.5
    header["CRPIX2"] = 32.5
    header["CRVAL1"] = -1020.0
    header["CRVAL2"] = 110.0
    header["CDELT1"] = 12.0
    header["CDELT2"] = 12.0
    header["DATE-OBS"] = "2026-04-03T19:58:30.006"
    header["HGLN_OBS"] = 54.3710463167
    header["HGLT_OBS"] = -0.875478602937
    header["DSUN_OBS"] = 1.496e11
    smap = map_from_data_header_compat(np.ones((64, 64), dtype=np.float32), header)
    smap.meta[_EMBEDDED_REFMAP_FLAG] = True

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        fov_definition_observer_key="earth",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
        fov=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
    )
    widget._refmap_display_notices = []
    widget._observer_warning_cache = set()
    widget._observer_coord_cache = {}
    widget._observer_metadata_cache = {}
    widget._observer_source_b3d = lambda: {}
    widget._prep_trace_counts = {}
    widget._prep_trace_order = []
    widget._last_status_base_text = ""
    widget._last_status_text = ""
    widget._status_callback = None
    widget._record_prepare_event = lambda _msg: None
    widget._refresh_status_text = lambda: None

    rotated = SimpleNamespace(data=np.zeros((12, 12)))
    with patch.object(MapBoxDisplayWidget, "_build_native_crop", return_value=(smap, None)) as native_crop_mock, patch.object(
        MapBoxDisplayWidget, "_reproject_map_for_display_observer", return_value=(rotated, None)
    ) as reproj_mock, patch.object(MapBoxDisplayWidget, "_apply_display_scaling", side_effect=lambda m, _k: m):
        out, coverage = widget._prepare_context_map(
            "stereo171",
            smap,
            prepare_variant=_CONTEXT_PREPARE_VARIANT_FOV_CROP,
        )

    assert out is rotated
    reproj_mock.assert_called_once()
    native_crop_mock.assert_called_once()
    assert coverage is None


def test_prepare_context_map_skips_display_crop_for_pyalign_embedded_maps():
    """See also test_pyalign_cross_observer_reprojection.py for full policy lock."""
    from types import SimpleNamespace
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget, _CONTEXT_PREPARE_VARIANT_FOV_CROP

    smap = SimpleNamespace(
        data=np.zeros((12, 12)),
        meta={"PYEMBED": True, "PYALIGN": True},
    )
    projected_fov = DisplayFovSelection(-900.0, 120.0, 880.0, 880.0)
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        fov_definition_observer_key="earth",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
        fov=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
    )
    widget._record_prepare_event = lambda _msg: None

    rotated = SimpleNamespace(data=np.zeros((12, 12)))
    with patch.object(MapBoxDisplayWidget, "_build_native_crop", return_value=(smap, None)) as native_crop_mock, patch.object(
        MapBoxDisplayWidget, "_submap_to_fov_selection_pixels"
    ) as crop_mock, patch.object(
        MapBoxDisplayWidget,
        "_fov_selection_projected_to_display_observer",
        return_value=projected_fov,
    ), patch.object(
        MapBoxDisplayWidget, "_reproject_map_for_display_observer", return_value=(rotated, projected_fov)
    ) as reproj_mock, patch.object(MapBoxDisplayWidget, "_apply_display_scaling", side_effect=lambda m, _k: m), patch.object(
        MapBoxDisplayWidget, "_is_non_earth_display_observer", return_value=True
    ):
        out, coverage = widget._prepare_context_map(
            "EOVSA_f1.418GHz",
            smap,
            prepare_variant=_CONTEXT_PREPARE_VARIANT_FOV_CROP,
        )

    assert out is rotated
    native_crop_mock.assert_not_called()
    crop_mock.assert_not_called()
    reproj_mock.assert_called_once()
    assert reproj_mock.call_args.kwargs["fov_override"] is projected_fov
    assert coverage is projected_fov


def test_fov_selection_projected_to_display_observer_returns_same_fov_for_shared_los():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="earth",
        fov_definition_observer_key="earth",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
    )
    fov = DisplayFovSelection(10.0, 20.0, 30.0, 40.0)
    with patch.object(MapBoxDisplayWidget, "_observers_share_los", return_value=True):
        out = widget._fov_selection_projected_to_display_observer(fov, "2026-04-03T19:46:37.800")
    assert out is fov


def test_bottom_map_loads_from_embedded_base_maps_in_filesystem_mode():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        map_source_mode="filesystem",
        map_files={"171": "/tmp/aia171.fits"},
        base_maps={"bz": np.ones((4, 4))},
        refmaps={},
        base_geometry=None,
        geometry=None,
    )
    widget._cache_lock = __import__("threading").Lock()
    widget._raw_map_cache = {}
    loaded = SimpleNamespace(data=np.zeros((4, 4)))

    with patch.object(MapBoxDisplayWidget, "_load_embedded_base_map", return_value=loaded) as embed_mock:
        out = widget._load_raw_map_for_source_mode("bz", "filesystem", purpose="bottom")

    assert out is loaded
    embed_mock.assert_called_once()


def test_context_map_falls_back_to_embedded_refmap_in_filesystem_mode():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        map_source_mode="filesystem",
        map_files={"171": "/tmp/aia171.fits"},
        base_maps={},
        refmaps={"20120712_044615_n4euA": {"data": np.ones((2, 2))}},
        base_geometry=None,
        geometry=None,
    )
    widget._cache_lock = __import__("threading").Lock()
    widget._raw_map_cache = {}
    loaded = SimpleNamespace(data=np.zeros((2, 2)))

    with patch.object(MapBoxDisplayWidget, "_load_embedded_refmap", return_value=loaded) as embed_mock:
        out = widget._load_raw_map_for_source_mode(
            "20120712_044615_n4euA",
            "filesystem",
            purpose="context",
        )

    assert out is loaded
    embed_mock.assert_called_once()


def test_filesystem_mode_loads_src_relpath_before_embedded_crop(tmp_path):
    from types import SimpleNamespace
    from astropy.io import fits

    from pyampp.io.refmaps import apply_refmap_source_cards

    model = tmp_path / "model.h5"
    source = tmp_path / "stereo_a_euvi" / "20120712_044615_n4euA.fts"
    source.parent.mkdir()
    source.write_bytes(b"fits")
    header = fits.Header()
    apply_refmap_source_cards(header, source, model_dir=tmp_path)
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        map_source_mode="filesystem",
        map_files={"171": str(tmp_path / "missing-aia.fits")},
        base_maps={},
        refmaps={
            "20120712_044615_n4euA": {
                "data": np.ones((2, 2)),
                "wcs_header": header.tostring(sep="\n", endcard=True),
            }
        },
        session_input=SimpleNamespace(data_dir=str(tmp_path / "jsoc"), gxmodel_dir=None),
    )
    widget._entry_box_path = model
    widget._cache_lock = __import__("threading").Lock()
    widget._raw_map_cache = {}
    loaded = SimpleNamespace(data=np.zeros((2, 2)))

    with patch("pyampp.gxbox.box_view2d.load_sunpy_map_compat", return_value=loaded) as load_mock, patch.object(
        MapBoxDisplayWidget, "_load_embedded_refmap"
    ) as embed_mock:
        out = widget._load_raw_map_for_source_mode(
            "20120712_044615_n4euA",
            "filesystem",
            purpose="context",
        )

    assert out is loaded
    load_mock.assert_called_once_with(str(source.resolve()))
    embed_mock.assert_not_called()


def test_filesystem_mode_uses_src_path_when_relative_root_misses(tmp_path):
    from types import SimpleNamespace
    from astropy.io import fits

    from pyampp.io.refmaps import apply_refmap_source_cards

    source = tmp_path / "stereo_a_euvi" / "euvi.fts"
    source.parent.mkdir()
    source.write_bytes(b"fits")
    header = fits.Header()
    apply_refmap_source_cards(header, source, model_dir=tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        map_files={},
        refmaps={"euvi": {"data": np.ones((2, 2)), "wcs_header": header.tostring(sep="\n", endcard=True)}},
        session_input=SimpleNamespace(data_dir=None, gxmodel_dir=None),
    )
    widget._entry_box_path = other / "model.h5"
    widget._cache_lock = __import__("threading").Lock()
    widget._raw_map_cache = {}
    loaded = SimpleNamespace(data=np.zeros((2, 2)))

    with patch("pyampp.gxbox.box_view2d.load_sunpy_map_compat", return_value=loaded) as load_mock:
        out = widget._load_raw_map_for_source_mode("euvi", "filesystem", purpose="context")

    assert out is loaded
    load_mock.assert_called_once_with(str(source.resolve()))


def test_embedded_only_context_selects_embedded_then_restores_filesystem():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._auto_embedded_source = False
    widget._default_map_source_mode = "filesystem"
    session = SimpleNamespace(map_source_mode="filesystem", data_dir="", gxmodel_dir=None)
    widget._state = SimpleNamespace(
        map_source_mode="filesystem",
        map_files={"171": "/tmp/aia171.fits"},
        refmaps={"20120712_044615_n4euA": {"data": np.ones((2, 2))}},
        session_input=session,
    )
    widget._entry_box_path = None

    widget._apply_embedded_source_preference("20120712_044615_n4euA")
    assert widget._state.map_source_mode == "embedded"
    assert session.map_source_mode == "embedded"

    widget._apply_embedded_source_preference("171")
    assert widget._state.map_source_mode == "filesystem"
    assert session.map_source_mode == "filesystem"
    assert widget._auto_embedded_source is False


def test_context_map_change_recomputes_view_instead_of_preserving_pixels():
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(selected_context_id="171")
    widget._background_cache_generation = 0
    calls = []

    widget._refresh_status_text = lambda: None
    widget._refresh_map_info = lambda: None
    widget._refresh_plot = lambda preserve_current_view=False: calls.append(preserve_current_view)
    widget._should_preserve_pixel_view = lambda: True
    widget._invalidate_display_prepared_cache = lambda: None
    widget._invalidate_geometry_dependent_display_maps = lambda: None

    MapBoxDisplayWidget.set_context_map_id(widget, "EOVSA_f1.418GHz")

    assert widget._state.selected_context_id == "EOVSA_f1.418GHz"
    assert calls == [False]


def test_set_map_source_mode_clears_prepared_and_raw_caches():
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(map_source_mode="filesystem")
    widget._map_summary_cache = {"Context:filesystem:box_fov:171": "stale"}
    widget._raw_map_cache = {"__rawmap__:filesystem:context:171": object()}
    prepared_entry = {"map": object(), "coverage_fov": None}
    widget._native_crop_cache = {"__native_crop__:filesystem:earth|earth:171": prepared_entry}
    widget._display_prepared_cache = {
        "__display__:filesystem:earth:box_fov:context:171": prepared_entry,
    }
    widget._loaded_map_cache = {"__context__:filesystem:earth:box_fov:171": object()}
    widget._background_cache_generation = 0
    widget._context_prewarm_generation = 0
    widget._cache_lock = __import__("threading").Lock()
    widget._current_axes = object()
    calls = []

    widget._refresh_status_text = lambda: None
    widget._refresh_map_info = lambda: None
    widget._refresh_plot = lambda preserve_current_view=False: calls.append(preserve_current_view)

    MapBoxDisplayWidget.set_map_source_mode(widget, "embedded")

    assert widget._state.map_source_mode == "embedded"
    assert widget._map_summary_cache == {}
    assert widget._raw_map_cache == {}
    assert widget._native_crop_cache == {}
    assert widget._display_prepared_cache == {}
    assert widget._loaded_map_cache == {}
    assert widget._background_cache_generation == 1
    assert widget._context_prewarm_generation == 1
    assert calls == [False]


def test_update_refmap_sources_schedules_background_prewarm_once():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        map_files={},
        refmaps={},
        session_input=SimpleNamespace(refmaps={}),
    )
    widget._map_summary_cache = {}
    widget._cache_lock = __import__("threading").Lock()
    widget._context_prewarm_generation = 0
    widget._invalidate_map_caches = lambda: None
    widget._refresh_map_info = lambda: None
    widget._refresh_plot = lambda preserve_current_view=False: None
    scheduled = []

    widget._schedule_context_prewarm = lambda: scheduled.append("prewarm")

    MapBoxDisplayWidget.update_refmap_sources(
        widget,
        map_files={"stereo304": "/tmp/stereo304.fits"},
        refmaps={"stereo304": {"data": [], "wcs_header": "SIMPLE = T"}},
    )

    assert scheduled == ["prewarm"]


def test_iter_warmable_context_map_keys_includes_embedded_refmaps():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        map_files={"magnetogram": "/tmp/B.fits", "stereo304": "/tmp/stereo304.fits"},
        refmaps={"stereo304": {"data": [], "wcs_header": "SIMPLE = T"}, "AIA_171": {"data": [], "wcs_header": "SIMPLE = T"}},
    )

    assert widget._iter_warmable_context_map_keys() == ["171", "stereo304"]


def test_record_prepare_event_skips_status_callback_off_gui_thread():
    from PyQt5.QtCore import QThread

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._prep_trace_counts = {}
    widget._prep_trace_order = []
    widget._last_status_base_text = "base"
    calls = []
    widget._status_callback = calls.append
    widget._on_gui_thread = lambda: False

    MapBoxDisplayWidget._record_prepare_event(widget, "context crop: stereo304")

    assert widget._prep_trace_counts == {}
    assert calls == []


def test_native_crop_cache_key_includes_geometry_and_source():
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        map_source_mode="embedded",
        fov_definition_observer_key="stereo-a",
        geometry_definition_observer_key="earth",
        fov=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
        geometry=None,
    )

    key = widget._native_crop_cache_key("20260403_195830_284A")

    assert key.startswith("__native_crop__:embedded:")
    assert key.endswith(":20260403_195830_284A")


def test_context_map_for_id_returns_prepared_map_without_runtime_crop():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        map_source_mode="embedded",
        selected_context_id="stereo171",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
        fov=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
        map_files={},
    )
    widget._view_mode = "box_fov"
    widget._cache_lock = __import__("threading").Lock()
    widget._display_prepared_cache = {}
    widget._record_prepare_event = lambda _msg: None

    prepared = SimpleNamespace(data=np.zeros((10, 10)))

    with patch.object(MapBoxDisplayWidget, "_load_raw_map", return_value=prepared), patch.object(
        MapBoxDisplayWidget, "_prepare_map_for_display", return_value=(prepared, None)
    ) as prepare_mock:
        out = widget._context_map_for_id("stereo171", "stereo171")

    assert out is prepared
    prepare_mock.assert_called_once()
    assert prepare_mock.call_args.kwargs["purpose"] == "context"


def test_cross_observer_context_skips_native_crop_before_reproject():
    from types import SimpleNamespace
    from unittest.mock import patch
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget, _CONTEXT_PREPARE_VARIANT_FOV_CROP
    from pyampp.gxbox.selector_api import DisplayFovSelection

    smap = SimpleNamespace(data=np.zeros((64, 64)), meta={})
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        fov_definition_observer_key="stereo-a",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
        fov=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
    )
    widget._record_prepare_event = lambda _msg: None
    widget._apply_display_scaling = lambda m, _k: m
    widget._apply_hmi_context_adjustments = lambda _k, m: m
    widget._reproject_fov_override_for_display = lambda *_a, **_k: object()
    widget._reproject_map_for_display_observer = lambda m, **kwargs: (m, None)

    with patch.object(MapBoxDisplayWidget, "_map_display_los_matches", return_value=False), patch.object(
        MapBoxDisplayWidget, "_is_native_spacecraft_payload", return_value=False
    ), patch.object(MapBoxDisplayWidget, "_get_native_cropped_map") as native_crop_mock:
        out, _cov = widget._prepare_map_for_display(
            "171",
            smap,
            purpose="context",
            use_native_crop=True,
        )

    assert out is smap
    native_crop_mock.assert_not_called()


def test_embedded_stereo_map_keeps_native_crop_even_when_los_match_is_ambiguous():
    from types import SimpleNamespace
    from unittest.mock import patch
    from astropy.io import fits
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget, _CONTEXT_PREPARE_VARIANT_FOV_CROP, _EMBEDDED_REFMAP_FLAG
    from pyampp.gxbox.boxutils import map_from_data_header_compat
    from pyampp.gxbox.selector_api import DisplayFovSelection

    header = fits.Header()
    header["TELESCOP"] = "STEREO"
    header["INSTRUME"] = "SECCHI"
    header["PYALIGN"] = False
    header["NAXIS"] = 2
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 32.5
    header["CRPIX2"] = 32.5
    header["CRVAL1"] = -1020.0
    header["CRVAL2"] = 110.0
    header["CDELT1"] = 12.0
    header["CDELT2"] = 12.0
    header["DATE-OBS"] = "2026-04-03T19:58:30.006"
    header["HGLN_OBS"] = 54.3710463167
    header["HGLT_OBS"] = -0.875478602937
    header["DSUN_OBS"] = 1.496e11
    smap = map_from_data_header_compat(np.ones((64, 64), dtype=np.float32), header)
    smap.meta[_EMBEDDED_REFMAP_FLAG] = True

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        fov_definition_observer_key="stereo-a",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
        fov=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
    )
    widget._status_callback = None
    widget._observer_warning_cache = set()
    widget._observer_metadata_cache = {}
    widget._refmap_display_notices = []
    widget._on_gui_thread = lambda: False
    widget._refresh_status_text = lambda: None
    widget._record_prepare_event = lambda _msg: None
    widget._apply_display_scaling = lambda m, _k: m
    widget._apply_hmi_context_adjustments = lambda _k, m: m
    cropped = SimpleNamespace(data=np.zeros((8, 8)))

    with patch.object(MapBoxDisplayWidget, "_map_display_los_matches", return_value=False), patch.object(
        MapBoxDisplayWidget, "_reproject_fov_override_for_display", return_value=None
    ), patch.object(
        MapBoxDisplayWidget, "_get_native_cropped_map", return_value=cropped
    ) as native_crop_mock, patch.object(
        MapBoxDisplayWidget, "_reproject_map_for_display_observer", return_value=(cropped, None)
    ):
        out, _cov = widget._prepare_context_map(
            "20260403_200030_171A",
            smap,
            prepare_variant=_CONTEXT_PREPARE_VARIANT_FOV_CROP,
        )

    native_crop_mock.assert_called_once()
    assert out is cropped


def test_embedded_stereo_without_telescop_header_still_crops_at_display():
    from types import SimpleNamespace
    from astropy.io import fits
    from astropy.time import Time
    from sunpy.coordinates import get_earth
    from pyampp.gxbox.box_view2d import (
        MapBoxDisplayWidget,
        _CONTEXT_PREPARE_VARIANT_FOV_CROP,
        _EMBEDDED_REFMAP_FLAG,
        _MIN_DISPLAY_MAP_SIDE,
    )
    from pyampp.gxbox.boxutils import map_from_data_header_compat
    from pyampp.gxbox.selector_api import DisplayFovSelection

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 512
    header["NAXIS2"] = 512
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 256.5
    header["CRPIX2"] = 256.5
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 3.2
    header["CDELT2"] = 3.2
    header["DATE-OBS"] = "2026-04-03T19:58:30.006"
    header["HGLN_OBS"] = 54.3710463167
    header["HGLT_OBS"] = -0.875478602937
    header["DSUN_OBS"] = 1.496e11
    header["PYALIGN"] = False
    smap = map_from_data_header_compat(np.ones((512, 512), dtype=np.float32), header)
    smap.meta[_EMBEDDED_REFMAP_FLAG] = True

    obstime = Time("2026-04-03T19:46:37.800")
    earth = get_earth(obstime)
    stereo_obs = smap.observer_coordinate

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._status_callback = None
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        fov_definition_observer_key="stereo-a",
        geometry_definition_observer_key="earth",
        fov=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
        session_input=SimpleNamespace(time_iso=obstime.isot),
        map_source_mode="embedded",
    )
    widget._observer_coord_cache = {"earth": earth, "stereo-a": stereo_obs}
    widget._observer_metadata_cache = {}
    widget._observer_warning_cache = set()
    widget._refmap_display_notices = []
    widget._observer_source_b3d = lambda: {}
    widget._record_prepare_event = lambda _msg: None
    widget._apply_display_scaling = lambda m, _k: m
    widget._view_mode = "box_fov"
    widget._on_gui_thread = lambda: False
    widget._refresh_status_text = lambda: None
    widget._observer_context = lambda key, obstime: SimpleNamespace(
        observer_coordinate=widget._observer_coord_cache.get(key, earth),
        date=obstime,
    )

    assert MapBoxDisplayWidget._is_native_spacecraft_payload(smap) is True
    out, _coverage = widget._prepare_context_map(
        "171",
        smap,
        prepare_variant=_CONTEXT_PREPARE_VARIANT_FOV_CROP,
    )
    shape = tuple(np.asarray(out.data).shape)
    assert shape != (512, 512)
    assert shape[0] >= _MIN_DISPLAY_MAP_SIDE
    assert shape[1] >= _MIN_DISPLAY_MAP_SIDE


def test_cross_observer_bottom_reprojects_at_display_anchor():
    from types import SimpleNamespace
    from unittest.mock import patch
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
    from pyampp.gxbox.selector_api import DisplayFovSelection

    smap = SimpleNamespace(data=np.ones((301, 300)), meta={})
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
        fov=DisplayFovSelection(0.0, 0.0, 100.0, 100.0),
        fov_definition_observer_key="earth",
    )
    widget.__dict__["_view_mode"] = "box_fov"
    widget._record_prepare_event = lambda _msg: None
    widget._apply_display_scaling = lambda m, _k: m

    with patch.object(MapBoxDisplayWidget, "_map_display_los_matches", return_value=False), patch.object(
        MapBoxDisplayWidget,
        "_fov_selection_projected_to_display_observer",
        return_value=DisplayFovSelection(1.0, 2.0, 80.0, 80.0),
    ), patch.object(
        MapBoxDisplayWidget,
        "_reproject_map_for_display_observer",
        return_value=(smap, None),
    ) as reproj_mock:
        out, _cov = widget._prepare_map_for_display("bz", smap, purpose="bottom")

    assert out is smap
    reproj_mock.assert_called_once()


def test_exportable_fov_selection_returns_none_when_projection_fails():
    from astropy.time import Time
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        fov_definition_observer_key="earth",
        fov=DisplayFovSelection(10.0, 20.0, 30.0, 40.0),
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
    )
    with patch.object(MapBoxDisplayWidget, "_display_obstime_anchor", return_value=Time("2026-04-03T19:46:37.800")), patch.object(
        MapBoxDisplayWidget, "_observers_share_los", return_value=False
    ), patch.object(MapBoxDisplayWidget, "_project_fov_between_observers", return_value=None):
        assert widget.exportable_fov_selection() is None


def test_exportable_fov_selection_returns_none_without_anchor():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="earth",
        fov_definition_observer_key="earth",
        fov=DisplayFovSelection(10.0, 20.0, 30.0, 40.0),
        session_input=SimpleNamespace(time_iso=""),
    )
    with patch.object(MapBoxDisplayWidget, "_display_obstime_anchor", return_value=None):
        assert widget.exportable_fov_selection() is None


def test_persist_selector_result_rejects_inconsistent_fov_box_observer(tmp_path):
    from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry
    from pyampp.gxbox.selector_api import BoxGeometrySelection, CoordMode, DisplayFovBoxSelection, SelectorDialogResult

    entry = tmp_path / "model.h5"
    result = SelectorDialogResult(
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        fov=DisplayFovSelection(0.0, 0.0, 100.0, 100.0),
    )
    fov_box = DisplayFovBoxSelection(
        center_x_arcsec=0.0,
        center_y_arcsec=0.0,
        width_arcsec=100.0,
        height_arcsec=100.0,
        z_min_mm=-100.0,
        z_max_mm=100.0,
        observer_key="earth",
    )
    observer_state = {
        "display_observer_key": "stereo-a",
        "display_fov_obstime": "2026-04-03T19:46:37.800",
    }
    box_data = {"observer": {"name": "earth"}, "refmaps": {}}

    with patch("pyampp.gxbox.gxbox_selector_view.load_model", return_value=box_data):
        ok = _persist_selector_result_to_entry(
            entry,
            result,
            fov_box=fov_box,
            observer_state=observer_state,
            output_path=tmp_path / "out.h5",
        )
    assert ok is False


def test_persist_selector_result_rejects_missing_display_fov_obstime(tmp_path):
    from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry
    from pyampp.gxbox.selector_api import BoxGeometrySelection, CoordMode, SelectorDialogResult

    entry = tmp_path / "model.h5"
    result = SelectorDialogResult(
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        fov=DisplayFovSelection(0.0, 0.0, 100.0, 100.0),
    )
    observer_state = {
        "display_observer_key": "earth",
        "display_fov_obstime": "unknown",
    }
    box_data = {"observer": {"name": "earth"}, "refmaps": {}}

    with patch("pyampp.gxbox.gxbox_selector_view.load_model", return_value=box_data):
        ok = _persist_selector_result_to_entry(
            entry,
            result,
            observer_state=observer_state,
            output_path=tmp_path / "out.h5",
        )
    assert ok is False


def test_fov_persistence_issue_reports_cross_observer_mismatch():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        fov_definition_observer_key="earth",
        fov=DisplayFovSelection(10.0, 20.0, 30.0, 40.0),
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
    )
    with patch.object(MapBoxDisplayWidget, "_display_obstime_cache_token", return_value="2026-04-03T19:46:37.800"), patch.object(
        MapBoxDisplayWidget, "exportable_fov_selection", return_value=None
    ), patch.object(MapBoxDisplayWidget, "_observer_label_for_key", side_effect=lambda key: str(key)):
        issue = widget.fov_persistence_issue()
    assert issue is not None
    assert "earth" in issue
    assert "stereo-a" in issue


def test_set_display_observer_preserves_fov_definition():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    original_fov = DisplayFovSelection(10.0, 20.0, 30.0, 40.0)
    widget._state = SimpleNamespace(
        display_observer_key="earth",
        fov_definition_observer_key="earth",
        fov=original_fov,
        fov_box=None,
        session_input=SimpleNamespace(time_iso="2026-04-03T19:46:37.800"),
        square_fov=False,
    )
    widget._view_mode = "box_fov"

    with patch.object(MapBoxDisplayWidget, "_enabled_observer_keys", return_value={"earth", "stereo-a"}), patch.object(
        MapBoxDisplayWidget, "_normalize_display_observer_state"
    ), patch.object(MapBoxDisplayWidget, "_invalidate_display_prepared_cache"), patch.object(
        MapBoxDisplayWidget, "_refresh_status_text"
    ), patch.object(MapBoxDisplayWidget, "_emit_observer_info"), patch.object(
        MapBoxDisplayWidget, "_schedule_observer_refresh"
    ), patch.object(MapBoxDisplayWidget, "_should_preserve_pixel_view", return_value=False):
        widget.set_display_observer_key("stereo-a")

    assert widget._state.display_observer_key == "stereo-a"
    assert widget._state.fov_definition_observer_key == "earth"
    assert widget._state.fov is original_fov
    assert widget._state.fov.center_x_arcsec == 10.0


def test_compute_fov_box_from_current_selection_uses_definition_observer():
    from types import SimpleNamespace

    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(
        display_observer_key="stereo-a",
        fov_definition_observer_key="earth",
        geometry_definition_observer_key="earth",
        fov=DisplayFovSelection(10.0, 20.0, 30.0, 40.0),
        square_fov=False,
    )
    widget._current_map = SimpleNamespace(date="2026-04-03T19:46:37.800")

    with patch.object(MapBoxDisplayWidget, "_observer_context", return_value=widget._current_map), patch.object(
        MapBoxDisplayWidget, "_build_legacy_box",
        return_value=SimpleNamespace(model_box_corners_world=lambda: object()),
    ), patch.object(
        MapBoxDisplayWidget,
        "_resolved_observer_for_map",
        return_value="earth",
    ) as resolve_mock, patch(
        "pyampp.gxbox.box_view2d.build_fov_box_from_user_hpc_and_red_box_world",
        return_value={
            "xc_arcsec": 10.0,
            "yc_arcsec": 20.0,
            "xsize_arcsec": 30.0,
            "ysize_arcsec": 40.0,
            "zmin_mm": -1.0,
            "zmax_mm": 1.0,
        },
    ):
        out = widget._compute_fov_box_from_current_selection()

    assert out is not None
    assert out.observer_key == "earth"
    resolve_mock.assert_called()
    assert resolve_mock.call_args[0][1] == "earth"


def test_persist_selector_result_can_clear_observer_fov(tmp_path):
    from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry
    from pyampp.gxbox.selector_api import BoxGeometrySelection, CoordMode, SelectorDialogResult

    entry = tmp_path / "model.h5"
    result = SelectorDialogResult(
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        fov=None,
    )
    box_data = {
        "observer": {
            "name": "earth",
            "fov": {"xc_arcsec": 0.0, "yc_arcsec": 0.0, "xsize_arcsec": 100.0, "ysize_arcsec": 100.0},
            "fov_box": {"xc_arcsec": 0.0, "yc_arcsec": 0.0, "xsize_arcsec": 100.0, "ysize_arcsec": 100.0},
        },
        "refmaps": {},
    }

    with patch("pyampp.gxbox.gxbox_selector_view.load_model", return_value=box_data), patch(
        "pyampp.gxbox.gxbox_selector_view.save_model"
    ) as save_model:
        ok = _persist_selector_result_to_entry(
            entry,
            result,
            observer_state={"display_observer_key": "stereo-a", "display_fov_obstime": "2026-04-03T19:46:37.800"},
            output_path=tmp_path / "out.h5",
            clear_observer_fov=True,
        )

    assert ok is True
    saved = save_model.call_args[0][0]
    assert saved["observer"]["name"] == "stereo-a"
    assert "fov" not in saved["observer"]
    assert "fov_box" not in saved["observer"]


def test_persist_2d_fov_without_box_writes_observer_key(tmp_path):
    """2D FOV-only saves must stamp observer_key so reload does not default to Earth."""
    from pyampp.gxbox.gxbox_selector_view import _persist_selector_result_to_entry
    from pyampp.gxbox.selector_api import BoxGeometrySelection, CoordMode, DisplayFovSelection, SelectorDialogResult

    entry = tmp_path / "model.h5"
    result = SelectorDialogResult(
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        fov=DisplayFovSelection(10.0, 20.0, 100.0, 80.0),
        square_fov=False,
    )
    box_data = {"observer": {"name": "earth"}, "refmaps": {}}

    with patch("pyampp.gxbox.gxbox_selector_view.load_model", return_value=box_data), patch(
        "pyampp.gxbox.gxbox_selector_view.save_model"
    ) as save_model:
        ok = _persist_selector_result_to_entry(
            entry,
            result,
            observer_state={
                "display_observer_key": "stereo-a",
                "display_fov_obstime": "2026-04-03T19:46:37.800",
            },
            fov_box=None,
            output_path=tmp_path / "out.h5",
        )

    assert ok is True
    saved = save_model.call_args[0][0]
    assert "fov_box" not in saved["observer"]
    fov = saved["observer"]["fov"]
    assert fov["observer_key"] == "stereo-a"
    assert saved["observer"]["name"] == "stereo-a"


def test_initialize_uses_display_observer_when_fov_box_missing():
    """Without fov_box, FOV definition observer must not silently become Earth."""
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
    from pyampp.gxbox.selector_api import (
        BoxGeometrySelection,
        CoordMode,
        DisplayFovSelection,
        SelectorSessionInput,
    )

    session = SelectorSessionInput(
        time_iso="2026-04-03T19:46:37.800",
        data_dir="",
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        fov=DisplayFovSelection(10.0, 20.0, 100.0, 80.0),
        fov_box=None,
        display_observer_key="stereo-a",
        fov_definition_observer_key="stereo-a",
        map_ids=("171",),
        initial_map_id="171",
    )
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._default_context_id = lambda _s: "171"
    widget._default_bottom_id = lambda _s: None
    widget._map_summary_cache = {}
    widget._observer_coord_cache = {}
    widget._observer_metadata_cache = {}
    widget._observer_warning_cache = {}
    widget._clear_prepare_trace = lambda: None
    widget._invalidate_map_caches = lambda: None
    widget._apply_embedded_source_preference = lambda _id: None
    widget._normalize_display_observer_state = lambda: None
    widget._refresh_status_text = lambda: None
    widget._refresh_map_info = lambda: None
    widget._emit_observer_info = lambda: None
    widget._update_fov_control_enabled_state = lambda: None
    widget._normalize_observer_key = staticmethod(MapBoxDisplayWidget._normalize_observer_key).__get__(widget, MapBoxDisplayWidget)

    widget.initialize(session)

    assert widget._state.display_observer_key == "stereo-a"
    assert widget._state.fov_definition_observer_key == "stereo-a"
    assert widget._state.fov_box is None


def test_initialize_falls_back_to_display_observer_without_fov_def_key():
    """Legacy 2D FOV entries with only observer.name still restore the FOV frame."""
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
    from pyampp.gxbox.selector_api import (
        BoxGeometrySelection,
        CoordMode,
        DisplayFovSelection,
        SelectorSessionInput,
    )

    session = SelectorSessionInput(
        time_iso="2026-04-03T19:46:37.800",
        data_dir="",
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        fov=DisplayFovSelection(10.0, 20.0, 100.0, 80.0),
        fov_box=None,
        display_observer_key="stereo-a",
        fov_definition_observer_key=None,
        map_ids=("171",),
        initial_map_id="171",
    )
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._default_context_id = lambda _s: "171"
    widget._default_bottom_id = lambda _s: None
    widget._map_summary_cache = {}
    widget._observer_coord_cache = {}
    widget._observer_metadata_cache = {}
    widget._observer_warning_cache = {}
    widget._clear_prepare_trace = lambda: None
    widget._invalidate_map_caches = lambda: None
    widget._apply_embedded_source_preference = lambda _id: None
    widget._normalize_display_observer_state = lambda: None
    widget._refresh_status_text = lambda: None
    widget._refresh_map_info = lambda: None
    widget._emit_observer_info = lambda: None
    widget._update_fov_control_enabled_state = lambda: None
    widget._normalize_observer_key = staticmethod(MapBoxDisplayWidget._normalize_observer_key).__get__(widget, MapBoxDisplayWidget)

    widget.initialize(session)

    assert widget._state.fov_definition_observer_key == "stereo-a"


def test_pre_accept_callback_keeps_dialog_open_on_cancel():
    """Mismatch Cancel / failed persist must not close Apply & Close."""
    from PyQt5.QtWidgets import QApplication, QDialog

    from pyampp.gxbox.fov_selector_gui import FovBoxSelectorDialog
    from pyampp.gxbox.selector_api import BoxGeometrySelection, CoordMode, SelectorSessionInput

    app = QApplication.instance() or QApplication([])
    session = SelectorSessionInput(
        time_iso="2026-04-03T19:46:37.800",
        data_dir="",
        geometry=BoxGeometrySelection(CoordMode.HPC, 0.0, 0.0, 4, 3, 2, 1400.0),
        map_ids=("171",),
        initial_map_id="171",
    )
    dialog = FovBoxSelectorDialog(session_input=session)
    dialog.set_pre_accept_callback(lambda: False)
    dialog.accept()
    assert dialog.result() != QDialog.Accepted
    assert dialog.accepted_selection() is None
