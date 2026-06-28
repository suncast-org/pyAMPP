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

    stereo = tmp_path / "stereo304.fits"
    stereo.write_text("placeholder")
    mock_build.return_value = {"stereo304": {"data": np.ones((4, 4)), "wcs_header": "SIMPLE  = T"}}
    box_data = {
        "base": {"index": np.bytes_(b"DATE-OBS = '2026-04-03T19:46:37.800'\n")},
        "refmaps": {},
    }

    embedded, skipped = _embed_external_refmaps_into_box_data(box_data, [str(stereo)])

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
        "base": {"index": np.bytes_(b"DATE-OBS = '2026-04-03T19:46:37.800'\n")},
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

    with patch.object(
        MapBoxDisplayWidget,
        "_display_observer_reproject_header_for_selection",
        return_value=None,
    ) as roi_mock, patch.object(
        MapBoxDisplayWidget,
        "_current_display_prepare_fov",
        return_value=DisplayFovSelection(-1021.28, 107.67, 801.64, 801.64),
    ) as implicit_fov_mock, patch.object(
        MapBoxDisplayWidget,
        "_resolve_display_observer_coord",
        return_value="stereo-observer",
    ), patch.object(
        MapBoxDisplayWidget,
        "_solar_disk_center_for_observer",
        return_value="disk-center",
    ):
        widget._reproject_map_for_display_observer(smap, fov_override=None)

    roi_mock.assert_called_once()
    assert roi_mock.call_args.args[3] is None
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
    with patch.object(
        smap,
        "reproject_to",
        return_value=reprojected,
    ) as reproj_mock:
        out, coverage = widget._reproject_map_for_display_observer(smap)

    reproj_mock.assert_called_once()
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


def test_context_map_change_recomputes_view_instead_of_preserving_pixels():
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(selected_context_id="171")
    calls = []

    widget._refresh_status_text = lambda: None
    widget._refresh_map_info = lambda: None
    widget._refresh_plot = lambda preserve_current_view=False: calls.append(preserve_current_view)
    widget._should_preserve_pixel_view = lambda: True

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


def test_cross_observer_bottom_skips_observer_reprojection():
    from types import SimpleNamespace
    from unittest.mock import patch
    from pyampp.gxbox.box_view2d import MapBoxDisplayWidget

    smap = SimpleNamespace(data=np.ones((301, 300)), meta={})
    widget = MapBoxDisplayWidget.__new__(MapBoxDisplayWidget)
    widget._state = SimpleNamespace(display_observer_key="stereo-a")
    widget._record_prepare_event = lambda _msg: None
    widget._apply_display_scaling = lambda m, _k: m

    with patch.object(MapBoxDisplayWidget, "_map_display_los_matches", return_value=False), patch.object(
        MapBoxDisplayWidget, "_reproject_map_for_display_observer"
    ) as reproj_mock:
        out, _cov = widget._prepare_map_for_display("bz", smap, purpose="bottom")

    assert out is smap
    reproj_mock.assert_not_called()
