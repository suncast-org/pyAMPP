import numpy as np
from astropy.io import fits

from pyampp.gxbox.box_view2d import MapBoxDisplayWidget
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
