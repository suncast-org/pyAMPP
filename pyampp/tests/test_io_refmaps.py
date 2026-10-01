import h5py
import numpy as np
import pytest
from astropy.io import fits

from pyampp.io import (
    add_fits_refmaps_from_dir_to_h5,
    add_fits_refmaps_to_h5,
    box_corners_world_from_model,
    build_fits_refmaps_for_model,
    discover_fits_refmap_map_ids,
    discover_fits_refmap_paths,
    list_embedded_refmap_ids,
    model_obstime_from_base_index,
    remove_refmaps_from_h5,
)
from pyampp.tests._fits_header import canonical_base_index_header


def _test_model_dict(*, base_date_obs: str = "2026-04-03T19:46:37.800") -> dict:
    return {
        "base": {"index": canonical_base_index_header(date_obs=base_date_obs)},
        "corona": {
            "bx": np.zeros((32, 32, 16), dtype=np.float32),
            "dr": np.array([0.05, 0.05, 0.05], dtype=np.float64),
        },
    }


def _hpc_header(
    *,
    shape=(8, 8),
    crpix=(4.5, 4.5),
    cdelt=(1.0, 1.0),
    date_obs="2026-04-03T20:00:00.000",
):
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = shape[1]
    header["NAXIS2"] = shape[0]
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = crpix[0]
    header["CRPIX2"] = crpix[1]
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = cdelt[0]
    header["CDELT2"] = cdelt[1]
    header["DATE-OBS"] = date_obs
    header["DSUN_OBS"] = 1.496e11
    header["RSUN_REF"] = 6.957e8
    header["RSUN_OBS"] = 959.63
    header["HGLN_OBS"] = 0.0
    header["HGLT_OBS"] = 0.0
    return header


def _write_refmap_model(path, *, base_date_obs="2026-04-03T19:46:37.800"):
    header = _hpc_header(shape=(4, 4), crpix=(2.5, 2.5))
    with h5py.File(path, "w") as h5f:
        base = h5f.create_group("base", track_order=True)
        base.create_dataset(
            "index",
            data=np.bytes_(canonical_base_index_header(date_obs=base_date_obs)),
        )
        corona = h5f.create_group("corona", track_order=True)
        corona.create_dataset("bx", data=np.zeros((32, 32, 16), dtype=np.float32))
        corona.create_dataset("dr", data=np.array([0.05, 0.05, 0.05], dtype=np.float64))
        refmaps = h5f.create_group("refmaps", track_order=True)
        group = refmaps.create_group("Bz_reference", track_order=True)
        group.attrs["order_index"] = np.int64(0)
        group.create_dataset("data", data=np.ones((4, 4), dtype=np.float32))
        group.create_dataset("wcs_header", data=np.bytes_(header.tostring(sep="\n", endcard=True)))


def _write_aia_fits(path, wavelength=171, date_obs="2026-04-03T20:00:00.000", shape=(128, 128)):
    header = _hpc_header(
        shape=shape,
        crpix=(shape[1] / 2.0, shape[0] / 2.0),
        cdelt=(2.0, 2.0),
        date_obs=date_obs,
    )
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = -5.0
    header["TELESCOP"] = "SDO/AIA"
    header["INSTRUME"] = "AIA_3"
    header["WAVELNTH"] = int(wavelength)
    header["WAVEUNIT"] = "angstrom"
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def _write_eovsa_fits(path, freq_hz, *, shape=(128, 128)):
    header = _hpc_header(shape=shape, crpix=(shape[1] / 2.0, shape[0] / 2.0), cdelt=(2.0, 2.0))
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = -5.0
    header["TELESCOP"] = "EOVSA"
    header["INSTRUME"] = "EOVSA"
    header["CRVAL3"] = float(freq_hz)
    header["CUNIT3"] = "Hz"
    fits.PrimaryHDU(data=np.ones(shape, dtype=np.float32), header=header).writeto(path)


def test_add_fits_refmaps_to_h5_crops_and_preserves_aia_metadata(tmp_path):
    model = tmp_path / "model.h5"
    source = tmp_path / "aia171.fits"
    _write_refmap_model(model)
    _write_aia_fits(source, wavelength=171)

    added = add_fits_refmaps_to_h5(model, [source])

    assert [item.map_id for item in added] == ["AIA_171"]
    with h5py.File(model, "r") as h5f:
        group = h5f["refmaps/AIA_171"]
        assert group.attrs["order_index"] == 1
        assert tuple(group["data"].shape) != (4, 4)
        assert group["data"].ndim == 2
        header = fits.Header.fromstring(group["wcs_header"][()].decode(), sep="\n")
        assert header["TELESCOP"] == "SDO/AIA"
        assert header["WAVELNTH"] == 171
        assert header["WAVEUNIT"] == "angstrom"
        assert header["PYALIGN"] is False


def test_add_fits_refmaps_preserves_native_date_obs(tmp_path):
    model = tmp_path / "model.h5"
    source = tmp_path / "aia171.fits"
    model_time = "2026-04-03T19:46:37.800"
    source_time = "2026-04-03T19:46:33.350"
    _write_refmap_model(model, base_date_obs=model_time)
    _write_aia_fits(source, wavelength=171, date_obs=source_time)

    with h5py.File(model, "r") as h5f:
        assert model_obstime_from_base_index(h5f) == model_time

    add_fits_refmaps_to_h5(model, [source], overwrite=True)

    with h5py.File(model, "r") as h5f:
        header = fits.Header.fromstring(h5f["refmaps/AIA_171/wcs_header"][()].decode(), sep="\n")
        assert header["DATE-OBS"] == source_time
        assert header["MODELT"] == model_time
        assert header["SRC_DATE"] == source_time
        assert header["PYALIGN"] is False
        assert header["PYEMBED"] is True


def test_box_corners_world_from_model_uses_corona_geometry(tmp_path):
    model = tmp_path / "model.h5"
    _write_refmap_model(model, base_date_obs="2026-04-03T19:46:37.800")
    with h5py.File(model, "r") as h5f:
        world = box_corners_world_from_model(h5f)
    assert world is not None
    assert len(world) == 8


def test_add_fits_refmaps_from_dir_to_h5_adds_all_fits(tmp_path):
    model = tmp_path / "model.h5"
    source_dir = tmp_path / "fits"
    source_dir.mkdir()
    _write_refmap_model(model)
    _write_eovsa_fits(source_dir / "eovsa_1.fits", 1.418334960938e9)
    _write_eovsa_fits(source_dir / "eovsa_2.fits", 2.873583984375e9)

    added = add_fits_refmaps_from_dir_to_h5(model, source_dir)

    assert [item.map_id for item in added] == ["EOVSA_f1.418GHz", "EOVSA_f2.874GHz"]
    with h5py.File(model, "r") as h5f:
        assert "EOVSA_f1.418GHz" in h5f["refmaps"]
        assert "EOVSA_f2.874GHz" in h5f["refmaps"]


def test_build_fits_refmaps_for_model_from_directory(tmp_path):
    source_dir = tmp_path / "fits"
    source_dir.mkdir()
    _write_eovsa_fits(source_dir / "eovsa_1.fits", 1.418334960938e9)
    _write_eovsa_fits(source_dir / "eovsa_2.fits", 2.873583984375e9)

    discovered = discover_fits_refmap_paths([source_dir])
    payloads = build_fits_refmaps_for_model(
        [source_dir],
        model_obstime="2026-04-03T19:46:37.800",
        model=_test_model_dict(),
    )

    assert [p.name for p in discovered] == ["eovsa_1.fits", "eovsa_2.fits"]
    assert set(payloads) == {"EOVSA_f1.418GHz", "EOVSA_f2.874GHz"}
    for payload in payloads.values():
        assert payload["data"].ndim == 2


def test_discover_fits_refmap_map_ids_known_only_excludes_hmi_wavelength(tmp_path):
    source_dir = tmp_path / "fits"
    source_dir.mkdir()
    aia = source_dir / "aia171.fits"
    _write_aia_fits(aia, wavelength=171)
    eovsa = source_dir / "eovsa.fits"
    _write_eovsa_fits(eovsa, 1.418334960938e9)

    hmi_header = _hpc_header()
    hmi_header["TELESCOP"] = "SDO/HMI"
    hmi_header["INSTRUME"] = "HMI"
    hmi_header["WAVELNTH"] = 6173
    hmi = source_dir / "hmi_field.fits"
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32), header=hmi_header).writeto(hmi)

    discovered = discover_fits_refmap_map_ids([source_dir], generic=False)

    assert discovered == {
        aia.resolve(): "AIA_171",
        eovsa.resolve(): "EOVSA_f1.418GHz",
    }


def test_build_fits_refmaps_for_model_known_only_uses_shared_discovery_policy(tmp_path):
    source_dir = tmp_path / "fits"
    source_dir.mkdir()
    _write_aia_fits(source_dir / "aia171.fits", wavelength=171)
    generic_header = _hpc_header()
    generic = source_dir / "generic_context.fits"
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32), header=generic_header).writeto(generic)

    payloads = build_fits_refmaps_for_model(
        [source_dir],
        model_obstime="2026-04-03T19:46:37.800",
        model=_test_model_dict(),
        generic=False,
    )

    assert set(payloads) == {"AIA_171"}


def test_add_fits_refmaps_to_h5_requires_overwrite_for_existing_id(tmp_path):
    model = tmp_path / "model.h5"
    source = tmp_path / "aia171.fits"
    _write_refmap_model(model)
    _write_aia_fits(source, wavelength=171)

    add_fits_refmaps_to_h5(model, [source])
    try:
        add_fits_refmaps_to_h5(model, [source])
    except ValueError as exc:
        assert "refmap already exists" in str(exc)
    else:
        raise AssertionError("expected duplicate refmap insertion to fail")

    add_fits_refmaps_to_h5(model, [source], overwrite=True)
    with h5py.File(model, "r") as h5f:
        assert "AIA_171" in h5f["refmaps"]


def _write_stereo_refmap_group(refmaps, map_id, *, shape=(8, 8)):
    header = _hpc_header(shape=shape, crpix=(shape[1] / 2.0, shape[0] / 2.0))
    header["TELESCOP"] = "STEREO"
    header["INSTRUME"] = "SECCHI"
    group = refmaps.create_group(map_id, track_order=True)
    group.attrs["order_index"] = np.int64(len(refmaps) - 1)
    group.create_dataset("data", data=np.ones(shape, dtype=np.float32))
    group.create_dataset("wcs_header", data=np.bytes_(header.tostring(sep="\n", endcard=True)))


def test_list_embedded_refmap_ids(tmp_path):
    model = tmp_path / "model.h5"
    _write_refmap_model(model)
    with h5py.File(model, "r+") as h5f:
        refmaps = h5f["refmaps"]
        _write_stereo_refmap_group(refmaps, "stereo304")

    assert list_embedded_refmap_ids(model) == ["Bz_reference", "stereo304"]


def test_remove_refmaps_from_h5_by_map_ids(tmp_path):
    model = tmp_path / "model.h5"
    _write_refmap_model(model)
    with h5py.File(model, "r+") as h5f:
        _write_stereo_refmap_group(h5f["refmaps"], "stereo304")

    removed = remove_refmaps_from_h5(model, ["stereo304"])

    assert [item.map_id for item in removed] == ["stereo304"]
    assert list_embedded_refmap_ids(model) == ["Bz_reference"]


def test_remove_refmaps_from_h5_by_telescope(tmp_path):
    model = tmp_path / "model.h5"
    _write_refmap_model(model)
    with h5py.File(model, "r+") as h5f:
        refmaps = h5f["refmaps"]
        _write_stereo_refmap_group(refmaps, "stereo304")
        _write_stereo_refmap_group(refmaps, "stereo171")

    removed = remove_refmaps_from_h5(model, telescope="STEREO")

    assert [item.map_id for item in removed] == ["stereo171", "stereo304"]
    assert list_embedded_refmap_ids(model) == ["Bz_reference"]


def test_remove_refmaps_from_h5_remove_all(tmp_path):
    model = tmp_path / "model.h5"
    _write_refmap_model(model)
    with h5py.File(model, "r+") as h5f:
        _write_stereo_refmap_group(h5f["refmaps"], "stereo304")

    removed = remove_refmaps_from_h5(model, remove_all=True)

    assert [item.map_id for item in removed] == ["Bz_reference", "stereo304"]
    assert list_embedded_refmap_ids(model) == []


def test_remove_refmaps_from_h5_empty_telescope_raises(tmp_path):
    model = tmp_path / "model.h5"
    _write_refmap_model(model)
    with h5py.File(model, "r+") as h5f:
        _write_stereo_refmap_group(h5f["refmaps"], "stereo304")

    with pytest.raises(ValueError, match="non-empty"):
        remove_refmaps_from_h5(model, telescope="")
    with pytest.raises(ValueError, match="non-empty"):
        remove_refmaps_from_h5(model, telescope="   ")

    assert list_embedded_refmap_ids(model) == ["Bz_reference", "stereo304"]


def test_crop_refmap_argument_warns_and_default_is_silent(tmp_path):
    import warnings

    model = tmp_path / "model.h5"
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert add_fits_refmaps_to_h5(model, []) == []
        assert add_fits_refmaps_from_dir_to_h5(model, empty_dir) == []

    with pytest.warns(DeprecationWarning, match="crop_refmap is ignored"):
        assert add_fits_refmaps_to_h5(model, [], crop_refmap="Bz_reference") == []
    with pytest.warns(DeprecationWarning, match="crop_refmap is ignored"):
        assert add_fits_refmaps_from_dir_to_h5(model, empty_dir, crop_refmap=None) == []


def test_model_context_keeps_execute_beside_geometry_contract(tmp_path):
    from pyampp.io.refmaps import _model_context_from_open_h5, _source_roots_from_model_context

    model = tmp_path / "model.h5"
    _write_refmap_model(model)
    with h5py.File(model, "r+") as h5f:
        meta = h5f.create_group("metadata")
        meta.create_dataset(
            "execute",
            data=np.bytes_("gx-fov2box --data-dir /cache --gxmodel-dir /models"),
        )
        contract = meta.create_group("geometry_contract")
        contract.create_dataset("obstime", data=np.bytes_("2026-04-03T19:46:37.800"))
        ctx = _model_context_from_open_h5(h5f)

    data_dir, gxmodel_dir = _source_roots_from_model_context(ctx)
    assert data_dir == "/cache"
    assert gxmodel_dir == "/models"
    assert ctx["metadata"]["geometry_contract"]["obstime"] == "2026-04-03T19:46:37.800"


def test_remove_refmaps_from_h5_missing_ok(tmp_path):
    model = tmp_path / "model.h5"
    _write_refmap_model(model)

    removed = remove_refmaps_from_h5(model, ["missing"], missing_ok=True)
    assert removed == []

    with pytest.raises(KeyError, match="refmap not found"):
        remove_refmaps_from_h5(model, ["missing"], missing_ok=False)


def test_remove_refmaps_from_h5_strict_is_atomic(tmp_path):
    model = tmp_path / "model.h5"
    _write_refmap_model(model)
    with h5py.File(model, "r+") as h5f:
        _write_stereo_refmap_group(h5f["refmaps"], "stereo304")

    # Existing id sorts before the missing one; a non-atomic loop would delete
    # stereo304 before raising.
    with pytest.raises(KeyError, match="refmap not found"):
        remove_refmaps_from_h5(model, ["stereo304", "missing"], missing_ok=False)

    assert list_embedded_refmap_ids(model) == ["Bz_reference", "stereo304"]


def test_refmap_source_cards_record_relative_root_and_absolute_fallback(tmp_path):
    from pyampp.io.refmaps import (
        REFMAP_SRC_ROOT_DATA_DIR,
        REFMAP_SRC_ROOT_MODEL_DIR,
        apply_refmap_source_cards,
        resolve_embedded_refmap_source,
    )

    model_dir = tmp_path / "model"
    data_dir = tmp_path / "jsoc"
    model_dir.mkdir()
    data_dir.mkdir()
    source = model_dir / "stereo_a_euvi" / "euvi.fts"
    source.parent.mkdir()
    source.write_bytes(b"fits")
    header = fits.Header()
    apply_refmap_source_cards(header, source, data_dir=data_dir, model_dir=model_dir)

    assert header["SRC_PATH"] == str(source.resolve())
    assert header["SRC_ROOT"] == REFMAP_SRC_ROOT_MODEL_DIR
    assert header["SRC_RELPATH"] == "stereo_a_euvi/euvi.fts"
    text = header.tostring(sep="\n", endcard=True)

    moved = tmp_path / "moved_model"
    moved_source = moved / "stereo_a_euvi" / "euvi.fts"
    moved_source.parent.mkdir(parents=True)
    moved_source.write_bytes(b"fits")
    assert resolve_embedded_refmap_source(text, model_dir=moved) == moved_source.resolve()

    cache = data_dir / "2012-07-12" / "aia.fits"
    cache.parent.mkdir()
    cache.write_bytes(b"fits")
    cache_header = fits.Header()
    apply_refmap_source_cards(cache_header, cache, data_dir=data_dir, model_dir=model_dir)
    assert cache_header["SRC_ROOT"] == REFMAP_SRC_ROOT_DATA_DIR
    assert cache_header["SRC_RELPATH"] == "2012-07-12/aia.fits"

    orphan = tmp_path / "elsewhere" / "euvi.fts"
    orphan.parent.mkdir()
    orphan.write_bytes(b"fits")
    orphan_header = fits.Header()
    apply_refmap_source_cards(orphan_header, orphan, data_dir=data_dir, model_dir=model_dir)
    assert "SRC_RELPATH" not in orphan_header
    assert resolve_embedded_refmap_source(
        orphan_header,
        data_dir=data_dir,
        model_dir=model_dir,
    ) == orphan.resolve()


def test_existing_file_under_rejects_path_escape(tmp_path):
    from pyampp.io.refmaps import (
        REFMAP_SRC_ROOT_DATA_DIR,
        _existing_file_under,
        resolve_embedded_refmap_source,
    )

    root = tmp_path / "data"
    root.mkdir()
    inside = root / "ok.fits"
    inside.write_bytes(b"fits")
    outside = tmp_path / "secret.fits"
    outside.write_bytes(b"fits")

    assert _existing_file_under(root, "ok.fits") == inside.resolve()
    assert _existing_file_under(root, "../secret.fits") is None
    assert _existing_file_under(root, str(outside)) is None

    header = fits.Header()
    header["SRC_ROOT"] = REFMAP_SRC_ROOT_DATA_DIR
    header["SRC_RELPATH"] = "../secret.fits"
    assert resolve_embedded_refmap_source(header, data_dir=root) is None
