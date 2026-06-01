import h5py
import numpy as np
from astropy.io import fits

from pyampp.io import (
    add_fits_refmaps_from_dir_to_h5,
    add_fits_refmaps_to_h5,
    build_fits_refmaps_for_model,
    discover_fits_refmap_map_ids,
    discover_fits_refmap_paths,
    model_obstime_from_base_index,
)
from pyampp.tests._fits_header import canonical_base_index_header


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


def _write_refmap_model(path, *, base_date_obs=None):
    header = _hpc_header(shape=(4, 4), crpix=(2.5, 2.5))
    with h5py.File(path, "w") as h5f:
        if base_date_obs is not None:
            base = h5f.create_group("base", track_order=True)
            base.create_dataset(
                "index",
                data=np.bytes_(canonical_base_index_header(date_obs=base_date_obs)),
            )
        refmaps = h5f.create_group("refmaps", track_order=True)
        group = refmaps.create_group("Bz_reference", track_order=True)
        group.attrs["order_index"] = np.int64(0)
        group.create_dataset("data", data=np.ones((4, 4), dtype=np.float32))
        group.create_dataset("wcs_header", data=np.bytes_(header.tostring(sep="\n", endcard=True)))


def _write_aia_fits(path, wavelength=171, date_obs="2026-04-03T20:00:00.000"):
    header = _hpc_header(shape=(8, 8), crpix=(4.5, 4.5), date_obs=date_obs)
    header["TELESCOP"] = "SDO/AIA"
    header["INSTRUME"] = "AIA_3"
    header["WAVELNTH"] = int(wavelength)
    header["WAVEUNIT"] = "angstrom"
    fits.PrimaryHDU(data=np.arange(64, dtype=np.float32).reshape(8, 8), header=header).writeto(path)


def _write_eovsa_fits(path, freq_hz):
    header = _hpc_header(shape=(8, 8), crpix=(4.5, 4.5))
    header["TELESCOP"] = "EOVSA"
    header["INSTRUME"] = "EOVSA"
    header["CRVAL3"] = float(freq_hz)
    header["CUNIT3"] = "Hz"
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32), header=header).writeto(path)


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
        assert group["data"].shape == (4, 4)
        header = fits.Header.fromstring(group["wcs_header"][()].decode(), sep="\n")
        assert header["TELESCOP"] == "SDO/AIA"
        assert header["WAVELNTH"] == 171
        assert header["WAVEUNIT"] == "angstrom"


def test_add_fits_refmaps_uses_base_index_time_for_model_alignment(tmp_path):
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
        assert header["DATE-OBS"] == model_time
        assert header["MODELT"] == model_time
        assert header["SRC_DATE"] == source_time
        assert header["PYALIGN"] is True


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
        target_fov=None,
    )

    assert [p.name for p in discovered] == ["eovsa_1.fits", "eovsa_2.fits"]
    assert set(payloads) == {"EOVSA_f1.418GHz", "EOVSA_f2.874GHz"}
    for payload in payloads.values():
        assert payload["data"].shape == (8, 8)


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
        target_fov=None,
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
