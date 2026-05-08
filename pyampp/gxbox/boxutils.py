from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from sunpy.map import Map, make_fitswcs_header

from pyampp.util.config import IDL_HMI_RSUN_M


def _decode_sav_text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _sav_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        scalar = arr.item()
    elif arr.size == 1:
        scalar = arr.flat[0]
    else:
        return None
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    if isinstance(scalar, (bytes, np.bytes_)):
        scalar = scalar.decode("utf-8", errors="ignore")
    return scalar


def serialize_sav_index_header(index: Any) -> str:
    """
    Convert an IDL FITSHEAD2STRUCT-like INDEX record into FITS header text.

    SAV entry boxes can carry INDEX as a restored IDL struct instead of a plain
    header string. Stringifying that struct produces an unusable Python tuple
    repr. This serializer reconstructs a standard FITS-style header payload so
    HDF5 outputs remain consumable by downstream FITS/WCS readers.
    """
    if index is None:
        return ""

    if isinstance(index, np.ndarray) and index.dtype.names is not None:
        if index.shape == ():
            index = index.item()
        elif index.size == 1:
            index = index.flat[0]
    elif isinstance(index, np.void):
        index = index

    dtype_names = getattr(getattr(index, "dtype", None), "names", None)
    if not dtype_names:
        return _decode_sav_text(index)

    header = fits.Header()
    date_obs_value = None
    for name in dtype_names:
        raw = index[name]
        keyword = name.replace("$", "-")
        if name == "DATE_D$OBS":
            # Keep the normalized DATE-OBS aliases rather than inventing a
            # non-standard long keyword that downstream code does not need.
            scalar = _sav_scalar(raw)
            if scalar not in (None, ""):
                date_obs_value = _decode_sav_text(scalar)
            continue
        if name in ("COMMENT", "HISTORY"):
            for item in np.asarray(raw, dtype=object).reshape(-1):
                text = _decode_sav_text(_sav_scalar(item) if np.asarray(item).shape == () else item).strip()
                if not text:
                    continue
                if name == "COMMENT":
                    header.add_comment(text)
                else:
                    header.add_history(text)
            continue
        scalar = _sav_scalar(raw)
        if scalar is None:
            continue
        if keyword == "SIMPLE":
            scalar = bool(int(scalar))
        header[keyword] = scalar
        if keyword in ("DATE-OBS", "DATE_OBS") and scalar not in ("", None):
            date_obs_value = _decode_sav_text(scalar)

    if date_obs_value:
        header["DATE-OBS"] = date_obs_value
        header["DATE_OBS"] = date_obs_value
        header["DATE"] = date_obs_value

    return header.tostring(sep="\n", endcard=True)

def _bilinear_sample(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, arr.shape[0] - 1)
    x1 = np.clip(x1, 0, arr.shape[0] - 1)
    y0 = np.clip(y0, 0, arr.shape[1] - 1)
    y1 = np.clip(y1, 0, arr.shape[1] - 1)

    wx = x - x0
    wy = y - y0

    f00 = arr[x0, y0]
    f10 = arr[x1, y0]
    f01 = arr[x0, y1]
    f11 = arr[x1, y1]

    return (f00 * (1 - wx) * (1 - wy) +
            f10 * wx * (1 - wy) +
            f01 * (1 - wx) * wy +
            f11 * wx * wy)


def _vxv(a: np.ndarray, b: np.ndarray, norm: bool = False) -> np.ndarray:
    c = np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ], dtype=float)
    if norm:
        n = np.sqrt((c * c).sum())
        if n > 0:
            c /= n
    return c


def compute_vertical_current(b0: np.ndarray,
                             b1: np.ndarray,
                             b2: np.ndarray,
                             wcs_header: str,
                             rsun_arcsec: float,
                             crpix1: float | None = None,
                             crpix2: float | None = None,
                             cdelt1_arcsec: float | None = None,
                             cdelt2_arcsec: float | None = None) -> np.ndarray:
    if crpix1 is None or crpix2 is None or cdelt1_arcsec is None or cdelt2_arcsec is None:
        header = fits.Header.fromstring(wcs_header, sep="\n")
        cdelt1 = header.get("CDELT1")
        cdelt2 = header.get("CDELT2")
        cunit1 = header.get("CUNIT1", "deg")
        cunit2 = header.get("CUNIT2", "deg")
        crpix1 = header.get("CRPIX1")
        crpix2 = header.get("CRPIX2")

        if cdelt1 is None or cdelt2 is None or crpix1 is None or crpix2 is None:
            raise ValueError("WCS header missing CDELT/CRPIX values.")

        cdelt1_arcsec = (cdelt1 * u.Unit(cunit1)).to_value(u.arcsec)
        cdelt2_arcsec = (cdelt2 * u.Unit(cunit2)).to_value(u.arcsec)

    jr = np.zeros_like(b0, dtype=float)
    nx, ny = b0.shape
    if nx < 3 or ny < 3:
        return jr

    delta = 0.5 * cdelt1_arcsec / rsun_arcsec
    x_idx = np.arange(1, nx - 1, dtype=float)[:, None]
    y_idx = np.arange(1, ny - 1, dtype=float)[None, :]

    x1 = (x_idx - float(crpix1)) * float(cdelt1_arcsec) / float(rsun_arcsec)
    x2 = (y_idx - float(crpix2)) * float(cdelt2_arcsec) / float(rsun_arcsec)
    r2 = x1 * x1 + x2 * x2
    mask = r2 < 0.95 ** 2
    if not np.any(mask):
        return jr

    x0 = np.sqrt(np.clip(1.0 - r2, 0.0, None))

    # Vectorized form of:
    #   e1 = cross([0,0,1], e0, norm=True)
    #   e2 = cross(e0, e1)
    e1_norm = np.sqrt(x0 * x0 + x1 * x1)
    e1_norm = np.where(e1_norm > 0.0, e1_norm, 1.0)
    e1_0 = -x1 / e1_norm
    e1_1 = x0 / e1_norm
    e1_2 = np.zeros_like(e1_0)

    e2_0 = -x2 * x0 / e1_norm
    e2_1 = -x2 * x1 / e1_norm
    e2_2 = (x0 * x0 + x1 * x1) / e1_norm

    x1s = np.stack(
        (
            -e1_1 * delta + x1,
            e1_1 * delta + x1,
            -e2_1 * delta + x1,
            e2_1 * delta + x1,
        ),
        axis=0,
    )
    x2s = np.stack(
        (
            -e1_2 * delta + x2,
            e1_2 * delta + x2,
            -e2_2 * delta + x2,
            e2_2 * delta + x2,
        ),
        axis=0,
    )

    xx1 = x1s * float(rsun_arcsec) / float(cdelt1_arcsec) + float(crpix1)
    xx2 = x2s * float(rsun_arcsec) / float(cdelt2_arcsec) + float(crpix2)

    b0_int = _bilinear_sample(b0, xx1, xx2) * 1e-4
    b1_int = _bilinear_sample(b1, xx1, xx2) * 1e-4
    b2_int = _bilinear_sample(b2, xx1, xx2) * 1e-4

    term_e2 = (
        (b0_int[1] * e2_0 + b1_int[1] * e2_1 + b2_int[1] * e2_2)
        - (b0_int[0] * e2_0 + b1_int[0] * e2_1 + b2_int[0] * e2_2)
    )
    term_e1 = (
        (b0_int[3] * e1_0 + b1_int[3] * e1_1 + b2_int[3] * e1_2)
        - (b0_int[2] * e1_0 + b1_int[2] * e1_1 + b2_int[2] * e1_2)
    )

    interior = (term_e2 - term_e1) / (2 * (delta * 696e6) * (4 * np.pi * 1e-7))
    interior = np.where(mask, interior, 0.0)
    jr[1:nx - 1, 1:ny - 1] = interior
    return jr


def _decode_sav_refmap_value(v: Any) -> Any:
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", "ignore")
    return v


def _sav_refmap_scalar(v: Any) -> Any:
    arr = np.asarray(v)
    if arr.shape == ():
        return arr.item()
    return arr.flat[0]


def _sav_refmap_text(v: Any) -> str:
    v = _decode_sav_refmap_value(v)
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return str(v.item())
        if v.size > 0:
            return str(_decode_sav_refmap_value(v.flat[0]))
        return ""
    return str(v)


def _sav_box_has_field(box: Any, name: str) -> bool:
    return name in (getattr(getattr(box, "dtype", None), "names", None) or ())


def _build_sav_refmap_wcs_header(
    data2d: np.ndarray,
    *,
    xc: float,
    yc: float,
    dx: float,
    dy: float,
    date_obs: str,
    xunits: str,
    yunits: str,
    rsun_obs: float | None = None,
    b0: float | None = None,
    l0: float | None = None,
) -> str:
    ny, nx = data2d.shape
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = int(nx)
    header["NAXIS2"] = int(ny)
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = xunits if xunits else "arcsec"
    header["CUNIT2"] = yunits if yunits else "arcsec"
    header["CRPIX1"] = (nx + 1.0) / 2.0
    header["CRPIX2"] = (ny + 1.0) / 2.0
    header["CRVAL1"] = float(xc)
    header["CRVAL2"] = float(yc)
    header["CDELT1"] = float(dx)
    header["CDELT2"] = float(dy)
    if date_obs:
        header["DATE-OBS"] = date_obs
        header["DATE_OBS"] = date_obs
    if rsun_obs is not None:
        header["RSUN_OBS"] = float(rsun_obs)
    if b0 is not None:
        header["HGLT_OBS"] = float(b0)
    if l0 is not None:
        header["HGLN_OBS"] = float(l0)
    return header.tostring(sep="\n", endcard=True)


def extract_sav_refmaps(box: Any) -> list[tuple[int, str, np.ndarray, str]]:
    """
    Extract GX/IDL BOX.REFMAPS into ordered HDF5-style payloads.
    """
    out: list[tuple[int, str, np.ndarray, str]] = []
    if not _sav_box_has_field(box, "REFMAPS"):
        return out

    try:
        refmaps_obj = box["REFMAPS"][0]
        omap = refmaps_obj["OMAP"][0]
        pointer = omap["POINTER"][0]
        ids = np.asarray(pointer["IDS"], dtype=object).ravel()
        ptrs = np.asarray(pointer["PTRS"], dtype=object).ravel()
    except Exception:
        return out

    used_names: dict[str, int] = {}
    order_index = 0
    for slot_id, entry in zip(ids, ptrs):
        if entry is None:
            continue
        slot_text = _sav_refmap_text(slot_id).strip()
        if not slot_text:
            continue
        try:
            rec = np.asarray(entry).reshape(-1)[0]
            names = rec.dtype.names or ()
        except Exception:
            continue
        if "DATA" not in names:
            continue

        data = np.asarray(rec["DATA"], dtype=np.float32)
        if data.shape and data.shape[0] == 1:
            data = np.asarray(data[0], dtype=np.float32)
        if data.ndim != 2:
            continue

        map_id = _sav_refmap_text(rec["ID"]) if "ID" in names else slot_text
        map_id = map_id.strip() if map_id else slot_text
        if not map_id:
            map_id = f"refmap_{slot_text}"

        base_name = map_id
        idx = used_names.get(base_name, 0)
        used_names[base_name] = idx + 1
        if idx > 0:
            map_id = f"{base_name}_{idx}"

        xc = float(_sav_refmap_scalar(rec["XC"])) if "XC" in names else 0.0
        yc = float(_sav_refmap_scalar(rec["YC"])) if "YC" in names else 0.0
        dx = float(_sav_refmap_scalar(rec["DX"])) if "DX" in names else 1.0
        dy = float(_sav_refmap_scalar(rec["DY"])) if "DY" in names else 1.0
        date_obs = _sav_refmap_text(rec["TIME"]) if "TIME" in names else ""
        xunits = _sav_refmap_text(rec["XUNITS"]).strip() if "XUNITS" in names else "arcsec"
        yunits = _sav_refmap_text(rec["YUNITS"]).strip() if "YUNITS" in names else "arcsec"
        rsun_obs = float(_sav_refmap_scalar(rec["RSUN"])) if "RSUN" in names else None
        b0 = float(_sav_refmap_scalar(rec["B0"])) if "B0" in names else None
        l0 = float(_sav_refmap_scalar(rec["L0"])) if "L0" in names else None

        wcs_header = _build_sav_refmap_wcs_header(
            data,
            xc=xc,
            yc=yc,
            dx=dx,
            dy=dy,
            date_obs=date_obs,
            xunits=xunits,
            yunits=yunits,
            rsun_obs=rsun_obs,
            b0=b0,
            l0=l0,
        )
        out.append((order_index, map_id, data, wcs_header))
        order_index += 1
    return out


def remap_vertical_current_inputs(
    map_bx: Map,
    map_by: Map,
    map_bz: Map,
    *,
    pad_factor: float = 1.1,
    algorithm: str = "exact",
) -> tuple[Map, Map, Map]:
    """
    Remap Bx/By/Bz onto an IDL-like reference WCS before current-density evaluation.

    The legacy IDL path first remaps the vector-field cutout onto a dedicated
    reference WCS that is slightly larger than the selected box footprint, then
    runs the current-density kernel on that remapped patch.  Using the same
    preconditioning here is safer than changing the kernel math itself.
    """
    data = np.asarray(map_bx.data)
    if data.ndim != 2:
        raise ValueError("Vertical-current inputs must be 2D maps.")

    ny, nx = data.shape
    nx_ref = max(2, int(round(nx * float(pad_factor))))
    ny_ref = max(2, int(round(ny * float(pad_factor))))
    cx = 0.5 * max(0, nx - 1)
    cy = 0.5 * max(0, ny - 1)
    center = map_bx.wcs.pixel_to_world(cx, cy)
    scale = u.Quantity(
        [
            abs(map_bx.scale.axis1.to_value(u.arcsec / u.pix)),
            abs(map_bx.scale.axis2.to_value(u.arcsec / u.pix)),
        ],
        u.arcsec / u.pix,
    )
    ref_header = make_fitswcs_header(
        (ny_ref, nx_ref),
        center,
        scale=scale,
        projection_code="TAN",
    )
    try:
        ref_header["rsun_ref"] = float(map_bx.rsun_meters.to_value(u.m))
    except Exception:
        pass

    ref_bx = map_bx.reproject_to(ref_header, algorithm=algorithm)
    ref_by = map_by.reproject_to(ref_header, algorithm=algorithm)
    ref_bz = map_bz.reproject_to(ref_header, algorithm=algorithm)
    return ref_bx, ref_by, ref_bz


def _sanitize_unit_like_header_values(header: fits.Header) -> fits.Header:
    """
    Normalize non-standard escaped unit strings found in some IDL-written FITS.
    Example: 'DN\\/s' -> 'DN/s'.
    """
    for key in ("BUNIT", "CUNIT1", "CUNIT2"):
        if key in header:
            value = header.get(key)
            if isinstance(value, str):
                fixed = value.replace("\\/", "/").replace("\\", "")
                if fixed != value:
                    header[key] = fixed
    return header


def _normalize_gx_header_values(header: fits.Header) -> fits.Header:
    """
    Normalize imported FITS/WCS metadata to the GX/HMI solar-radius convention.

    SunPy otherwise falls back to its own photospheric radius when RSUN_REF is
    missing, which introduces a systematic mismatch with the GX/IDL path.
    """
    header = _sanitize_unit_like_header_values(header)
    header["RSUN_REF"] = float(IDL_HMI_RSUN_M)
    return header


def _header_from_any(header_like) -> fits.Header:
    if isinstance(header_like, fits.Header):
        header = header_like.copy()
    else:
        header = fits.Header()
        for key, value in dict(header_like).items():
            if value is None or str(key).lower() == "keycomments":
                continue
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8", "ignore")
            try:
                header[str(key)] = value
            except Exception:
                continue
    return _normalize_gx_header_values(header)


def map_from_data_header_compat(data, header) -> Map:
    """
    Build a SunPy map while enforcing unit sanitation and GX-compatible RSUN_REF.
    """
    return Map(data, _header_from_any(header))


def _normalize_loaded_map_rsun(smap: Map) -> Map:
    try:
        rsun_m = u.Quantity(smap.rsun_meters).to_value(u.m)
    except Exception:
        rsun_m = None
    if rsun_m is not None and np.isclose(rsun_m, IDL_HMI_RSUN_M, rtol=0.0, atol=1e-6):
        return smap
    return map_from_data_header_compat(np.asarray(smap.data), smap.meta)


def _first_image_hdu(hdulist):
    for hdu in hdulist:
        if getattr(hdu, "data", None) is not None:
            return hdu
    raise ValueError("No image HDU with data found in FITS file.")


def load_sunpy_map_compat(path_or_data, header=None):
    """
    Load a SunPy map while tolerating IDL-style escaped FITS unit strings.

    If a direct `Map(path)` fails due to unit parsing, this falls back to opening
    the FITS file, sanitizing unit-like header fields, and building the map from
    `(data, header)`.
    """
    if header is not None:
        return map_from_data_header_compat(path_or_data, header)

    try:
        return _normalize_loaded_map_rsun(Map(path_or_data))
    except Exception as exc:
        msg = str(exc)
        if ("did not parse as unit" not in msg) and ("not a valid unit" not in msg):
            raise

        with fits.open(path_or_data) as hdul:
            image_hdu = _first_image_hdu(hdul)
            data = image_hdu.data
            safe_header = image_hdu.header.copy()
        return map_from_data_header_compat(data, safe_header)


def observer_ephemeris_from_map(source_map) -> tuple[dict[str, float | str], tuple[str, ...]]:
    ephemeris: dict[str, float | str] = {}
    missing: list[str] = []
    if source_map is None:
        return {}, ("DATE-OBS", "HGLN_OBS", "HGLT_OBS", "DSUN_OBS", "RSUN_REF")
    obs_time = getattr(source_map, "date", None)
    if obs_time is not None:
        try:
            ephemeris["obs_date"] = Time(obs_time).isot
        except Exception:
            pass
    observer = getattr(source_map, "observer_coordinate", None)
    if observer is not None:
        try:
            obs_hgs = observer.transform_to(HeliographicStonyhurst(obstime=obs_time))
            ephemeris["hgln_obs_deg"] = float(obs_hgs.lon.to_value(u.deg))
            ephemeris["hglt_obs_deg"] = float(obs_hgs.lat.to_value(u.deg))
            ephemeris["dsun_cm"] = float(obs_hgs.radius.to_value(u.cm))
        except Exception:
            pass
    meta = getattr(source_map, "meta", None)
    if meta is not None:
        try:
            if "date-obs" in meta and "obs_date" not in ephemeris:
                ephemeris["obs_date"] = Time(meta["date-obs"]).isot
            elif "date_obs" in meta and "obs_date" not in ephemeris:
                ephemeris["obs_date"] = Time(meta["date_obs"]).isot
        except Exception:
            pass
        for src_key, dst_key in (
            ("hgln_obs", "hgln_obs_deg"),
            ("hglt_obs", "hglt_obs_deg"),
        ):
            value = meta.get(src_key)
            if value is not None and dst_key not in ephemeris:
                try:
                    ephemeris[dst_key] = float(value)
                except Exception:
                    pass
        if meta.get("dsun_obs") is not None and "dsun_cm" not in ephemeris:
            try:
                ephemeris["dsun_cm"] = float(u.Quantity(meta["dsun_obs"], u.m).to_value(u.cm))
            except Exception:
                pass
        if meta.get("rsun_ref") is not None and "rsun_cm" not in ephemeris:
            try:
                ephemeris["rsun_cm"] = float(u.Quantity(meta["rsun_ref"], u.m).to_value(u.cm))
            except Exception:
                pass
    if getattr(source_map, "rsun_meters", None) is not None and "rsun_cm" not in ephemeris:
        try:
            ephemeris["rsun_cm"] = float(u.Quantity(source_map.rsun_meters).to_value(u.cm))
        except Exception:
            pass
    for required_key, card_name in (
        ("obs_date", "DATE-OBS"),
        ("hgln_obs_deg", "HGLN_OBS"),
        ("hglt_obs_deg", "HGLT_OBS"),
        ("dsun_cm", "DSUN_OBS"),
        ("rsun_cm", "RSUN_REF"),
    ):
        if required_key not in ephemeris:
            missing.append(card_name)
    return ephemeris, tuple(missing)


def observer_ephemeris_from_fits_file(path: str | bytes | PathLike[str]) -> tuple[dict[str, float | str], tuple[str, ...]]:
    source_map = load_sunpy_map_compat(path)
    return observer_ephemeris_from_map(source_map)


def _observer_ephemeris_missing(ephemeris: dict[str, float | str]) -> tuple[str, ...]:
    missing: list[str] = []
    for required_key, card_name in (
        ("obs_date", "DATE-OBS"),
        ("hgln_obs_deg", "HGLN_OBS"),
        ("hglt_obs_deg", "HGLT_OBS"),
        ("dsun_cm", "DSUN_OBS"),
        ("rsun_cm", "RSUN_REF"),
    ):
        if required_key not in ephemeris:
            missing.append(card_name)
    return tuple(missing)


def _coerce_nested_scalar(value):
    current = value
    while isinstance(current, np.ndarray):
        if current.shape == ():
            current = current.item()
            continue
        if current.size != 1:
            return None
        current = current.reshape(-1)[0]
    if isinstance(current, np.generic):
        return current.item()
    return current


def _node_field_items(node):
    if isinstance(node, dict):
        return list(node.items())
    names = getattr(getattr(node, "dtype", None), "names", None)
    if not names:
        return []
    items = []
    for name in names:
        try:
            items.append((name, node[name]))
        except Exception:
            continue
    return items


def _observer_pb0r_from_header(header) -> dict[str, float | str]:
    if header is None:
        return {}
    pb0r: dict[str, float | str] = {}
    for source_key, target_key in (
        ("DATE-OBS", "obs_date"),
        ("DATE_OBS", "obs_date"),
        ("HGLT_OBS", "b0_deg"),
        ("HGLN_OBS", "l0_deg"),
        ("RSUN_OBS", "rsun_arcsec"),
        ("CROTA2", "p_deg"),
        ("SOLAR_P", "p_deg"),
    ):
        value = header.get(source_key)
        if value is None:
            continue
        try:
            if target_key == "obs_date":
                pb0r[target_key] = Time(value).isot
            else:
                pb0r[target_key] = float(value)
        except Exception:
            continue
    return pb0r


def _merge_observer_fields(
    fields,
    ephemeris: dict[str, float | str],
    pb0r: dict[str, float | str],
) -> None:
    values = {}
    for name, raw_value in fields:
        key = str(name).upper()
        value = _coerce_nested_scalar(raw_value)
        if value is None:
            continue
        values[key] = value

    for source_key in ("DATE-OBS", "DATE_OBS", "OBS_DATE"):
        if source_key in values and "obs_date" not in ephemeris:
            try:
                ephemeris["obs_date"] = Time(values[source_key]).isot
            except Exception:
                pass
        if source_key in values and "obs_date" not in pb0r:
            try:
                pb0r["obs_date"] = Time(values[source_key]).isot
            except Exception:
                pass

    for source_key, target_key in (
        ("HGLN_OBS", "hgln_obs_deg"),
        ("HGLT_OBS", "hglt_obs_deg"),
        ("CRLN_OBS", "crln_obs_deg"),
        ("CRLT_OBS", "crlt_obs_deg"),
    ):
        if source_key in values and target_key not in ephemeris:
            try:
                ephemeris[target_key] = float(values[source_key])
            except Exception:
                pass

    for source_key, target_key in (
        ("DSUN_OBS", "dsun_cm"),
        ("RSUN_REF", "rsun_cm"),
    ):
        if source_key in values and target_key not in ephemeris:
            try:
                ephemeris[target_key] = float(u.Quantity(values[source_key], u.m).to_value(u.cm))
            except Exception:
                pass

    for source_key, target_key in (
        ("B0_DEG", "b0_deg"),
        ("L0_DEG", "l0_deg"),
        ("RSUN_ARCSEC", "rsun_arcsec"),
        ("P_DEG", "p_deg"),
    ):
        if source_key in values and target_key not in pb0r:
            try:
                pb0r[target_key] = float(values[source_key])
            except Exception:
                pass

    for header_key in ("WCS_HEADER", "HEADER", "INDEX"):
        if header_key not in values:
            continue
        header = _header_from_text(values[header_key])
        header_ephemeris = _observer_ephemeris_from_header(header)
        for key, value in header_ephemeris.items():
            ephemeris.setdefault(key, value)
        header_pb0r = _observer_pb0r_from_header(header)
        for key, value in header_pb0r.items():
            pb0r.setdefault(key, value)


def _walk_nested_nodes(root):
    seen: set[int] = set()

    def visit(node):
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        yield node

        if isinstance(node, dict):
            for value in node.values():
                yield from visit(value)
            return

        if isinstance(node, np.ndarray):
            if node.dtype == object:
                for value in node.flat[:16]:
                    yield from visit(value)
                return
            if node.dtype.names:
                for value in node.flat[:8]:
                    yield from visit(value)
            return

        names = getattr(getattr(node, "dtype", None), "names", None)
        if names:
            for name in names:
                try:
                    yield from visit(node[name])
                except Exception:
                    continue

    yield from visit(root)


def _fill_hgs_from_carrington(ephemeris: dict[str, float | str]) -> None:
    if "hgln_obs_deg" in ephemeris and "hglt_obs_deg" in ephemeris:
        return
    if "crln_obs_deg" not in ephemeris or "crlt_obs_deg" not in ephemeris:
        return
    if "dsun_cm" not in ephemeris or "obs_date" not in ephemeris:
        return
    try:
        obstime = Time(ephemeris["obs_date"])
        obs_hgc = SkyCoord(
            lon=float(ephemeris["crln_obs_deg"]) * u.deg,
            lat=float(ephemeris["crlt_obs_deg"]) * u.deg,
            radius=float(ephemeris["dsun_cm"]) * u.cm,
            frame=HeliographicCarrington(observer="self", obstime=obstime),
        )
        obs_hgs = obs_hgc.transform_to(HeliographicStonyhurst(obstime=obstime))
        ephemeris.setdefault("hgln_obs_deg", float(obs_hgs.lon.to_value(u.deg)))
        ephemeris.setdefault("hglt_obs_deg", float(obs_hgs.lat.to_value(u.deg)))
    except Exception:
        return


def _observer_ephemeris_from_pb0r_dict(
    pb0r: dict[str, float | str],
    *,
    rsun_cm: float | str | None = None,
) -> dict[str, float | str]:
    from .observer_restore import build_ephemeris_from_pb0r

    ephemeris = build_ephemeris_from_pb0r(
        b0_deg=pb0r.get("b0_deg"),
        l0_deg=pb0r.get("l0_deg"),
        rsun_arcsec=pb0r.get("rsun_arcsec"),
        obs_date=pb0r.get("obs_date"),
        rsun_cm=rsun_cm,
    )
    if not isinstance(ephemeris, dict):
        return {}
    return {str(key): value for key, value in ephemeris.items()}


def observer_ephemeris_from_sav_file(path: str | bytes | PathLike[str]) -> tuple[dict[str, float | str], tuple[str, ...]]:
    path_obj = Path(path)
    ephemeris: dict[str, float | str] = {}
    pb0r: dict[str, float | str] = {}

    try:
        from .gx_fov2box import _load_entry_box_any

        loaded = _load_entry_box_any(path_obj)
    except Exception:
        loaded = None

    if isinstance(loaded, dict):
        observer = loaded.get("observer")
        if isinstance(observer, dict):
            raw_ephemeris = observer.get("ephemeris")
            if isinstance(raw_ephemeris, dict):
                ephemeris.update(
                    {
                        str(key): value
                        for key, value in raw_ephemeris.items()
                        if key in ("obs_date", "hgln_obs_deg", "hglt_obs_deg", "dsun_cm", "rsun_cm")
                    }
                )
            raw_pb0r = observer.get("pb0r")
            if isinstance(raw_pb0r, dict):
                pb0r.update(
                    {
                        str(key): value
                        for key, value in raw_pb0r.items()
                        if key in ("obs_date", "b0_deg", "l0_deg", "p_deg", "rsun_arcsec")
                    }
                )
        header = _model_observer_header(loaded)
        header_ephemeris = _observer_ephemeris_from_header(header)
        for key, value in header_ephemeris.items():
            ephemeris.setdefault(key, value)
        header_pb0r = _observer_pb0r_from_header(header)
        for key, value in header_pb0r.items():
            pb0r.setdefault(key, value)

    if _observer_ephemeris_missing(ephemeris):
        from scipy.io import readsav

        data = readsav(str(path_obj), python_dict=True, verbose=False)
        for node in _walk_nested_nodes(data):
            fields = _node_field_items(node)
            if not fields:
                continue
            _merge_observer_fields(fields, ephemeris, pb0r)
            if not _observer_ephemeris_missing(ephemeris):
                break

    _fill_hgs_from_carrington(ephemeris)

    if _observer_ephemeris_missing(ephemeris) and pb0r:
        derived = _observer_ephemeris_from_pb0r_dict(pb0r, rsun_cm=ephemeris.get("rsun_cm"))
        for key, value in derived.items():
            ephemeris.setdefault(key, value)

    ephemeris = {
        key: ephemeris[key]
        for key in ("obs_date", "hgln_obs_deg", "hglt_obs_deg", "dsun_cm", "rsun_cm")
        if key in ephemeris
    }
    return ephemeris, _observer_ephemeris_missing(ephemeris)


def observer_ephemeris_from_reference_file(path: str | bytes | PathLike[str]) -> tuple[dict[str, float | str], tuple[str, ...]]:
    path_obj = Path(path)
    lower_name = path_obj.name.lower()
    if lower_name.endswith((".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")):
        return observer_ephemeris_from_fits_file(path_obj)
    if lower_name.endswith(".sav"):
        return observer_ephemeris_from_sav_file(path_obj)
    raise ValueError(f"Unsupported reference-file type: {path_obj.suffix or path_obj.name}")


def _compact_label_parts(*parts) -> str:
    values: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if not text or text.lower() in {value.lower() for value in values}:
            continue
        values.append(text)
    return " / ".join(values)


def observer_reference_details_from_file(path: str | bytes | PathLike[str]) -> dict[str, str]:
    path_obj = Path(path)
    details = {
        "label": path_obj.stem,
        "source": path_obj.name,
    }
    lower_name = path_obj.name.lower()
    if lower_name.endswith((".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")):
        try:
            smap = load_sunpy_map_compat(path_obj)
        except Exception:
            return details
        meta = getattr(smap, "meta", {}) or {}
        observatory = getattr(smap, "observatory", None) or meta.get("telescop")
        detector = getattr(smap, "detector", None) or meta.get("instrume")
        nickname = getattr(smap, "nickname", None)
        content = meta.get("content") or meta.get("id")
        label = (
            _compact_label_parts(observatory, detector)
            or str(nickname or "").strip()
            or str(content or "").strip()
            or path_obj.stem
        )
        details["label"] = label
        return details
    if lower_name.endswith(".sav"):
        try:
            from .gx_fov2box import _load_entry_box_any

            loaded = _load_entry_box_any(path_obj)
        except Exception:
            return details
        if not isinstance(loaded, dict):
            return details
        observer = loaded.get("observer")
        metadata = loaded.get("metadata")
        if isinstance(observer, dict):
            label_value = observer.get("label")
            if label_value is None:
                label_value = observer.get("name")
            label = decode_meta_text(label_value).strip()
            if label:
                details["label"] = label
        if isinstance(metadata, dict):
            model_id = decode_meta_text(metadata.get("id") or "").strip()
            if model_id:
                details["label"] = model_id
        return details
    return details


from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicCarrington, HeliographicStonyhurst
from sunpy.map import all_coordinates_from_map
from PyQt5.QtWidgets import  QMessageBox

import numpy as np
from astropy.io import fits
import h5py


def _decode_meta_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, (bytes, bytearray)):
            return item.decode("utf-8", "ignore")
        return str(item)
    return str(value)


def _header_from_text(header_text):
    text = _decode_meta_text(header_text)
    if not text:
        return None
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    try:
        return fits.Header.fromstring(text, sep="\n")
    except Exception:
        return None


def _observer_ephemeris_from_header(header):
    if header is None:
        return {}
    ephemeris = {}
    try:
        if header.get("DATE-OBS") is not None:
            ephemeris["obs_date"] = Time(header["DATE-OBS"]).isot
    except Exception:
        pass
    try:
        if header.get("HGLN_OBS") is not None:
            ephemeris["hgln_obs_deg"] = float(header["HGLN_OBS"])
        if header.get("HGLT_OBS") is not None:
            ephemeris["hglt_obs_deg"] = float(header["HGLT_OBS"])
        if header.get("DSUN_OBS") is not None:
            ephemeris["dsun_cm"] = float(u.Quantity(header["DSUN_OBS"], u.m).to_value(u.cm))
        if header.get("RSUN_REF") is not None:
            ephemeris["rsun_cm"] = float(u.Quantity(header["RSUN_REF"], u.m).to_value(u.cm))
    except Exception:
        pass
    return ephemeris


def _model_observer_header(box_b3d: dict):
    refmaps = box_b3d.get("refmaps", {}) if isinstance(box_b3d, dict) else {}
    if isinstance(refmaps, dict):
        for key in ("Bz_reference", "Ic_reference"):
            payload = refmaps.get(key)
            if isinstance(payload, dict) and payload.get("wcs_header") is not None:
                header = _header_from_text(payload.get("wcs_header"))
                if header is not None:
                    return header
    return None


def normalize_observer_metadata(box_b3d: dict) -> dict:
    if not isinstance(box_b3d, dict):
        return box_b3d
    observer = box_b3d.get("observer")
    observer = dict(observer) if isinstance(observer, dict) else {}
    ephemeris = dict(observer.get("ephemeris", {})) if isinstance(observer.get("ephemeris"), dict) else {}
    required = {"obs_date", "hgln_obs_deg", "hglt_obs_deg", "dsun_cm", "rsun_cm"}
    incomplete = not required.issubset(ephemeris.keys())
    if incomplete:
        header = _model_observer_header(box_b3d)
        ephemeris.update({k: v for k, v in _observer_ephemeris_from_header(header).items() if k not in ephemeris})
    if observer.get("name") is None:
        observer["name"] = "earth"
    if observer.get("label") is None:
        observer["label"] = {
            "earth": "Earth",
            "sdo": "SDO",
            "solar orbiter": "Solar Orbiter",
            "stereo-a": "STEREO-A",
            "stereo-b": "STEREO-B",
            "custom": "Custom",
        }.get(str(observer.get("name", "earth")).strip().lower(), "Earth")
    if isinstance(observer.get("fov_box"), dict) and observer["fov_box"].get("observer_key") is None and observer.get("name"):
        observer["fov_box"] = dict(observer["fov_box"])
        observer["fov_box"]["observer_key"] = str(observer["name"])
    if ephemeris:
        observer["ephemeris"] = {
            key: ephemeris[key]
            for key in ("obs_date", "hgln_obs_deg", "hglt_obs_deg", "dsun_cm", "rsun_cm")
            if key in ephemeris
        }
    if observer:
        box_b3d["observer"] = observer
    return box_b3d

def hmi_disambig(azimuth_map, disambig_map, method=2):
    """
    Combine HMI disambiguation result with azimuth.

    :param azimuth_map: sunpy.map.Map, The azimuth map.
    :type azimuth_map: sunpy.map.Map
    :param disambig_map: sunpy.map.Map, The disambiguation map.
    :type disambig_map: sunpy.map.Map
    :param method: int, Method index (0: potential acute, 1: random, 2: radial acute). Default is 2.
    :type method: int, optional

    :return: map_azimuth: sunpy.map.Map, The azimuth map with disambiguation applied.
    :rtype: sunpy.map.Map
    """
    # Load data from FITS files
    azimuth = azimuth_map.data
    disambig = disambig_map.data

    # Check dimensions of the arrays
    if azimuth.shape != disambig.shape:
        raise ValueError("Dimension of two images do not agree")

    # Fix disambig to ensure it is integer
    disambig = disambig.astype(int)

    # Validate method index
    if method < 0 or method > 2:
        method = 2
        print("Invalid disambiguation method, set to default method = 2")

    # Apply disambiguation method
    disambig_corr = disambig // (2 ** method)  # Move the corresponding bit to the lowest
    index = disambig_corr % 2 != 0  # True where bits indicate flip

    # Update azimuth where flipping is needed
    azimuth[index] += 180
    azimuth = azimuth % 360  # Ensure azimuth stays within [0, 360]

    map_azimuth = map_from_data_header_compat(azimuth, azimuth_map.meta)
    return map_azimuth


def hmi_b2ptr(map_field, map_inclination, map_azimuth):
    '''
    Converts the magnetic field components from SDO/HMI
    data from field strength, inclination, and azimuth into components of the
    magnetic field in the local heliographic coordinate system (B_phi, B_theta, B_r).

    This function transforms the magnetic field vector in the local (xi, eta, zeta)
    system to the heliographic (phi, theta, r) system. The transformation accounts for
    the observer's position and orientation relative to the Sun.

    :param map_field: `sunpy.map.Map`,
        The magnetic field strength map from HMI, given in Gauss.

    :param map_inclination: `sunpy.map.Map`,
        The magnetic field inclination angle map from HMI, in degrees, where 0 degrees is
        parallel to the radial direction.

    :param map_azimuth: `sunpy.map.Map`,
        The magnetic field azimuth angle map from HMI, in degrees, measured counterclockwise
        from the north in the plane perpendicular to the radial direction.

    :return: tuple,
        A tuple containing the three magnetic field component maps in the heliographic
        coordinate system:

        - `map_bp` (`sunpy.map.Map`): The magnetic field component in the phi direction (B_phi).
        - `map_bt` (`sunpy.map.Map`): The magnetic field component in the theta direction (B_theta).
        - `map_br` (`sunpy.map.Map`): The magnetic field component in the radial direction (B_r).

    :example:

    .. code-block:: python

        # Load the HMI field, inclination, and azimuth maps
        map_field = sunpy.map.Map('hmi_field.fits')
        map_inclination = sunpy.map.Map('hmi_inclination.fits')
        map_azimuth = sunpy.map.Map('hmi_azimuth.fits')

        # Convert to heliographic coordinates
        map_bp, map_bt, map_br = hmi_b2ptr(map_field, map_inclination, map_azimuth)

    '''
    sz = map_field.data.shape
    ny, nx = sz

    field = map_field.data
    gamma = np.deg2rad(map_inclination.data)
    psi = np.deg2rad(map_azimuth.data)

    b_xi = -field * np.sin(gamma) * np.sin(psi)
    b_eta = field * np.sin(gamma) * np.cos(psi)
    b_zeta = field * np.cos(gamma)

    foo = all_coordinates_from_map(map_field).transform_to(HeliographicStonyhurst)
    phi = foo.lon
    lambda_ = foo.lat

    b = np.deg2rad(map_field.fits_header["crlt_obs"])
    p = np.deg2rad(-map_field.fits_header["crota2"])

    phi, lambda_ = np.deg2rad(phi), np.deg2rad(lambda_)

    sinb, cosb = np.sin(b), np.cos(b)
    sinp, cosp = np.sin(p), np.cos(p)
    sinphi, cosphi = np.sin(phi), np.cos(phi)  # nx*ny
    sinlam, coslam = np.sin(lambda_), np.cos(lambda_)  # nx*ny

    k11 = coslam * (sinb * sinp * cosphi + cosp * sinphi) - sinlam * cosb * sinp
    k12 = - coslam * (sinb * cosp * cosphi - sinp * sinphi) + sinlam * cosb * cosp
    k13 = coslam * cosb * cosphi + sinlam * sinb
    k21 = sinlam * (sinb * sinp * cosphi + cosp * sinphi) + coslam * cosb * sinp
    k22 = - sinlam * (sinb * cosp * cosphi - sinp * sinphi) - coslam * cosb * cosp
    k23 = sinlam * cosb * cosphi - coslam * sinb
    k31 = - sinb * sinp * sinphi + cosp * cosphi
    k32 = sinb * cosp * sinphi + sinp * cosphi
    k33 = - cosb * sinphi

    bptr = np.zeros((3, ny, nx))

    bptr[0, :, :] = k31 * b_xi + k32 * b_eta + k33 * b_zeta
    bptr[1, :, :] = k21 * b_xi + k22 * b_eta + k23 * b_zeta
    bptr[2, :, :] = k11 * b_xi + k12 * b_eta + k13 * b_zeta

    header = map_field.fits_header
    map_bp = map_from_data_header_compat(bptr[0, :, :], header)
    map_bt = map_from_data_header_compat(bptr[1, :, :], header)
    map_br = map_from_data_header_compat(bptr[2, :, :], header)

    return map_bp, map_bt, map_br


def validate_number(func):
    """
    Decorator to validate if the input in the widget is a number.

    :param func: function,
        The function to wrap.
    :return: function ,
        The wrapped function.
    """

    def wrapper(self, widget, *args, **kwargs):
        try:
            if hasattr(widget, "value"):
                float(widget.value())
            else:
                float(widget.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")
            return
        return func(self, widget, *args, **kwargs)

    return wrapper


def set_QLineEdit_text_pos(line_edit, text):
    """
    Sets the text of the QLineEdit and moves the cursor to the beginning.

    :param line_edit: QLineEdit,
        The QLineEdit widget.
    :param text: str,
        The text to set.
    """
    line_edit.setText(text)
    line_edit.setCursorPosition(0)


def read_gxsim_b3d_sav(savfile):
    """
    Read B3D data from a ``.sav`` file and save it as a ``.gxbox`` file.

    :param savfile: str,
        The path to the ``.sav`` file.
    """
    from scipy.io import readsav
    import pickle
    bdata = readsav(savfile)
    bx = bdata['box']['bx'][0]
    by = bdata['box']['by'][0]
    bz = bdata['box']['bz'][0]
    # boxfile = 'hmi.sharp_cea_720s.8088.20220330_172400_TAI.Br.gxbox'
    # with open(boxfile, 'rb') as f:
    #     gxboxdata = pickle.load(f)
    gxboxdata = {}
    gxboxdata['b3d'] = {}
    gxboxdata['b3d']['corona'] = {}
    gxboxdata['b3d']['corona']['bx'] = bx.swapaxes(0,2)
    gxboxdata['b3d']['corona']['by'] = by.swapaxes(0,2)
    gxboxdata['b3d']['corona']['bz'] = bz.swapaxes(0,2)
    gxboxdata['b3d']['corona']["attrs"] = {"model_type": "nlfff"}
    boxfilenew = savfile.replace('.sav', '.gxbox')
    with open(boxfilenew, 'wb') as f:
        pickle.dump(gxboxdata, f)
    print(f'{savfile} is saved as {boxfilenew}')
    return boxfilenew

def read_b3d_h5(filename):
    """
    Read B3D data from an HDF5 file and populate a dictionary.

    The resulting dictionary will contain keys corresponding to different
    magnetic field models (e.g., 'corona' for coronal fields and 'chromo' for
    chromospheric fields), and each model will have sub-keys for the magnetic
    field components (e.g., 'bx', 'by', 'bz').

    :param filename: str,
        The path to the HDF5 file.
    :return: dict,
        A dictionary containing the B3D data.

    :example:

    .. code-block:: python

        b3dbox = read_b3d_h5('path_to_file.h5')

        # Get the coronal field components
        bx_cor = b3dbox['corona']['bx']
        by_cor = b3dbox['corona']['by']
        bz_cor = b3dbox['corona']['bz']
    """
    def _read_h5_node(node):
        if isinstance(node, h5py.Group):
            out = {}
            for key in node.keys():
                out[key] = _read_h5_node(node[key])
            return out
        if node.shape == ():
            return node[()]
        return node[:]

    box_b3d = {}
    with h5py.File(filename, 'r') as hdf_file:
        for model_type in hdf_file.keys():
            group = hdf_file[model_type]
            target_type = model_type
            model_attr = None
            if model_type in ("nlfff", "pot", "potential", "bounds"):
                target_type = "corona"
                if model_type == "potential":
                    model_attr = "pot"
                elif model_type in ("bounds",):
                    model_attr = "bnd"
                else:
                    model_attr = model_type
            if target_type not in box_b3d:
                box_b3d[target_type] = {}
            component_names = list(group.keys())
            if target_type == "refmaps":
                def _refmap_sort_key(name: str):
                    obj = group[name]
                    if isinstance(obj, h5py.Group) and "order_index" in obj.attrs:
                        return (0, int(obj.attrs["order_index"]))
                    return (1, name)

                component_names = sorted(component_names, key=_refmap_sort_key)
            for component in component_names:
                ds = group[component]
                box_b3d[target_type][component] = _read_h5_node(ds)
            if len(group.attrs.keys()) > 0 or model_attr is not None:
                attrs = dict(group.attrs)
                if model_attr is not None and "model_type" not in attrs:
                    attrs["model_type"] = model_attr
                if "attrs" in box_b3d[target_type]:
                    box_b3d[target_type]["attrs"].update(attrs)
                else:
                    box_b3d[target_type]["attrs"] = attrs
    return box_b3d

def write_b3d_h5(filename, box_b3d):
    """
    Write B3D data to an HDF5 file from a dictionary.

    :param filename: str,
        The path to the HDF5 file.
    :param box_b3d: dict,
        A dictionary containing the B3D data to be written.
    """

    def _encode_dataset_value(value):
        if isinstance(value, (str, bytes, np.bytes_)):
            return np.bytes_(value)
        if value is None:
            return np.bytes_("")

        arr = np.asarray(value)
        if arr.dtype.kind != "O":
            return value

        if arr.shape == ():
            scalar = arr.item()
            if isinstance(scalar, (str, bytes, np.bytes_)):
                return np.bytes_(scalar)
            if scalar is None:
                return np.bytes_("")
            try:
                return np.asarray(scalar)
            except Exception:
                return np.bytes_(str(scalar))

        # Object arrays are serialized as fixed-width bytes arrays for HDF5 safety.
        as_text = np.vectorize(lambda x: "" if x is None else str(x), otypes=[str])(arr)
        return as_text.astype("S")

    def _set_group_attrs(group, attrs):
        if not isinstance(attrs, dict):
            return
        for key, value in attrs.items():
            group.attrs[key] = _encode_dataset_value(value)

    def _write_node(group, key, value, *, target_type=None):
        if isinstance(value, dict):
            sub = group.create_group(key)
            attrs = value.get("attrs", {}) if isinstance(value.get("attrs"), dict) else {}
            for sub_key, sub_val in value.items():
                if sub_key == "attrs":
                    continue
                _write_node(sub, sub_key, sub_val, target_type=target_type)
            _set_group_attrs(sub, attrs)
            return

        if target_type in ("chromo", "lines") and key == "voxel_status":
            group.create_dataset(key, data=np.asarray(value, dtype=np.uint8))
            return
        group.create_dataset(key, data=_encode_dataset_value(value))

    with h5py.File(filename, 'w') as hdf_file:
        for model_type, components in box_b3d.items():
            if components is None:
                print(f"Warning: {model_type} components are None, skipping.")
                continue
            if model_type == "metadata":
                group = hdf_file.create_group("metadata")
                for key, value in components.items():
                    _write_node(group, key, value, target_type="metadata")
                continue
            target_type = model_type
            attrs = components.get("attrs", {}) if isinstance(components, dict) else {}
            if model_type in ("nlfff", "pot", "potential", "bounds"):
                target_type = "corona"
                if "model_type" not in attrs:
                    attrs = dict(attrs)
                    if model_type == "potential":
                        attrs["model_type"] = "pot"
                    elif model_type == "bounds":
                        attrs["model_type"] = "bnd"
                    else:
                        attrs["model_type"] = model_type
            if target_type in hdf_file:
                # Avoid collisions if multiple legacy groups map to corona
                continue
            if target_type == "refmaps":
                group = hdf_file.create_group(target_type, track_order=True)
            else:
                group = hdf_file.create_group(target_type)
            refmap_idx = 0
            for component, data in components.items():
                if component == "attrs":
                    continue
                if target_type == "refmaps" and isinstance(data, dict):
                    sub = group.create_group(component, track_order=True)
                    sub.attrs["order_index"] = np.int64(refmap_idx)
                    refmap_idx += 1
                    sub_attrs = data.get("attrs", {}) if isinstance(data.get("attrs"), dict) else {}
                    for sub_key, sub_val in data.items():
                        if sub_key == "attrs":
                            continue
                        _write_node(sub, sub_key, sub_val, target_type=target_type)
                    _set_group_attrs(sub, sub_attrs)
                else:
                    _write_node(group, component, data, target_type=target_type)
            if attrs:
                _set_group_attrs(group, attrs)


def update_line_seeds_h5(filename, line_seeds):
    """
    Update only the ``line_seeds`` group in an existing HDF5 model file.

    Parameters
    ----------
    filename : str
        Target ``.h5`` model path.
    line_seeds : dict | None
        Serialized ``line_seeds`` payload. If falsy/non-dict, any existing
        ``line_seeds`` group is removed.
    """

    def _write_node(group, key, value):
        if isinstance(value, dict):
            sub = group.create_group(key)
            attrs = value.get("attrs", {}) if isinstance(value.get("attrs"), dict) else {}
            for sub_key, sub_val in value.items():
                if sub_key == "attrs":
                    continue
                _write_node(sub, sub_key, sub_val)
            if attrs:
                sub.attrs.update(attrs)
            return
        if isinstance(value, str):
            group.create_dataset(key, data=np.bytes_(value))
            return
        group.create_dataset(key, data=value)

    with h5py.File(filename, 'r+') as hdf_file:
        if "line_seeds" in hdf_file:
            del hdf_file["line_seeds"]
        if not isinstance(line_seeds, dict) or not line_seeds:
            return
        group = hdf_file.create_group("line_seeds")
        attrs = line_seeds.get("attrs", {}) if isinstance(line_seeds.get("attrs"), dict) else {}
        for key, value in line_seeds.items():
            if key == "attrs":
                continue
            _write_node(group, key, value)
        if attrs:
            group.attrs.update(attrs)


# Backward-compatible SFQ exports.
from pyampp.sfq import get_str_mag as get_str_mag
from pyampp.sfq import sfq_frame as sfq_frame
from pyampp.sfq import sfq_step1 as sfq_step1
from pyampp.sfq import sfq_clean as sfq_clean
from pyampp.sfq import pex_bl_ as pex_bl_
from pyampp.sfq import pex_bl as pex_bl
from pyampp.sfq import pot_vmag as pot_vmag
