from __future__ import annotations

from astropy.constants import R_sun
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import numpy as np
from sunpy.coordinates import (
    HeliographicStonyhurst,
    get_body_heliographic_stonyhurst,
    get_earth,
    get_horizons_coord,
)
from sunpy.coordinates import sun


_HORIZONS_TARGETS = {
    "solar orbiter": "Solar Orbiter",
    "stereo-a": "STEREO-A",
    "stereo-b": "STEREO-B",
}

_OBSERVER_CACHE: dict[tuple[str, str], SkyCoord | None] = {}
_OBSERVER_STATUS_CACHE: dict[tuple[str, str], tuple[SkyCoord | None, str, str | None]] = {}
_NETWORK_ERROR_MARKERS = (
    "temporary failure",
    "name or service not known",
    "name resolution",
    "nodename nor servname",
    "network is unreachable",
    "failed to establish a new connection",
    "max retries exceeded",
    "connection refused",
    "connection reset",
    "connection aborted",
    "remote end closed connection",
    "timed out",
    "timeout",
    "ssl",
    "proxyerror",
    "urlopen error",
    "service unavailable",
)


def decode_meta_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, (bytes, bytearray)):
            return item.decode("utf-8", "ignore")
        return str(item)
    return str(value)


def normalize_observer_key(observer_key: str | None) -> str:
    raw = observer_key
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "ignore")
    key = str(raw or "earth").strip().lower()
    aliases = {
        "custom": "custom",
        "sdo": "sdo",
        "sdo/aia": "sdo",
        "sdo/hmi": "sdo",
        "earth": "earth",
        "terra": "earth",
        "solo": "solar orbiter",
        "solar-orbiter": "solar orbiter",
        "solarorbiter": "solar orbiter",
        "solar orbiter": "solar orbiter",
        "stereo a": "stereo-a",
        "stereo-a": "stereo-a",
        "stereoa": "stereo-a",
        "stereo ahead": "stereo-a",
        "stereo b": "stereo-b",
        "stereo-b": "stereo-b",
        "stereob": "stereo-b",
        "stereo behind": "stereo-b",
    }
    return aliases.get(key, key)


def _observer_coord_from_ephemeris(ephemeris: dict, obs_time=None) -> SkyCoord | None:
    if not isinstance(ephemeris, dict):
        return None
    try:
        hgln = ephemeris.get("hgln_obs_deg")
        hglt = ephemeris.get("hglt_obs_deg")
        dsun_cm = ephemeris.get("dsun_cm")
        if hgln is None or hglt is None or dsun_cm is None:
            return None
        when = obs_time if obs_time is not None else ephemeris.get("obs_date", ephemeris.get("obs_time"))
        if when is not None and not isinstance(when, Time):
            when = Time(when)
        return SkyCoord(
            lon=float(hgln) * u.deg,
            lat=float(hglt) * u.deg,
            radius=u.Quantity(float(dsun_cm), u.cm),
            frame=HeliographicStonyhurst(obstime=when),
        )
    except Exception:
        return None


def _header_from_text(header_text) -> fits.Header | None:
    text = decode_meta_text(header_text)
    if not text:
        return None
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    try:
        return fits.Header.fromstring(text, sep="\n")
    except Exception:
        return None


def _observer_coord_from_header(header: fits.Header | None, obs_time=None) -> SkyCoord | None:
    if header is None:
        return None
    try:
        hgln = header.get("HGLN_OBS")
        hglt = header.get("HGLT_OBS")
        dsun = header.get("DSUN_OBS")
        if hgln is None or hglt is None or dsun is None:
            return None
        when = obs_time or header.get("DATE-OBS") or header.get("DATE_OBS")
        if when is not None and not isinstance(when, Time):
            when = Time(when)
        return SkyCoord(
            lon=float(hgln) * u.deg,
            lat=float(hglt) * u.deg,
            radius=u.Quantity(float(dsun), u.m),
            frame=HeliographicStonyhurst(obstime=when),
        )
    except Exception:
        return None


def ephemeris_from_fits_header(header: fits.Header | None) -> tuple[dict[str, float | str], tuple[str, ...]]:
    if header is None:
        return {}, ("DATE-OBS", "HGLN_OBS", "HGLT_OBS", "DSUN_OBS", "RSUN_REF")
    ephemeris: dict[str, float | str] = {}
    missing: list[str] = []
    raw_obs_date = header.get("DATE-OBS", header.get("DATE_OBS"))
    if raw_obs_date is None:
        missing.append("DATE-OBS")
    else:
        try:
            ephemeris["obs_date"] = Time(raw_obs_date).isot
        except Exception:
            missing.append("DATE-OBS")
    for key in ("HGLN_OBS", "HGLT_OBS"):
        value = header.get(key)
        if value is None:
            missing.append(key)
            continue
        try:
            ephemeris["hgln_obs_deg" if key == "HGLN_OBS" else "hglt_obs_deg"] = float(value)
        except Exception:
            missing.append(key)
    dsun_obs = header.get("DSUN_OBS")
    if dsun_obs is None:
        missing.append("DSUN_OBS")
    else:
        try:
            ephemeris["dsun_cm"] = float(u.Quantity(dsun_obs, u.m).to_value(u.cm))
        except Exception:
            missing.append("DSUN_OBS")
    rsun_ref = header.get("RSUN_REF")
    if rsun_ref is None:
        missing.append("RSUN_REF")
    else:
        try:
            ephemeris["rsun_cm"] = float(u.Quantity(rsun_ref, u.m).to_value(u.cm))
        except Exception:
            missing.append("RSUN_REF")
    return ephemeris, tuple(dict.fromkeys(missing))


def _cache_key(observer_key: str | None, obs_time: Time | None) -> tuple[str, str]:
    key = normalize_observer_key(observer_key)
    if obs_time is None:
        return key, ""
    when = obs_time if isinstance(obs_time, Time) else Time(obs_time)
    return key, when.isot


def _classify_observer_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _NETWORK_ERROR_MARKERS):
        return "offline"
    return "unavailable"


def _resolve_named_observer_with_status(observer_key: str | None, obs_time: Time):
    key = normalize_observer_key(observer_key)
    cache_key = _cache_key(key, obs_time)
    if cache_key in _OBSERVER_STATUS_CACHE:
        return _OBSERVER_STATUS_CACHE[cache_key]

    coord = None
    status = "unavailable"
    detail = None
    if key in _HORIZONS_TARGETS:
        target = _HORIZONS_TARGETS[key]
        try:
            coord = get_horizons_coord(target, obs_time)
            coord = coord.transform_to(HeliographicStonyhurst(obstime=obs_time))
            status = "available"
        except Exception as exc:
            status = _classify_observer_exception(exc)
            detail = str(exc)
    else:
        try:
            coord = get_body_heliographic_stonyhurst(key, obs_time)
            status = "available"
        except Exception as exc:
            status = "unavailable"
            detail = str(exc)

    _OBSERVER_STATUS_CACHE[cache_key] = (coord, status, detail)
    _OBSERVER_CACHE[cache_key] = coord
    return coord, status, detail


def _stored_observer_key(b3d: dict) -> str | None:
    observer_meta = b3d.get("observer", {}) if isinstance(b3d, dict) else {}
    if not isinstance(observer_meta, dict):
        return None
    fov_box = observer_meta.get("fov_box", {})
    if isinstance(fov_box, dict) and fov_box.get("observer_key") is not None:
        return normalize_observer_key(fov_box.get("observer_key"))
    if observer_meta.get("name") is not None:
        return normalize_observer_key(observer_meta.get("name"))
    return None


def resolve_observer_from_metadata(b3d: dict, observer_key: str | None, obs_time=None) -> SkyCoord | None:
    if not isinstance(b3d, dict):
        return None
    requested_key = normalize_observer_key(observer_key)
    stored_key = _stored_observer_key(b3d)
    if stored_key is not None and requested_key != stored_key:
        return None

    observer_meta = b3d.get("observer", {})
    if not isinstance(observer_meta, dict):
        observer_meta = {}

    coord = _observer_coord_from_ephemeris(observer_meta.get("ephemeris", {}), obs_time=obs_time)
    if coord is not None:
        observer_name = normalize_observer_key(observer_meta.get("name"))
        if requested_key == "custom" or observer_name == requested_key:
            return coord

    return None


def resolve_sdo_observer_from_b3d(b3d: dict, obs_time=None) -> SkyCoord | None:
    if not isinstance(b3d, dict):
        return None
    refmaps = b3d.get("refmaps", {})
    if not isinstance(refmaps, dict):
        refmaps = {}
    for key in ("Bz_reference", "Ic_reference"):
        payload = refmaps.get(key)
        if not isinstance(payload, dict):
            continue
        coord = _observer_coord_from_header(_header_from_text(payload.get("wcs_header")), obs_time=obs_time)
        if coord is not None:
            return coord
    return None


def resolve_named_observer(observer_key: str | None, obs_time: Time):
    coord, status, _detail = _resolve_named_observer_with_status(observer_key, obs_time)
    if status == "available":
        return coord
    return None


def probe_observer_availability(b3d: dict, observer_key: str | None, obs_time: Time):
    key = normalize_observer_key(observer_key)
    if key in {"earth", "custom"}:
        return "available", None
    if key == "sdo":
        if resolve_sdo_observer_from_b3d(b3d, obs_time=obs_time) is not None:
            return "available", None
        return "unavailable", "Saved/source map metadata does not contain an SDO observer."
    _coord, status, detail = _resolve_named_observer_with_status(key, obs_time)
    return status, detail


def resolve_observer(b3d: dict, observer_key: str | None, obs_time: Time):
    coord, _warning, _used_key = resolve_observer_with_info(b3d, observer_key, obs_time)
    return coord


def resolve_observer_parameters_from_ephemeris(
    ephemeris: dict | None,
    *,
    observer_key: str | None = None,
    obs_time=None,
):
    """
    Resolve observer geometry and derived solar angles from stored ephemeris cards.

    Parameters
    ----------
    ephemeris : dict | None
        Expected keys are ``hgln_obs_deg``, ``hglt_obs_deg``, ``dsun_cm`` and optionally
        ``rsun_cm`` and ``obs_date``.
    observer_key : str | None
        Fallback observer name used only when the ephemeris cards are incomplete.
    obs_time : parseable time or `astropy.time.Time`, optional
        Observation time. If omitted, ``ephemeris['obs_date']`` is used.

    Returns
    -------
    dict
        Keys:
        ``observer_coordinate``, ``observer_key``, ``obs_time``, ``b0_deg``, ``l0_deg``,
        ``p_deg``, ``dsun_cm``, ``rsun_cm``, ``rsun_arcsec``, ``source``.

    Notes
    -----
    ``b0_deg`` and ``l0_deg`` follow the legacy GX/IDL ``pb0r`` convention:
    they are the observer Stonyhurst latitude and longitude, respectively.
    ``p_deg`` is currently returned as ``None``. SunPy's public ``sun.P()`` helper is Earth-only,
    and a general spacecraft-safe image-north position-angle implementation should be added
    separately rather than guessed here.
    """
    when = obs_time
    if when is None and isinstance(ephemeris, dict):
        when = ephemeris.get("obs_date", ephemeris.get("obs_time"))
    if when is not None and not isinstance(when, Time):
        when = Time(when)

    coord = _observer_coord_from_ephemeris(ephemeris or {}, obs_time=when)
    source = "ephemeris"
    used_key = normalize_observer_key(observer_key)
    if coord is None:
        if when is None:
            return None
        coord = resolve_named_observer(observer_key, when)
        source = "observer_name"
        if coord is None:
            return None

    try:
        hgs = coord.transform_to(HeliographicStonyhurst(obstime=when))
    except Exception:
        hgs = coord

    rsun_cm = None
    if isinstance(ephemeris, dict) and ephemeris.get("rsun_cm") is not None:
        try:
            rsun_cm = float(ephemeris.get("rsun_cm"))
        except Exception:
            rsun_cm = None

    rsun_arcsec = None
    if rsun_cm is not None:
        try:
            ratio = float(rsun_cm) / float(coord.radius.to_value(u.cm))
            ratio = float(np.clip(ratio, -1.0, 1.0))
            rsun_arcsec = float(np.arcsin(ratio) * u.rad.to(u.arcsec))
        except Exception:
            rsun_arcsec = None

    return {
        "observer_coordinate": coord,
        "observer_key": used_key,
        "obs_time": when,
        "b0_deg": float(hgs.lat.to_value(u.deg)),
        "l0_deg": float(hgs.lon.to_value(u.deg)),
        "p_deg": float(sun.P(when).to_value(u.deg)) if used_key == "earth" and when is not None else None,
        "dsun_cm": float(coord.radius.to_value(u.cm)),
        "rsun_cm": rsun_cm,
        "rsun_arcsec": rsun_arcsec,
        "source": source,
    }


def build_pb0r_metadata_from_ephemeris(
    ephemeris: dict | None,
    *,
    observer_key: str | None = None,
    obs_time=None,
):
    params = resolve_observer_parameters_from_ephemeris(
        ephemeris,
        observer_key=observer_key,
        obs_time=obs_time,
    )
    if params is None:
        return None
    pb0r = {
        "obs_date": params["obs_time"].isot if isinstance(params.get("obs_time"), Time) else params.get("obs_time"),
        "b0_deg": params.get("b0_deg"),
        "l0_deg": params.get("l0_deg"),
        "p_deg": params.get("p_deg"),
        "rsun_arcsec": params.get("rsun_arcsec"),
    }
    return {key: value for key, value in pb0r.items() if value is not None}


def build_ephemeris_from_pb0r(
    *,
    b0_deg,
    l0_deg,
    rsun_arcsec,
    obs_date=None,
    rsun_cm=None,
):
    try:
        b0_value = float(b0_deg)
        l0_value = float(l0_deg)
        rsun_arcsec_value = float(rsun_arcsec)
    except Exception:
        return None
    if not np.isfinite(b0_value) or not np.isfinite(l0_value) or not np.isfinite(rsun_arcsec_value):
        return None
    if rsun_arcsec_value <= 0:
        return None
    if rsun_cm is None:
        rsun_cm_value = float(R_sun.cgs.value)
    else:
        try:
            rsun_cm_value = float(rsun_cm)
        except Exception:
            return None
    try:
        rsun_rad = float((rsun_arcsec_value * u.arcsec).to_value(u.rad))
        if not np.isfinite(rsun_rad) or rsun_rad <= 0:
            return None
        dsun_cm_value = float(rsun_cm_value / np.sin(rsun_rad))
    except Exception:
        return None
    ephemeris = {
        "hgln_obs_deg": l0_value,
        "hglt_obs_deg": b0_value,
        "dsun_cm": dsun_cm_value,
        "rsun_cm": rsun_cm_value,
    }
    if obs_date:
        ephemeris["obs_date"] = str(obs_date)
    return ephemeris


def resolve_observer_with_info(b3d: dict, observer_key: str | None, obs_time: Time):
    requested_key = normalize_observer_key(observer_key)

    if requested_key == "earth":
        return get_earth(obs_time), None, "earth"

    if requested_key == "sdo":
        sdo_coord = resolve_sdo_observer_from_b3d(b3d, obs_time=obs_time)
        if sdo_coord is not None:
            return sdo_coord, None, "sdo"
        warning = (
            "Observer 'sdo' could not be restored from saved/source map metadata; "
            "falling back to Earth observer for this session."
        )
        return get_earth(obs_time), warning, "earth"

    coord = resolve_observer_from_metadata(b3d, requested_key, obs_time=obs_time)
    if coord is not None:
        return coord, None, requested_key

    coord = resolve_named_observer(requested_key, obs_time)
    if coord is not None:
        return coord, None, requested_key

    earth_coord = get_earth(obs_time)
    warning = (
        f"Observer '{requested_key}' could not be restored from saved metadata "
        f"and remote ephemeris was unavailable; falling back to Earth observer for this session."
    )
    return earth_coord, warning, "earth"
