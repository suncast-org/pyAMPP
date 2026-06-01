import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import numpy as np
from sunpy.net import Fido, attrs as a

from pyampp.util.config import *


class SDOImageDownloader:
    """
    Download SDO image products using either the legacy Fido path or a DRMS
    path that mirrors the IDL gx_box_jsoc_get_fits workflow more closely.
    """

    SUPPORTED_BACKENDS = ("fido", "drms")
    DRMS_MAX_WORKERS = 3
    DRMS_HMI_MAX_WORKERS = 1
    DRMS_EXPORT_RETRIES = 24
    DRMS_THROTTLE_BACKOFF_SECONDS = 5

    def __init__(
        self,
        time,
        uv=True,
        euv=True,
        hmi=True,
        data_dir=DOWNLOAD_DIR,
        backend="drms",
        force_download=False,
        poll_seconds=5,
        drms_sequential=False,
    ):
        self.time = time if isinstance(time, Time) else Time(time)
        self.uv = uv
        self.euv = euv
        self.hmi = hmi
        self.backend = str(backend or "drms").strip().lower()
        self.force_download = bool(force_download)
        self.poll_seconds = max(1, int(poll_seconds))
        self.drms_sequential = bool(drms_sequential)
        self._drms_throttle_seen = False
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported downloader backend '{backend}'. "
                f"Expected one of: {', '.join(self.SUPPORTED_BACKENDS)}"
            )

        self.path = os.path.join(data_dir, self.time.datetime.strftime("%Y-%m-%d"))
        self._cache_index_path = Path(self.path) / "index.json"
        self._cache_index_lock = threading.RLock()
        self._drms_client_local = threading.local()
        self._prepare_directory()
        self.existence_report = self._check_files_exist(self.path)

    def _prepare_directory(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print(f"Created directory: {self.path}")
        else:
            print(f"Using existing directory: {self.path}")

    def _generate_filename_patterns(self, base_dir):
        patterns = {
            "euv": {
                pb: [
                    os.path.join(base_dir, f"aia.lev1_euv_12s.*.{pb}.image_lev1.fits"),
                    os.path.join(base_dir, f"aia.lev1_euv_12s.*.image.{pb}.fits"),
                ]
                for pb in AIA_EUV_PASSBANDS
            },
            "uv": {
                pb: [
                    os.path.join(base_dir, f"aia.lev1_uv_24s.*.{pb}.image_lev1.fits"),
                    os.path.join(base_dir, f"aia.lev1_uv_24s.*.image.{pb}.fits"),
                ]
                for pb in AIA_UV_PASSBANDS
            },
            "hmi_b": {
                seg: [
                    os.path.join(base_dir, f"hmi.b_720s.*_TAI.{seg}.fits"),
                    os.path.join(base_dir, f"hmi.B_720s.*_TAI.{seg}.fits"),
                ]
                for seg in HMI_B_SEGMENTS
            },
            "hmi_m": {
                "magnetogram": [
                    os.path.join(base_dir, "hmi.m_720s.*_TAI*.magnetogram.fits"),
                    os.path.join(base_dir, "hmi.M_720s.*_TAI*.magnetogram.fits"),
                ]
            },
            "hmi_ic": {
                "continuum": [
                    os.path.join(base_dir, "hmi.ic_nolimbdark_720s.*_TAI*.continuum.fits"),
                    os.path.join(base_dir, "hmi.Ic_noLimbDark_720s.*_TAI*.continuum.fits"),
                ]
            },
        }
        return patterns

    def _check_files_exist(self, datadir, returnfilelist=False):
        patterns = self._generate_filename_patterns(datadir)
        existence_report = {}

        time_tolerances = {
            "euv": timedelta(seconds=12),
            "uv": timedelta(seconds=24),
            "hmi_b": timedelta(seconds=720),
            "hmi_m": timedelta(seconds=720),
            "hmi_ic": timedelta(seconds=720),
        }

        def file_within_tolerance(filepath, tolerance):
            filename = os.path.basename(filepath)
            match_aia = re.search(r"\d{4}-\d{2}-\d{2}T\d{6}Z", filename)
            if match_aia:
                file_dt = datetime.strptime(match_aia.group(), "%Y-%m-%dT%H%M%SZ")
                return abs((file_dt - self.time.datetime).total_seconds()) <= tolerance.total_seconds()

            match_hmi = re.search(r"\d{8}_\d{6}_TAI", filename)
            if match_hmi:
                file_dt = datetime.strptime(match_hmi.group(), "%Y%m%d_%H%M%S_TAI")
                return abs((file_dt - self.time.datetime).total_seconds()) <= tolerance.total_seconds()

            return False

        def find_matching_files(pattern_spec, tolerance):
            if isinstance(pattern_spec, str):
                pattern_spec = [pattern_spec]
            all_matches = []
            for pattern in pattern_spec:
                all_matches.extend(glob(pattern))
            unique_matches = sorted(set(all_matches))
            return [f for f in unique_matches if file_within_tolerance(f, tolerance)]

        if returnfilelist:
            for category, patterns_dict in patterns.items():
                for key, pattern in patterns_dict.items():
                    found_files = find_matching_files(pattern, time_tolerances[category])
                    existence_report[key] = found_files[0] if found_files else None
        else:
            for category, patterns_dict in patterns.items():
                existence_report[category] = {}
                for key, pattern in patterns_dict.items():
                    found_files = find_matching_files(pattern, time_tolerances[category])
                    existence_report[category][key] = bool(found_files)
        return existence_report

    def download_images(self):
        print(f"Using downloader backend: {self.backend}")
        if self.force_download:
            print("Force download enabled: bypassing local cache hits")
        if self.backend == "drms":
            self.existence_report = self._download_images_drms()
        else:
            self.existence_report = self._download_images_fido()
        return self.existence_report

    def _download_images_fido(self):
        all_files = {}
        if self.euv:
            self._handle_euv(all_files)
        if self.uv:
            self._handle_uv(all_files)
        if self.hmi:
            self._handle_hmi(all_files)

        files_to_download = list(all_files.values())
        if files_to_download:
            self._fetch(files_to_download, overwrite=self.force_download)
        return self._check_files_exist(self.path, returnfilelist=True)

    def _download_images_drms(self):
        self._drms_throttle_seen = False
        if self.drms_sequential:
            print("DRMS sequential mode enabled: forcing single-worker downloads")
        files = self._check_files_exist(self.path, returnfilelist=True)
        for key, path in list(files.items()):
            if path and not self._fits_has_map_metadata(path):
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
                files[key] = None
        if self.force_download:
            for key in list(files.keys()):
                files[key] = None

        hmi_tasks = [
            ("hmi.B_720s", "field", "field"),
            ("hmi.B_720s", "inclination", "inclination"),
            ("hmi.B_720s", "azimuth", "azimuth"),
            ("hmi.B_720s", "disambig", "disambig"),
            ("hmi.M_720s", "magnetogram", "magnetogram"),
            ("hmi.Ic_noLimbDark_720s", "continuum", "continuum"),
        ]
        hmi_jobs = []
        context_jobs = []
        if self.hmi:
            for idx, (series, segment, key) in enumerate(hmi_tasks, start=1):
                if not files.get(key):
                    hmi_jobs.append(
                        {
                            "key": key,
                            "series": series,
                            "segment": segment,
                            "wave": None,
                            "time_window": 720,
                            "label": f"DRMS HMI: {idx}/{len(hmi_tasks)} ({segment})",
                        }
                    )

        if self.euv:
            for idx, wave in enumerate(AIA_EUV_PASSBANDS, start=1):
                if not files.get(wave):
                    context_jobs.append(
                        {
                            "key": wave,
                            "series": "aia.lev1_euv_12s",
                            "segment": "image",
                            "wave": wave,
                            "time_window": 12,
                            "label": f"DRMS AIA EUV: {idx}/{len(AIA_EUV_PASSBANDS)} ({wave})",
                        }
                    )

        if self.uv:
            for idx, wave in enumerate(AIA_UV_PASSBANDS, start=1):
                if not files.get(wave):
                    context_jobs.append(
                        {
                            "key": wave,
                            "series": "aia.lev1_uv_24s",
                            "segment": "image",
                            "wave": wave,
                            "time_window": 24,
                            "label": f"DRMS AIA UV: {idx}/{len(AIA_UV_PASSBANDS)} ({wave})",
                        }
                    )

        def _run_jobs(jobs, workers, label):
            if not jobs:
                return []
            failed = []
            max_workers = max(1, min(workers, len(jobs)))
            if len(jobs) > 1:
                print(f"DRMS: downloading {len(jobs)} {label} products with up to {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_job = {}
                for job in jobs:
                    print(job["label"])
                    future_to_job[
                        pool.submit(
                            self._drms_get_fits,
                            job["series"],
                            job["segment"],
                            wave=job["wave"],
                            time_window=job["time_window"],
                        )
                    ] = job
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        files[job["key"]] = future.result() or ""
                        if not files[job["key"]]:
                            failed.append(job)
                    except Exception as exc:
                        print(
                            "DRMS task failed for "
                            f"{job['series']}{{{job['segment']}}}: {exc}"
                        )
                        files[job["key"]] = ""
                        failed.append(job)
            return failed

        # HMI requests are serialized to avoid JSOC pending-export throttling.
        hmi_workers = 1 if self.drms_sequential else self.DRMS_HMI_MAX_WORKERS
        _run_jobs(hmi_jobs, hmi_workers, "HMI")
        # Context AIA products keep parallelism from PR40.
        context_workers = 1 if self.drms_sequential else self.DRMS_MAX_WORKERS
        failed_context = _run_jobs(context_jobs, context_workers, "AIA")
        if (
            failed_context
            and context_workers > 1
            and self._drms_throttle_seen
        ):
            retry_jobs = [job for job in failed_context if not files.get(job["key"])]
            if retry_jobs:
                print(
                    "DRMS throttling detected during AIA parallel fetch; "
                    "retrying missing AIA products sequentially"
                )
                self._drms_throttle_seen = False
                _run_jobs(retry_jobs, 1, "AIA retry")

        verified_files = self._check_files_exist(self.path, returnfilelist=True)
        for key, path in files.items():
            if path and not verified_files.get(key):
                verified_files[key] = path
        return verified_files

    def _handle_euv(self, all_files):
        if self.force_download:
            missing_euv = AIA_EUV_PASSBANDS
        elif self.existence_report:
            missing_euv = [pb for pb, exists in self.existence_report.get("euv", {}).items() if not exists]
        else:
            missing_euv = AIA_EUV_PASSBANDS

        if missing_euv:
            missing_euv = [int(pb) for pb in missing_euv]
            wavelength_attr = a.AttrOr([a.Wavelength(pb * u.AA) for pb in missing_euv])
            all_files["euv"] = self._search(
                "aia.lev1_euv_12s",
                wavelength=wavelength_attr,
                segments=a.jsoc.Segment("image"),
            )

    def _handle_uv(self, all_files):
        if self.force_download:
            missing_uv = AIA_UV_PASSBANDS
        elif self.existence_report:
            missing_uv = [pb for pb, exists in self.existence_report.get("uv", {}).items() if not exists]
        else:
            missing_uv = AIA_UV_PASSBANDS

        if missing_uv:
            missing_uv = [int(pb) for pb in missing_uv]
            wavelength_attr = a.AttrOr([a.Wavelength(pb * u.AA) for pb in missing_uv])
            all_files["uv"] = self._search(
                "aia.lev1_uv_24s",
                wavelength=wavelength_attr,
                segments=a.jsoc.Segment("image"),
            )

    def _handle_hmi(self, all_files):
        if self.force_download:
            missing_hmi_b = HMI_B_SEGMENTS
            missing_hmi_m = True
            missing_hmi_ic = True
        elif self.existence_report:
            missing_hmi_b = [seg for seg, exists in self.existence_report.get("hmi_b", {}).items() if not exists]
            missing_hmi_m = not self.existence_report.get("hmi_m", {}).get("magnetogram", True)
            missing_hmi_ic = not self.existence_report.get("hmi_ic", {}).get("continuum", True)
        else:
            missing_hmi_b = HMI_B_SEGMENTS
            missing_hmi_m = True
            missing_hmi_ic = True

        if missing_hmi_b:
            segment_attr = a.AttrAnd([a.jsoc.Segment(seg) for seg in missing_hmi_b])
            all_files["hmi_b"] = self._search("hmi.B_720s", segments=segment_attr)
        if missing_hmi_m:
            all_files["hmi_m"] = self._search("hmi.M_720s", segments=a.jsoc.Segment("magnetogram"))
        if missing_hmi_ic:
            all_files["hmi_ic"] = self._search("hmi.Ic_noLimbDark_720s", segments=a.jsoc.Segment("continuum"))

    def _search(self, series, segments=None, wavelength=None):
        notify_email = jsoc_notify_email()
        search_attrs = [a.Time(self.time, self.time), a.jsoc.Series(series), a.jsoc.Notify(notify_email)]
        if wavelength:
            search_attrs.insert(-1, wavelength)
        if segments:
            search_attrs.insert(-1, segments)

        print(f"Searching for {series} with attributes {search_attrs}")
        result = Fido.search(*search_attrs)
        print(f"Found {len(result)} records for download.")
        return result

    def _fetch(self, files_to_download, streams=5, overwrite=False):
        fetched_files = Fido.fetch(*files_to_download, path=self.path, overwrite=overwrite, max_conn=streams)
        print(f"Downloaded {len(fetched_files)} files.")
        return fetched_files

    def _load_cache_index(self):
        if not self._cache_index_path.exists():
            return {}
        try:
            payload = json.loads(self._cache_index_path.read_text())
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        entries = payload.get("entries", payload)
        return entries if isinstance(entries, dict) else {}

    def _save_cache_index(self, entries):
        payload = {"version": 1, "entries": entries}
        cache_dir = self._cache_index_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, indent=2, sort_keys=True)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(cache_dir),
                prefix=".index.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_file.write(serialized)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                tmp_path = Path(tmp_file.name)
            os.replace(tmp_path, self._cache_index_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _cache_lookup(self, query):
        with self._cache_index_lock:
            entries = self._load_cache_index()
            rel_path = entries.get(query)
            if not rel_path:
                return ""
            fpath = Path(self.path) / rel_path
            if fpath.exists() and self._fits_has_map_metadata(fpath):
                return str(fpath)
            if fpath.exists():
                try:
                    fpath.unlink()
                except Exception:
                    pass
            entries.pop(query, None)
            self._save_cache_index(entries)
            return ""

    def _cache_store(self, query, file_path):
        with self._cache_index_lock:
            entries = self._load_cache_index()
            entries[query] = os.path.basename(file_path)
            self._save_cache_index(entries)

    def _get_drms_client(self):
        client = getattr(self._drms_client_local, "client", None)
        if client is None:
            try:
                import drms
            except Exception as exc:
                raise RuntimeError("drms is required for --download-backend drms") from exc
            client = drms.Client()
            self._drms_client_local.client = client
        return client

    @staticmethod
    def _series_time_mode(series):
        s = str(series or "").strip().lower()
        if s.startswith("hmi."):
            return "tai"
        if s.startswith("aia."):
            return "utc"
        return "utc"

    @staticmethod
    def _query_window_seconds(series, nominal_window):
        s = str(series or "").strip().lower()
        window = max(1, int(round(float(nominal_window))))
        if s.startswith("hmi."):
            # Query one full cadence on either side, then choose the nearest
            # record explicitly. This matches the IDL outcome more reliably
            # than a half-cadence DRMS recordset window.
            return window * 2
        return window

    @classmethod
    def _jsoc_time_string(cls, t, series):
        tt = Time(t)
        mode = cls._series_time_mode(series)
        if mode == "tai":
            return tt.tai.strftime("%Y.%m.%d_%H:%M:%S_TAI")
        return tt.utc.strftime("%Y.%m.%d_%H:%M:%S")

    @classmethod
    def _make_query_recordset(cls, t1, t2, series, wave=None):
        duration_sec = max(1, int(round((Time(t2) - Time(t1)).to_value("sec"))))
        query = f"{series}[{cls._jsoc_time_string(t1, series)}/{duration_sec}s]"
        if wave is not None:
            query += f"[{wave}]"
        return query

    @classmethod
    def _make_query_key(cls, t1, t2, series, segment, wave=None):
        return cls._make_query_recordset(t1, t2, series, wave=wave) + f"{{{segment}}}"

    @staticmethod
    def _parse_jsoc_time(value) -> Optional[Time]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "nat", "none"}:
            return None
        try:
            return Time(text)
        except Exception:
            pass
        match = re.match(
            r"^(?P<y>\d{4})\.(?P<m>\d{2})\.(?P<d>\d{2})_(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})(?:\.(?P<frac>\d+))?(?:_(?P<scale>TAI|UTC))?$",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        parts = match.groupdict()
        frac = parts.get("frac") or "0"
        iso = (
            f"{parts['y']}-{parts['m']}-{parts['d']}T"
            f"{parts['h']}:{parts['mi']}:{parts['s']}.{frac}"
        )
        scale = (parts.get("scale") or "TAI").lower()
        try:
            return Time(iso, format="isot", scale=scale)
        except Exception:
            return None

    @staticmethod
    def _filename_time_token(jsoc_time_text):
        return str(jsoc_time_text).replace(".", "").replace(":", "")

    @classmethod
    def _make_local_filename(cls, series, segment, t_rec, wave=None):
        time_token = cls._filename_time_token(t_rec)
        if wave is not None:
            return f"{series}.{time_token}.{segment}.{wave}.fits"
        return f"{series}.{time_token}.{segment}.fits"

    def _select_nearest_record(self, key_df, seg_df, segment):
        if key_df is None or key_df.empty:
            return None

        valid_index = list(key_df.index)
        if seg_df is not None and not seg_df.empty:
            seg_col = next((col for col in seg_df.columns if str(col).lower() == str(segment).lower()), None)
            if seg_col is None and len(seg_df.columns) == 1:
                seg_col = seg_df.columns[0]
            if seg_col is not None:
                valid_index = [
                    idx for idx in valid_index if str(seg_df.at[idx, seg_col]).strip().lower() not in {"", "nan", "none"}
                ]

        best = None
        target = self.time.tai
        cols = {str(col).lower(): col for col in key_df.columns}
        obs_col = cols.get("t_obs")
        rec_col = cols.get("t_rec")

        for record in valid_index:
            row = key_df.loc[record]
            obs_time = self._parse_jsoc_time(row[obs_col]) if obs_col is not None else None
            rec_time = self._parse_jsoc_time(row[rec_col]) if rec_col is not None else None
            sample_time = obs_time or rec_time
            if sample_time is None:
                continue
            delta = abs((sample_time.tai - target).to_value("sec"))
            if best is None or delta < best["delta"]:
                best = {
                    "record": str(record),
                    "delta": float(delta),
                    "t_obs": row[obs_col] if obs_col is not None else None,
                    "t_rec": row[rec_col] if rec_col is not None else None,
                }
        return best

    def _download_from_url(self, url, target_path, timeout=60):
        target = Path(target_path)
        tmp = target.with_name(target.name + ".part")
        try:
            with urlopen(url, timeout=timeout) as response, open(tmp, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            tmp.replace(target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    @staticmethod
    def _sanitize_header_key(name):
        key = str(name or "").strip().upper()
        if not key or len(key) > 8:
            return ""
        if not re.fullmatch(r"[A-Z0-9_\-]+", key):
            return ""
        return key

    @staticmethod
    def _sanitize_header_value(value):
        if value is None:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode(errors="ignore")
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in {"nan", "none", "null"}:
                return None
            return value
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (list, tuple, dict, set)):
            return None
        return value

    @classmethod
    def _build_header_from_keyword_row(cls, row):
        header = fits.Header()
        skip_keys = {
            "SIMPLE",
            "BITPIX",
            "EXTEND",
            "PCOUNT",
            "GCOUNT",
            "BSCALE",
            "BZERO",
            "BLANK",
            "CHECKSUM",
            "DATASUM",
            "COMMENT",
            "HISTORY",
        }
        for name, value in row.items():
            key = cls._sanitize_header_key(name)
            if not key or key in skip_keys or key.startswith("NAXIS"):
                continue
            cleaned = cls._sanitize_header_value(value)
            if cleaned is None:
                continue
            try:
                header[key] = cleaned
            except Exception:
                continue

        if "DATE-OBS" not in header:
            for source_key in ("DATE_OBS", "DATE__OB", "T_OBS", "T_REC"):
                if source_key in header:
                    header["DATE-OBS"] = header[source_key]
                    break
        if "DATE-OBS" in header and "DATE_OBS" not in header:
            header["DATE_OBS"] = header["DATE-OBS"]
        return header

    @staticmethod
    def _extract_normalized_image_data(raw_path):
        with fits.open(raw_path, do_not_scale_image_data=True, memmap=False) as hdul:
            source_hdu = None
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                if data is not None and getattr(data, "ndim", 0) >= 2:
                    source_hdu = hdu
                    break
            if source_hdu is None:
                raise RuntimeError(f"No image data found in {raw_path}")

            raw_data = np.asarray(source_hdu.data)
            bscale = source_hdu.header.get("BSCALE", 1.0)
            bzero = source_hdu.header.get("BZERO", 0.0) or 0.0

            data = raw_data.astype(np.float32, copy=False)
            if float(bscale) != 1.0:
                data = data * np.float32(bscale)
            if float(bzero) != 0.0:
                data = data + np.float32(bzero)
            return data

    def _query_record_header(self, record):
        client = self._get_drms_client()
        try:
            key_df = client.query(record, key="**ALL**", rec_index=False, convert_numeric=True)
        except Exception as exc:
            print(f"DRMS keyword query failed for {record}: {exc}")
            return None
        if key_df is None or len(key_df) == 0:
            print(f"DRMS keyword query returned no metadata for {record}")
            return None
        return self._build_header_from_keyword_row(key_df.iloc[0])

    def _normalize_drms_export(self, raw_path, local_path, record):
        header = self._query_record_header(record)
        if header is None:
            return ""
        try:
            data = self._extract_normalized_image_data(raw_path)
            fits.PrimaryHDU(data=data, header=header).writeto(local_path, overwrite=True)
        except Exception as exc:
            print(f"DRMS normalization failed for {record}: {exc}")
            return ""
        return str(local_path)

    @staticmethod
    def _fits_has_map_metadata(path):
        try:
            with fits.open(path) as hdul:
                hdr = hdul[0].header
                return any(k in hdr for k in ("CTYPE1", "DATE-OBS", "T_REC", "T_OBS"))
        except Exception:
            return False

    def _drms_get_fits(self, series, segment, wave=None, time_window=12):
        query_window = self._query_window_seconds(series, time_window)
        t1 = self.time - (query_window / 2.0) * u.s
        t2 = self.time + (query_window / 2.0) * u.s
        query_key = self._make_query_key(t1, t2, series, segment, wave=wave)

        if not self.force_download:
            cached = self._cache_lookup(query_key)
            if cached:
                print(f"  cache hit: {os.path.basename(cached)}")
                return cached

        client = self._get_drms_client()
        recordset = self._make_query_recordset(t1, t2, series, wave=wave)
        keys = ["T_REC", "T_OBS"]
        try:
            key_df, seg_df = client.query(
                recordset,
                key=keys,
                seg=[segment],
                rec_index=True,
                convert_numeric=False,
            )
        except Exception as exc:
            print(f"DRMS query failed for {query_key}: {exc}")
            return ""

        chosen = self._select_nearest_record(key_df, seg_df, segment)
        if not chosen:
            print(f"No DRMS records found for {query_key}")
            return ""

        export_record = f"{chosen['record']}{{{segment}}}"
        print(f"  nearest record: {chosen['record']}")
        urls_df = None
        for attempt in range(1, self.DRMS_EXPORT_RETRIES + 1):
            try:
                request = client.export(
                    export_record,
                    method="url_quick",
                    protocol="fits",
                    email=jsoc_notify_email(),
                    filenamefmt=False,
                )
                request.wait(sleep=self.poll_seconds)
                urls_df = request.urls
                break
            except Exception as exc:
                msg = str(exc)
                throttled = "pending export requests" in msg.lower() or "status=7" in msg.lower()
                if throttled and attempt < self.DRMS_EXPORT_RETRIES:
                    self._drms_throttle_seen = True
                    backoff = max(self.poll_seconds, self.DRMS_THROTTLE_BACKOFF_SECONDS)
                    print(
                        f"DRMS export throttled for {export_record}; retry {attempt}/{self.DRMS_EXPORT_RETRIES} "
                        f"after {backoff}s"
                    )
                    time.sleep(backoff)
                    continue
                print(f"DRMS export failed for {export_record}: {exc}")
                return ""

        if urls_df is None or len(urls_df) == 0:
            print(f"DRMS export returned no URLs for {export_record}")
            return ""

        local_name = self._make_local_filename(series, segment, chosen["t_rec"], wave=wave)
        local_path = Path(self.path) / local_name
        if self.force_download and local_path.exists():
            try:
                local_path.unlink()
            except Exception:
                pass
        if not local_path.exists():
            tmp_raw = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp_file:
                    tmp_raw = Path(tmp_file.name)
                self._download_from_url(str(urls_df.iloc[0]["url"]), tmp_raw)
                print(f"  normalizing: {local_name}")
                normalized = self._normalize_drms_export(tmp_raw, local_path, chosen["record"])
                if not normalized:
                    return ""
            except Exception as exc:
                print(f"DRMS download failed for {export_record}: {exc}")
                return ""
            finally:
                if tmp_raw is not None:
                    tmp_raw.unlink(missing_ok=True)
        if not self._fits_has_map_metadata(local_path):
            try:
                local_path.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"DRMS download produced invalid FITS metadata for {export_record}")
            return ""

        self._cache_store(query_key, str(local_path))
        return str(local_path)
