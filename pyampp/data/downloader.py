from pyampp.util.config import *
import os
import astropy.units as u
from sunpy.net import Fido, attrs as a
from glob import glob
from datetime import timedelta
import re

class SDOImageDownloader:
    """
    A class to download SDO (Solar Dynamics Observatory) image data for various instruments.

    :param time: The specific time for which to download SDO data.
    :type time: Astropy.Time
    :param uv: Flag to download ultraviolet (UV) images, defaults to True.
    :type uv: bool, optional
    :param euv: Flag to download extreme ultraviolet (EUV) images, defaults to True.
    :type euv: bool, optional
    :param hmi: Flag to download Helioseismic and Magnetic Imager (HMI) images, defaults to True.
    :type hmi: bool, optional
    """

    def __init__(self, time, uv=True, euv=True, hmi=True, data_dir = DOWNLOAD_DIR):
        """
        Initializes the downloader with specified configurations and prepares the download directory.
        """
        self.time = time
        self.uv = uv
        self.euv = euv
        self.hmi = hmi
        self.path = os.path.join(data_dir, time.datetime.strftime('%Y%m%d'))
        self._prepare_directory()
        self.existence_report = self._check_files_exist(self.path)

    def _prepare_directory(self):
        """
        Prepares the directory for storing downloaded files. Creates the directory if it does not exist.
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print(f"Created directory: {self.path}")
        else:
            print(f"Using existing directory: {self.path}")

    def _generate_filename_patterns(self, base_dir):
        """
        Generates a dictionary of filename patterns for different SDO data categories.

        :param base_dir: The base directory to include in the file patterns.
        :type base_dir: str
        :return: A dictionary with keys as data categories and values as filename patterns.
        :rtype: dict
        """
        patterns = {
            'euv': {
                pb: os.path.join(base_dir, f'aia.lev1_euv_12s.*.{pb}.image_lev1.fits') for pb in AIA_EUV_PASSBANDS
            },
            'uv': {
                pb: os.path.join(base_dir, f'aia.lev1_uv_24s.*.{pb}.image_lev1.fits') for pb in AIA_UV_PASSBANDS
            },
            'hmi_b': {
                seg: os.path.join(base_dir, f'hmi.b_720s.*_TAI.{seg}.fits') for seg in HMI_B_SEGMENTS
            },
            'hmi_m': {
                'magnetogram': os.path.join(base_dir, 'hmi.m_720s.*_TAI*.magnetogram.fits')
            },
            'hmi_ic': {
                'continuum': os.path.join(base_dir, 'hmi.ic_nolimbdark_720s.*_TAI*.continuum.fits')
            }
        }
        return patterns


    def _check_files_exist(self, datadir, returnfilelist=False):
        """
        Checks if the files exist in the specified directory based on pre-defined patterns.

        :param datadir: The directory where files are expected to be located.
        :type datadir: str
        :param returnfilelist: Whether to return the first file found for each pattern, defaults to False.
        :type returnfilelist: bool, optional
        :return: A dictionary indicating file existence or paths for each data category.
        :rtype: dict
        """
        patterns = self._generate_filename_patterns(datadir)
        existence_report = {}

        time_tolerances = {
            'euv': timedelta(seconds=12),
            'uv': timedelta(seconds=24),
            'hmi_b': timedelta(seconds=720),
            'hmi_m': timedelta(seconds=720),
            'hmi_ic': timedelta(seconds=720)
        }

        def file_within_tolerance(filepath, tolerance):
            filename = os.path.basename(filepath)
            timestamp_str = re.search(r'\d{4}-\d{2}-\d{2}T\d{6}Z', filename)
            if not timestamp_str:
                timestamp_str = re.search(r'\d{8}_\d{6}_TAI', filename)
            if timestamp_str:
                file_time = self.time.strptime(timestamp_str.group(), '%Y%m%d_%H%M%S_TAI' if '_' in timestamp_str.group() else '%Y-%m-%dT%H%M%SZ')
                return round(abs(file_time - self.time).sec, 2) <= tolerance.total_seconds()
            return False

        if returnfilelist:
            for category, patterns_dict in patterns.items():
                for key, pattern in patterns_dict.items():
                    found_files = glob(pattern)
                    found_files = glob(pattern)
                    found_files = [f for f in found_files if file_within_tolerance(f, time_tolerances[category])]
                    existence_report[key] = found_files[0] if found_files else None
        else:
            for category, patterns_dict in patterns.items():
                existence_report[category] = {}
                for key, pattern in patterns_dict.items():
                    found_files = glob(pattern)
                    found_files = [f for f in found_files if file_within_tolerance(f, time_tolerances[category])]
                    existence_report[category][key] = bool(found_files)
        return existence_report



    def download_images(self):
        """
        Initiates the download of missing SDO images based on the existence report.

        :return: An updated existence report after attempting to download missing files.
        :rtype: dict
        """
        all_files = {}
        if self.euv:
            self._handle_euv(all_files)
        if self.uv:
            self._handle_uv(all_files)
        if self.hmi:
            self._handle_hmi(all_files)

        files_to_download = list(all_files.values())
        if len(files_to_download) > 0:
            print(files_to_download)
            self._fetch(files_to_download)
        # Re-check file existence after downloads to update the report
        self.existence_report = self._check_files_exist(self.path, returnfilelist=True)
        return self.existence_report

    def _handle_euv(self, all_files):
        """
        Handles the downloading of missing EUV (Extreme Ultraviolet) data files.

        :param all_files: Dictionary to collect information about the downloaded files.
        :type all_files: dict
        """
        if self.existence_report:
            missing_euv = [pb for pb, exists in self.existence_report.get('euv', {}).items() if not exists]
            # print(f"Missing EUV passbands: {missing_euv}")
        else:
            missing_euv = AIA_EUV_PASSBANDS

        if len(missing_euv) > 0:
            missing_euv = [int(pb) for pb in missing_euv]
            wavelength_attr = a.AttrOr([a.Wavelength(pb * u.AA) for pb in missing_euv])

            all_files[f'euv'] = self._search('aia.lev1_euv_12s', wavelength=wavelength_attr,
                                                            segments=a.jsoc.Segment('image'))

    def _handle_uv(self, all_files):
        """
        Handles the downloading of missing UV (Ultraviolet) data files.

        :param all_files: Dictionary to collect information about the downloaded files.
        :type all_files: dict
        """
        if self.existence_report:
            missing_uv = [pb for pb, exists in self.existence_report.get('uv', {}).items() if not exists]
            # print(f"Missing UV segment: {missing_uv}")
        else:
            missing_uv = AIA_UV_PASSBANDS
            print("No existence report provided, downloading all UV segments.")

        if len(missing_uv) > 0:
            missing_uv = [int(pb) for pb in missing_uv]
            wavelength_attr = a.AttrOr([a.Wavelength(pb * u.AA) for pb in missing_uv])

            all_files[f'uv'] = self._search('aia.lev1_uv_24s', wavelength=wavelength_attr,
                                                           segments=a.jsoc.Segment('image'))

    def _handle_hmi(self, all_files):
        """
        Handles the downloading of missing HMI (Helioseismic and Magnetic Imager) data files.

        :param all_files: Dictionary to collect information about the downloaded files.
        :type all_files: dict
        """
        if self.existence_report:
            missing_hmi_b = [seg for seg, exists in self.existence_report.get('hmi_b', {}).items() if not exists]
            # print(f"Missing HMI B segments: {missing_hmi_b}")
            missing_hmi_m = not self.existence_report.get('hmi_m', {}).get('magnetogram', True)
            # print(f"Missing HMI M segments: {missing_hmi_m}")
            missing_hmi_ic = not self.existence_report.get('hmi_ic', {}).get('continuum', True)
            # print(f"Missing HMI IC segments: {missing_hmi_ic}")
        else:
            missing_hmi_b = HMI_B_SEGMENTS
            print("No existence report provided, downloading all HMI B segments.")
            missing_hmi_m = ['magnetogram']
            print("No existence report provided, downloading all HMI M segments.")
            missing_hmi_ic = ['continuum']
            print("No existence report provided, downloading all HMI IC segments.")

        if len(missing_hmi_b) > 0:
            segment_attr = a.AttrAnd([a.jsoc.Segment(seg) for seg in missing_hmi_b])
            all_files[f'hmi_b'] = self._search('hmi.B_720s', segments=segment_attr)
        if missing_hmi_m:
            all_files['hmi_m'] = self._search('hmi.M_720s', segments=a.jsoc.Segment('magnetogram'))
        if missing_hmi_ic:
            all_files['hmi_ic'] = self._search('hmi.Ic_noLimbDark_720s', segments=a.jsoc.Segment('continuum'))

    def _search(self, series, segments=None, wavelength=None):
        """
        Searches for and fetches files from JSOC based on series, segments, and wavelength.

        :param series: The data series to query.
        :type series: str
        :param segments: The data segments to include in the search, optional.
        :type segments: sunpy.net.attrs.Segment, optional
        :param wavelength: The wavelength filter to apply in the search, optional.
        :type wavelength: astropy.units.Quantity, optional
        :return: A list of paths to the downloaded files.
        :rtype: list
        """
        search_attrs = [a.Time(self.time, self.time), a.jsoc.Series(series), a.jsoc.Notify(JSOC_NOTIFY_EMAIL)]
        if wavelength:
            search_attrs.insert(-1, wavelength)  # Insert wavelength before notify
        if segments:
            search_attrs.insert(-1, segments)  # Insert segments before notify

        print(f"Searching for {series} with attributes {search_attrs}")
        result = Fido.search(*search_attrs)
        print(f"Found {len(result)} records for download.")
        return result

    def _fetch(self, files_to_download, streams=5):
        fetched_files = Fido.fetch(*files_to_download, path=self.path, overwrite=False, max_conn=streams)
        print(f"Downloaded {len(fetched_files)} files.")
        return fetched_files