import os

def get_base_directory():
    """ Get the base directory depending on the OS. """
    home_dir = os.path.expanduser('~')
    return os.path.join(home_dir, 'pyampp')

def setup_directories():
    """ Create and return the necessary directories for the package. """
    base_dir = get_base_directory()

    # Define subdirectories within the base directory
    sample_data_dir = os.path.join(base_dir, 'sample')
    download_dir = os.path.join(base_dir, 'download')

    # Ensure these directories exist
    os.makedirs(sample_data_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    return sample_data_dir, download_dir

def aia_euv_passbands():
    """ Return the passbands for the SDO/AIA instrument. """
    return ['94', '131', '171', '193', '211', '304', '335']

def aia_uv_passbands():
    """ Return the passbands for the SDO/AIA instrument. """
    return ['1600']

def hmi_b_segments():
    """ Return the segments for the HMI magnetogram. """
    return ['field', 'inclination', 'azimuth', 'disambig']

def jsoc_notify_email():
    """ Return the email address for JSOC notifications. """
    return "suncasa-group@njit.edu"

def hmi_b_products():
    """ Return the products for the HMI magnetogram. """
    return ['br', 'bp', 'bt']

# Set up the directories when this module is imported
SAMPLE_DATA_DIR, DOWNLOAD_DIR = setup_directories()
AIA_EUV_PASSBANDS = aia_euv_passbands()
AIA_UV_PASSBANDS = aia_uv_passbands()
JSOC_NOTIFY_EMAIL = jsoc_notify_email()
HMI_B_SEGMENTS = hmi_b_segments()
HMI_B_PRODUCTS = hmi_b_products()
