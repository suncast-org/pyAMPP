import numpy as np


def _closest_idx(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _histogram_cdf(values: np.ndarray, nbins: int, data_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.array([]), np.array([])
    if not np.isfinite(data_range[0]) or not np.isfinite(data_range[1]):
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        data_range = (vmin, vmax)
    hist, bin_edges = np.histogram(values[finite], bins=nbins, range=data_range)
    xbin = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cdf = np.cumsum(hist, dtype=np.float64)
    if cdf.size:
        cdf = cdf / cdf[-1]
    return xbin, cdf


def decompose(mag, cont):
    mag_qs = 10  # 10 Gauss for QS
    thr_plage = 3  # MF in plage is thr_plage times stronger than QS

    sub = np.abs(mag) < mag_qs
    count = np.count_nonzero(sub)
    if count > 0:
        cutoff_qs = float(np.sum(cont[sub]) / count)
    else:
        finite_cont = np.asarray(cont, dtype=float)
        finite_mask = np.isfinite(finite_cont)
        if np.any(finite_mask):
            cutoff_qs = float(np.nanmedian(finite_cont[finite_mask]))
        else:
            cutoff_qs = 1.0

    if not np.isfinite(cutoff_qs):
        cutoff_qs = 1.0

    nbins = cont.size
    data_range = (float(np.min(cont)), float(np.max(cont)))

    # all pixels in FOV (including sunspots) - kept for parity with IDL flow
    xbin, cdf = _histogram_cdf(cont.ravel(), nbins, data_range)
    if cdf.size:
        cutoff_b = xbin[_closest_idx(cdf, 0.75)]
        cutoff_f = xbin[_closest_idx(cdf, 0.97)]
    else:
        cutoff_b = cutoff_f = data_range[1]

    # exclude sunspots
    sub = cont > (cutoff_qs * 0.9)
    xbin, cdf = _histogram_cdf(cont[sub].ravel(), nbins, data_range)
    if cdf.size:
        cutoff_b = xbin[_closest_idx(cdf, 0.75)]
        cutoff_f = xbin[_closest_idx(cdf, 0.97)]
    else:
        cutoff_b = cutoff_f = data_range[1]

    
    # creating decomposition mask
    model_mask = np.zeros(cont.shape, dtype=np.int32)
    abs_mag = np.abs(mag)

    # umbra
    sub = cont <= (0.65 * cutoff_qs)
    n_umbra = np.count_nonzero(sub)
    if n_umbra != 0:
        model_mask[sub] = 7

    # penumbra
    sub = (cont > (0.65 * cutoff_qs)) & (cont <= (0.9 * cutoff_qs))
    n_penumbra = np.count_nonzero(sub)
    if n_penumbra != 0:
        model_mask[sub] = 6

    # enhanced NW
    sub = (cont > cutoff_f) & (cont <= (1.19 * cutoff_qs))
    n_enw = np.count_nonzero(sub)
    if n_enw != 0:
        model_mask[sub] = 3

    # NW lane
    sub = (cont > cutoff_b) & (cont <= cutoff_f)
    n_nw = np.count_nonzero(sub)
    if n_nw != 0:
        model_mask[sub] = 2

    # IN
    sub = (cont > (0.9 * cutoff_qs)) & (cont <= cutoff_b)
    n_in = np.count_nonzero(sub)
    if n_in != 0:
        model_mask[sub] = 1

    # plage
    sub = (cont > (0.95 * cutoff_qs)) & (cont <= cutoff_f) & (abs_mag > (thr_plage * mag_qs))
    n_plage = np.count_nonzero(sub)
    if n_plage != 0:
        model_mask[sub] = 4

    # facula
    sub = (cont > (1.01 * cutoff_qs)) & (abs_mag > (thr_plage * mag_qs))
    n_facula = np.count_nonzero(sub)
    if n_facula != 0:
        model_mask[sub] = 5

    return model_mask
