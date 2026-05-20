from __future__ import annotations

import warnings

import numpy as np

from pyampp.gx_chromo.decompose import decompose


def test_decompose_no_quiet_sun_pixels_emits_no_runtime_warning() -> None:
    mag = np.full((4, 4), 50.0, dtype=np.float64)
    cont = np.linspace(0.8, 1.2, 16, dtype=np.float64).reshape(4, 4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        mask = decompose(mag, cont)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert mask.shape == cont.shape
    assert mask.dtype == np.int32
