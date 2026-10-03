"""SFQ (Super Fast and Quality) azimuth disambiguation for pyAMPP.

Vendored from the Python port in Sergey Anfinogentov's SFQ repository
(vit1-irk contribution, merged upstream), the same source gx_simulator
vendors as its ``sfq`` submodule:

  https://github.com/Sergey-Anfinogentov/SFQ

Reference: Rudenko, G. V. & Anfinogentov, S. A. (2014),
"Very Fast and Accurate Azimuth Disambiguation of Vector Magnetograms",
Solar Physics, Volume 289, Issue 5, pp. 1499-1516.

Please cite that paper and the upstream repository when publishing work
that uses these routines.
"""

from .disambig import sfq_clean, sfq_disambig, sfq_frame, sfq_step1
from .data import get_str_mag
from .field import pot_vmag

__all__ = [
    "sfq_disambig",
    "sfq_frame",
    "sfq_step1",
    "sfq_clean",
    "get_str_mag",
    "pot_vmag",
]
