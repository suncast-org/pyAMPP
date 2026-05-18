from __future__ import annotations

"""Test helper: exposes SAV->HDF5 converter from the canonical io location."""

from pyampp.io._sav_convert import build_h5_from_sav, main

__all__ = ["build_h5_from_sav", "main"]


if __name__ == "__main__":
    main()
