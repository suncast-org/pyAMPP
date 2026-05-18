"""
CLI utility: export a supported pyAMPP model to canonical pyAMPP HDF5.

This CLI is a thin wrapper around the canonical pyAMPP loader/writer contract:
    source model -> load_model -> canonical in-memory structure -> save_model -> H5

Usage::

    python -m pyampp.util.export_model --model-path model.sav --out-h5 model.h5

Or if registered as a console script::

    pyampp-export-model --model-path model.sav --out-h5 model.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyampp.io import load_model, save_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a supported pyAMPP model to canonical pyAMPP HDF5 format."
    )
    p.add_argument("--model-path", type=Path, required=True, help="Path to input model (.sav or .h5).")
    p.add_argument("--out-h5", type=Path, required=True, help="Output HDF5 path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model = load_model(args.model_path)
    save_model(model, args.out_h5)
    print(f"Wrote: {args.out_h5}")


if __name__ == "__main__":
    main()