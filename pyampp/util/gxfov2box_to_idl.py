#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py


def _decode_scalar(value) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _load_execute_from_h5(path: Path) -> str:
    with h5py.File(path, "r") as h5:
        if "metadata/execute" not in h5:
            raise KeyError(f"Missing metadata/execute in {path}")
        return _decode_scalar(h5["metadata/execute"][()]).strip()


def _strip_command_name(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens
    candidates = {"gx-fov2box", "gx_fov2box"}
    exe = Path(tokens[0]).name.lower()
    if exe in candidates:
        return tokens[1:]
    if len(tokens) >= 4 and exe.startswith("python") and tokens[1] == "-m" and tokens[2] == "pyampp.gxbox.gx_fov2box":
        return tokens[3:]
    return tokens


def _parse_python_command(command: str) -> Tuple[List[str], Dict[str, str], List[str]]:
    tokens = _strip_command_name(shlex.split(command))
    kw: Dict[str, str] = {}
    bool_flags: List[str] = []
    i = 0
    arity = {
        "--time": 1,
        "--coords": 2,
        "--box-dims": 3,
        "--dx-km": 1,
        "--pad-frac": 1,
        "--data-dir": 1,
        "--gxmodel-dir": 1,
        "--stop-after": 1,
        "--observer-name": 1,
        "--fov-xc": 1,
        "--fov-yc": 1,
        "--fov-xsize": 1,
        "--fov-ysize": 1,
        "--reduce-passed": 1,
        "--entry-box": 1,
    }
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            n = arity.get(token, 0)
            if n == 0:
                bool_flags.append(token)
                i += 1
                continue
            vals = tokens[i + 1:i + 1 + n]
            if len(vals) != n:
                raise ValueError(f"Incomplete argument for {token}")
            kw[token] = " ".join(vals)
            i += 1 + n
            continue
        i += 1
    return tokens, kw, bool_flags


def _iso_to_idl_time(value: str) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.strftime("%d-%b-%y %H:%M:%S")


def _quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _build_idl_execute(kw: Dict[str, str], bool_flags: List[str]) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    args: List[str] = ["gx_fov2box"]
    mapped: Dict[str, str] = {}
    unmapped: Dict[str, str] = {}

    if "--time" in kw:
        idl_time = _iso_to_idl_time(kw["--time"])
        args.append(_quote(idl_time))
        mapped["TIME"] = idl_time

    coord_mode = None
    if "--hpc" in bool_flags:
        coord_mode = "HPC"
    elif "--hgc" in bool_flags:
        coord_mode = "HGC"
    elif "--hgs" in bool_flags:
        coord_mode = "HGS"

    if "--coords" in kw:
        coords = kw["--coords"].split()
        if len(coords) == 2:
            if coord_mode == "HPC":
                args.append(f"CENTER_ARCSEC=[{coords[0]}, {coords[1]}]")
                mapped["CENTER_ARCSEC"] = kw["--coords"]
            elif coord_mode in {"HGC", "HGS"}:
                args.append(f"CENTER_DEG=[{coords[0]}, {coords[1]}]")
                mapped["CENTER_DEG"] = kw["--coords"]
                args.append("/" + coord_mode)
            else:
                unmapped["--coords"] = kw["--coords"]

    direct_map = {
        "--box-dims": ("SIZE_PIX", lambda v: "[" + ", ".join(v.split()) + "]"),
        "--dx-km": ("DX_KM", lambda v: v),
        "--pad-frac": ("PAD_FRAC", lambda v: v),
        "--data-dir": ("TMP_DIR", _quote),
        "--gxmodel-dir": ("OUT_DIR", _quote),
        "--entry-box": ("ENTRY_BOX", _quote),
        "--reduce-passed": ("REDUCE_PASSED", lambda v: v),
        "--observer-name": ("OBSERVER_NAME", _quote),
        "--fov-xc": ("FOV_XC", lambda v: v),
        "--fov-yc": ("FOV_YC", lambda v: v),
        "--fov-xsize": ("FOV_XSIZE", lambda v: v),
        "--fov-ysize": ("FOV_YSIZE", lambda v: v),
    }
    for flag, (key, fmt) in direct_map.items():
        if flag in kw:
            args.append(f"{key}={fmt(kw[flag])}")
            mapped[key] = kw[flag]

    bool_map = {
        "--cea": "CEA",
        "--top": "TOP",
        "--euv": "EUV",
        "--uv": "UV",
        "--sfq": "SFQ",
        "--save-empty-box": "SAVE_EMPTY_BOX",
        "--save-potential": "SAVE_POTENTIAL",
        "--save-bounds": "SAVE_BOUNDS",
        "--save-nas": "SAVE_NAS",
        "--save-gen": "SAVE_GEN",
        "--save-chr": "SAVE_CHR",
        "--empty-box-only": "EMPTY_BOX_ONLY",
        "--potential-only": "POTENTIAL_ONLY",
        "--nlfff-only": "NLFFF_ONLY",
        "--generic-only": "GENERIC_ONLY",
        "--use-potential": "USE_POTENTIAL",
        "--skip-lines": "SKIP_LINES",
        "--center-vox": "CENTER_VOX",
        "--square-fov": "SQUARE_FOV",
        "--jump2potential": "JUMP2POTENTIAL",
        "--jump2bounds": "JUMP2BOUNDS",
        "--jump2nlfff": "JUMP2NLFFF",
        "--jump2lines": "JUMP2LINES",
        "--jump2chromo": "JUMP2CHROMO",
        "--rebuild": "REBUILD",
        "--rebuild-from-none": "REBUILD_FROM_NONE",
        "--info": "INFO",
    }
    for flag in bool_flags:
        if flag in {"--hpc", "--hgc", "--hgs"}:
            continue
        key = bool_map.get(flag)
        if key is None:
            unmapped[flag] = "1"
            continue
        args.append("/" + key)
        mapped[key] = "1"

    if "--stop-after" in kw:
        unmapped["--stop-after"] = kw["--stop-after"]

    return ", ".join(args), mapped, unmapped


def _format_procedure(proc_name: str, execute: str) -> str:
    return (
        f"pro {proc_name}\n"
        "  compile_opt idl2\n"
        f"  {execute}\n"
        "end\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate a gx-fov2box Python command or HDF5 metadata/execute into an IDL gx_fov2box call."
    )
    parser.add_argument("--command", type=str, help="Python gx-fov2box command string.")
    parser.add_argument("--h5", type=Path, help="HDF5 model file; reads metadata/execute.")
    parser.add_argument("--out-pro", type=Path, help="Optional output .pro file path.")
    parser.add_argument("--out-json", type=Path, help="Optional output JSON report path.")
    parser.add_argument(
        "--proc-name",
        type=str,
        default="run_gx_fov2box_generated",
        help="Procedure name to use when writing --out-pro.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print both source Python command and translated IDL call.",
    )
    args = parser.parse_args()

    if not args.command and not args.h5:
        parser.error("Provide one of --command or --h5.")
    if args.command and args.h5:
        parser.error("Use either --command or --h5, not both.")

    if args.h5:
        source_command = _load_execute_from_h5(args.h5)
        source = str(args.h5)
    else:
        source_command = args.command or ""
        source = "cli"

    _, kw, bool_flags = _parse_python_command(source_command)
    execute, mapped, unmapped = _build_idl_execute(kw, bool_flags)

    if args.verbose:
        print("Python gx-fov2box:")
        print(source_command)
        print()
        print("IDL gx_fov2box:")
        print(execute)
        print()
    else:
        print(execute)

    report = {
        "source": source,
        "python_command": source_command,
        "idl_execute": execute,
        "mapped_keywords": mapped,
        "unmapped_flags": unmapped,
    }

    if args.out_pro:
        args.out_pro.parent.mkdir(parents=True, exist_ok=True)
        args.out_pro.write_text(_format_procedure(args.proc_name, execute), encoding="utf-8")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if unmapped:
        for key, value in unmapped.items():
            print(f"WARNING: unmapped {key}={value}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
