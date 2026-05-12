#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyampp.util.config import DOWNLOAD_DIR, GXMODEL_DIR


REFERENCE_STAGE_DATE = "2020-11-26"
DEFAULT_IDL_STAGE_DIR = Path(GXMODEL_DIR) / REFERENCE_STAGE_DATE
DEFAULT_DATA_DIR = Path(DOWNLOAD_DIR)
DEFAULT_ARTIFACT_ROOT = Path(GXMODEL_DIR).parent / "stage_parity_artifacts"

PYAMPP_ROOT = REPO_ROOT
EXPORT_MODEL_SCRIPT = PYAMPP_ROOT / "pyampp" / "util" / "export_model.py"
GX_FOV2BOX_SCRIPT = PYAMPP_ROOT / "pyampp" / "gxbox" / "gx_fov2box.py"


@dataclass(frozen=True)
class StageStep:
    entry_stage: str
    target_stage: str
    jump_flag: str
    save_flag: str
    stop_after: str
    compare_paths: tuple[str, ...]
    target_suffixes: tuple[str, ...]


STAGE_STEPS: tuple[StageStep, ...] = (
    StageStep(
        entry_stage="NONE",
        target_stage="POT",
        jump_flag="--jump2potential",
        save_flag="--save-potential",
        stop_after="pot",
        compare_paths=("corona/bx", "corona/by", "corona/bz"),
        target_suffixes=(".POT.h5",),
    ),
    StageStep(
        entry_stage="POT",
        target_stage="BND",
        jump_flag="--jump2bounds",
        save_flag="--save-bounds",
        stop_after="bnd",
        compare_paths=("corona/bx", "corona/by", "corona/bz"),
        target_suffixes=(".BND.h5",),
    ),
    StageStep(
        entry_stage="BND",
        target_stage="NAS",
        jump_flag="--jump2nlfff",
        save_flag="--save-nas",
        stop_after="nas",
        compare_paths=("corona/bx", "corona/by", "corona/bz"),
        target_suffixes=(".NAS.h5",),
    ),
    StageStep(
        entry_stage="NAS",
        target_stage="GEN",
        jump_flag="--jump2lines",
        save_flag="--save-gen",
        stop_after="gen",
        compare_paths=("corona/bx", "corona/by", "corona/bz"),
        target_suffixes=(".NAS.GEN.h5", ".POT.GEN.h5"),
    ),
    StageStep(
        entry_stage="GEN",
        target_stage="CHR",
        jump_flag="--jump2chromo",
        save_flag="--save-chr",
        stop_after="chr",
        compare_paths=(
            "corona/bx",
            "corona/by",
            "corona/bz",
            "chromo/bx",
            "chromo/by",
            "chromo/bz",
        ),
        target_suffixes=(".NAS.CHR.h5", ".NAS.GEN.CHR.h5", ".POT.CHR.h5", ".POT.GEN.CHR.h5"),
    ),
)

STAGE_ORDER: tuple[str, ...] = ("NONE", "POT", "BND", "NAS", "GEN", "CHR")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonicalize IDL gx_fov2box SAV stages, run one-stage pyAMPP resume steps, "
            "and compare the resulting HDF5 cubes against exported IDL targets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--idl-stage-dir",
        type=Path,
        default=DEFAULT_IDL_STAGE_DIR,
        help=(
            "Directory containing the staged IDL SAV files. The documented reference "
            f"workflow defaults to the {REFERENCE_STAGE_DATE} folder under the pyAMPP gx_models root."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="gx_fov2box --data-dir to use for resume runs.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=(
            "Directory where exported IDL H5 files, pyAMPP outputs, logs, and the final report are stored."
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to invoke export_model.py and gx_fov2box.py.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the artifact root before generating new outputs.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-5,
        help="Relative tolerance used for allclose checks in the comparison report.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance used for allclose checks in the comparison report.",
    )
    return parser.parse_args()


def _stage_from_path(path: Path) -> str | None:
    name = path.name.upper()
    for suffix, stage in (
        (".NAS.GEN.CHR.SAV", "CHR"),
        (".POT.GEN.CHR.SAV", "CHR"),
        (".NAS.CHR.SAV", "CHR"),
        (".POT.CHR.SAV", "CHR"),
        (".NAS.GEN.SAV", "GEN"),
        (".POT.GEN.SAV", "GEN"),
        (".NAS.SAV", "NAS"),
        (".BND.SAV", "BND"),
        (".POT.SAV", "POT"),
        (".NONE.SAV", "NONE"),
    ):
        if name.endswith(suffix):
            return stage
    return None


def _discover_idl_stage_files(idl_stage_dir: Path) -> dict[str, Path]:
    stage_files: dict[str, Path] = {}
    for candidate in sorted(idl_stage_dir.glob("*.sav")):
        stage = _stage_from_path(candidate)
        if stage is None:
            continue
        stage_files[stage] = candidate

    required = {"NONE", "POT", "BND", "NAS", "GEN", "CHR"}
    missing = sorted(required - set(stage_files))
    if missing:
        raise FileNotFoundError(
            f"Missing required IDL stage SAV files in {idl_stage_dir}: {', '.join(missing)}"
        )
    return stage_files


def _run_command(command: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(command))
        log_file.write("\n\nOUTPUT:\n")
        log_file.flush()
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(command)}")


def _progress(message: str) -> None:
    print(message, flush=True)


def _export_one_idl_stage(
    *,
    python_exe: Path,
    stage: str,
    sav_path: Path,
    export_dir: Path,
    log_dir: Path,
) -> Path:
    out_h5 = export_dir / f"{sav_path.stem}.h5"
    _progress(f"[export] {stage}: {sav_path.name} -> {out_h5.name}")
    command = [
        str(python_exe),
        str(EXPORT_MODEL_SCRIPT),
        "--model-path",
        str(sav_path),
        "--out-h5",
        str(out_h5),
    ]
    _run_command(command, log_dir / f"export_{stage.lower()}.log", PYAMPP_ROOT)
    _progress(f"[export] done: {stage}")
    return out_h5


def _ensure_exported_stage(
    *,
    python_exe: Path,
    stage: str,
    stage_files: dict[str, Path],
    export_dir: Path,
    log_dir: Path,
    exported_stage_h5: dict[str, Path],
) -> Path:
    if stage in exported_stage_h5:
        return exported_stage_h5[stage]
    exported_stage_h5[stage] = _export_one_idl_stage(
        python_exe=python_exe,
        stage=stage,
        sav_path=stage_files[stage],
        export_dir=export_dir,
        log_dir=log_dir,
    )
    return exported_stage_h5[stage]


def _scan_h5_files(root: Path) -> set[Path]:
    return {p.resolve() for p in root.rglob("*.h5")}


def _pick_generated_file(new_files: set[Path], suffixes: tuple[str, ...]) -> Path:
    matches = sorted(
        path for path in new_files if any(path.name.endswith(suffix) for suffix in suffixes)
    )
    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely identify generated stage file. "
            f"Suffixes={suffixes!r}, matches={[str(p) for p in matches]}"
        )
    return matches[0]


def _compare_arrays(a: np.ndarray, b: np.ndarray, *, rtol: float, atol: float) -> dict[str, Any]:
    if a.shape != b.shape:
        return {
            "shape_match": False,
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
            "allclose": False,
        }

    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a64) & np.isfinite(b64)
    if not np.any(finite):
        return {
            "shape_match": True,
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
            "allclose": False,
            "reason": "no_finite_overlap",
        }

    av = a64[finite]
    bv = b64[finite]
    diff = av - bv
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    mean_abs_ref = float(np.mean(np.abs(av)))
    rel_mae = float(mae / mean_abs_ref) if mean_abs_ref > 0 else None
    corr = None
    if av.size > 1 and np.std(av) > 0 and np.std(bv) > 0:
        corr = float(np.corrcoef(av, bv)[0, 1])
    allclose = bool(np.allclose(av, bv, rtol=rtol, atol=atol, equal_nan=False))
    return {
        "shape_match": True,
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "allclose": allclose,
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "mean_abs_ref": mean_abs_ref,
        "rel_mae": rel_mae,
        "corr": corr,
        "finite_count": int(av.size),
    }


def _read_dataset(path: Path, dataset_path: str) -> np.ndarray | None:
    with h5py.File(path, "r") as handle:
        if dataset_path not in handle:
            return None
        return np.asarray(handle[dataset_path])


def _compare_stage_outputs(
    *,
    exported_target: Path,
    produced_target: Path,
    dataset_paths: tuple[str, ...],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "exported_target": str(exported_target),
        "produced_target": str(produced_target),
        "datasets": {},
        "missing": {"exported": [], "produced": []},
    }
    for dataset_path in dataset_paths:
        exported_arr = _read_dataset(exported_target, dataset_path)
        produced_arr = _read_dataset(produced_target, dataset_path)
        if exported_arr is None:
            metrics["missing"]["exported"].append(dataset_path)
            continue
        if produced_arr is None:
            metrics["missing"]["produced"].append(dataset_path)
            continue
        metrics["datasets"][dataset_path] = _compare_arrays(
            exported_arr,
            produced_arr,
            rtol=rtol,
            atol=atol,
        )
    metrics["missing"]["exported"].sort()
    metrics["missing"]["produced"].sort()
    metrics["allclose"] = all(
        entry.get("allclose", False) for entry in metrics["datasets"].values()
    ) and not metrics["missing"]["exported"] and not metrics["missing"]["produced"]
    return metrics


def _run_stage_resume_step(
    *,
    python_exe: Path,
    entry_h5: Path,
    data_dir: Path,
    branch_output_root: Path,
    step: StageStep,
    log_path: Path,
) -> Path:
    before = _scan_h5_files(branch_output_root)
    _progress(
        f"[resume] {step.entry_stage} -> {step.target_stage}: "
        f"{entry_h5.name}"
    )
    command = [
        str(python_exe),
        str(GX_FOV2BOX_SCRIPT),
        "--entry-box",
        str(entry_h5),
        "--data-dir",
        str(data_dir),
        "--gxmodel-dir",
        str(branch_output_root),
        step.save_flag,
        "--stop-after",
        step.stop_after,
        step.jump_flag,
    ]
    _run_command(command, log_path, PYAMPP_ROOT)
    after = _scan_h5_files(branch_output_root)
    new_files = after - before
    produced = _pick_generated_file(new_files, step.target_suffixes)
    _progress(f"[resume] produced: {produced.name}")
    return produced


def _prepare_artifact_root(artifact_root: Path, *, clean: bool) -> None:
    if clean and artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = _parse_args()

    idl_stage_dir = args.idl_stage_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    python_exe = args.python.expanduser().resolve()

    if not EXPORT_MODEL_SCRIPT.exists():
        raise FileNotFoundError(f"export_model.py not found: {EXPORT_MODEL_SCRIPT}")
    if not GX_FOV2BOX_SCRIPT.exists():
        raise FileNotFoundError(f"gx_fov2box.py not found: {GX_FOV2BOX_SCRIPT}")
    if not idl_stage_dir.exists():
        raise FileNotFoundError(f"IDL stage directory not found: {idl_stage_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    _progress(f"[start] artifact root: {artifact_root}")
    _progress(f"[start] IDL stage dir: {idl_stage_dir}")
    _progress(f"[start] data dir: {data_dir}")
    if args.clean:
        _progress("[start] cleaning previous artifacts")
    _prepare_artifact_root(artifact_root, clean=args.clean)

    export_dir = artifact_root / "idl_exported"
    pyampp_dir = artifact_root / "pyampp_generated"
    log_dir = artifact_root / "logs"
    report_dir = artifact_root / "reports"
    export_dir.mkdir(parents=True, exist_ok=True)
    pyampp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    stage_files = _discover_idl_stage_files(idl_stage_dir)
    _progress("[discover] found IDL stages: " + ", ".join(STAGE_ORDER))
    exported_stage_h5: dict[str, Path] = {}

    report: dict[str, Any] = {
        "idl_stage_dir": str(idl_stage_dir),
        "data_dir": str(data_dir),
        "artifact_root": str(artifact_root),
        "python": str(python_exe),
        "rtol": args.rtol,
        "atol": args.atol,
        "stage_files": {stage: str(path) for stage, path in stage_files.items()},
        "exported_stage_h5": {stage: str(path) for stage, path in exported_stage_h5.items()},
        "steps": [],
    }

    for step in STAGE_STEPS:
        entry_h5 = _ensure_exported_stage(
            python_exe=python_exe,
            stage=step.entry_stage,
            stage_files=stage_files,
            export_dir=export_dir,
            log_dir=log_dir,
            exported_stage_h5=exported_stage_h5,
        )
        target_h5 = _ensure_exported_stage(
            python_exe=python_exe,
            stage=step.target_stage,
            stage_files=stage_files,
            export_dir=export_dir,
            log_dir=log_dir,
            exported_stage_h5=exported_stage_h5,
        )
        branch_output_root = pyampp_dir / f"from_{step.entry_stage.lower()}"
        branch_output_root.mkdir(parents=True, exist_ok=True)
        produced_target = _run_stage_resume_step(
            python_exe=python_exe,
            entry_h5=entry_h5,
            data_dir=data_dir,
            branch_output_root=branch_output_root,
            step=step,
            log_path=log_dir / f"resume_{step.entry_stage.lower()}_to_{step.target_stage.lower()}.log",
        )
        _progress(
            f"[compare] {step.entry_stage} -> {step.target_stage}: "
            f"{target_h5.name} vs {produced_target.name}"
        )
        comparison = _compare_stage_outputs(
            exported_target=target_h5,
            produced_target=produced_target,
            dataset_paths=step.compare_paths,
            rtol=args.rtol,
            atol=args.atol,
        )
        _progress(
            f"[compare] result {step.entry_stage} -> {step.target_stage}: "
            f"allclose={comparison['allclose']}"
        )
        report["steps"].append(
            {
                "entry_stage": step.entry_stage,
                "target_stage": step.target_stage,
                "entry_h5": str(entry_h5),
                "exported_target_h5": str(target_h5),
                "produced_target_h5": str(produced_target),
                "compare_paths": list(step.compare_paths),
                "comparison": comparison,
            }
        )

    report["exported_stage_h5"] = {stage: str(path) for stage, path in exported_stage_h5.items()}

    report_path = report_dir / "gx_idl2py_stage_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print(f"Exported IDL stage H5 files: {export_dir}")
    print(f"pyAMPP generated stage H5 files: {pyampp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
