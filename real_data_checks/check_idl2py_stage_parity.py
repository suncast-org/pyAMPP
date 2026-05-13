#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
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
    entry_sav_suffixes: tuple[str, ...]
    target_sav_suffixes: tuple[str, ...]
    jump_flag: str
    save_flag: str
    stop_after: str
    compare_paths: tuple[str, ...]
    target_suffixes: tuple[str, ...]
    naming_note: str | None = None


STAGE_STEPS: tuple[StageStep, ...] = (
    StageStep(
        entry_stage="NONE",
        target_stage="POT",
        entry_sav_suffixes=(".NONE.SAV",),
        target_sav_suffixes=(".POT.SAV",),
        jump_flag="--jump2potential",
        save_flag="--save-potential",
        stop_after="pot",
        compare_paths=("corona/bx", "corona/by", "corona/bz"),
        target_suffixes=(".POT.h5",),
    ),
    StageStep(
        entry_stage="POT",
        target_stage="BND",
        entry_sav_suffixes=(".POT.SAV",),
        target_sav_suffixes=(".BND.SAV",),
        jump_flag="--jump2bounds",
        save_flag="--save-bounds",
        stop_after="bnd",
        compare_paths=("corona/bx", "corona/by", "corona/bz"),
        target_suffixes=(".BND.h5",),
    ),
    StageStep(
        entry_stage="BND",
        target_stage="NAS",
        entry_sav_suffixes=(".BND.SAV",),
        target_sav_suffixes=(".NAS.SAV",),
        jump_flag="--jump2nlfff",
        save_flag="--save-nas",
        stop_after="nas",
        compare_paths=("corona/bx", "corona/by", "corona/bz"),
        target_suffixes=(".NAS.h5",),
    ),
    StageStep(
        entry_stage="NAS",
        target_stage="GEN",
        entry_sav_suffixes=(".NAS.SAV",),
        target_sav_suffixes=(".NAS.GEN.SAV",),
        jump_flag="--jump2lines",
        save_flag="--save-gen",
        stop_after="gen",
        compare_paths=("corona/bx", "corona/by", "corona/bz"),
        target_suffixes=(".NAS.GEN.h5", ".POT.GEN.h5"),
    ),
    StageStep(
        entry_stage="GEN",
        target_stage="CHR",
        entry_sav_suffixes=(".NAS.GEN.SAV",),
        target_sav_suffixes=(".NAS.CHR.SAV", ".NAS.GEN.CHR.SAV", ".POT.CHR.SAV", ".POT.GEN.CHR.SAV"),
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
        naming_note=(
            "IDL usually labels the CHR target as NAS.CHR, while pyAMPP may emit "
            "NAS.GEN.CHR because the Python workflow exposes the explicit NAS-to-CHR jump."
        ),
    ),
)

STAGE_ORDER: tuple[str, ...] = ("NONE", "POT", "BND", "NAS", "GEN", "CHR")


def _stage_selector_values() -> tuple[str, ...]:
    values: list[str] = []
    for step in STAGE_STEPS:
        values.append(step.target_stage)
        values.append(f"{step.entry_stage}->{step.target_stage}")
        values.append(f"{step.entry_stage}:{step.target_stage}")
    return tuple(values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonicalize IDL gx_fov2box SAV stages for comparison, run one-stage "
            "pyAMPP resume steps directly from the IDL SAV entry models, and compare "
            "the resulting HDF5 cubes against exported IDL targets."
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
        "--dry-run",
        action="store_true",
        help="Report the per-stage resume/export/compare actions without executing commands or writing artifacts.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate only the JSON report from an existing artifact tree without rerunning resume or export commands.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help=(
            "Restrict execution to a single transition while still regenerating the final report. "
            "Accepted values are the target stage name or an explicit ENTRY->TARGET transition, for example "
            "NAS, 'BND->NAS', or BND:NAS."
        ),
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


def _flag_explicit_on_cli(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


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


def _pick_idl_stage_file(
    idl_stage_dir: Path,
    suffixes: tuple[str, ...],
    *,
    label: str,
) -> Path:
    matches = sorted(
        candidate for candidate in idl_stage_dir.glob("*.sav") if any(candidate.name.upper().endswith(suffix) for suffix in suffixes)
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            "Could not uniquely identify IDL stage SAV file. "
            f"label={label!r}, suffixes={suffixes!r}, matches={[str(path) for path in matches]}"
        )
    return matches[0]


def _run_command(command: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(shlex.join(command))
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


def _build_export_command(*, python_exe: Path, sav_path: Path, out_h5: Path) -> list[str]:
    return [
        str(python_exe),
        str(EXPORT_MODEL_SCRIPT),
        "--model-path",
        str(sav_path),
        "--out-h5",
        str(out_h5),
    ]


def _build_resume_command(
    *,
    python_exe: Path,
    entry_model: Path,
    data_dir: Path,
    branch_output_root: Path,
    step: StageStep,
) -> list[str]:
    return [
        str(python_exe),
        str(GX_FOV2BOX_SCRIPT),
        "--entry-box",
        str(entry_model),
        "--data-dir",
        str(data_dir),
        "--gxmodel-dir",
        str(branch_output_root),
        step.save_flag,
        "--stop-after",
        step.stop_after,
        step.jump_flag,
    ]


def _planned_export_path(*, export_dir: Path, sav_path: Path) -> Path:
    return export_dir / f"{sav_path.stem}.h5"


def _export_one_idl_stage(
    *,
    python_exe: Path,
    stage: str,
    sav_path: Path,
    export_dir: Path,
    log_dir: Path,
) -> Path:
    out_h5 = _planned_export_path(export_dir=export_dir, sav_path=sav_path)
    _progress(f"[export] {stage}: {sav_path.name} -> {out_h5.name}")
    command = _build_export_command(python_exe=python_exe, sav_path=sav_path, out_h5=out_h5)
    _run_command(command, log_dir / f"export_{stage.lower()}.log", PYAMPP_ROOT)
    _progress(f"[export] done: {stage}")
    return out_h5


def _ensure_exported_stage(
    *,
    python_exe: Path,
    label: str,
    sav_path: Path,
    export_dir: Path,
    log_dir: Path,
    exported_stage_h5: dict[str, Path],
) -> Path:
    cache_key = str(sav_path.resolve())
    if cache_key in exported_stage_h5:
        return exported_stage_h5[cache_key]
    exported_stage_h5[cache_key] = _export_one_idl_stage(
        python_exe=python_exe,
        stage=label,
        sav_path=sav_path,
        export_dir=export_dir,
        log_dir=log_dir,
    )
    return exported_stage_h5[cache_key]


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


def _pick_existing_file(root: Path, suffixes: tuple[str, ...], *, label: str) -> Path:
    matches = sorted(
        path.resolve() for path in root.rglob("*.h5") if any(path.name.endswith(suffix) for suffix in suffixes)
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            "Could not uniquely identify existing artifact file. "
            f"label={label!r}, suffixes={suffixes!r}, matches={[str(path) for path in matches]}"
        )
    return matches[0]


def _require_existing_path(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required existing artifact for {label}: {path}")
    return path.resolve()


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
    entry_model: Path,
    data_dir: Path,
    branch_output_root: Path,
    step: StageStep,
    log_path: Path,
) -> Path:
    before = _scan_h5_files(branch_output_root)
    _progress(
        f"[resume] {step.entry_stage} -> {step.target_stage}: "
        f"{entry_model.name}"
    )
    command = _build_resume_command(
        python_exe=python_exe,
        entry_model=entry_model,
        data_dir=data_dir,
        branch_output_root=branch_output_root,
        step=step,
    )
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


def _load_existing_report_defaults(artifact_root: Path) -> dict[str, Path]:
    report_path = artifact_root / "reports" / "gx_idl2py_stage_parity_report.json"
    if not report_path.exists():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    defaults: dict[str, Path] = {}
    for key in ("idl_stage_dir", "data_dir"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            defaults[key] = Path(value).expanduser().resolve()
    return defaults


def _normalize_stage_selector(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    value = raw_value.strip().upper()
    if not value:
        raise ValueError("--stage cannot be empty")
    if value.endswith("-"):
        raise ValueError(
            "Invalid --stage selector. Unquoted ENTRY->TARGET values are interpreted by the shell as redirection. "
            "Quote the selector, for example '--stage \"BND->NAS\"', or use the shell-safe form '--stage BND:NAS'."
        )
    if ":" in value:
        entry_stage, target_stage = (part.strip() for part in value.split(":", 1))
        if not entry_stage or not target_stage:
            raise ValueError(f"Invalid --stage selector: {raw_value!r}")
        value = f"{entry_stage}->{target_stage}"
    if "->" in value:
        entry_stage, target_stage = (part.strip() for part in value.split("->", 1))
        if not entry_stage or not target_stage:
            raise ValueError(f"Invalid --stage selector: {raw_value!r}")
        value = f"{entry_stage}->{target_stage}"
    valid = {item.upper() for item in _stage_selector_values()}
    if value not in valid:
        raise ValueError(
            "Unsupported --stage selector. "
            f"Expected one of: {', '.join(_stage_selector_values())}. "
            "If using ENTRY->TARGET, quote it or use ENTRY:TARGET to avoid shell redirection."
        )
    return value


def _step_matches_selector(step: StageStep, selector: str | None) -> bool:
    if selector is None:
        return True
    if selector == step.target_stage.upper():
        return True
    return selector == f"{step.entry_stage}->{step.target_stage}"


def _selected_steps(selector: str | None) -> tuple[StageStep, ...]:
    return tuple(step for step in STAGE_STEPS if _step_matches_selector(step, selector))


def _collect_stage_report_entry(
    *,
    step: StageStep,
    entry_sav: Path,
    target_sav: Path,
    exported_target: Path,
    produced_target: Path,
    rtol: float,
    atol: float,
    progress_prefix: str,
) -> dict[str, Any]:
    _progress(
        f"[{progress_prefix}] {step.entry_stage} -> {step.target_stage}: "
        f"{exported_target.name} vs {produced_target.name}"
    )
    comparison = _compare_stage_outputs(
        exported_target=exported_target,
        produced_target=produced_target,
        dataset_paths=step.compare_paths,
        rtol=rtol,
        atol=atol,
    )
    _progress(
        f"[{progress_prefix}] result {step.entry_stage} -> {step.target_stage}: "
        f"allclose={comparison['allclose']}"
    )
    return {
        "entry_stage": step.entry_stage,
        "target_stage": step.target_stage,
        "entry_sav": str(entry_sav),
        "target_sav": str(target_sav),
        "entry_box": str(entry_sav),
        "entry_box_format": "sav",
        "exported_target_h5": str(exported_target),
        "produced_target_h5": str(produced_target),
        "compare_paths": list(step.compare_paths),
        "naming_note": step.naming_note,
        "comparison": comparison,
    }


def _report_dry_run_step(
    *,
    python_exe: Path,
    entry_sav: Path,
    target_sav: Path,
    data_dir: Path,
    branch_output_root: Path,
    export_dir: Path,
    log_dir: Path,
    step: StageStep,
) -> dict[str, Any]:
    resume_log = log_dir / f"resume_{step.entry_stage.lower()}_to_{step.target_stage.lower()}.log"
    export_log = log_dir / f"export_{step.target_stage.lower()}.log"
    target_h5 = _planned_export_path(export_dir=export_dir, sav_path=target_sav)
    resume_command = _build_resume_command(
        python_exe=python_exe,
        entry_model=entry_sav,
        data_dir=data_dir,
        branch_output_root=branch_output_root,
        step=step,
    )
    export_command = _build_export_command(
        python_exe=python_exe,
        sav_path=target_sav,
        out_h5=target_h5,
    )

    _progress(f"[dry-run] {step.entry_stage} -> {step.target_stage}")
    _progress(f"[dry-run] resume log: {resume_log}")
    _progress(f"[dry-run] resume command: {shlex.join(resume_command)}")
    _progress(f"[dry-run] export log: {export_log}")
    _progress(f"[dry-run] export command: {shlex.join(export_command)}")
    _progress(f"[dry-run] compare target: {target_h5}")
    _progress(f"[dry-run] compare datasets: {', '.join(step.compare_paths)}")
    _progress(f"[dry-run] expected produced suffixes: {', '.join(step.target_suffixes)}")
    if step.naming_note:
        _progress(f"[dry-run] naming note: {step.naming_note}")

    return {
        "entry_stage": step.entry_stage,
        "target_stage": step.target_stage,
        "entry_sav": str(entry_sav),
        "target_sav": str(target_sav),
        "entry_box": str(entry_sav),
        "entry_box_format": "sav",
        "planned_resume_log": str(resume_log),
        "planned_resume_command": resume_command,
        "planned_export_log": str(export_log),
        "planned_export_command": export_command,
        "planned_exported_target_h5": str(target_h5),
        "planned_produced_target_suffixes": list(step.target_suffixes),
        "compare_paths": list(step.compare_paths),
        "naming_note": step.naming_note,
    }


def main() -> int:
    args = _parse_args()

    artifact_root = args.artifact_root.expanduser().resolve()
    prior_defaults = _load_existing_report_defaults(artifact_root)
    idl_stage_dir_arg = args.idl_stage_dir.expanduser()
    data_dir_arg = args.data_dir.expanduser()
    idl_stage_dir = prior_defaults.get("idl_stage_dir", idl_stage_dir_arg).resolve()
    data_dir = prior_defaults.get("data_dir", data_dir_arg).resolve()
    if _flag_explicit_on_cli("--idl-stage-dir"):
        idl_stage_dir = idl_stage_dir_arg.resolve()
    if _flag_explicit_on_cli("--data-dir"):
        data_dir = data_dir_arg.resolve()
    python_exe = args.python.expanduser().resolve()
    selected_stage = _normalize_stage_selector(args.stage)

    if not EXPORT_MODEL_SCRIPT.exists():
        raise FileNotFoundError(f"export_model.py not found: {EXPORT_MODEL_SCRIPT}")
    if not GX_FOV2BOX_SCRIPT.exists():
        raise FileNotFoundError(f"gx_fov2box.py not found: {GX_FOV2BOX_SCRIPT}")
    if not idl_stage_dir.exists():
        raise FileNotFoundError(f"IDL stage directory not found: {idl_stage_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    _progress(f"[start] artifact root: {artifact_root}")
    if prior_defaults and not _flag_explicit_on_cli("--idl-stage-dir") and "idl_stage_dir" in prior_defaults:
        _progress(f"[start] reusing IDL stage dir from existing report: {idl_stage_dir}")
    if prior_defaults and not _flag_explicit_on_cli("--data-dir") and "data_dir" in prior_defaults:
        _progress(f"[start] reusing data dir from existing report: {data_dir}")
    _progress(f"[start] IDL stage dir: {idl_stage_dir}")
    _progress(f"[start] data dir: {data_dir}")
    if args.dry_run:
        _progress("[start] dry run enabled; commands will be reported but not executed")
    if args.report_only:
        _progress("[start] report-only mode enabled; existing artifacts will be reused")
    if selected_stage:
        _progress(f"[start] selected transition: {selected_stage}")
    if args.clean:
        _progress("[start] cleaning previous artifacts")
    if args.dry_run and args.report_only:
        raise ValueError("--dry-run and --report-only are mutually exclusive")
    if args.clean and args.report_only:
        raise ValueError("--clean cannot be used with --report-only")
    if args.clean and selected_stage:
        raise ValueError("--clean cannot be used with --stage; a single-stage rerun reuses the other stage artifacts for the regenerated report")

    export_dir = artifact_root / "idl_exported"
    pyampp_dir = artifact_root / "pyampp_generated"
    log_dir = artifact_root / "logs"
    report_dir = artifact_root / "reports"
    if args.report_only:
        _require_existing_path(artifact_root, label="artifact root")
        _require_existing_path(export_dir, label="export directory")
        _require_existing_path(pyampp_dir, label="pyAMPP output directory")
        report_dir.mkdir(parents=True, exist_ok=True)
    elif not args.dry_run:
        _prepare_artifact_root(artifact_root, clean=args.clean)
        export_dir.mkdir(parents=True, exist_ok=True)
        pyampp_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

    stage_files = _discover_idl_stage_files(idl_stage_dir)
    _progress("[discover] found IDL stages: " + ", ".join(STAGE_ORDER))
    selected_steps = _selected_steps(selected_stage)
    if not selected_steps:
        raise ValueError(f"No stage step matches selector: {selected_stage}")
    exported_stage_h5: dict[str, Path] = {}

    report: dict[str, Any] = {
        "idl_stage_dir": str(idl_stage_dir),
        "data_dir": str(data_dir),
        "artifact_root": str(artifact_root),
        "python": str(python_exe),
        "entry_box_mode": "sav",
        "comparison_mode": "exported_h5",
        "dry_run": bool(args.dry_run),
        "report_only": bool(args.report_only),
        "stage_selector": selected_stage,
        "rtol": args.rtol,
        "atol": args.atol,
        "stage_files": {stage: str(path) for stage, path in stage_files.items()},
        "exported_stage_h5": {},
        "steps": [],
    }

    stage_inputs: dict[str, tuple[Path, Path]] = {}
    for step in STAGE_STEPS:
        entry_sav = _pick_idl_stage_file(
            idl_stage_dir,
            step.entry_sav_suffixes,
            label=f"{step.entry_stage} entry",
        )
        target_sav = _pick_idl_stage_file(
            idl_stage_dir,
            step.target_sav_suffixes,
            label=f"{step.target_stage} target",
        )
        stage_inputs[f"{step.entry_stage}->{step.target_stage}"] = (entry_sav, target_sav)

    if args.dry_run:
        for step in selected_steps:
            entry_sav, target_sav = stage_inputs[f"{step.entry_stage}->{step.target_stage}"]
            branch_output_root = pyampp_dir / f"from_{step.entry_stage.lower()}"
            report["steps"].append(
                _report_dry_run_step(
                    python_exe=python_exe,
                    entry_sav=entry_sav,
                    target_sav=target_sav,
                    data_dir=data_dir,
                    branch_output_root=branch_output_root,
                    export_dir=export_dir,
                    log_dir=log_dir,
                    step=step,
                )
            )
        _progress("[dry-run] complete")
        return 0

    if not args.report_only:
        for step in selected_steps:
            entry_sav, target_sav = stage_inputs[f"{step.entry_stage}->{step.target_stage}"]
            branch_output_root = pyampp_dir / f"from_{step.entry_stage.lower()}"
            if selected_stage is not None and branch_output_root.exists():
                shutil.rmtree(branch_output_root)
            branch_output_root.mkdir(parents=True, exist_ok=True)
            produced_target = _run_stage_resume_step(
                python_exe=python_exe,
                entry_model=entry_sav,
                data_dir=data_dir,
                branch_output_root=branch_output_root,
                step=step,
                log_path=log_dir / f"resume_{step.entry_stage.lower()}_to_{step.target_stage.lower()}.log",
            )
            target_h5 = _ensure_exported_stage(
                python_exe=python_exe,
                label=step.target_stage,
                sav_path=target_sav,
                export_dir=export_dir,
                log_dir=log_dir,
                exported_stage_h5=exported_stage_h5,
            )
            _progress(
                f"[rerun] completed {step.entry_stage} -> {step.target_stage}: "
                f"{target_h5.name} vs {produced_target.name}"
            )

    for step in STAGE_STEPS:
        entry_sav, target_sav = stage_inputs[f"{step.entry_stage}->{step.target_stage}"]
        branch_output_root = pyampp_dir / f"from_{step.entry_stage.lower()}"
        branch_output_root = _require_existing_path(
            branch_output_root,
            label=f"{step.entry_stage} branch root",
        )
        produced_target = _pick_existing_file(
            branch_output_root,
            step.target_suffixes,
            label=f"{step.entry_stage} -> {step.target_stage} produced target",
        )
        target_h5 = _require_existing_path(
            _planned_export_path(export_dir=export_dir, sav_path=target_sav),
            label=f"{step.target_stage} exported target",
        )
        exported_stage_h5[str(target_sav.resolve())] = target_h5
        report_prefix = "report-only" if args.report_only else ("compare" if selected_stage is None else "report")
        report["steps"].append(
            _collect_stage_report_entry(
                step=step,
                entry_sav=entry_sav,
                target_sav=target_sav,
                exported_target=target_h5,
                produced_target=produced_target,
                rtol=args.rtol,
                atol=args.atol,
                progress_prefix=report_prefix,
            )
        )

    report["exported_stage_h5"] = {
        key: str(path) for key, path in exported_stage_h5.items()
    }

    report_path = report_dir / "gx_idl2py_stage_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print(f"Exported IDL stage H5 files: {export_dir}")
    print(f"pyAMPP generated stage H5 files: {pyampp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
