#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_COMPARE_PATHS: tuple[str, ...] = (
    "base/bx",
    "base/by",
    "base/bz",
    "base/ic",
)

CORONA_Z0_COMPARE_PATHS: tuple[str, ...] = (
    "corona/bx[0]",
    "corona/by[0]",
    "corona/bz[0]",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank OBS->NONE reprojection sweep outputs against an IDL reference H5. "
            "Writes JSON and Markdown summaries with a closeness ranking."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scan-root", type=Path, required=True, help="Root directory containing reproject_* outputs.")
    parser.add_argument("--reference-h5", type=Path, required=True, help="IDL-exported NONE reference H5 file.")
    parser.add_argument(
        "--glob",
        type=str,
        default="reproject_*",
        help="Directory glob under --scan-root used to discover reprojection option folders.",
    )
    parser.add_argument(
        "--target-suffix",
        type=str,
        default=".NONE.h5",
        help="File suffix used to locate produced target files in each option folder.",
    )
    parser.add_argument(
        "--include-corona-z0",
        action="store_true",
        help="Also compare z=0 corona slices (corona/bx[0], by[0], bz[0]).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <scan-root>/reproject_ranking_obsnone.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output Markdown table path. Defaults to <scan-root>/reproject_ranking_obsnone.md",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    return parser.parse_args()


def _read_dataset(path: Path, dataset_path: str) -> np.ndarray | None:
    import re

    match = re.fullmatch(r"(?P<base>[^\[]+)\[(?P<index>\d+)\]", dataset_path)
    base_path = dataset_path
    slice_index: int | None = None
    if match is not None:
        base_path = match.group("base")
        slice_index = int(match.group("index"))

    with h5py.File(path, "r") as handle:
        if base_path not in handle:
            return None
        data = np.asarray(handle[base_path])
    if slice_index is None:
        return data
    if data.ndim == 0 or slice_index >= data.shape[0]:
        return None
    return np.asarray(data[slice_index])


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


def _compute_stage_closeness(comparison: dict[str, Any]) -> dict[str, Any]:
    dataset_errors: dict[str, float] = {}
    worst_relative_error = 0.0
    for dataset_name, metrics in comparison.get("datasets", {}).items():
        if not isinstance(metrics, dict):
            continue
        if not metrics.get("shape_match", True):
            error_value = 1.0
        else:
            rel_mae = metrics.get("rel_mae")
            if rel_mae is None:
                error_value = 0.0 if metrics.get("allclose", False) else 1.0
            else:
                error_value = max(0.0, min(1.0, float(rel_mae)))
        dataset_errors[dataset_name] = error_value
        worst_relative_error = max(worst_relative_error, error_value)

    mean_relative_error = (
        float(sum(dataset_errors.values()) / len(dataset_errors))
        if dataset_errors else 0.0
    )
    exact_mismatch_penalty = 0.0
    composite_error = (
        0.50 * mean_relative_error
        + 0.35 * worst_relative_error
        + 0.15 * exact_mismatch_penalty
    )
    return {
        "dataset_errors": dataset_errors,
        "mean_relative_error": mean_relative_error,
        "worst_relative_error": worst_relative_error,
        "exact_mismatch_penalty": exact_mismatch_penalty,
        "composite_error": composite_error,
        "closeness_score": 1.0 - composite_error,
    }


def _compare_candidate(
    *,
    reference_h5: Path,
    candidate_h5: Path,
    compare_paths: tuple[str, ...],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "reference_h5": str(reference_h5),
        "candidate_h5": str(candidate_h5),
        "datasets": {},
        "missing": {"reference": [], "candidate": []},
    }

    for dataset_path in compare_paths:
        ref_arr = _read_dataset(reference_h5, dataset_path)
        can_arr = _read_dataset(candidate_h5, dataset_path)
        if ref_arr is None:
            metrics["missing"]["reference"].append(dataset_path)
            continue
        if can_arr is None:
            metrics["missing"]["candidate"].append(dataset_path)
            continue
        metrics["datasets"][dataset_path] = _compare_arrays(ref_arr, can_arr, rtol=rtol, atol=atol)

    metrics["missing"]["reference"].sort()
    metrics["missing"]["candidate"].sort()
    metrics["allclose"] = all(
        entry.get("allclose", False)
        for entry in metrics["datasets"].values()
    ) and not metrics["missing"]["reference"] and not metrics["missing"]["candidate"]
    metrics["stage_closeness"] = _compute_stage_closeness(metrics)
    return metrics


def _discover_candidates(scan_root: Path, dir_glob: str, target_suffix: str) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for option_dir in sorted(scan_root.glob(dir_glob)):
        if not option_dir.is_dir():
            continue
        matches = sorted(option_dir.rglob(f"*{target_suffix}"))
        if len(matches) != 1:
            continue
        option_name = option_dir.name
        if option_name.startswith("reproject_"):
            option_name = option_name[len("reproject_"):]
        candidates.append((option_name, matches[0]))
    return candidates


def _to_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| rank | option | closeness_score | composite_error | mean_relative_error | worst_relative_error |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {option} | {score:.6f} | {comp:.6f} | {mean:.6f} | {worst:.6f} |".format(
                rank=row["rank"],
                option=row["option"],
                score=float(row["closeness_score"]),
                comp=float(row["composite_error"]),
                mean=float(row["mean_relative_error"]),
                worst=float(row["worst_relative_error"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    scan_root = args.scan_root.expanduser().resolve()
    reference_h5 = args.reference_h5.expanduser().resolve()
    out_json = (args.out_json.expanduser().resolve() if args.out_json else scan_root / "reproject_ranking_obsnone.json")
    out_md = (args.out_md.expanduser().resolve() if args.out_md else scan_root / "reproject_ranking_obsnone.md")

    if not scan_root.exists():
        raise FileNotFoundError(f"Scan root does not exist: {scan_root}")
    if not reference_h5.exists():
        raise FileNotFoundError(f"Reference H5 does not exist: {reference_h5}")

    candidates = _discover_candidates(scan_root, args.glob, args.target_suffix)
    if not candidates:
        raise RuntimeError(
            f"No candidates found under {scan_root} with dir glob {args.glob!r} and target suffix {args.target_suffix!r}."
        )

    compare_paths = DEFAULT_COMPARE_PATHS
    if args.include_corona_z0:
        compare_paths = CORONA_Z0_COMPARE_PATHS + DEFAULT_COMPARE_PATHS

    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for option, candidate_h5 in candidates:
        comparison = _compare_candidate(
            reference_h5=reference_h5,
            candidate_h5=candidate_h5,
            compare_paths=compare_paths,
            rtol=args.rtol,
            atol=args.atol,
        )
        closeness = comparison["stage_closeness"]
        row = {
            "option": option,
            "candidate_h5": str(candidate_h5),
            "closeness_score": float(closeness["closeness_score"]),
            "composite_error": float(closeness["composite_error"]),
            "mean_relative_error": float(closeness["mean_relative_error"]),
            "worst_relative_error": float(closeness["worst_relative_error"]),
            "exact_mismatch_penalty": float(closeness["exact_mismatch_penalty"]),
        }
        rows.append(row)
        results.append(
            {
                "option": option,
                "candidate_h5": str(candidate_h5),
                "comparison": comparison,
            }
        )

    rows.sort(key=lambda item: (-float(item["closeness_score"]), item["option"]))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    payload = {
        "scan_root": str(scan_root),
        "reference_h5": str(reference_h5),
        "compare_paths": list(compare_paths),
        "rtol": args.rtol,
        "atol": args.atol,
        "ranking": rows,
        "results": results,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown_table(rows), encoding="utf-8")

    print(f"Wrote JSON: {out_json}")
    print(f"Wrote Markdown: {out_md}")
    if rows:
        best = rows[0]
        print(
            "Best option: "
            f"{best['option']} (closeness_score={best['closeness_score']:.6f}, rank=1)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())