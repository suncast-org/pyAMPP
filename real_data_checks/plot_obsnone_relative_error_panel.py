#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a side-by-side PNG panel of OBS->NONE base-field relative error maps "
            "for selected reprojection methods."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ranking-json", type=Path, required=True, help="Ranking JSON produced by rank_obsnone_reproject_scan.py")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["adaptive", "exact", "interpolation"],
        help="Methods/options to include in panel",
    )
    p.add_argument("--out-png", type=Path, required=True, help="Output PNG path")
    p.add_argument(
        "--eps",
        type=float,
        default=1.0e-6,
        help="Small denominator floor used for relative error",
    )
    p.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.0,
        help="Shared color scale upper bound percentile across all displayed maps",
    )
    return p.parse_args()


def _load_components(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        bx = np.asarray(f["base/bx"], dtype=np.float64)
        by = np.asarray(f["base/by"], dtype=np.float64)
        bz = np.asarray(f["base/bz"], dtype=np.float64)
    return bx, by, bz


def _relative_error_map(
    ref_bx: np.ndarray,
    ref_by: np.ndarray,
    ref_bz: np.ndarray,
    cand_bx: np.ndarray,
    cand_by: np.ndarray,
    cand_bz: np.ndarray,
    eps: float,
) -> np.ndarray:
    dmag = np.sqrt((cand_bx - ref_bx) ** 2 + (cand_by - ref_by) ** 2 + (cand_bz - ref_bz) ** 2)
    rmag = np.sqrt(ref_bx**2 + ref_by**2 + ref_bz**2)
    return dmag / np.maximum(rmag, eps)


def main() -> int:
    args = _parse_args()
    ranking_json = args.ranking_json.expanduser().resolve()
    out_png = args.out_png.expanduser().resolve()

    payload = json.loads(ranking_json.read_text(encoding="utf-8"))
    reference_h5 = Path(payload["reference_h5"]).expanduser().resolve()

    by_option: dict[str, dict] = {
        item["option"]: item for item in payload.get("ranking", []) if isinstance(item, dict)
    }
    requested = [m for m in args.methods if m in by_option]
    if not requested:
        raise RuntimeError(f"None of requested methods are present in ranking JSON: {args.methods}")

    result_by_option: dict[str, dict] = {
        item["option"]: item for item in payload.get("results", []) if isinstance(item, dict)
    }

    ref_bx, ref_by, ref_bz = _load_components(reference_h5)

    panels: list[dict] = []
    all_values: list[np.ndarray] = []
    for method in requested:
        result = result_by_option.get(method)
        if result is None:
            continue
        cand_h5 = Path(result["candidate_h5"]).expanduser().resolve()
        cand_bx, cand_by, cand_bz = _load_components(cand_h5)
        if cand_bx.shape != ref_bx.shape:
            raise RuntimeError(
                f"Shape mismatch for method={method}: ref={ref_bx.shape}, candidate={cand_bx.shape}"
            )
        rel = _relative_error_map(ref_bx, ref_by, ref_bz, cand_bx, cand_by, cand_bz, args.eps)
        rel = np.asarray(rel, dtype=np.float64)
        finite = np.isfinite(rel)
        if np.any(finite):
            all_values.append(rel[finite])
        panels.append(
            {
                "method": method,
                "rank": by_option[method].get("rank"),
                "closeness": by_option[method].get("closeness_score"),
                "rel_map": rel,
            }
        )

    if not panels:
        raise RuntimeError("No valid methods found to plot")

    if all_values:
        concat = np.concatenate(all_values)
        vmax = float(np.percentile(concat, args.vmax_percentile))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(concat)) if concat.size else 1.0
    else:
        vmax = 1.0
    vmax = max(vmax, 1.0e-8)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.1), constrained_layout=True)
    if n == 1:
        axes = [axes]

    mappable = None
    for ax, panel in zip(axes, panels):
        rel = panel["rel_map"]
        mappable = ax.imshow(rel, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        rank = panel["rank"]
        closeness = panel["closeness"]
        ax.set_title(f"{panel['method']} | rank {rank}")
        ax.set_xlabel(f"closeness={float(closeness):.6f}")
        ax.set_xticks([])
        ax.set_yticks([])

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.9, pad=0.02)
        cbar.set_label("Relative vector error |dB| / |Bref|")

    fig.suptitle("OBS->NONE Base-Field Relative Error Maps by Reprojection Method")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    print(f"Wrote: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
