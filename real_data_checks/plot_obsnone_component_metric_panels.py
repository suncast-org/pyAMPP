#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("bx", "by", "bz")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build component-wise OBS->NONE parity panels for selected reprojection methods, "
            "including relative residual and symmetric normalized difference metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ranking-json", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=["adaptive", "exact", "interpolation"])
    p.add_argument("--out-relative-png", type=Path, required=True)
    p.add_argument("--out-symdiff-png", type=Path, required=True)
    p.add_argument("--eps", type=float, default=1.0e-6)
    p.add_argument(
        "--vmax-relative-percentile",
        type=float,
        default=99.0,
        help="Shared vmax percentile for absolute relative residual maps",
    )
    p.add_argument(
        "--vlim-symdiff",
        type=float,
        default=1.0,
        help="Symmetric display limit for normalized difference maps",
    )
    return p.parse_args()


def _load_base_components(h5_path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for comp in COMPONENTS:
            out[comp] = np.asarray(f[f"base/{comp}"], dtype=np.float64)
    return out


def _abs_relative_residual(ref: np.ndarray, cand: np.ndarray, eps: float) -> np.ndarray:
    return np.abs(cand - ref) / np.maximum(np.abs(ref), eps)


def _sym_normalized_difference(ref: np.ndarray, cand: np.ndarray, eps: float) -> np.ndarray:
    # Guaranteed in [-1, 1] with absolute denominator (up to epsilon regularization).
    return (ref - cand) / (np.abs(ref) + np.abs(cand) + eps)


def _collect_method_payload(payload: dict, methods: list[str]) -> list[dict]:
    rank_by_option = {
        item["option"]: item for item in payload.get("ranking", []) if isinstance(item, dict)
    }
    result_by_option = {
        item["option"]: item for item in payload.get("results", []) if isinstance(item, dict)
    }

    selected: list[dict] = []
    for method in methods:
        if method not in rank_by_option or method not in result_by_option:
            continue
        selected.append(
            {
                "method": method,
                "rank": rank_by_option[method].get("rank"),
                "closeness": rank_by_option[method].get("closeness_score"),
                "candidate_h5": Path(result_by_option[method]["candidate_h5"]).expanduser().resolve(),
            }
        )
    return selected


def _plot_relative_panel(
    methods: list[dict],
    ref_components: dict[str, np.ndarray],
    eps: float,
    vmax_percentile: float,
    out_png: Path,
) -> None:
    maps: dict[tuple[int, str], np.ndarray] = {}
    all_vals: list[np.ndarray] = []
    for i, entry in enumerate(methods):
        cand = _load_base_components(entry["candidate_h5"])
        for comp in COMPONENTS:
            m = _abs_relative_residual(ref_components[comp], cand[comp], eps)
            maps[(i, comp)] = m
            finite = np.isfinite(m)
            if np.any(finite):
                all_vals.append(m[finite])

    if all_vals:
        vmax = float(np.percentile(np.concatenate(all_vals), vmax_percentile))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    else:
        vmax = 1.0

    rows = len(methods)
    cols = len(COMPONENTS)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])

    mappable = None
    for i, entry in enumerate(methods):
        for j, comp in enumerate(COMPONENTS):
            ax = axes[i, j]
            im = ax.imshow(maps[(i, comp)], origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
            mappable = im
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{entry['method']} | {comp} | rank {entry['rank']}")
            if j == 0:
                ax.set_ylabel(f"closeness={float(entry['closeness']):.6f}")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.92, pad=0.02)
        cbar.set_label("abs relative residual |Py - IDL| / max(|IDL|, eps)")

    fig.suptitle("OBS->NONE Base Components: Absolute Relative Residual")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_symdiff_panel(
    methods: list[dict],
    ref_components: dict[str, np.ndarray],
    eps: float,
    vlim: float,
    out_png: Path,
) -> None:
    rows = len(methods)
    cols = len(COMPONENTS)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])

    mappable = None
    for i, entry in enumerate(methods):
        cand = _load_base_components(entry["candidate_h5"])
        for j, comp in enumerate(COMPONENTS):
            ax = axes[i, j]
            m = _sym_normalized_difference(ref_components[comp], cand[comp], eps)
            m = np.clip(m, -1.0, 1.0)
            im = ax.imshow(m, origin="lower", cmap="coolwarm", vmin=-abs(vlim), vmax=abs(vlim))
            mappable = im
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{entry['method']} | {comp} | rank {entry['rank']}")
            if j == 0:
                ax.set_ylabel(f"closeness={float(entry['closeness']):.6f}")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.92, pad=0.02)
        cbar.set_label("sym diff (IDL - Py) / (|IDL| + |Py| + eps)")

    fig.suptitle("OBS->NONE Base Components: Symmetric Normalized Difference")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.ranking_json.expanduser().resolve().read_text(encoding="utf-8"))
    reference_h5 = Path(payload["reference_h5"]).expanduser().resolve()
    methods = _collect_method_payload(payload, args.methods)
    if not methods:
        raise RuntimeError(f"No requested methods found in ranking JSON: {args.methods}")

    ref_components = _load_base_components(reference_h5)

    _plot_relative_panel(
        methods=methods,
        ref_components=ref_components,
        eps=args.eps,
        vmax_percentile=args.vmax_relative_percentile,
        out_png=args.out_relative_png.expanduser().resolve(),
    )
    _plot_symdiff_panel(
        methods=methods,
        ref_components=ref_components,
        eps=args.eps,
        vlim=args.vlim_symdiff,
        out_png=args.out_symdiff_png.expanduser().resolve(),
    )

    print(f"Wrote: {args.out_relative_png.expanduser().resolve()}")
    print(f"Wrote: {args.out_symdiff_png.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
