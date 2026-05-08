from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from pyampp.io import load_geometry_contract_and_observer_from_h5

app = typer.Typer(help="Inspect only geometry_contract and observer metadata from an HDF5 model.")


def _observer_summary(observer: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(observer, dict):
        return None
    ephemeris = observer.get("ephemeris") if isinstance(observer.get("ephemeris"), dict) else {}
    return {
        "name": observer.get("name"),
        "label": observer.get("label"),
        "source": observer.get("source"),
        "has_ephemeris": bool(ephemeris),
        "ephemeris_keys": sorted(ephemeris.keys()),
    }


@app.command()
def main(
    path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    json_output: bool = typer.Option(False, "--json", help="Print JSON output."),
    require_contract: bool = typer.Option(
        False,
        "--require-contract",
        help="Exit non-zero if geometry_contract is missing.",
    ),
) -> None:
    """Inspect thin model metadata (geometry_contract + observer)."""
    thin = load_geometry_contract_and_observer_from_h5(path)
    if thin is None:
        payload = {
            "path": str(path),
            "has_geometry_contract": False,
            "observer": None,
        }
        if json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"{path}")
            print("geometry_contract: missing")
            print("observer: skipped (no thin model payload)")
        if require_contract:
            raise typer.Exit(code=2)
        raise typer.Exit(code=0)

    contract = thin["metadata"]["geometry_contract"]
    observer_info = _observer_summary(thin.get("observer"))
    payload = {
        "path": str(path),
        "has_geometry_contract": True,
        "geometry_contract": {
            "nx": int(contract.nx),
            "ny": int(contract.ny),
            "nz": int(contract.nz),
            "dr_x": float(contract.dr_x),
            "dr_y": float(contract.dr_y),
            "dr_z": float(contract.dr_z),
            "frame": str(contract.frame),
            "obstime": str(contract.obstime),
            "inferred_from": contract.inferred_from,
        },
        "observer": observer_info,
    }
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise typer.Exit(code=0)

    print(f"{path}")
    print("geometry_contract: present")
    print(
        "dims: "
        f"{payload['geometry_contract']['nx']}x"
        f"{payload['geometry_contract']['ny']}x"
        f"{payload['geometry_contract']['nz']}"
    )
    print(
        "dr_rsun: "
        f"({payload['geometry_contract']['dr_x']}, "
        f"{payload['geometry_contract']['dr_y']}, "
        f"{payload['geometry_contract']['dr_z']})"
    )
    print(f"frame: {payload['geometry_contract']['frame']}")
    print(f"obstime: {payload['geometry_contract']['obstime']}")
    print(f"inferred_from: {payload['geometry_contract']['inferred_from']}")

    if observer_info is None:
        print("observer: absent")
    else:
        print("observer: present")
        print(f"observer_name: {observer_info['name']}")
        print(f"observer_label: {observer_info['label']}")
        print(f"observer_source: {observer_info['source']}")
        print(f"observer_has_ephemeris: {observer_info['has_ephemeris']}")
        if observer_info["ephemeris_keys"]:
            print(f"observer_ephemeris_keys: {', '.join(observer_info['ephemeris_keys'])}")


if __name__ == "__main__":
    app()
