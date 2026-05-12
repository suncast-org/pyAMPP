from __future__ import annotations

from pathlib import Path

import typer

from pyampp.io import export_thin_model

app = typer.Typer(help="Export metadata-only thin HDF5 (full metadata + optional observer) from any supported model file.")


@app.command()
def main(
    source_model: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    output_h5: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for the thin file (defaults to <source_stem>_metadata.h5).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail if geometry contract cannot be completed during source restore.",
    ),
) -> None:
    """Create a portable metadata-only thin model HDF5."""
    out = export_thin_model(source_model, output_h5=output_h5, strict=strict)
    print(out)


if __name__ == "__main__":
    app()