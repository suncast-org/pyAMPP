from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py
import typer

app = typer.Typer(help="Print a tree view of an HDF5 file.")


def _format_attrs(attrs: dict[str, Any], max_len: int | None) -> str:
    if not attrs:
        return ""
    parts = []
    for key, value in attrs.items():
        text = f"{key}={value!r}"
        if max_len is not None and len(text) > max_len:
            text = text[: max_len - 3] + "..."
        parts.append(text)
    return " {" + ", ".join(parts) + "}"


def _matches_filter(full_path: str, name: str, flt: Optional[str]) -> bool:
    if not flt:
        return True
    flt = flt.lower()
    return flt in full_path.lower() or flt in name.lower()


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return value


def _print_metadata_node(prefix: str, node: h5py.Group | h5py.Dataset) -> None:
    if isinstance(node, h5py.Group):
        print(f"{prefix}/")
        for key in node.keys():
            _print_metadata_node(f"{prefix}/{key}", node[key])
        return

    if node.shape == ():
        val = _decode_scalar(node[()])
        print(f"{prefix}: {val}")
        return

    print(f"{prefix}: <dataset shape={node.shape} dtype={node.dtype}>")


def _print_metadata_values(meta: h5py.Group) -> None:
    for key in meta.keys():
        _print_metadata_node(f"metadata/{key}", meta[key])


def _print_observer_summary(h5f: h5py.File) -> None:
    observer = h5f.get("observer")
    if not isinstance(observer, h5py.Group):
        return
    if "name" in observer:
        print(f"observer/name: {_decode_scalar(observer['name'][()])}")
    if "label" in observer:
        print(f"observer/label: {_decode_scalar(observer['label'][()])}")
    if "source" in observer:
        print(f"observer/source: {_decode_scalar(observer['source'][()])}")
    pb0r = observer.get("pb0r")
    if not isinstance(pb0r, h5py.Group):
        return
    for key in pb0r.keys():
        print(f"observer/pb0r/{key}: {_decode_scalar(pb0r[key][()])}")


def _print_group(
    group: h5py.Group,
    prefix: str,
    show_attrs: bool,
    max_attr_len: int | None,
    max_depth: int | None,
    current_depth: int,
    flt: Optional[str],
    base_path: str,
) -> None:
    if max_depth is not None and current_depth > max_depth:
        return
    keys = list(group.keys())
    for idx, name in enumerate(keys):
        is_last = idx == len(keys) - 1
        branch = "└── " if is_last else "├── "
        child = group[name]
        full_path = f"{base_path}/{name}"
        if isinstance(child, h5py.Dataset):
            shape = child.shape
            dtype = child.dtype
            attrs = dict(child.attrs) if show_attrs else {}
            attr_text = _format_attrs(attrs, max_attr_len)
            if _matches_filter(full_path, name, flt):
                print(f"{prefix}{branch}{name} {shape} {dtype}{attr_text}")
        else:
            attrs = dict(child.attrs) if show_attrs else {}
            attr_text = _format_attrs(attrs, max_attr_len)
            if _matches_filter(full_path, name, flt):
                print(f"{prefix}{branch}{name}/{attr_text}")
            extension = "    " if is_last else "│   "
            _print_group(
                child,
                prefix + extension,
                show_attrs,
                max_attr_len,
                max_depth,
                current_depth + 1,
                flt,
                full_path,
            )


@app.command()
def main(
    ctx: typer.Context,
    path: Optional[Path] = typer.Argument(None, exists=True, file_okay=True, dir_okay=False, readable=True),
    show_attrs: bool = typer.Option(False, "--attrs", help="Show dataset/group attributes."),
    max_attr_len: int | None = typer.Option(120, "--attr-max", help="Max length for each attribute entry."),
    max_depth: int | None = typer.Option(None, "--max-depth", help="Limit recursion depth."),
    flt: Optional[str] = typer.Option(None, "--filter", help="Only show paths matching this string."),
    no_metadata: bool = typer.Option(False, "--no-metadata", help="Do not print metadata/* values."),
    meta_only: bool = typer.Option(False, "--meta", help="Print only metadata/* values (no tree)."),
) -> None:
    """Print a tree of groups/datasets with shapes and dtypes."""
    if path is None:
        print(ctx.get_help())
        raise typer.Exit(code=0)
    with h5py.File(path, "r") as h5f:
        if not meta_only:
            print(f"{path}")
            _print_group(h5f, "", show_attrs, max_attr_len, max_depth, 0, flt, "")
        if (not no_metadata) and "metadata" in h5f:
            _print_metadata_values(h5f["metadata"])
        if (not no_metadata) and (not meta_only):
            _print_observer_summary(h5f)


if __name__ == "__main__":
    app()
