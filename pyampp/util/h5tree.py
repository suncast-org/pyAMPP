from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import typer

from pyampp.io import load_model, save_model
from pyampp.io.model import _prepare_model_for_h5_write

app = typer.Typer(help="Print a tree view of an HDF5 file.")


def _format_label(path: Path, *, normalized_view: bool) -> str:
    if not normalized_view:
        return str(path)
    return f"{path} (normalized view)"


def _encode_preview_value(value: Any, *, key: str | None = None, target_type: str | None = None) -> Any:
    if target_type in ("chromo", "lines") and key == "voxel_status":
        return np.asarray(value, dtype=np.uint8)
    if isinstance(value, (str, bytes, np.bytes_)):
        return np.bytes_(value)
    if value is None:
        return np.bytes_("")

    arr = np.asarray(value)
    if arr.dtype.kind != "O":
        return value

    if arr.shape == ():
        scalar = arr.item()
        if isinstance(scalar, (str, bytes, np.bytes_)):
            return np.bytes_(scalar)
        if scalar is None:
            return np.bytes_("")
        try:
            return np.asarray(scalar)
        except Exception:
            return np.bytes_(str(scalar))

    as_text = np.vectorize(lambda x: "" if x is None else str(x), otypes=[str])(arr)
    return as_text.astype("S")


def _preview_shape_dtype(value: Any, *, key: str | None = None, target_type: str | None = None) -> tuple[tuple[int, ...], np.dtype[Any]]:
    encoded = _encode_preview_value(value, key=key, target_type=target_type)
    arr = np.asarray(encoded)
    return arr.shape, arr.dtype


def _print_metadata_object(prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        print(f"{prefix}/")
        for key, child in value.items():
            if key == "attrs":
                continue
            _print_metadata_object(f"{prefix}/{key}", child)
        return

    shape, dtype = _preview_shape_dtype(value)
    if shape == ():
        scalar = _encode_preview_value(value)
        if isinstance(scalar, (bytes, bytearray, np.bytes_)):
            scalar = _decode_scalar(scalar)
        elif isinstance(scalar, np.ndarray) and scalar.shape == ():
            scalar = scalar.item()
        print(f"{prefix}: {scalar}")
        return

    print(f"{prefix}: <dataset shape={shape} dtype={dtype}>")


def _print_metadata_values_from_model(metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        if key == "attrs":
            continue
        _print_metadata_object(f"metadata/{key}", value)


def _print_observer_summary_from_model(model: dict[str, Any]) -> None:
    observer = model.get("observer")
    if not isinstance(observer, dict):
        return
    if "name" in observer:
        print(f"observer/name: {_decode_scalar(_encode_preview_value(observer['name']))}")
    if "label" in observer:
        print(f"observer/label: {_decode_scalar(_encode_preview_value(observer['label']))}")
    if "source" in observer:
        print(f"observer/source: {_decode_scalar(_encode_preview_value(observer['source']))}")
    pb0r = observer.get("pb0r")
    if not isinstance(pb0r, dict):
        return
    for key, value in pb0r.items():
        if key == "attrs":
            continue
        print(f"observer/pb0r/{key}: {_decode_scalar(_encode_preview_value(value))}")


def _print_group_from_model(
    group: dict[str, Any],
    prefix: str,
    show_attrs: bool,
    max_attr_len: int | None,
    max_depth: int | None,
    current_depth: int,
    flt: Optional[str],
    base_path: str,
    target_type: str | None,
) -> None:
    if max_depth is not None and current_depth > max_depth:
        return
    keys = [key for key in group.keys() if key != "attrs"]
    for idx, name in enumerate(keys):
        is_last = idx == len(keys) - 1
        branch = "└── " if is_last else "├── "
        child = group[name]
        full_path = f"{base_path}/{name}"
        if isinstance(child, dict):
            attrs = child.get("attrs", {}) if show_attrs and isinstance(child.get("attrs"), dict) else {}
            attr_text = _format_attrs(attrs, max_attr_len)
            if _matches_filter(full_path, name, flt):
                print(f"{prefix}{branch}{name}/{attr_text}")
            extension = "    " if is_last else "│   "
            _print_group_from_model(
                child,
                prefix + extension,
                show_attrs,
                max_attr_len,
                max_depth,
                current_depth + 1,
                flt,
                full_path,
                target_type,
            )
            continue

        shape, dtype = _preview_shape_dtype(child, key=name, target_type=target_type)
        attrs = {} if not show_attrs else {}
        attr_text = _format_attrs(attrs, max_attr_len)
        if _matches_filter(full_path, name, flt):
            print(f"{prefix}{branch}{name} {shape} {dtype}{attr_text}")


def _print_h5_tree(
    path: Path,
    *,
    label: str,
    show_attrs: bool,
    max_attr_len: int | None,
    max_depth: int | None,
    flt: Optional[str],
    no_metadata: bool,
    meta_only: bool,
) -> None:
    with h5py.File(path, "r") as h5f:
        if not meta_only:
            print(label)
            _print_group(h5f, "", show_attrs, max_attr_len, max_depth, 0, flt, "")
        if (not no_metadata) and "metadata" in h5f:
            _print_metadata_values(h5f["metadata"])
        if (not no_metadata) and (not meta_only):
            _print_observer_summary(h5f)


def _print_model_tree(
    path: Path,
    model: dict[str, Any],
    *,
    show_attrs: bool,
    max_attr_len: int | None,
    max_depth: int | None,
    flt: Optional[str],
    no_metadata: bool,
    meta_only: bool,
) -> None:
    payload = _prepare_model_for_h5_write(model)
    if not meta_only:
        print(_format_label(path, normalized_view=True))
        _print_group_from_model(
            payload,
            "",
            show_attrs,
            max_attr_len,
            max_depth,
            0,
            flt,
            "",
            None,
        )
    metadata = payload.get("metadata")
    if (not no_metadata) and isinstance(metadata, dict):
        _print_metadata_values_from_model(metadata)
    if (not no_metadata) and (not meta_only):
        _print_observer_summary_from_model(payload)


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
    save_normalized: Path | None = typer.Option(
        None,
        "--save-normalized",
        help="For SAV input, also write the normalized canonical HDF5 to this path.",
    ),
) -> None:
    """Print a tree of groups/datasets with shapes and dtypes."""
    if path is None:
        print(ctx.get_help())
        raise typer.Exit(code=0)
    suffix = path.suffix.lower()
    if suffix == ".h5":
        _print_h5_tree(
            path,
            label=_format_label(path, normalized_view=False),
            show_attrs=show_attrs,
            max_attr_len=max_attr_len,
            max_depth=max_depth,
            flt=flt,
            no_metadata=no_metadata,
            meta_only=meta_only,
        )
        return

    if suffix != ".sav":
        raise typer.BadParameter(
            f"Unsupported input format for {path}; expected .h5 or .sav file"
        )

    model = load_model(path)
    if save_normalized is not None:
        save_model(model, save_normalized)
        _print_h5_tree(
            save_normalized,
            label=_format_label(save_normalized, normalized_view=False),
            show_attrs=show_attrs,
            max_attr_len=max_attr_len,
            max_depth=max_depth,
            flt=flt,
            no_metadata=no_metadata,
            meta_only=meta_only,
        )
        return

    _print_model_tree(
        path,
        model,
        show_attrs=show_attrs,
        max_attr_len=max_attr_len,
        max_depth=max_depth,
        flt=flt,
        no_metadata=no_metadata,
        meta_only=meta_only,
    )


if __name__ == "__main__":
    app()
