import numpy as np

from pyampp.gxbox.gxampp import (
    _contains_entry_field_payload,
    _has_template_execute_metadata,
    _split_process_output_text,
)


def test_split_process_output_text_handles_partial_line_boundaries():
    complete_lines, partial = _split_process_output_text("partial", " line\nnext")

    assert complete_lines == ["partial line"]
    assert partial == "next"


def test_split_process_output_text_normalizes_crlf_and_flushes_complete_tail():
    complete_lines, partial = _split_process_output_text("", "alpha\r\nbeta\rgamma\n")

    assert complete_lines == ["alpha", "beta", "gamma"]
    assert partial == ""


def test_contains_entry_field_payload_accepts_corona_bx():
    model = {
        "corona": {
            "bx": np.zeros((2, 2, 2), dtype=np.float32),
        }
    }
    assert _contains_entry_field_payload(model) is True


def test_contains_entry_field_payload_rejects_metadata_only_thin():
    model = {
        "metadata": {
            "id": "thin_only",
        },
        "observer": {
            "name": "earth",
        },
    }
    assert _contains_entry_field_payload(model) is False


def test_has_template_execute_metadata_accepts_nonempty_execute():
    model = {
        "metadata": {
            "execute": "gx_fov2box --time 2020-01-01T00:00:00 --hpc",
        }
    }
    assert _has_template_execute_metadata(model) is True


def test_has_template_execute_metadata_rejects_missing_or_empty_execute():
    assert _has_template_execute_metadata({"metadata": {}}) is False
    assert _has_template_execute_metadata({"metadata": {"execute": "   "}}) is False