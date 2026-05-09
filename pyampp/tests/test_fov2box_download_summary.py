from __future__ import annotations

from astropy.time import Time

from pyampp.gxbox import gx_fov2box


class _FakeDownloader:
    def __init__(self, *args, **kwargs):
        self.existence_report = {}

    def download_images(self):
        return {
            "field": "/fake/field.fits",
            "inclination": "/fake/inclination.fits",
            "azimuth": "/fake/azimuth.fits",
            "disambig": "",
            "continuum": "/fake/continuum.fits",
            "magnetogram": "",
        }


def test_load_hmi_maps_prints_required_and_optional_missing_summary(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(gx_fov2box, "SDOImageDownloader", _FakeDownloader)

    maps, info = gx_fov2box._load_hmi_maps_from_downloader(
        Time("2025-11-26T15:47:52"),
        tmp_path,
        euv=True,
        uv=True,
        strict_required=False,
    )

    out = capsys.readouterr().out

    assert maps == {}
    assert "Download summary:" in out
    assert "required missing" in out
    assert "optional missing" in out
    assert "disambig" in out
    assert "magnetogram" in out
    assert "94" in out
    assert "1600" in out
    assert "missing_required" in info
    assert "missing_optional" in info
