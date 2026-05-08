from pyampp.util.config import SAMPLE_DATA_DIR, DOWNLOAD_DIR
from .gx_box2id import gx_box2id
from .gx_voxelid import gx_voxelid
from .selector_api import (
    BoxGeometrySelection,
    CoordMode,
    DisplayFovSelection,
    GeometrySelectionConsumer,
    SelectorDialogResult,
    SelectorSessionInput,
)

_LAZY_EXPORTS = [
    'MapBoxViewState',
    'MapBoxDisplayWidget',
    'FovBoxSelectorDialog',
    'run_fov_box_selector',
    'gxbox_select_main',
]

__all__ = [
    'SAMPLE_DATA_DIR',
    'DOWNLOAD_DIR',
    'gx_voxelid',
    'gx_box2id',
    'CoordMode',
    'BoxGeometrySelection',
    'DisplayFovSelection',
    'SelectorDialogResult',
    'SelectorSessionInput',
    'GeometrySelectionConsumer',
]


def __getattr__(name):
    if name in {'MapBoxViewState', 'MapBoxDisplayWidget'}:
        from .box_view2d import MapBoxDisplayWidget, MapBoxViewState
        exports = {
            'MapBoxViewState': MapBoxViewState,
            'MapBoxDisplayWidget': MapBoxDisplayWidget,
        }
        return exports[name]
    if name in {'FovBoxSelectorDialog', 'run_fov_box_selector'}:
        from .fov_selector_gui import FovBoxSelectorDialog, run_fov_box_selector
        exports = {
            'FovBoxSelectorDialog': FovBoxSelectorDialog,
            'run_fov_box_selector': run_fov_box_selector,
        }
        return exports[name]
    if name == 'gxbox_select_main':
        from .gxbox_selector_view import main as gxbox_select_main
        return gxbox_select_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_LAZY_EXPORTS))
