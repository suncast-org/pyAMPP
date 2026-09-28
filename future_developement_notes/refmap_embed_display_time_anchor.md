# Refmap embed + display time anchor — implementation plan

Branch: `feat/stereo-refmap-display`.

`00058cd` is the historical pre-plan checkpoint (before embed wiring). The display rules below are the current branch behaviour: observer LOS canvas, native-grid base overlays.

## Goals

1. Embed reference maps via `refmap_crop.py` (spatial crop only; no model-time alignment at embed).
2. Anchor gxbox-view2d display geometry, overlays, and FOV export to the **selected context refmap observation time**.
3. Remove redundant **SDO** display-LOS selector (Earth covers SDO/AIA/HMI).
4. Show **OBS time + Δt vs model** on the matplotlib figure banner (persists on save).
5. Drop refmap-vs-model time **warnings** (user responsibility to pick nearby refmaps).

**Out of scope:** model field synthesis, NLFFF/base grid construction, chromospheric cubes — unchanged.

---

## Phase 1 — Display time anchor + UI (completed)

### 1.1 Time anchor policy

- **Display obstime** = `DATE-OBS` of the **selected context refmap** (fallback: model time if map has no date).
- Observer ephemeris for display reprojection, red/blue overlays, and FOV export use this anchor.
- Model geometry (world box corners) is projected at the display anchor (rigid rotation assumption).

### 1.2 Figure banner

Extend existing `fig.text` banner (top-left) with compact text:

```text
OBS 2026-04-03T19:46:37.800 (Δt=+2.3h)
```

Δt formatting (shortest readable unit):

| |Δt| | Format |
|-----|--------|
| &lt; 0.05 s | `Δt=0` |
| &lt; 90 s | `Δt=+12s` |
| &lt; 90 min | `Δt=+2.3min` or `+45min` |
| else | `Δt=+2.3h` |

Signed delta: refmap time − model time.

### 1.3 GUI status line

Optional duplicate of anchor time in status text (`display_time=…`, `Δt=…`); primary export-facing label stays on `fig.text`.

### 1.4 Remove SDO display-LOS option

- Drop `("sdo", "SDO")` from `_DISPLAY_OBSERVER_OPTIONS`.
- Map persisted `display_observer_key="sdo"` → `"earth"` in `_normalize_observer_key`.
- Remove SDO-only ephemeris fallback in `_resolve_display_observer_metadata`.
- Keep `TELESCOP=SDO/AIA` in map headers unchanged.

### 1.5 Remove refmap time-skew warnings

- Remove `_warn_if_refmap_model_time_skew` calls and related status notices.

### 1.6 Cache keys

Include display obstime ISO in `_display_prepared_cache_key` (and related tokens).
Invalidate display cache on **context refmap** change (`set_context_map_id`).

### 1.7 Tests

- Unit tests for `_format_time_delta_short` and `_format_display_time_banner`.
- Adjust any tests that assumed SDO in display observer options.

---

## Phase 2 — Refmap embed via `refmap_crop.py` (completed)

### 2.1 Unified embed pipeline

Replace model-time reprojection in `pyampp/io/refmaps.py`:

- **`build_refmap_payload_for_model`**: call `crop_refmap_spatial()` with `box_corners_world` from model geometry.
- Output: cropped `data` + `wcs_header` at **native map observer/time**.
- Set `PYALIGN=False` (or document new semantics: native embed, not model-aligned).
- Preserve audit headers: `SRC_DATE`, `MODELT`, `PYEMBED`.

### 2.2 Wire entry points

- `add_fits_refmaps_to_h5` / `build_fits_refmaps_for_model`
- `gxbox_selector_view._embed_external_refmaps_into_box_data`
- `gx_fov2box` external refmap collection (replace `target_fov` reproject with crop-from-box-corners)

### 2.3 Requirements

- Model HDF5 must expose red-box world corners (or geometry to compute them) at embed time.
- Spacecraft native maps: `pangle_policy="auto"` in `crop_refmap_spatial`.
- No intermediate cropped FITS required; in-memory `refmaps/` dict → H5 on save.

### 2.4 Tests

- Extend `test_io_refmaps.py`: embedded maps keep native `DATE-OBS`, not forced to `MODELT`.
- End-to-end: embed AIA + STEREO refmap, reload H5, gxbox display at context anchor.

---

## Phase 3 — Display reprojection alignment (completed)

The selector canvas is the **display observer's helioprojective line of sight**. Every context map and the model boxes are drawn on that frame. The Carrington `base/index` header is stored model geometry (CEA). It is not the context axes and not the empty-canvas WCS.

With native-time embeds:

- **Native LOS** (helioprojective map whose observer matches the display observer): show the embedded crop; overlays at the display anchor.
- **HMI context products** (magnetogram, continuum, field, inclination, azimuth, disambig): rotate for display (`rotate(order=3)`). Do not resample them onto `base/index`. Observer LOS reprojection, when a map is not already helioprojective, is `_reproject_map_for_display_observer`.
- **Cross-observer helioprojective maps**: `reproject_refmap_to_observer` / `reproject_map_to_target_observer_fov` at **display anchor time** with `mask_off_limb=True` via `mask_pixels_not_visible_from_source`.
- **Model-grid base overlays** (Carrington CEA and any other identified non-helioprojective base map): leave the pixels on their **native WCS** after display scaling. `plot(..., autoalign=True)` warps that grid onto the observer LOS context axes. Do not send them through `display_observer_reproject_header_for_fov`: that path reads plate scale as helioprojective arcsec per pixel, so a CEA degree-per-pixel scale collapses the overlay onto a 32-pixel canvas and tears it into quadrants. Helioprojective bottom maps still crop or reproject in the display observer frame.
- **Geometry scaffold (Context = none)**: empty canvas is an observer helioprojective map at model time (`make_empty_observer_fov_map`, or the geometry header when that header is already helioprojective and the observers share a line of sight). A Carrington / CEA header is rejected as the drawing frame (`_is_known_non_los_map`). Model boxes follow those axes, so they stay in the observer LOS.

Cache: `_display_prepared_cache_key` includes `(context_id, display_obstime, display_observer, view_mode, purpose)`.

API exports: `reproject_refmap_to_observer`, `reproject_map_to_target_observer_fov`, `mask_pixels_not_visible_from_source`, `make_empty_observer_fov_map`.

---

## Phase 4 — FOV export for synthesis (completed)

- `MapBoxDisplayWidget.exportable_fov_selection()` — blue FOV in display-observer HPC at display time anchor.
- `MapBoxDisplayWidget.exportable_fov_box_selection()` — 3D FOV box at anchor with `observer_key = display_observer_key`.
- Selector **Save / Apply** uses exportable FOV/box (not raw form values in definition frame).
- Persisted `observer.fov` includes `obstime` = display anchor; ephemeris resolution prefers `display_fov_obstime` from observer persistence state.

### 4.1 Session vs save policy

- **Display observer switch** reprojects maps/overlays only; it does **not** recompute or overwrite `_state.fov`, `fov_box`, or `fov_definition_observer_key`. Users may inspect another LOS and switch back without losing the prior FOV.
- **Save / Apply** uses exportable FOV/box (display observer + anchor). If incompatible, a dialog offers: recompute for current display observer, save without FOV, or cancel. No implicit recompute on observer change.

---

## Rollback

Historical only. `00058cd` is the checkpoint from before embed wiring and before the observer-LOS display rules. Resetting there drops the current canvas behaviour.

```bash
git checkout feat/stereo-refmap-display
git reset --hard 00058cd   # pre-plan checkpoint; not the current display rules
```

---

## Open decisions (resolved)

| Question | Decision |
|----------|----------|
| Display time anchor | Selected context refmap `DATE-OBS` |
| Δt display | On `fig.text` banner + status |
| SDO LOS selector | Remove |
| Map vs model Δt warnings | Remove |
| Model-time display for cache efficiency | Rejected — correctness over cache hits |
| Context canvas | Display observer helioprojective LOS. Carrington `base/index` is model geometry, not the drawing frame |
| Model-grid base overlay | Stay on native WCS; `autoalign` onto the observer LOS context |
