# Geometry Contract Enforcement: Design and Implementation

## Summary

This feature introduces **Geometry Contract Enforcement** to pyAMPP, a metadata
completion system that eliminates fallback-based coordinate inference in model
geometry. By completing and storing Tier 1 (intrinsic box) and Tier 2 (world
embedding) metadata at model load time, downstream geometry functions (in
`gximagecomputing` and elsewhere) can trust a single, authoritative source of
truth.

The architecture is now centered on a single canonical model I/O module:

- `pyampp.io.model` is the required application-level restore/save path
- model restore from H5 and SAV goes through centralized completion/normalization
- geometry consumers consume completed metadata and should not re-infer it

## Problem Statement

### Current State: Branching Complexity
Models loaded from SAV/H5 files have incomplete or scattered metadata:
- Box dimensions may be implied (from cube shape) or explicit (from metadata)
- Voxel resolution stored in multiple places (corona.dr, chromo.dr, execute text)
- World anchor (lon/lat/radius) derived from index, execute, or defaults
- Observation time inferred from multiple metadata layers

**Result:** `observer_geometry.py` implements three fallback paths for corner computation:
1. Index-based corners
2. Execute-based corners
3. Model cube-based corners

Each path requires separate parsing and validation logic, making the code complex and difficult to audit for correctness.

### Architectural Goal
Enforce a **canonical geometry contract** at load time, so:
- All models have identical Tier 1 (intrinsic box) and Tier 2 (world embedding) metadata
- Geometry module reads completed metadata once, never infers dynamically
- Models saved to H5 cache the completed contract; unsaved models re-complete on reload

## Architecture

### Tier 1: Intrinsic Box Geometry (Mandatory)
Box dimensions and resolution in physical units:
- **Nx, Ny, Nz**: pixels, from coronal cube shape (never chromo)
- **dr_x, dr_y, dr_z**: voxel resolution in units of $R_\odot$
- **Rsun**: solar radius in meters (fixed to HMI value: 6.957×10⁸ m)

**Rationale:** The coronal cube is uniform in all three dimensions. Chromosphere is non-uniform by design and cannot serve as resolution ground truth.

### Tier 2: World Embedding (Fallback to Defaults)
Placement and orientation of the box in solar coordinates:
- **anchor_lon_deg**: box anchor longitude in HeliographicStonyhurst degrees
- **anchor_lat_deg**: box anchor latitude in HeliographicStonyhurst degrees
- **anchor_radius_rsun**: box anchor radius in units of $R_\odot$
- **frame**: coordinate frame identifier
- **obstime**: observation time (ISO timestamp)

**Fallback Priority:**
1. Index-based metadata (WCS headers)

Current implementation policy:
- Tier 2 is inferred from base/index metadata only.
- If Tier 2 fields cannot be inferred, `complete_geometry_contract()` returns `None`
   (or raises in strict mode).

### Storage in H5

Completed contract stored at `model["metadata"]["geometry_contract"]`:
```
metadata/
  ├─ execute: str
  ├─ id: str
  ├─ obstime: str
  ├─ ...
  └─ geometry_contract/
      ├─ nx: int
      ├─ ny: int
      ├─ nz: int
      ├─ dr_x: float
      ├─ dr_y: float
      ├─ dr_z: float
      ├─ rsun_m: float
      ├─ anchor_lon_deg: float
      ├─ anchor_lat_deg: float
      ├─ anchor_radius_rsun: float
      ├─ frame: str
      ├─ obstime: str
      └─ inferred_from: str  (provenance)
```

## Implementation

### Files Modified/Created

1. **pyampp/geometry/contract.py** (new)
   - `GeometryContract`: dataclass for the complete contract
   - `complete_geometry_contract()`: main completion function
   - `world_corners_from_geometry_contract()`: red-box SkyCoord constructor from contract metadata
   - Helper functions for inferring each field
   - Serialization/deserialization methods

2. **pyampp/geometry/core.py**
   - `build_fov_box_from_red_box_world()`: canonical inscribing blue FOV-box constructor
   - `build_fov_box_from_user_hpc_and_red_box_world()`: user HPC rectangle + LOS-safe z constructor
   - `world_to_local_cartesian_mm()`: shared world-to-local conversion helper for viewers

3. **pyampp/geometry/__init__.py**
   - Export contract/core APIs for public use

4. **pyampp/io/model.py** (new)
   - `load_model()`: canonical provenance-agnostic loader with contract enforcement
   - `save_model()`: canonical save path with contract persistence
   - `load_model_metadata()`: canonical metadata/observer inspection helper

5. **pyampp/io/__init__.py** (new)
   - Exposes public model I/O APIs

6. **pyampp/__init__.py**
   - Exposes `io` and `geometry` as public package imports

7. **pyampp/util/build_h5_from_sav.py**
   - `_apply_geometry_contract_to_h5()`: delegation shim to canonical io-model completion API
   - Called at end of `build_h5_from_sav()` before returning
   - Non-fatal: returns False when contract remains incomplete in non-strict mode

8. **pyampp/gxbox/box_view2d.py**
   - Uses geometry-module constructors for blue FOV box generation
   - Recomputes blue-box z extent from red-box geometry when user edits HPC FOV

9. **pyampp/gxbox/box_view3d.py**
   - Uses geometry-module corner reconstruction and local conversion helpers

10. **pyampp/tests/test_geometry_contract.py** (new)
   - Comprehensive tests for contract inference
   - Tests for serialization/deserialization
   - Tests for fallback behavior and strict mode

11. **pyampp/tests/test_geometry_core.py**
   - Tests for new blue-box constructors (auto-inscribing and user-defined HPC)

### Key Functions

#### complete_geometry_contract(model_dict, *, strict=False)
Main entry point for contract completion.
- **Input:** Loaded model dictionary (from H5 or SAV)
- **Output:** `GeometryContract` or None (if Tier 1 incomplete)
- **strict=False:** Return None when required fields are missing
- **strict=True:** Raise ValueError if required fields cannot be inferred

#### world_corners_from_geometry_contract(contract, *, obstime=None)
Build the canonical 8-corner red-box world `SkyCoord` from a completed contract.
- Uses contract dimensions/resolution for extents
- Uses anchor/frame/obstime to place box in solar coordinates

#### build_fov_box_from_red_box_world(world, ...)
Compute the canonical inscribing observer-aligned 3D blue FOV box from red-box world corners.

#### build_fov_box_from_user_hpc_and_red_box_world(world, ...)
Build a user-defined HPC x/y blue FOV box while preserving LOS-safe z extents derived from the red box.

#### infer_box_dims(model_dict)
Extract (Nx, Ny, Nz) from corona cube shape.
- Prefers `corona.bx`, falls back to `corona.by`, `corona.bz`
- Returns None if no suitable field found

#### infer_voxel_resolution(model_dict)
Extract (dr_x, dr_y, dr_z) from corona.dr.
- Assumes uniform resolution in three directions
- Returns None if corona.dr not found

#### infer_world_anchor_from_index(model_dict, obstime)
Parse WCS metadata (CRVAL1/CRVAL2/RSUN_REF from INDEX).
- Returns None if index metadata incomplete

#### infer_world_anchor_defaults(obstime)
Fallback to disk center (lon=0, lat=0, radius=1 R☉).

## Integration Points

### 1. Canonical Model Restore/Save Path

All application-level model restore/save should use `pyampp.io`:

```python
from pyampp import io

model = io.load_model("model.h5")
# or
model = io.load_model("model.sav")

io.save_model(model, "updated_model.h5")
```

Behavior:

- load: complete or reuse `metadata/geometry_contract`, normalize observer metadata
- save: persist completed contract so next load can reuse stored fields

### 2. SAV → H5 Conversion
After `build_h5_from_sav()` writes the H5 file, it calls `_apply_geometry_contract_to_h5()`.

### 3. Geometry and Viewer Consumption
- 2D/3D gxbox viewers consume geometry-module constructors for red/blue box paths.
- Blue box from user HPC selection is now generated by geometry helpers that preserve z safety.

### 4. Model Loading in gximagecomputing

Downstream usage should import directly from public pyAMPP surfaces:

- `from pyampp import io, geometry`
- load via `io.load_model`
- consume geometry contract and geometry constructors from `pyampp.geometry`

## Backward Compatibility

- Existing low-level readers remain available for compatibility/internal usage.
- Old models without persisted contracts still load (completion happens on load).
- New models and upgraded saves persist contracts in H5 for faster/cleaner reload.
- Models not saved to H5 re-complete on each load (acceptable for throwaway work).

## Migration Path

1. **Phase 1 (this PR):** Contract module and SAV→H5 integration ✅
2. **Phase 2 (in progress):** Geometry-module constructors for red/blue boxes and viewer adoption
3. **Phase 3 (follow-up):** Update gximagecomputing to consume contract-built red box directly

## Testing

### Unit Tests
- `test_geometry_contract.py`: Contract inference logic
- Tests for dimension inference, resolution inference, anchor inference
- Tests for serialization/deserialization
- Tests for strict vs. permissive mode
- `test_geometry_core.py`: Blue FOV-box constructors and projection helpers

### Integration Tests (manual, for now)
```bash
# Convert old SAV model
python -m pyampp.util.build_h5_from_sav --sav old.sav --out-h5 new.h5

# Verify contract is stored
h5dump new.h5 | grep geometry_contract
```

## Design Decisions

### Why Only Corona for Resolution?
Chromosphere is non-uniform by construction (varies with height). Only corona.dr is reliable.

### Why No Tier-2 Defaults in Current Contract?
Tier-2 defaults can silently hide provenance and placement errors. Current behavior
requires inferable world metadata for a complete contract.

### Why Store in metadata.geometry_contract?
Separates model data (corona, chromo, base) from metadata/provenance. Clear namespace avoids conflicts with existing fields.

### Why RSUN_HMI_METERS Fixed?
HMI convention (695700 km) is standard across SDO/pyAMPP/sunpy. Models built with different Rsun should re-normalize or accept implicit mismatch.

## Future Work

1. **gximagecomputing Integration:** Consume contract-to-red-box constructor in runtime geometry paths
2. **Contract Validation:** Add post-load validation in geometry functions
3. **Performance:** Cache loaded contracts in memory to avoid re-reading H5

## References

- Related issue: geometry module fallback branching
- Upstream: sunpy conventions for Rsun and coordinate frames
- Contract idea: inspired by data serialization best practices (protobuf, Apache Arrow)
