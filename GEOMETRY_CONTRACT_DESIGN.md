# Geometry Contract Enforcement: Design and Implementation

## Summary

This feature introduces **Geometry Contract Enforcement** to pyAMPP, a metadata completion system that eliminates fallback-based coordinate inference in model geometry. By completing and storing Tier 1 (intrinsic box) and Tier 2 (world embedding) metadata at model load time, downstream geometry functions (in `gximagecomputing` and elsewhere) can trust a single, authoritative source of truth.

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
2. Execute-based metadata (parsed command line)
3. Defaults (disk center at 1 $R_\odot$, obstime="2020-01-01T00:00:00")

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
   - Helper functions for inferring each field
   - Serialization/deserialization methods

2. **pyampp/geometry/__init__.py**
   - Export contract API for public use

3. **pyampp/util/build_h5_from_sav.py**
   - `_apply_geometry_contract_to_h5()`: completion step after SAV→H5 conversion
   - Called at end of `build_h5_from_sav()` before returning
   - Non-fatal: skips silently if any step fails

4. **pyampp/tests/test_geometry_contract.py** (new)
   - Comprehensive tests for contract inference
   - Tests for serialization/deserialization
   - Tests for fallback behavior and strict mode

### Key Functions

#### complete_geometry_contract(model_dict, *, strict=False)
Main entry point for contract completion.
- **Input:** Loaded model dictionary (from H5 or SAV)
- **Output:** `GeometryContract` or None (if Tier 1 incomplete)
- **strict=False:** Use defaults for Tier 2, return None if Tier 1 incomplete
- **strict=True:** Raise ValueError if any field cannot be inferred

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

### 1. SAV → H5 Conversion
After `build_h5_from_sav()` writes the H5 file, it calls `_apply_geometry_contract_to_h5()`:
```python
def build_h5_from_sav(...):
    # ... existing conversion logic ...
    _apply_geometry_contract_to_h5(out_h5)  # NEW
    return out_h5
```

### 2. Model Loading in gximagecomputing
(Future work, not part of this PR)
- Add optional function to read completed contract from H5
- Pass contract to geometry functions instead of inferring

### 3. gxbox-view2d
(Future work, not part of this PR)
- Use completed contract for box restoration
- Eliminate fallback branches in observer_geometry.py

## Backward Compatibility

- **No breaking changes:** Contract completion is optional and non-fatal
- Old models without contracts still work (completion happens on load)
- New models automatically cache contracts in H5 (benefit on re-load)
- Models not saved to H5 re-complete on each load (no performance benefit, but correct)

## Migration Path

1. **Phase 1 (this PR):** Contract module and SAV→H5 integration ✅
2. **Phase 2 (follow-up):** Update gximagecomputing geometry to read contracts
3. **Phase 3 (follow-up):** Enforce strict mode in new pipeline code

## Testing

### Unit Tests
- `test_geometry_contract.py`: Contract inference logic
- Tests for dimension inference, resolution inference, anchor inference
- Tests for serialization/deserialization
- Tests for strict vs. permissive mode

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

### Why Fallback to Defaults for Tier 2?
Some models lack explicit world metadata. Defaulting to disk center is a reasonable conservative choice that allows geometry to continue functioning.

### Why Store in metadata.geometry_contract?
Separates model data (corona, chromo, base) from metadata/provenance. Clear namespace avoids conflicts with existing fields.

### Why RSUN_HMI_METERS Fixed?
HMI convention (695700 km) is standard across SDO/pyAMPP/sunpy. Models built with different Rsun should re-normalize or accept implicit mismatch.

## Future Work

1. **EXECUTE Parsing:** Full integration of execute-based anchor inference (currently stubbed)
2. **Contract Validation:** Add post-load validation in geometry functions
3. **Strict Pipeline Mode:** Opt-in mode that rejects models without valid contracts
4. **Performance:** Cache loaded contracts in memory to avoid re-reading H5

## References

- Related issue: geometry module fallback branching
- Upstream: sunpy conventions for Rsun and coordinate frames
- Contract idea: inspired by data serialization best practices (protobuf, Apache Arrow)
