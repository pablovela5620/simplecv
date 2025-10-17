# ExoEgo RRD Loader Notes (October 2025)

## Key Findings So Far

- **RRD contents:** `/world/gt/coco133_xyz` exists with positions + confidences; `/world/gt/env_mesh` also present with vertex colors encoded as packed `uint32` RGBA.
- **Ego cameras:** The recording logs three ego streams (`left`, `rgb`, `right`), but only the `rgb` camera includes calibration metadata (pinhole + transform). `left`/`right` are monochrome feeds without intrinsics, so the current loader must ignore them when building `PinholeParameters`.
- **Viewer failure mode:** `RRDEgoSequence.align_cams_and_videos` previously returned three video names even when only one camera was calibrated, provoking `AssertionError: Mismatched ego video assets (1) and names (3)` inside `setup_scene`.
- **Ground-truth logging:** `log_exoego_batch` expects an `ExoEgoLabels` struct containing the COCO stack. If `load_labels()` fails to populate `xyzc_stack`, nothing is sent to `/world/gt/coco133_xyz`.
- **Mesh experiments:** Attempting to pipe `/world/gt/env_mesh` through `ExoEgoLabels` exposed timestamp unit issues (nanoseconds vs. seconds) and colour channel decoding pitfalls. We rolled those changes back for now.

## Agreed Reset

1. **Undo mesh integration:** treat the environment mesh as out-of-scope until the base sequence works end-to-end.
2. **Restore timestamp behaviour:** revert to the original timeline handling so we only touch logic required for ego/coco fixes.

## Next Steps

1. **Stabilise ego calibration handling** ✅
   - Filtered the ego streams to keep only calibrated cameras when building `PinholeParameters`.
   - Viewer now opens with the single `rgb` ego video and no longer asserts.
2. **Make sure COCO-133 3D logging fires** ✅
   - Corrected the loader to read `/world/gt/coco133_xyz`; every frame now carries valid keypoints.
3. **Regression test** ✅
   - `pixi run -e dev python tools/view_exoego.py rrd --rrd-path <rrd>` completes successfully (only expected metadata warnings remain).

## TODO

1. Keep mono ego feeds visible once calibration lands (current warning reminds us videos are remuxed but not logged).
2. Revisit environment mesh ingestion once ego + 3D GT are solid.

We can revisit mesh logging (and any colour decoding) after those two fundamentals are solid.
