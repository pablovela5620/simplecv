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

## Environment Mesh Revisit (October 17, 2025)

- **Loader wiring:** Added an `EnvironmentMesh` adapter to `RRDSequence` so `/world/gt/env_mesh` is decoded on ingest. It fetches vertex positions + triangles, promotes optional normals, and unpacks packed `uint32` RGBA colors into `uint8[*,4]`.
- **Viewer relogging:** `visualize_exo_ego` now replays the mesh (guarded by a `log_env_mesh` toggle) before blueprint setup so the static environment is visible alongside cameras.
- **Validation:** Running `pixi run -e dev view-exoego-data rrd --rrd-path /mnt/8tb/data/exoego-self-collected/gus/17600630913N_staticRandomCupStack-annotated.rrd` now relogs the mesh without warnings. Directly instantiating `RRDSequence` confirms a non-empty mesh with `vertex_positions.shape == (136188, 3)` and the first vertex `[0.13119504, 1.27617, -0.45922568]`, matching the viewer screenshot.
- **Follow-ups:** Once we confirm with live data, expand the doc with a note on expected timeline (`video_time` vs. static) and add a regression task in `pixi` to smoke-test mesh extraction.
We can revisit mesh logging (and any colour decoding) after those two fundamentals are solid.
