# Metric Calibration Benchmark (RRD, 2025‑11‑25 session-new.rrd)

**Run date:** 2025-12-10  
**Command:** `pixi run python ./tools/run_metric_calib.py --rr-config.headless --calib-ts-nano 195000000000 rrd --rrd-path /mnt/8tb/data/exoego-self-collected/quest+oak+exo/2025-11-25/2ebc849c-80e6-4952-af80-0725e900d1eb/session-new.rrd`  
**Env:** default pixi (beartype disabled). Rerun viewer suppressed via `--rr-config.headless`.

## Wall-clock & resources
- Baseline (before fast path): **3m 40.9s**; user CPU: **2987.3s**; system: **74.4s**; CPU util: **1385%** (~13–14 cores).  
- After time-aligned projection fast path: **1m 48.3s**; user CPU: **1121.1s**; system: **31.2s**; CPU util: **1063%** (~10–11 cores). Max RSS unchanged (~11.6 GB).
- Cached Recording + time-aligned everywhere (current): **31.2s**; user CPU: **134.1s**; system: **8.4s**; CPU util: **456%** (~5 cores). Profile at `/tmp/metric_calib_profile_cached.pstats`.
- Relative speedup: **~7× faster wall clock** vs baseline (220.9s → 31.2s).
- Max RSS: **11.6 GB**.  
- I/O: ~5.3 GB writes, ~61 MB reads (per `/usr/bin/time -v`).  
- Warnings: four exo cams skipped due to missing metadata; onnxruntime fell back to CPU (`CUDAExecutionProvider` unavailable); timm import deprecation.

## Hotspots (cProfile, full run; profile saved at `/tmp/metric_calib_profile.pstats`)
| Rank | Function (file:line) | Cum. time | Calls | Notes |
| --- | --- | --- | --- | --- |
| 1 | `project_brown_conrady_grid` (brown_conrady.py:298) | **175.1s** | 1189 | Core projector + distortion; dominates runtime. |
| 2 | `log_exoego_batch` (view_exoego.py:398) | 111.3s | 1 | Columnar logging of 3D/2D labels; internally calls projector. |
| 3 | `world_to_cam_batched` (base_camera.py:9) | 72.6s | 1189 | Part of projection pipeline. |
| 4 | `cam_to_image_batched` (brown_conrady.py:17) | 60.8s | 1189 | Pixel-space projection. |
| 5 | `apply_brown_conrady_distortion_batch` (brown_conrady.py:182) | 33.7s | 883 | Lens distortion stage. |
| 6 | `load_recording` (rerun_bindings) | 16.8s | 5 | Multiple opens of the 2.7 GB RRD. |
| 7 | Wilor hand keypoint forward (`hand_keypoints.py:244` → `wilor.py`) | 8.9s | 4 | ONNXRuntime CPU (CUDA provider missing). |
| 8 | `torch.conv_transpose2d` | 5.9s | 12 | From Wilor refinement net. |

Observed user-facing phases:  
- Hand detection/keypoints on 5 ego cams: ~8–9s.  
- Reproject GT into each ego cam (5 cams × 147 frames): ~70s total.  
- Remaining time spent in bulk label logging (3D + UV) and projector kernels.

## Bottlenecks & root causes
1) **Projection kernel overhead (major)**: Repeated Brown–Conrady projection for ~14.6k frames across 5 cams with 133 keypoints each. Previously, a “frames == views” pattern in `log_reprojected_gt_uv` triggered an F×F outer-product inside `world_to_cam_batched` before selecting the diagonal (O(F²) work). The new time-aligned fast path eliminates that for ego reprojection, cutting total runtime from 3m41s → 1m48s.  
2) **Multiple recording loads (≈17s)**: `RRDExoEgoSequence` opens the same RRD several times (exo, ego, labels).  
3) **Hand model on CPU**: ONNXRuntime could not use CUDA, so Wilor runs on CPU (≈9s per run, small but avoidable).  
4) **Large logging volume**: Columnar send of 133-keypoint tracks for every frame/cam drives both projection compute and ~5 GB of writes.  
5) **Missing exo cam metadata**: Skips exo projections; not a speed issue but signals upstream data gap.

## Recommendations (prioritized)
1) **Reduce projection volume**  
   - Add a CLI option to decimate label frames during logging/reprojection (e.g., every Nth frame or time range around the calib timestamp).  
   - Allow disabling `log_reprojected_gt_uv` when not needed; it alone accounts for ~70s.
2) **Fuse/accelerate projection math**  
   - Detect the “frames == views” case and run an O(F) path (`xyz_cam = einsum('fij,fkj->fki', cam_T_world, xyz_h)`) instead of the current outer-product/diagonal pattern; synthetic benchmarks show ~60× speedup for `world_to_cam_batched` in this scenario and would eliminate most of the 175 s hotspot.  
   - Increase batch size in `log_reprojected_gt_uv` and `log_exoego_batch` (currently 100) to amortize Python overhead; 11.6 GB headroom suggests 500–1000 is safe.  
   - JIT the Brown–Conrady pipeline (Numba or Cython) or port to torch/JAX to leverage SIMD/GPU; `project_brown_conrady_grid` + `world_to_cam_batched` are the primary targets.  
   - Precompute per-camera distortion maps and use `cv2.remap`-style lookup for dense points, or cache `K` and inverse once per camera per batch.
3) **Single recording load**  
   - Thread a shared `Recording` object through `RRDExoEgoConfig` so exo, ego, and labels reuse one load; should shave ~15–17s and reduce I/O.  
4) **Enable GPU for Wilor**  
   - Install `onnxruntime-gpu` and ensure CUDA driver is visible; will drop the hand keypoint stage from ~9s to sub-second on GPU.  
5) **Logging scope control**  
   - Provide flags to skip ego/exo UV logging or to log only a subset of cameras; this directly cuts projection compute and Rerun write volume.

## How to inspect the profile
- `snakeviz /tmp/metric_calib_profile.pstats` or `gprof2dot -f pstats /tmp/metric_calib_profile.pstats | dot -Tpdf -o profile.pdf`.

## Extra notes on distortion cost
- Per-call benchmarking of `apply_brown_conrady_distortion_batch` on full-sequence-sized data (14,613 frames × 5 views × 133 kpts) is ~0.48 s; the 33 s cum in profile arises because the function is invoked ~1.2k times on tiny batches (driven by the same outer-product/diagonal pattern above).  
- The current distortion path loops over views; a fully vectorized version without tilt clocks the same runtime (~0.48 s) on the full batch, so the loop itself is not the primary culprit. The real hit is repeated small-batch calls.  
- Tilt terms (tau_x, tau_y) trigger per-call trig; when zero (common), we can guard those branches and skip the extra math. Providing a flag to bypass distortion for logging-only scenarios would cut ~20% of the projection time when distortion isn’t needed.

## Grid vs diagonal (what changed and why it matters)
- **Grid (`project_brown_conrady_grid`)**: Designed for many views of the *same frame*. It builds a `[n_views, 4, 4]` stack and projects all frames against all views (outer product), then you pick what you need. Great for multi-view triangulation; bad when `n_views == n_frames` because you pay O(F²) before taking the diagonal.
- **Diagonal (`project_brown_conrady_diagonal`)**: One camera pose per frame. Does an O(F) einsum (`cam_T_world @ xyz_h`) and projects each frame only through its matching pinhole—no outer product, no diagonal pick. This is now used for ego reprojection.
- Practical impact: Ego reprojection stopped calling the projector ~1.2k times on tiny slices and now runs once per camera over all frames. Wall clock dropped ~50% versus baseline; tqdm “4/5 cams” now reflects per-camera work, not batching overhead.

## Current bottlenecks (after fast path)
Top cumulative times from `/tmp/metric_calib_profile_fastpath.pstats`:
1) `project_brown_conrady_grid` — **65.5s** (442 calls). Still used in `log_exoego_batch` for ego GT logging with batch_size=100, triggering many small calls. Fix: swap this site to `project_brown_conrady_diagonal` (one call per cam) or raise batch size to full sequence.
2) `world_to_cam_batched` / `cam_to_image_batched` — **~48s combined**; same root cause as above (hundreds of small batched calls).
3) `apply_brown_conrady_distortion_batch` — **15.6s**; will shrink once #1 is addressed.
4) `load_recording` — **17.0s** across 5 loads; share a single `Recording` instance across exo/ego/labels to save ~15–17s.
5) Wilor hand keypoints (ONNXRuntime CPU) — **~10s**; enabling `onnxruntime-gpu` or switching provider to CUDA would remove most of this.

Next quick win: change the ego GT logging (`log_exoego_batch`) to use the time-aligned projector; expected to drop ~60–70s from the remaining ~111s profile.

## View CLI benchmark (`tools/view_exoego.py`)
- Command: `pixi run python tools/view_exoego.py rrd --rrd-path ... --rr-config.headless true`
- Wall: **14.85s** via /usr/bin/time; tool reports “Total time taken: 5.13s” (wall includes process startup).
- Profile: `/tmp/view_exoego_profile.pstats` (13.15s cum).
- Top hotspots (cumtime):
  1) `base_config.setup` (RRDSequence init) — 7.97s (includes single recording load).
  2) `visualize_exo_ego` — 5.21s total.
  3) `log_exoego_batch` — 3.69s (columnar logging).
  4) `project_brown_conrady_diagonal` — 2.44s cum (5 cams, one call each).
  5) `_distort_normalized_points` — 1.33s cum.
  6) Remaining time is split across camera param construction and video asset logging; exo projections are skipped due to missing metadata.
  - Hands/Wilor are not invoked in this CLI path.

Breakdown of `base_config.setup` (7.97s):
- `load_recording` (once) — ~3.24s.
- `load_ego_cams` — ~3.21s total:
  - `compute_transformation_matrices` inside `PinholeParameters.__post_init__`: ~1.38s across **76,110 calls** (5 cams × 15,222 frames), each doing an `np.linalg.inv` + compose.
  - `_load_extrinsics_series`: 0.89s (stacking 4×4s per cam across all frames).
  - `_load_distortion`: 0.45s (Brown–Conrady coeffs per cam).
  - `_load_intrinsics`: 0.02s (negligible).
- Asset video extraction/logging (`log_video` / `AssetVideo`): ~1.09s.
- Remaining glue: <0.2s.

Ideas to trim setup:
- Short-circuit `compute_transformation_matrices` when `cam_T_world` / `world_T_cam` are already stored in the RRD; skip the per-frame inverse (~1.4s on this clip, more on longer ones).
- Vectorize inversion: compute both transforms over the full `[n_frames, 4, 4]` stack once per cam (5 calls) instead of per-frame `__post_init__` (76k calls).
- Lazy-init ego `PinholeParameters` (build on first use) or allow frame decimation in the viewer so unused frames never instantiate camera params.
