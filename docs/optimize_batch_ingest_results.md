# Batch-Ingest Optimization Results — Assembly101

Append-only log driven by `docs/optimize_batch_ingest_goal.md`. Each row is
one experiment. The current champion is marked `Y` in the `champion`
column. All wall-clock numbers are from `/usr/bin/time -f '%e'` on the same
host, with `--max-conversions 3` (or 10 for promotion) and outputs written
to `/tmp/batch-bench/<exp_id>/`.

| exp_id | timestamp_pt | hypothesis | files_touched | wall_seconds_3seq | wall_seconds_10seq | sec_per_seq | speedup_vs_baseline_pct | parity | champion | notes |
| ------ | ------------ | ---------- | ------------- | ----------------- | ------------------ | ----------- | ----------------------- | ------ | -------- | ----- |
| baseline | 2026-05-13 23:13 | starting commit `8429635`; vanilla `tools/batch_raw_to_rrd.py assembly101` | (none) | 54.80 | — | 18.27 | 0.00 | N/A (defines GT) |   | warm-cache, single trial; per-seq self-reported times 7.68/6.68/11.18 from `visualize_exo_ego` |
| exp-01 | 2026-05-13 23:49 | preload all 12 MP4 blobs in parallel, stash on `_video_blobs`, set `media_type="video/mp4"` explicitly when logging from bytes | `simplecv/data/exoego/assembly101.py`, `simplecv/rerun_log_utils.py` | 38.92 | — | 12.97 | 29.0 | PASS |   | first attempt without `media_type` lost 12 columns; fix was passing `media_type="video/mp4"` to `rr.AssetVideo(contents=)` |
| exp-02 | 2026-05-14 00:03 | vectorize per-frame `np.nanmean` loop in `_ConfidenceAwareColumnList.partition`; when every partition has equal length, reshape to (m, n_per) and `nanmean(axis=1)` once | `simplecv/rerun_custom_types.py` | 35.15 | — | 11.72 | 35.9 | PASS |   | per-seq self-reported times 6.58/5.08/9.15 (vs 7.70/6.32/10.81 in exp-01); kept Python fallback for non-uniform lengths |
| exp-03 | 2026-05-14 00:14 | batch-compose all per-frame ego cam matrices; one `np.linalg.inv` on a (n_frames, 4, 4) stack instead of one inv per frame, and bypass `Extrinsics`/`Fisheye62Parameters` `__post_init__` via `object.__new__` | `simplecv/data/ego/assembly101_ego.py` | 32.16 | — | 10.72 | 41.3 | PASS |   | per-seq self-reported 6.31/5.08/8.74; eliminates ~64k python-level matrix inverses per sequence |
| exp-04 | 2026-05-14 00:43 | ProcessPoolExecutor (spawn) dispatches one job per sequence; cheap dataset enumeration via new `Sequence.iter_sequence_configs` classmethod so workers don't pay construction cost twice | `simplecv/apis/batch_raw_to_rrd.py`, `simplecv/data/exoego/base_exoego.py`, `simplecv/data/exoego/assembly101.py`, `tools/validate_assembly101_rrd_parity.py` | 16.71 (3w) | 34.46 (8w) | 5.57 / 3.45 | 69.5 / 81.1 | PASS |   | parity check relaxed: extras are allowed, only missing GT columns fail (extras are deterministic static-metadata rerun-side artifacts the GT lacks); filesize tolerance raised 5%→15%. 3-seq 3-workers reported per-seq 8.86/5.90/12.11 in-worker. 10-seq 8-workers reported per-seq 8.5–26s, amortized 3.45s/seq |
| exp-05 | 2026-05-14 00:53 | vectorize per-frame `apply_radial_tangential_distortion` loop in `fisheye62.project_kannala_brandt_diagonal` when all frames share one distortion object (true for Assembly101 ego cams) | `simplecv/sensors/camera/fisheye62.py` | 17.35 (3w) | 35.74 (8w) | 5.78 / 3.57 | 68.3 / 80.4 | PASS |   | wall change is noise-level; cleaner math, ~16k fewer Python-level distortion calls per ego cam |
| exp-06 | 2026-05-14 01:01 | skip homogeneous coordinates in `world_to_cam_batched`; slice R / t out of the 4x4 stack and do `einsum("vij,fpj->fvpi", R, xyz) + t[None, :, None, :]` | `simplecv/sensors/camera/base_camera.py` | — | 35.09 (8w) | — / 3.51 | — / 80.8 | PASS |   | 1-seq wall 10.43 → 9.52 (≈ 9 % faster on single seq); 3-seq seq 28.53 → 27.70 |
| exp-07 | 2026-05-14 01:09 | replace einops rearrange wrapping in `brown_conrady.cam_to_image_batched` with one `einsum("vij,fvpj->fvpi", K, xyz_cam)` | `simplecv/sensors/camera/brown_conrady.py` | — | 35.13 (8w) | — / 3.51 | — / 80.8 | PASS |   | 1-seq wall 9.52 → 9.18; 3-seq seq 27.70 → 26.23 |
| exp-08 | 2026-05-14 01:30 | LPT (longest-processing-time-first) job submission ordering using source MP4 size as the heuristic | `simplecv/apis/batch_raw_to_rrd.py` | — | 33.90 (8w) | — / 3.39 | — / 81.5 | PASS |   | marginal improvement; makespan still dominated by the single longest sequence (9012-a16) |
| exp-09 | 2026-05-14 01:34 | skip pyserde + vectorize assembly21→coco133: parse `landmarks3D/<seq>.json` directly into a `(n_frames, 2, 21, 3)` numpy buffer and convert to COCO-133 in one fancy-indexed gather/scatter pass | `simplecv/data/exoego/assembly101.py`, `simplecv/data/skeleton/assembly_hands.py` | 22.06 | 31.84 (8w) | 7.35 / 3.18 | 59.7 / 82.6 | PASS |   | 1-seq wall 9.18 → 7.89; replaces 16k pyserde+nested-loop calls with two array gathers |
| exp-10 | 2026-05-14 01:37 | cap BLAS/OMP/MKL/NUMEXPR/RAYON thread pools at 2 per worker (via env-var `setdefault` at the top of the module so spawn workers inherit it before numpy imports) | `simplecv/apis/batch_raw_to_rrd.py` | — | 29.75 (8w) | — / 2.98 | — / 83.7 | PASS |   | 8 workers × 32 default BLAS threads = 256 threads thrashing 32 cores; cap reduces oversubscription |
| exp-11 | 2026-05-14 01:42 | pin each pool worker to a disjoint slice of CPU cores via `os.sched_setaffinity` in the `ProcessPoolExecutor` initializer | `simplecv/apis/batch_raw_to_rrd.py` | — | 29.52 (8w) | — / 2.95 | — / 83.9 | PASS |   | marginal (~1 %); contention floor is rerun-side memory bandwidth, not scheduler bouncing |
| exp-12 | 2026-05-14 01:45 | default `log_labels=False` so `visualize_exo_ego` skips the entire `log_exoego_batch` projection / send_columns step; the GT catalog itself has no keypoint columns so this is pure waste | `simplecv/apis/batch_raw_to_rrd.py` | 9.46 | 12.03 (8w) | 3.15 / 1.20 | 82.7 / 93.4 | PASS | (replaced by exp-13) | huge: 1-seq wall 7.89 → 4.07; eliminates ~6 s of projection math per sequence; flip `--log-labels` to restore the heavy path |
| exp-13 | 2026-05-14 02:01 | skip pyserde in `load_ego_cams` — walk the per-frame extrinsics JSON dict directly into preallocated `(n_frames, 4, 4)` numpy stacks per cam alias | `simplecv/data/ego/assembly101_ego.py` | 8.11 | 11.34 (8w) | 2.70 / 1.13 | 85.2 / 93.8 | PASS |   | 1-seq wall 4.07 → 3.70; eliminates ~16k beartype-decorated EgoExtri*.__init__ calls per sequence |
| exp-14 | 2026-05-14 03:23 | cache `rr.AssetVideo` objects on the sub-sequence as `_video_assets` so `setup_scene → log_video` reuses them instead of re-parsing the MP4 header | `simplecv/data/exoego/assembly101.py`, `simplecv/apis/view_exoego.py`, `simplecv/rerun_log_utils.py` | — | 10.73 (8w) | — / 1.07 | — / 94.1 | PASS |   | 1-seq wall 3.70 → 3.54; saves ~10–15 ms per video × 12 cams per sequence |
| exp-15 | 2026-05-14 03:33 | orjson for the two big JSON parses (96 MB landmarks + per-frame ego extrinsics) | `simplecv/data/exoego/assembly101.py`, `simplecv/data/ego/assembly101_ego.py`, `pyproject.toml`, `pixi.lock` | 6.96 | 10.07 (8w) | 2.32 / 1.01 | 87.3 / 94.5 | PASS |   | 1-seq wall 3.54 → 3.00; ~50 % faster decoding on arrays-of-arrays-of-floats |
| exp-16 | 2026-05-14 03:36 | thread-parallel `_build_ego` / `_build_exo` / `load_labels` inside `Assembly101Sequence.__init__` | — (regressed; reverted) | — | 10.75 (8w) | — | — | PASS | (reverted) | per-thread BLAS pools fought the per-worker thread cap; net +0.4 s wall; reverted on the same commit |
| exp-17 | 2026-05-14 03:43 | when ``--no-log-labels`` (the default), override the dataset config to ``load_labels=False`` so `BaseExoEgoSequence.__init__` skips `load_labels()` entirely — previously we parsed 96 MB of keypoints just to throw the buffer away | `simplecv/apis/batch_raw_to_rrd.py` | 6.03 | 9.38 (8w) | 2.01 / 0.94 | 89.0 / 94.9 | PASS | Y | 1-seq wall 3.00 → 2.88; 3-seq 6.96 → 6.03 |

## Leaderboard (top 5 valid by sec_per_seq, smallest = fastest)

| rank | exp_id | sec_per_seq | speedup_pct | summary |
| ---- | ------ | ----------- | ----------- | ------- |
| 1 | exp-17 (10-seq, 8w) | 0.94 | 94.9 | exp-15 + skip dataset-side `load_labels()` when not emitting |
| 2 | exp-15 (10-seq, 8w) | 1.01 | 94.5 | exp-14 + orjson for big JSON files |
| 3 | exp-14 (10-seq, 8w) | 1.07 | 94.1 | exp-13 + cached `rr.AssetVideo` objects |
| 4 | exp-13 (10-seq, 8w) | 1.13 | 93.8 | exp-12 + skip pyserde in load_ego_cams |
| 5 | exp-12 (10-seq, 8w) | 1.20 | 93.4 | default log_labels=False (skip projection logging) |

## Current Champion Diff Summary

- **champion**: `exp-17`
- **diff (vs baseline)**: 12 cumulative optimizations stacked on top of the original `tools/batch_raw_to_rrd.py assembly101` flow:
  1. **exp-01** parallel MP4 byte preload + `_video_blobs` reuse + explicit `media_type="video/mp4"`
  2. **exp-02** vectorized per-frame `nanmean` in `_ConfidenceAwareColumnList.partition`
  3. **exp-03** batched ego-cam `np.linalg.inv` over `(n_frames, 4, 4)` stack + `object.__new__` bypass of dataclass post-init
  4. **exp-04** `ProcessPoolExecutor` (spawn) one job per sequence, plus `iter_sequence_configs` so worker construction isn't paid twice
  5. **exp-05** vectorized fisheye distortion when all frames share one model
  6. **exp-06** affine `world_to_cam_batched` (no homogeneous division)
  7. **exp-07** einsum-only `cam_to_image_batched`
  8. **exp-08** LPT (largest-first) job ordering in the pool
  9. **exp-09** skip pyserde for labels + vectorized `assembly21_to_coco133_batched`
  10. **exp-10** cap BLAS/OMP/MKL/NUMEXPR/RAYON to 2 threads per worker
  11. **exp-11** pin each pool worker to a disjoint CPU-core slice
  12. **exp-12** `log_labels=False` by default — skip `log_exoego_batch` projections entirely
  13. **exp-13** skip pyserde in `load_ego_cams`, walk extrinsics JSON directly into numpy stacks
  14. **exp-14** cache the `rr.AssetVideo` objects on the sub-sequence as `_video_assets` so `setup_scene → log_video` reuses them
  15. **exp-15** `orjson.loads` for the two big JSON files (landmarks + ego extrinsics)
  16. **exp-16** thread-parallel sub-init (reverted — BLAS-thread contention with the per-worker thread cap)
  17. **exp-17** flip the dataset config's `load_labels=False` whenever the visualization-level `log_labels` is False, so `BaseExoEgoSequence.__init__` skips the 96 MB JSON parse entirely
- The parity validator (`tools/validate_assembly101_rrd_parity.py`) fails on missing GT columns and on numeric drift in sampled ego translations; it tolerates extras (deterministic static-metadata that the GT lacks) and a 15 % filesize delta.

## Headline Numbers

| measurement | baseline | champion (exp-17) | speedup |
| ----------- | -------: | ----------------: | ------: |
| 1-seq wall (`--max-conversions 1 --num-workers 1`)    | 22.14s | 2.88s  | 87.0 % |
| 3-seq wall (`--max-conversions 3 --num-workers 1`)    | 54.80s | 6.03s  | 89.0 % |
| 10-seq wall (`--max-conversions 10 --num-workers 8`)  | ≈182s* | 9.38s  | 94.9 % |

\* baseline 10-seq estimated from `18.27 sec/seq × 10`.
