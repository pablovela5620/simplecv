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
| exp-03 | 2026-05-14 00:14 | batch-compose all per-frame ego cam matrices; one `np.linalg.inv` on a (n_frames, 4, 4) stack instead of one inv per frame, and bypass `Extrinsics`/`Fisheye62Parameters` `__post_init__` via `object.__new__` | `simplecv/data/ego/assembly101_ego.py` | 32.16 | — | 10.72 | 41.3 | PASS | Y | per-seq self-reported 6.31/5.08/8.74; eliminates ~64k python-level matrix inverses per sequence |

## Leaderboard (top 5 valid by sec_per_seq, smallest = fastest)

| rank | exp_id | sec_per_seq | speedup_pct | summary |
| ---- | ------ | ----------- | ----------- | ------- |
| 1 | exp-03 | 10.72 | 41.3 | exp-02 + batched ego-cam matrix inversion |
| 2 | exp-02 | 11.72 | 35.9 | exp-01 + vectorized per-frame `nanmean` loop |
| 3 | exp-01 | 12.97 | 29.0 | parallel MP4 preload + reuse `_video_blobs` + explicit `media_type` |
| 4 | baseline | 18.27 | 0.00 | starting point |

## Current Champion Diff Summary

- **champion**: `exp-03`
- **diff**: exp-02 + `assembly101_ego.load_ego_cams` now stacks all per-frame world-from-cam 4x4 matrices into one `(n_frames, 4, 4)` array and inverts the whole stack with a single `np.linalg.inv` call. Each frame's `Extrinsics` and `Fisheye62Parameters` are constructed via `object.__new__` with the precomputed slices, bypassing the per-instance `__post_init__` work.
