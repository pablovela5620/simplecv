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
| exp-02 | 2026-05-14 00:03 | vectorize per-frame `np.nanmean` loop in `_ConfidenceAwareColumnList.partition`; when every partition has equal length, reshape to (m, n_per) and `nanmean(axis=1)` once | `simplecv/rerun_custom_types.py` | 35.15 | — | 11.72 | 35.9 | PASS | Y | per-seq self-reported times 6.58/5.08/9.15 (vs 7.70/6.32/10.81 in exp-01); kept Python fallback for non-uniform lengths |

## Leaderboard (top 5 valid by sec_per_seq, smallest = fastest)

| rank | exp_id | sec_per_seq | speedup_pct | summary |
| ---- | ------ | ----------- | ----------- | ------- |
| 1 | exp-02 | 11.72 | 35.9 | exp-01 + vectorized per-frame `nanmean` loop |
| 2 | exp-01 | 12.97 | 29.0 | parallel MP4 preload + reuse `_video_blobs` + explicit `media_type` |
| 3 | baseline | 18.27 | 0.00 | starting point |

## Current Champion Diff Summary

- **champion**: `exp-02`
- **diff**: exp-01 (parallel MP4 byte preload + `_video_blobs` reuse + explicit `media_type="video/mp4"`) plus vectorized `nanmean` in `_ConfidenceAwareColumnList.partition` (replaces ~706k python-level `nanmean(segment)` calls with one `nanmean(axis=1)` on a reshaped (m, n_per) array).
