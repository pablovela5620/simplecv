# Batch-Ingest Optimization Results — Assembly101

Append-only log driven by `docs/optimize_batch_ingest_goal.md`. Each row is
one experiment. The current champion is marked `Y` in the `champion`
column. All wall-clock numbers are from `/usr/bin/time -f '%e'` on the same
host, with `--max-conversions 3` (or 10 for promotion) and outputs written
to `/tmp/batch-bench/<exp_id>/`.

| exp_id | timestamp_pt | hypothesis | files_touched | wall_seconds_3seq | wall_seconds_10seq | sec_per_seq | speedup_vs_baseline_pct | parity | champion | notes |
| ------ | ------------ | ---------- | ------------- | ----------------- | ------------------ | ----------- | ----------------------- | ------ | -------- | ----- |
| baseline | 2026-05-13 23:13 | starting commit `8429635`; vanilla `tools/batch_raw_to_rrd.py assembly101` | (none) | 54.80 | — | 18.27 | 0.00 | N/A (defines GT) | Y | warm-cache, single trial; per-seq self-reported times 7.68/6.68/11.18 from `visualize_exo_ego` |

## Leaderboard (top 5 valid by sec_per_seq, smallest = fastest)

| rank | exp_id | sec_per_seq | speedup_pct | summary |
| ---- | ------ | ----------- | ----------- | ------- |
| 1 | baseline | 18.27 | 0.00 | starting point |

## Current Champion Diff Summary

- **champion**: `baseline`
- **diff**: none (starting state)
