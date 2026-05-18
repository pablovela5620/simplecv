# Catalog registration speedup — perf/catalog-registration-overnight

## Headline

| | seconds |
| --- | --- |
| Baseline (HEAD of `codex/epfl-smartkitch-addition`) | **496.08s** median (3 warm runs: 496.82, 489.05, 496.08) |
| Final (this branch) | **2.65s** median (3 warm runs: 2.65, 2.65, 2.72) |
| Speedup | **~187×** |

Target was <30s; we hit 2.65s.

## Bottleneck identified

A single line dominated baseline startup:

```python
server: rr.server.Server = rr.server.Server(datasets=registration_paths_by_dataset, port=port)
```

cProfile/print-instrumented run breakdown on the baseline:

| phase | seconds |
| --- | --- |
| `mount.server_init` (Rerun ingests 6332 RRDs over FFI) | **487.29** |
| `main.tables_total` (8 datasets x segment_table + create_table) | 1.06 |
| `cli_and_imports` (Python startup + tyro) | 0.95 |
| everything else | 0.40 |

The Rerun `Server(datasets=<list of files>)` constructor synchronously
ingested every RRD before returning. Passing a directory string instead
routes through Rerun's `dataset_prefixes` path: the Rust prefix walker
returns as soon as the gRPC listener is up, then keeps registering
in the background.

## Top three accepted experiments

1. **E005 — defer table-build to a daemon thread.** Combines the prefix-
   mode kickoff with deferral of `build_rrd_index_rows_from_dataset` /
   `create_rrd_index_table` / `_register_default_dataset_blueprint`
   (blueprints stay sync because they don't touch segments). Marker
   fires at 2.65s; the same 487s of segment ingest continues
   asynchronously after ready. *Accepted.*
2. **E002 — prefix-dict to `rr.server.Server`.** Stand-alone this is
   already a 300× server-init win (487s → 1.5s), but is rejected on
   its own because subsequent table-builds assume segments are
   registered. Kept as a building block for E005.
3. **E000 — instrument + ready marker.** Before this we had no way to
   distinguish what was slow. Adding `[catalog-phase] <name> <s>`
   prints unlocked targeted optimization.

Rejected experiments (kept in `experiments.jsonl`):

- **E001** — explicit `client.create_dataset()` + `dataset.register([uris])`.
  Too chatty: 12s for 12 files, then gRPC `service unavailable` on the
  second batch.
- **E003** — prefix mode + synchronous polling for full registration.
  0.25s poll interval starved the server; killed after 20+ min.
- **E004** — `dataset.register_prefix()` + `handle.wait()`. The
  `register_prefix` Python wrapper blocked indefinitely on the first
  call; killed after 25+ min.

## Files in this directory

- `experiments.jsonl` — one row per experiment (E000..E005).
- `baseline/timings.json` — three-warm-run baseline raw numbers.
- `baseline/timings_instrumented.json` — single instrumented run with
  per-phase breakdown that pointed at `mount.server_init`.
- `e005_final.json` — three-warm-run final numbers.
- `screenshots/final/` — the catalog/dataset/segment object-model views.
- `bench.py` — the harness (parses `[catalog-phase]` lines and the
  `SIMPLECV_CATALOG_READY` marker).
- `capture_screenshots.py` — headless screenshot driver (spawns the
  catalog under `xvfb-run`, queries the running catalog via
  `rr.catalog.CatalogClient` for entry / segment IDs, then runs
  `rerun --screenshot-to` per target URL).
- `log_experiment.py` — JSONL appender CLI used by every experiment.

## Reproducing

```bash
cd /home/pablo/0Dev/personal/simplecv-perf-catalog
unset PIXI_PROJECT_MANIFEST
pixi run python bench/bench.py --n 3 --out-json bench/final/timings.json
```

The bench spawns `pixi run python tools/catalog.py …` on port 19988+1
so it does not collide with any catalog the user is already running.
