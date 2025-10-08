# RRD Viewer Startup Latency

## Observed Runtime
- Command: `pixi run -e dev python tools/view_exoego.py --rr-config.no-headless --no-log-labels --no-log-mano rrd --no-load-labels --rrd-path /home/pablo/0Dev/personal/simplecv/data/exoego-examples/adil-correct/adil3/data+rbl.rrd`
- Wall-clock time: ~98–104 s per run (measured repeatedly on the dev box)

## What the CLI Does Before Anything Appears
1. Tyro parses the config and instantiates `RRDSequence`, which in turn tries to build ego and exo sub-sequences (`simplecv/data/exoego/rrd_exoego.py:34`).
2. `RRDExoSequence.load_video_paths` and `RRDEgoSequence.load_video_paths` remux every camera stream on demand (`simplecv/data/exo/rrd_exo.py:47`, `simplecv/data/ego/rrd_ego.py:42`). Each call:
   - Opens the `.rrd` file via `rr.dataframe.load_recording`.
   - Pulls the raw Annex-B H.264 samples with `read_h264_samples_from_rrd` (`simplecv/rerun_log_utils.py:259`), which combines all Arrow chunks into a contiguous buffer in memory.
   - Pipes the buffer through PyAV’s `mux_h264_to_mp4`, decoding/rewriting every packet to a temporary mp4 (`simplecv/rerun_log_utils.py:228`).
   - Repeats this sequentially for every exo and ego sensor on every run.
3. Only after all mp4 files exist do we construct the `MultiVideoReader`, which opens each file with OpenCV (`simplecv/video_io.py:281`), so the viewer can finally start logging video frames.

## Why It’s Slow
| Bottleneck | Evidence | Effect |
|------------|----------|--------|
| Per-run remuxing | Temp directories are recreated inside `load_video_paths`, forcing a full remux even if nothing changed (`simplecv/data/exo/rrd_exo.py:47`, `simplecv/data/ego/rrd_ego.py:42`). | ~12–15 s per camera on the test recording, linear with camera count; dominates total time. |
| Full data copy | `samples.combine_chunks().flatten(recursive=True)` materialises the entire video column before PyAV sees it (`simplecv/rerun_log_utils.py:234`). | Adds large memcpy + alloc per stream (hundreds of MB each). |
| Single-threaded pipeline | Cameras are processed strictly one after another; PyAV remux is CPU-bound and not parallelised. | No overlap => total latency = sum of all streams. |
| MultiVideoReader eager init | It blocks until every mp4 exists because `VideoReader` needs real paths (`simplecv/video_io.py:281`). | Viewer can’t start logging until remux finishes. |

## Proposed Improvements
1. **Persistent remux cache**
   - Cache mp4 outputs under a stable path (e.g. `~/.cache/simplecv/rrd/<hash>/<entity>.mp4`) keyed by RRD mtime + entity name.
   - Skip PyAV if a valid cached file exists; only remux streams that are missing or stale.
   - Add a CLI flag/env to purge the cache when desired.

2. **Parallel remuxing**
   - Process camera streams concurrently (e.g. `concurrent.futures.ProcessPoolExecutor`) once caching identifies the ones that still need work.
   - Bound worker count to CPU cores; expect close to N/cores speedup.

3. **Streaming-friendly logging**
   - Extend `log_video` to accept `bytes` and log an `rr.AssetVideo` directly; when the RRD already stores `AssetVideo` blobs, we can bypass remuxing by reusing them (`simplecv/rerun_log_utils.py:188` already exposes helper code).
   - For pure H.264 streams, consider logging them as `VideoStream` nodes instead of materialising mp4, letting Rerun decode on the fly.

4. **Lazy `MultiVideoReader`**
   - Defer `VideoReader` construction until we actually iterate frames. This lets the blueprint appear immediately once at least one mp4 is ready, improving perceived responsiveness.

## Suggested Next Steps
1. Implement the cache abstraction (item 1) and add instrumentation to log cache hits/misses.
2. Once caching is in, benchmark again; if total time is still dominated by remaining remuxes, add parallelism (item 2).
3. Investigate direct `AssetVideo` logging to potentially eliminate remux entirely for future recordings (item 3).
4. Update developer docs so contributors know about the cache and how to warm it or purge it.
