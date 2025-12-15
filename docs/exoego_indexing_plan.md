# ExoEgo indexing and timeline plan (2025-12-14)

## Goals
- Allow all ExoEgo datasets **except `rrd_exoego`** to be indexed by either an integer frame number or a nanosecond-relative timestamp.
- Define one canonical timeline per dataset instance that downstream logging and data access can rely on.
- Start by implementing the scheme for `HocapSequence`, then roll it out to the other adapters.

## Current state / gaps
- `BaseExoEgoSequence.__len__` and `__getitem__` are stubs; child sequences (e.g., `hocap`, `assembly101`, `umetrack`) also return `None`.
- Video access goes through `MultiVideoReader`, which only aligns by min-frame-count and ignores timing.
- Labels may carry `timestamps_ns` but are usually aligned implicitly by array index.
- Rerun logging (`log_video`) already exposes `frame_timestamps_ns = AssetVideo.read_frame_timestamps_nanos()`, but this information is not stored on the dataset objects.

## Canonical timeline definition
1. For each stream that can be sampled (`ego` videos, `exo` videos, optional label timeline), read the monotonic frame timestamps via `rr.AssetVideo(path).read_frame_timestamps_nanos()`. These are already relative to the start of each video, so no extra normalization is needed.
2. Collect each stream’s **end time** (`timestamp[-1]`).
3. Set `canonical_end_ns = min(end_times)`; this is “shortest by duration,” not by frame count.
4. Choose the canonical timestamp grid as:
   - Prefer the stream whose end time equals `canonical_end_ns`; use its timestamps clipped to `<= canonical_end_ns`.
   - If label timestamps exist and end sooner, they become canonical; otherwise fall back to the earliest-ending video stream.
5. Store:
   - `canonical_timestamps_ns: Int[np.ndarray, "n_events"]`
   - `stream_timestamps_ns: dict[str, Int[np.ndarray, "n_frames"]]` keyed by camera/stream name.

Rationale: preserves irregular spacing (dropped frames) while guaranteeing every canonical sample has data available (by clamping each stream to its last in-range frame).

## Canonical timeline loading API
- Add an abstract method on `BaseExoEgoSequence`:
  ```python
  @abstractmethod
  def load_stream_timestamps_ns(self) -> dict[str, Int[np.ndarray, "n_frames"]]:
      """Per-stream nanosecond timestamps (relative to first frame). Keys should match stream names."""
  ```
- Each dataset implements it by reading `AssetVideo.read_frame_timestamps_nanos()` for every ego/exo video (and any label timeline if available).
- `BaseExoEgoSequence.__init__` will call this to populate:
  - `self.stream_timestamps_ns` (dict)
  - `self.canonical_timestamps_ns` (selected as described above)
  - `self.canonical_end_ns`
- This makes the requirement explicit for all datasets (except `rrd_exoego`, handled separately later).

## Indexing semantics
- Accept exactly **one** of `idx: int` or `ts_nano: np.timedelta64` in `BaseExoEgoSequence.__getitem__` (helpers `at_idx(idx)` and `at_ts(ts_nano)` are available for readability).
  - **Int path:** treat as position on `canonical_timestamps_ns`.
  - **Timestamp path (`np.timedelta64`)**: clamp into `[canonical_timestamps_ns[0], canonical_end_ns]`, then map to nearest-at-or-before canonical sample via `searchsorted(..., side="right") - 1`.
- Use shared helpers (lift `timestamp_to_frame_index` from `metric_exoego_calib.py`) for consistent mapping of timestamps to frame indices per stream.
- `__len__` returns `len(canonical_timestamps_ns)`.

## Data retrieval at a timestamp
For a requested canonical timestamp `t_ns`:
- **Per video stream:** `frame_idx = timestamp_to_frame_index(t_ns, stream_timestamps_ns[name])`; read frame via the existing `VideoReader`.
- **Labels:** 
  - If `ExoEgoLabels.timestamps_ns` is present, pick the same mapping strategy.
  - If absent, assume label stack is aligned by index; clamp to last label frame if `idx` exceeds label length.
- **Outputs:** return an `ExoEgoSample` containing ego/exo `BGRList` frames, matching camera parameters, and (clamped) labels for that timestamp. When visualizing, also log label points (e.g., COCO-133) alongside the sampled frames for parity with `view_exoego.py`.

Note: a future refactor of `VideoReader`/`MultiVideoReader` to accept nanosecond timestamps directly would remove the index↔timestamp shim; until then, use the mapping helper above.

## Hocap implementation snapshot (done)
- `HocapSequence.__getitem__` now enforces “exactly one of idx (int) or ts_nano (np.timedelta64)”, clamps timestamps, and maps frames via `timestamp_to_frame_index`.
- Convenience accessors: `seq[idx]` or `seq.at_idx(idx)` for integers, `seq.at_ts(np.timedelta64(...))` for timestamp-based lookups (no keywords needed at call sites).
- `load_stream_timestamps_ns` reads `AssetVideo.read_frame_timestamps_nanos()` for every ego/exo video (and labels when present).
- Canonical timeline selection lives in `BaseExoEgoSequence.__init__`; `canonical_end_ns` uses the shortest-duration stream.
- Rerun logging parity via `tools/check_exoego_indexing.py`:
  - Cameras logged with `log_pinhole`, including `image_plane_distance`, ego transforms per frame, exo static.
  - Videos logged at `/world/{ego|exo}/{cam}/pinhole/video` on timeline `video_time` (BGR, JPEG-compressed).
  - Sampled frames logged side-by-side under `/sample/index/{ego|exo}/{cam}`; blueprint mirrors `view_exoego.py` but shows video + sample per tab.
- Labels logged as `Points3DWithConfidence` with COCO-133 ids, gradient-colored by confidence, and annotation context registered at `/`.
- CLI gained `--cfg.max-frames` to cap logging for quick tests (default None = all).

## Roll-out status (2025-12-15)
- Implemented and smoke-tested (150-frame runs via `tools/check_exoego_indexing.py` without dev env): `assembly101`, `umetrack`, `ego_dex`, `ego100k` (still skip `rrd_exoego`).
- `ExoEgoSample` accepts both `PinholeParameters` and `Fisheye62Parameters` to cover mixed rigs.

## Open questions / next steps
- Confirm whether any streams should be excluded from canonical selection for those datasets (depth, IR, etc.).
- Keep clamping strategy? (current behavior: clamp per-stream frame index to last available; labels clamp if shorter than canonical).
- Consider refactoring video readers to accept `np.timedelta64` directly once timestamp plumbing is stable.
- Shared helpers now live in `BaseExoEgoSequence`:
  - `_resolve_canonical(idx, ts_nano)` maps inputs to `(canonical_idx, ts_ns)`.
  - `_sample_ego(ts_ns)` / `_sample_exo(ts_ns)` fetch frames + cam params.
  - `_sample_labels(canonical_idx, ts_ns)` clamps labels to timestamp/idx.
  - Convenience accessors `at_idx`, `at_ts` wrap `__getitem__`.
