# Distortion Logging Plan (Brown‑Conrady only)

Goal: keep distortion coefficients in the `.rrd` alongside existing pinhole logs without doing rectification or recompressing videos. Viewer overlays will still rely on the undistorted model, but downstream tools can read the stored coefficients.

## Scope & constraints
- Only Brown–Conrady distortion (what exo/ego ingests today).
- No image rectification or new AV1 renders; we preserve original videos.
- Keep standard `rr.Pinhole` logging for compatibility; add lightweight custom components for distortion.

## Steps
- **Custom components**: In `simplecv/rerun_custom_types.py`, define descriptors and batches for:
  - `DistortionModel` (string, fixed value `"brown_conrady"`).
  - `DistortionCoefficients` (float32 vector; accept 5–14 terms but store as-length array).
  - Convenience wrapper `CameraDistortion.as_component_batches()` returning model+coeffs.
- **Pinhole wrapper**: Add `PinholeWithDistortion(rr.AsComponents)` that:
  - Emits the regular `rr.Pinhole` batch first.
  - Appends the distortion batches when provided.
  - Factory `from_camera(PinholeParameters | Fisheye62Parameters)` that pulls `camera.distortion` if present.
- **Logging helper**: Update `log_pinhole` in `simplecv/rerun_log_utils.py` to log `PinholeWithDistortion` instead of raw `rr.Pinhole`; add flag `include_distortion: bool = True`.
- **Ingestion path**: Ensure `oak_calib_to_pinhole` keeps populating `PinholeParameters.distortion` (k1–k6, p1–p2, s1–s4, tau_x, tau_y) and that calls to `log_pinhole` pass it through.
- **Viewer helpers**: `view_exoego.setup_scene` currently calls `log_pinhole` for exo cams but logs ego cams with direct `rr.Pinhole`. Switch those ego logs to `log_pinhole(..., include_distortion=True)` (or the wrapper directly) so both exo and ego paths emit the distortion components without altering layout/blueprint behavior.
- **Viewer discoverability**: Optionally add a `rr.TextLog` under each camera node summarizing the model and first few coefficients for quick inspection.
- **Testing**: Add a small test that logs `PinholeWithDistortion`, reloads the recording, and asserts presence/values of `DistortionModel` and `DistortionCoefficients` columns; keep jaxtyping annotations.
- **Docs snippet**: Document usage: build `PinholeParameters` -> `log_pinhole(..., include_distortion=True)`; note that the viewer does not apply distortion.

## Future (out of scope now)
- Add Kannala–Brandt support if/when fisheye ingest arrives.
- Optional rectified video path plus paired rectified intrinsics.
- TODO: wire up reading the distortion components in `simplecv/data/ego/rrd_ego.py`.

## TODO: Brown–Conrady undistort helper for triangulation

Goal: provide a batched undistort utility so triangulation can keep operating in pixel space (P = K[R|t]) while removing lens distortion first. Primary implementation is our own numerical inverse; OpenCV is only a validation oracle, not a hard dependency.

Planned steps
- Add `undistort_brown_conrady_batch` to `simplecv/sensors/camera/brown_conrady.py`.
  - Inputs: distorted UV (optionally with confidence) shaped `[n_frames, n_views, n_kpts, 2 or 3]`, per-view K stack, per-view `BrownConradyDistortion | None`.
  - Implementation:
    - Normalize by K per view.
    - Run a small fixed-point/Newton iteration to invert `_distort_normalized_points` in normalized coords; seed with the distorted point.
    - Re-apply K to return undistorted pixel coordinates; pass confidences through unchanged.
    - Optionally (guarded by env flag) compare against `cv2.undistortPoints(..., P=K)` and warn on large deltas for validation only.
  - Outputs: undistorted pixel UV; shape mirrors input.
- Testing: add Hypothesis-based unit test (`tests/test_brown_conrady_undistort.py` or extend the existing distortion test) that:
  - Samples random K and Brown–Conrady coefficients (k1–k6, p1–p2, s1–s4, τx/τy).
  - Generates random distorted UV, runs the new helper, and compares against `cv2.undistortPoints(..., P=K)` with tight tolerance (e.g., `atol ~1e-6` px).
  - Covers identity distortion, multi-view batches, and confidence passthrough.
- Integration: at triangulation call sites, call the helper to undistort 2D measurements before passing them to existing `batch_triangulate` (no change to projection matrices).
