# Camera Projection + Triangulation Audit

## TL;DR
- `simplecv/camera_parameters.py` already encapsulates Brown–Conrady intrinsics/extrinsics for pinhole cameras and sketches a `Fisheye62Parameters`, but distortion is not yet wired into logging or triangulation.
- Quest/Project Aria style fisheye handling lives in `simplecv/umetrack_temp/*` and duplicates several helpers (projection, distortion) that also appear in `camera_parameters.py`.
- Rerun visualization currently goes through `log_pinhole` (`simplecv/rerun_log_utils.py:45`) regardless of lens model, so fisheye lenses are logged as simple pinholes.
- Triangulation utilities (`simplecv/ops/triangulate.py`) assume OpenCV-style pinhole projection matrices; they ignore distortion and depend on precomputed `P = K [R | t]`.
- Deliverable: unify camera model abstractions, provide model-specific projection/unprojection paths, and add Rerun logging hooks for each camera family (pinhole ± distortion, ultra-wide, Aria/Quest fisheye).

---

## Current Building Blocks

### Rerun logging (`simplecv/rerun_log_utils.py`)
- `RerunTyroConfig` (`simplecv/rerun_log_utils.py:18`) bootstraps Rerun sessions and stores a `RecordingStream`.
- `log_pinhole` (`simplecv/rerun_log_utils.py:45`) logs:
  - Intrinsics through `rr.Pinhole(image_from_camera=K, camera_xyz=ViewCoordinates[convention])`.
  - Extrinsics via `rr.Transform3D` with `cam_T_world` (child-from-parent).
  - Assumes a perspective projection; `camera.distortion` is ignored.
- Other helpers (`log_video`, `read_h264_samples_from_rrd`, `write_asset_video_blob`) are camera-model agnostic.

### Camera parameter dataclasses (`simplecv/camera_parameters.py`)
- `Distortion` (`simplecv/camera_parameters.py:9`) captures Brown–Conrady coefficients (radial k1–k6, tangential p1–p2).
- `Extrinsics` (`simplecv/camera_parameters.py:29`) accepts either world-to-camera or camera-to-world inputs and auto-computes both transforms (`cam_T_world`, `world_T_cam`) plus rotation/translation pairs.
- `Intrinsics` (`simplecv/camera_parameters.py:63`) stores focal lengths, principal point, optional resolution, and camera convention (`"RDF"` or `"RUB"`). Automatically derives `k_matrix` and infers height/width if omitted.
- `PinholeParameters` (`simplecv/camera_parameters.py:104`) combines intrinsics/extrinsics, computes the projection matrix `K @ cam_T_world[:3, :]`, and optionally holds a `Distortion`.
- `Fisheye62Parameters` (`simplecv/camera_parameters.py:127`) mirrors the pinhole structure for KB62-like fisheye setups but currently shares the same projection matrix logic.
- Utility functions:
  - `rescale_intri` rescales an `Intrinsics` instance to a new resolution (`simplecv/camera_parameters.py:176`).
  - `perspective_projection`, `arctan_projection`, `apply_radial_tangential_distortion`, and `fisheye_projection` (`simplecv/camera_parameters.py:188-313`) project world points given a `PinholeParameters` or `Fisheye62Parameters`. `fisheye_projection` normalizes, distorts, denormalizes, and masks out-of-bounds points.

### Quest / Project Aria stack (`simplecv/umetrack_temp/*`)
- `FisheyeCameraParameter` and lightweight `PinholeCameraParameter` (`simplecv/umetrack_temp/camera_models.py`) manage intrinsics, distortion vectors (up to k6/p2), and extrinsics (`set_KRT`, `intrinsic33`, getters).
- `Camera` (`simplecv/umetrack_temp/cameras.py`) wraps either parameter class and provides:
  - `world_to_camera`/`camera_to_world` transforms via cached `cam_T_world` and `world_T_cam`.
  - `camera_to_image` dispatch: perspective projection for pinhole, `arctan_projection` + Brown–Conrady distortion for fisheye.
  - `image_to_camera` (unit-ray back-projection) for pinholes only.
- `projection.project_points` (`simplecv/umetrack_temp/projection.py`) chains the above to produce 2D points with NaN masking for invalid rays.
- `view_umetrack_data.py` uses this stack to:
  - Ingest Quest recordings (`DataStream`).
  - Update `cam_T_world` per frame and log images via `log_camera` (`simplecv/apis/view_umetrack_data.py:168`), which currently calls Rerun `rr.Pinhole` even for fisheye lenses.

### Triangulation utilities (`simplecv/ops/triangulate.py`)
- `projectN3` and `proj_3d_vectorized` project homogeneous 3D joints through provided projection matrices—again assuming pinhole + no distortion.
- `batch_triangulate` (`simplecv/ops/triangulate.py:50`) performs linear least-squares triangulation with confidence weighting, expecting:
  - Inputs in OpenCV coordinate convention.
  - `projection_matrices` shaped `(n_views, 3, 4)` as `K [R | t]`.
  - Visibility gating via `min_views`, returning `xyz` + aggregated confidence.

---

## Coverage vs Target Camera Families

| Use case                                    | Current entry points                                           | Ready today? | Notes |
|---------------------------------------------|----------------------------------------------------------------|--------------|-------|
| 1. Standard pinhole (e.g. iPhone main)      | `PinholeParameters`, `log_pinhole`, `perspective_projection`, `batch_triangulate` | ✅ | Distortion slot exists but unused; triangulation assumes undistorted correspondences. |
| 2. Mildly distorted pinhole (iPhone ultra-wide) | `Distortion` + `apply_radial_tangential_distortion` (not yet integrated), `projection_matrix` still pinhole-only | ⚠️ partial | Need consistent normalization/distortion when projecting, logging, and deserializing to Rerun. |
| 3. Highly distorted Brown–Conrady (OAK-D W) | `PinholeParameters` + high-order `Distortion`, `apply_radial_tangential_distortion` | ⚠️ partial | DepthAI reports a 14-coefficient Brown–Conrady model (k1–k6, p1,p2, s1–s4, τx,τy); we need to honour it end-to-end rather than reaching for the KB62 placeholder. |
| 4. Project Aria / Quest fisheye             | `umetrack_temp` modules (`FisheyeCameraParameter`, `Camera.camera_to_image`, `log_camera`) | ⚠️ separate | Works inside Quest pipeline but duplicates code and reports fisheye lenses as `rr.Pinhole` to Rerun. |

---

## Gaps & Opportunities

- **Log-time fidelity**: Need `rr.Fisheye` (or custom blueprint) logging path so ultra-wide/KB62 lenses are visualized correctly instead of reusing `log_pinhole`.
- **Distortion-aware projections**: `PinholeParameters.distortion` is unused; triangulation ignores lens models. We need a strategy for undistortion/distortion when going between image ↔ ray ↔ world.
- **Code duplication**: `camera_parameters.py` and `umetrack_temp/cameras.py` maintain parallel implementations of projection math. Consolidating will reduce drift.
- **Transform conventions**: Ensure all APIs expose both `cam_T_world` and `world_T_cam` (current `Extrinsics` already handles it) and document the expected convention for triangulation.
- **Quest pipeline integration**: Transition Quest/Aria helpers into the main camera module, keeping any Aria-specific distortion extensions (tangential/thin-prism) explicit.
- **Unit handling**: DepthAI and Quest assets may use different translation units; triangulation routines assume meters. Centralizing scaling metadata would avoid subtle bugs.

---

## Rerun Support & Interim Workarounds

- **Status quo**: Rerun’s camera archetypes only cover pinhole projection today; fisheye distortion requires pre-projecting data outside the viewer. The upstream issue tracks exactly this gap and notes that users currently “do it manually for accurate projection.”citeturn2open0
- **Workaround 1 — project + log 2D overlays**: Our existing Quest pipeline mirrors this advice by performing all distortion-aware projection in Python and then logging the resulting 2D features (`rr.Image`, 2D keypoints) via `view_umetrack_data.py`.  
  - *Pros*: Zero additional geometry, predictable performance, leverages the exact distortion math already maintained for inference.  
  - *Cons*: Viewers see only baked 2D data—no way to reproject other primitives (e.g., LiDAR, 3D boxes) on the fly; any camera pose change demands recomputation and re-upload of the overlay.
- **Workaround 2 — textured mesh warp**: A community workaround encodes the undistortion map into `rr.Mesh3D` UVs: build a planar quad grid matching the rectified plane, populate `vertex_texcoords` with undistorted UVs, and stream the original distorted image as the texture. Toggling the mesh on/off reproduces the rectified view without ever resampling on the CPU.
  - *Pros*: Keeps the distorted frame as-is, lets Rerun handle reprojection for anything that can sample the mesh, and centralizes the warp inside the viewer. Once the mesh is uploaded, only the texture needs updating per frame.
  - *Cons*: The mesh must be dense enough to approximate the warp (piecewise-linear error shrinks with subdivision), adding GPU draw cost and CPU upload overhead. Texture updates remain per-frame; with very wide FOVs you may need thousands of quads to avoid visible aliasing, which risks becoming the new bottleneck. This path still doesn’t help log-time `rr.Pinhole` consumers like transforms, so additional bookkeeping is required if other data needs consistent projection.
- **Choosing between them**: When you already own the projection math (e.g., for triangulation) and primarily need overlays, sticking to projected 2D annotations avoids runtime surprises. The mesh trick gains flexibility for layered scene visualization but trades correctness for tessellation density; if we adopt it, we should benchmark how many triangles we need for OAK-D W / Aria-grade distortion and whether Rerun’s current texture upload path can keep up with our frame rates.

---

## Proposed Refactor Plan

1. **Unify camera model representations**
   - Introduce a base protocol (e.g. `CameraModel`) with required accessors: `intrinsics`, `extrinsics`, `project_world_points`, `pixel_to_ray`.
   - Refactor `PinholeParameters` and `Fisheye62Parameters` to implement this interface.
   - Migrate `simplecv/umetrack_temp` classes to reuse the shared dataclasses (or merge them outright).

2. **Extend projection + triangulation utilities**
   - Provide explicit distortion-aware projection paths (normalize → distort/undistort).
   - Add helpers to produce distortion-aware Rerun components and to generate rectified projection matrices for triangulation.
   - Update `batch_triangulate` to accept either:
     - Pre-undistorted correspondences.
     - A camera model that can supply rays (triangulate via ray intersection, not just `K[R|t]`).

3. **Rerun logging alignment**
   - Create `log_camera_model` that dispatches to:
     - `rr.Pinhole` for perspective cameras.
     - `rr.Fisheye`/custom archetype for KB62-style cameras.
   - Include distortion metadata (radial/tangential coefficients) so viewers/processors can undistort consistently.
   - Ensure ViewCoordinates reflect `camera_conventions`.

4. **Quest/Aria integration**
   - Lift Quest-specific fisheye math into the shared module, preserving KB62 (k1–k6, p1–p2) and extendable slots for thin-prism / tilt if needed.
   - Replace `log_camera` in `view_umetrack_data.py` with the unified logging helper.

5. **Testing + validation**
   - Build regression tests that round-trip points through project→unproject for each camera family.
   - Add Rerun smoke tests to confirm logged archetypes render as expected.
   - Validate triangulation on synthetic scenes covering all four camera types.

---

## Open Questions

- Do we want to store distortion on disk (JSON, NPZ) in Brown–Conrady order or align with vendor-specific parameter order (DepthAI, Aria)?
- Should triangulation stay purely P-matrix based, or move to a ray–ray intersection approach for fisheye lenses?
- What Rerun archetypes (or blueprint customizations) best represent KB62 / KB624 fisheye models today?
- How much Quest-specific calibration (e.g. per-frame extrinsics, rolling shutter) do we need to surface in the shared interface?

---

## Next Steps Checklist

1. Merge `FisheyeCameraParameter` capabilities into `Fisheye62Parameters` (or vice versa) and delete duplicate math.
2. Implement `log_camera_model` that inspects `camera.model_type` and selects the right Rerun component.
3. Add distortion-aware projection utilities and document expected workflows (undistort before triangulation vs. ray-based solving).
4. Refactor Quest `view_umetrack_data` to consume the unified camera API and update downstream tools accordingly.
