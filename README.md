# Simple CV
Utility Computer Vision functions that I often use
<p align="center">
  <img src="media/depth-fusion.gif" alt="example output" width="720" />
</p>

## Install
Make sure you have the [Pixi](https://pixi.sh/latest/#installation) package manager installed
```bash
git clone https://github.com/pablovela5620/simplecv.git
cd simplecv
pixi shell
```
this will drop you into the activated environment (currently only linux/macos)

## Run Examples
### See all avaiable tasks
```bash
pixi task list
```

### Rerun Environment
The default Pixi environment uses the released `rerun-sdk[datafusion]>=0.32`, so normal Rerun commands and catalog tasks should run without `-e rerun-prerelease`.

The `rerun-prerelease` environment is kept in `pyproject.toml` as a spare lane for testing future prerelease wheels, but avoid using it unless a specific unreleased Rerun feature is required.

### Visualize Polycam Data
Quick example
```
pixi run view-polycam-data
```

If you have a polycam zip file or extracted directory (from within the pixi shell)
```
python tools/view_polycam.py --polycam-zip-path $PATH-TO-POLYCAM-ZIP
```

### Ingest Exo/Ego Recordings
Ingest synchronized exo/ego captures into Rerun (spawns the viewer unless told otherwise).
```bash
simplecv-ingest-exoego --exoego-dir data/exoego-examples/adil-correct/adil3/
```

#### Handy flags
- `--reencode-to-av1` ensures every clip is resized to ≤720p and re-encoded to AV1 MP4 before logging.
- `--rr-config.headless` disables the Rerun UI (useful for automated runs).
- `--rr-config.connect` or `--rr-config.serve` reuse an external/remote Rerun viewer.

The CLI is Tyro-based, so tab completion and `--help` are available by default.

### Video Cache
Visualizing RRD-based exo/ego datasets remuxes the embedded video streams once and caches the resulting MP4s under `~/.cache/simplecv/exoego_videos`. Subsequent runs reuse these files, eliminating the 30 s+ extraction hit per recording.

- Set `SIMPLECV_VIDEO_CACHE=/path/to/cache` to override the cache root (for example, to keep it on a faster disk).
- Set `SIMPLECV_VIDEO_CACHE_DISABLE=1` to opt out entirely; the remux step will run every time.
- The cache auto-invalidates if the source `.rrd` changes (mtime or size). To reclaim disk space manually, delete the directory shown above.

### Batch raw → RRD ingestion (ExoEgo Forge catalog)

Convert raw on-disk recordings into the catalog `.rrd` layout used by
`docs/optimize_batch_ingest_goal.md`. The CLI takes a dataset subcommand
(e.g. `assembly101`, `hocap`, `aria-gen2`, ...) plus a save directory and
optional knobs for parallelism / scope.

```bash
# 10 Assembly101 sequences, 8 worker processes, write into the
# preferred long-term storage path on the workstation
pixi run python tools/batch_raw_to_rrd.py \
    --rrd-save-dir /mnt/8tb/data/exoego-forge-catalog \
    --max-conversions 10 \
    --num-workers 8 \
    --force \
    assembly101
```

**Storage location.** `--rrd-save-dir` defaults to `data/exoego-forge-catalog`
(the in-repo path). For full-corpus ingestion, prefer
`/mnt/8tb/data/exoego-forge-catalog` on this workstation — `data/` is on
the smaller root volume.

**Parallelism.** `--num-workers N` dispatches each sequence to a separate
process via `ProcessPoolExecutor` (spawn). With 8 workers a 10-sequence
Assembly101 batch finishes in ~11 s on a 32-core box (down from ~3 min
sequentially before optimization; see
`docs/optimize_batch_ingest_results.md`).

**Skipping keypoint streams.** By default `log_labels=False`, which
matches the existing read-only GT catalog under `data/exoego-forge-catalog/`
(its RRDs have no `coco133_xyz`/`coco133_uv` columns). Pass `--log-labels`
to additionally emit the 3D-keypoint and per-camera 2D projection
streams (≈5× slower per sequence).

### Batch Processing ExoEgo from S3

Process multiple ExoEgo sequences from S3 in batch. The pipeline downloads, cuts, and optionally ingests recordings.

**Environments:**
- Use `gpu` environment for NVENC AV1 encoding (requires RTX 40+ GPU)
- Never use `dev` for batch (beartype slows processing)

**Cut Only (download + cut videos):**
```bash
pixi run -e gpu python tools/exoego_tools/batch_process_s3.py \
    --s3-bucket YOUR_BUCKET_ID \
    --profile YOUR_AWS_PROFILE \
    --output-dir /path/to/output \
    --parallel-workers 4 \
    --cut-only
```

**Full Pipeline (cut + ingest to RRD):**
```bash
pixi run -e gpu python tools/exoego_tools/batch_process_s3.py \
    --s3-bucket YOUR_BUCKET_ID \
    --profile YOUR_AWS_PROFILE \
    --output-dir /path/to/output \
    --parallel-workers 4
```

**Re-ingest Only (regenerate RRDs from cut data):**
```bash
pixi run -e gpu python tools/exoego_tools/batch_process_s3.py \
    --s3-bucket YOUR_BUCKET_ID \
    --profile YOUR_AWS_PROFILE \
    --output-dir /path/to/output \
    --reingest-only
```

**State Management:**
- Progress tracked in `manifest.json` in output directory
- Ctrl+C is safe - restart resumes from last checkpoint
- Completed sequences are skipped on restart


## T265 SLAM
- **Env:** `t265` feature includes `librealsense==2.53.1` and `pyrealsense2==2.53.1.4623` (see `pyproject.toml`).
- **Verify CLI:** `pixi run -e t265 which rs-enumerate-devices`.
- **Enumerate:** `pixi run -e t265 rs-enumerate-devices`.
- **If you see RS2_USB_STATUS_ACCESS:** install udev rules so user-space can access the device and upload firmware.
  - `curl -fsSL https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules | sudo tee /etc/udev/rules.d/99-realsense-libusb.rules >/dev/null`
  - `sudo udevadm control --reload-rules && sudo udevadm trigger`
  - You should already be in `plugdev`; otherwise: `sudo usermod -aG plugdev $USER` then re-login.
  - Replug the T265.
- **Watch re-enumeration:** `watch -n 1 "lsusb | rg -i '(realsense|t265|8087:0b37|03e7:2150)'"` → expect flip from `03e7:2150` (Movidius boot) to `8087:0b37` (T265).
- **Run logger:** `pixi run -e t265 python tools/t265_slam.py --timeout-ms 1000`.
- **Pixi tasks:** `pixi run -e t265 t265-enum`, `pixi run -e t265 t265-probe`.



## Notation for Transformation Matrices

__TL;DR:__ `world_T_cam == world_from_cam`  
This repo uses the notation "cam_T_world" to denote a transformation from world to camera points (extrinsics). The intention is to make it so that the coordinate frame names would match on either side of the variable when used in multiplication from *right to left*:

    cam_points = cam_T_world @ world_points

`world_T_cam` denotes camera pose (from cam to world coords). `ref_T_src` denotes a transformation from a source to a reference view.  
Finally this notation allows for representing both rotations and translations such as: `world_R_cam` and `world_t_cam`
