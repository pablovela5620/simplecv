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
