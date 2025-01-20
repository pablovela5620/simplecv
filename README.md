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



## Notation for Transformation Matrices

__TL;DR:__ `world_T_cam == world_from_cam`  
This repo uses the notation "cam_T_world" to denote a transformation from world to camera points (extrinsics). The intention is to make it so that the coordinate frame names would match on either side of the variable when used in multiplication from *right to left*:

    cam_points = cam_T_world @ world_points

`world_T_cam` denotes camera pose (from cam to world coords). `ref_T_src` denotes a transformation from a source to a reference view.  
Finally this notation allows for representing both rotations and translations such as: `world_R_cam` and `world_t_cam`