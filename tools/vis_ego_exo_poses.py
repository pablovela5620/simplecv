#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
import rerun as rr
import argparse
from pathlib import Path

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix"""
    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def create_camera_frustum(intrinsics, depth=0.5, scale=1.0):
    """Create camera frustum points for visualization"""
    fx = intrinsics['camera_matrix'][0][0] * scale
    fy = intrinsics['camera_matrix'][1][1] * scale
    cx = intrinsics['camera_matrix'][0][2] * scale
    cy = intrinsics['camera_matrix'][1][2] * scale
    w, h = intrinsics['image_size']
    w, h = w * scale, h * scale
    
    # Camera frustum corners at depth
    corners_image = np.array([
        [0, 0],      # Top-left
        [w, 0],      # Top-right  
        [w, h],      # Bottom-right
        [0, h]       # Bottom-left
    ])
    
    # Convert image coordinates to 3D camera coordinates
    corners_3d = []
    for u, v in corners_image:
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        corners_3d.append([x, y, z])
    
    corners_3d = np.array(corners_3d)
    
    # Add camera center
    camera_center = np.array([[0, 0, 0]])
    all_points = np.vstack([camera_center, corners_3d])
    
    # Define lines connecting camera center to frustum corners
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Frustum rectangle
    ]
    
    return all_points, lines

def create_generic_intrinsics():
    """Create generic camera intrinsics for left/right cameras"""
    return {
        'camera_matrix': [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0], 
            [0.0, 0.0, 1.0]
        ],
        'image_size': [1280, 720]
    }

def create_ground_plane(size=10.0, grid_lines=20, z_level=0.0, flip_ground=False):
    """Create a ground plane with grid lines"""
    # Create grid points
    half_size = size / 2
    step = size / grid_lines
    
    # Grid lines parallel to X-axis (varying Y)
    x_lines = []
    for i in range(grid_lines + 1):
        y = -half_size + i * step
        line = [[-half_size, y, z_level], [half_size, y, z_level]]
        x_lines.append(line)
    
    # Grid lines parallel to Y-axis (varying X) 
    y_lines = []
    for i in range(grid_lines + 1):
        x = -half_size + i * step
        line = [[x, -half_size, z_level], [x, half_size, z_level]]
        y_lines.append(line)
    
    all_lines = x_lines + y_lines
    
    # Apply flip transformation if requested
    if flip_ground:
        flip_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0], 
            [0, 0, -1]
        ])
        flipped_lines = []
        for line in all_lines:
            flipped_line = []
            for point in line:
                flipped_point = flip_matrix @ np.array(point)
                flipped_line.append(flipped_point.tolist())
            flipped_lines.append(flipped_line)
        all_lines = flipped_lines
    
    return all_lines

def log_static_camera(camera_name, camera_data, entity_path_prefix="", flip_ground=False):
    """Log a static camera pose with frustum"""
    entity_path = f"{entity_path_prefix}/{camera_name}"
    
    # Extract pose
    R = np.array(camera_data['extrinsics']['R'])
    t = np.array(camera_data['extrinsics']['tvec'])
    
    # Apply ground flip transformation if requested
    if flip_ground:
        # 180° rotation around X-axis to flip Y and Z
        flip_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0], 
            [0, 0, -1]
        ])
        R = flip_matrix @ R
        t = flip_matrix @ t
    
    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    
    # Log camera transform
    rr.log(f"{entity_path}/transform", rr.Transform3D(translation=t, mat3x3=R))
    
    # Create and log frustum
    frustum_points, frustum_lines = create_camera_frustum(camera_data['intrinsics'])
    
    # Transform frustum points to world coordinates
    frustum_points_homogeneous = np.hstack([frustum_points, np.ones((frustum_points.shape[0], 1))])
    frustum_world = (transform @ frustum_points_homogeneous.T).T[:, :3]
    
    # Log frustum
    frustum_strips = []
    for line in frustum_lines:
        strip = [frustum_world[line[0]], frustum_world[line[1]]]
        frustum_strips.append(strip)
    
    rr.log(
        f"{entity_path}/frustum",
        rr.LineStrips3D(
            strips=frustum_strips,
            colors=[(255, 100, 100, 255)] * len(frustum_strips)
        )
    )
    
    # Log camera position as a point
    rr.log(
        f"{entity_path}/position", 
        rr.Points3D(
            positions=[t],
            radii=[0.02],
            colors=[(255, 100, 100, 255)]
        )
    )

def log_dynamic_camera(camera_name, poses_df, intrinsics, entity_path_prefix="", color=(100, 255, 100, 255), flip_ground=False):
    """Log dynamic camera poses with frustums over time"""
    entity_path = f"{entity_path_prefix}/{camera_name}"
    
    for idx, row in poses_df.iterrows():
        frame_idx = int(row['frame_idx'])
        
        # Set timeline
        rr.set_time_sequence("frame", frame_idx)
        
        # Extract pose
        t = np.array([row['tx'], row['ty'], row['tz']])
        R = quaternion_to_rotation_matrix(row['qx'], row['qy'], row['qz'], row['qw'])
        
        # Apply ground flip transformation if requested
        if flip_ground:
            # 180° rotation around X-axis to flip Y and Z
            flip_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0], 
                [0, 0, -1]
            ])
            R = flip_matrix @ R
            t = flip_matrix @ t
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        # Log camera transform
        rr.log(f"{entity_path}/transform", rr.Transform3D(translation=t, mat3x3=R))
        
        # Create and log frustum (smaller for dynamic cameras)
        frustum_points, frustum_lines = create_camera_frustum(intrinsics, depth=0.3, scale=0.5)
        
        # Transform frustum points to world coordinates
        frustum_points_homogeneous = np.hstack([frustum_points, np.ones((frustum_points.shape[0], 1))])
        frustum_world = (transform @ frustum_points_homogeneous.T).T[:, :3]
        
        # Log frustum
        frustum_strips = []
        for line in frustum_lines:
            strip = [frustum_world[line[0]], frustum_world[line[1]]]
            frustum_strips.append(strip)
        
        rr.log(
            f"{entity_path}/frustum",
            rr.LineStrips3D(
                strips=frustum_strips,
                colors=[color] * len(frustum_strips)
            )
        )
        
        # Log camera position as a point
        rr.log(
            f"{entity_path}/position", 
            rr.Points3D(
                positions=[t],
                radii=[0.015],
                colors=[color]
            )
        )

def main():
    parser = argparse.ArgumentParser(description='Visualize camera poses in Rerun')
    parser.add_argument('--data-dir', help='Directory containing camera data (searched recursively)')
    parser.add_argument('--app-id', default='camera_poses', help='Rerun application ID')
    parser.add_argument('--flip-ground', action='store_true', help='Flip scene over ground plane (rotate 180° around X-axis)')
    parser.add_argument('--add-ground-plane', action='store_true', help='Add a ground plane at Z=0 for reference')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Initialize Rerun
    rr.init(args.app_id)
    rr.spawn()
    
    print("Loading camera data...")
    
    # Load static cameras recursively
    static_cameras = {}
    # Known calibration file name patterns
    calibration_patterns = [
        'p1Calibration.json',
        'p2Calibration.json',
        'p3Calibration.json',
        'calibration.json'
    ]
    found_calibration_files = []
    for pattern in calibration_patterns:
        found_calibration_files.extend(list(data_dir.rglob(pattern)))
    # De-duplicate paths
    found_calibration_files = sorted(set(found_calibration_files), key=lambda p: str(p))
    for camera_path in found_calibration_files:
        try:
            with open(camera_path, 'r') as f:
                camera_data = json.load(f)
            camera_name = camera_data.get('camera_name')
            if not camera_name:
                # Fall back to parent directory name (e.g., p1, p2, p3)
                camera_name = camera_path.parent.name
                camera_data['camera_name'] = camera_name
            # Only log as static if extrinsics are present
            extrinsics = camera_data.get('extrinsics', {})
            if not (isinstance(extrinsics, dict) and 'R' in extrinsics and 'tvec' in extrinsics):
                print(f"Skipping static camera without extrinsics: {camera_name} from {camera_path}")
                continue
            # Only keep first instance per camera name
            if camera_name not in static_cameras:
                static_cameras[camera_name] = camera_data
                print(f"Loaded static camera: {camera_name} from {camera_path}")
        except Exception as e:
            print(f"Warning: failed to load calibration from {camera_path}: {e}")
    
    # Load dynamic camera poses (prefer *_world.csv for ego)
    def pick_first(paths):
        return paths[0] if len(paths) > 0 else None
    
    # Prefer files under an 'ego' folder if multiple exist
    def sort_ego_first(paths):
        def score(p):
            parts = set(p.parts)
            in_ego = 0 if 'ego' in parts else 1
            # shorter path first as a tie-breaker
            return (in_ego, len(p.parts), str(p))
        return sorted(paths, key=score)
    
    left_world_candidates = sort_ego_first(list(data_dir.rglob('left_poses_world.csv')))
    right_world_candidates = sort_ego_first(list(data_dir.rglob('right_poses_world.csv')))
    left_regular_candidates = sort_ego_first(list(data_dir.rglob('left_poses.csv')))
    right_regular_candidates = sort_ego_first(list(data_dir.rglob('right_poses.csv')))
    
    left_poses_path = pick_first(left_world_candidates) or pick_first(left_regular_candidates)
    right_poses_path = pick_first(right_world_candidates) or pick_first(right_regular_candidates)
    
    left_poses = None
    right_poses = None
    
    if left_poses_path is not None and left_poses_path.exists():
        left_poses = pd.read_csv(left_poses_path)
        print(f"Loaded left poses from {left_poses_path}: {len(left_poses)} frames")
    
    if right_poses_path is not None and right_poses_path.exists():
        right_poses = pd.read_csv(right_poses_path)
        print(f"Loaded right poses from {right_poses_path}: {len(right_poses)} frames")
    
    print("Logging camera poses to Rerun...")
    
    # Log static cameras (no timeline, always visible)
    rr.set_time_sequence("frame", 0)
    for camera_name, camera_data in static_cameras.items():
        log_static_camera(camera_name, camera_data, "cameras/static", flip_ground=args.flip_ground)
        print(f"Logged static camera: {camera_name}")
    
    # Create generic intrinsics for dynamic cameras
    generic_intrinsics = create_generic_intrinsics()
    
    # Log dynamic cameras
    if left_poses is not None:
        print("Logging left camera poses...")
        log_dynamic_camera(
            "left", 
            left_poses, 
            generic_intrinsics, 
            "cameras/dynamic",
            color=(100, 255, 100, 255),  # Green
            flip_ground=args.flip_ground
        )
    
    if right_poses is not None:
        print("Logging right camera poses...")
        log_dynamic_camera(
            "right", 
            right_poses, 
            generic_intrinsics, 
            "cameras/dynamic",
            color=(100, 100, 255, 255),  # Blue
            flip_ground=args.flip_ground
        )
    
    # Add world coordinate system
    rr.set_time_sequence("frame", 0)
    
    # Apply flip to world axes if requested
    world_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if args.flip_ground:
        flip_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0], 
            [0, 0, -1]
        ])
        world_axes = (flip_matrix @ world_axes.T).T
    
    rr.log(
        "world/axes",
        rr.Arrows3D(
            vectors=world_axes.tolist(),
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            radii=[0.01, 0.01, 0.01]
        )
    )
    
    print("Visualization complete! Check the Rerun viewer.")
    print("Static cameras (p1, p2, p3) are shown in red and remain constant.")
    print("Dynamic cameras (left, right) are shown in green/blue and change over time.")
    print("Use the timeline controls to scrub through the dynamic poses.")

if __name__ == '__main__':
    main()