#!/usr/bin/env python3
"""
Insta360 X5 Video → PLY Point Cloud Pipeline

This script takes a 360° video (MP4/INSV) from an Insta360 X5 camera,
extracts frames, runs Structure-from-Motion (SfM) via COLMAP,
performs dense reconstruction, and exports a colored .ply point cloud.

Usage:
    python make_ply.py <video_path> [--output output.ply] [--fps 1] [--quality high]

Requirements:
    pip install pycolmap open3d opencv-python-headless numpy plyfile torch
"""

import argparse
import sys
import os
import shutil
import time
import subprocess
from pathlib import Path

import cv2
import numpy as np


def extract_frames(video_path, output_dir, fps=1, max_frames=200):
    """Extract frames from 360° video at specified FPS."""
    print(f"\n[1/5] Extracting frames from {video_path} at {fps} fps...")

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"   Video: {width}x{height}, {video_fps:.1f}fps, {duration:.1f}s, {total_frames} frames")

    frame_interval = max(1, int(video_fps / fps))
    extracted = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or extracted >= max_frames:
            break

        if frame_idx % frame_interval == 0:
            # Resize if too large (for faster processing)
            h, w = frame.shape[:2]
            if w > 3840:
                scale = 3840 / w
                frame = cv2.resize(frame, (3840, int(h * scale)))

            filename = f"frame_{extracted:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted += 1

            if extracted % 10 == 0:
                print(f"   Extracted {extracted} frames...")

        frame_idx += 1

    cap.release()
    print(f"   Done: {extracted} frames extracted to {output_dir}")
    return extracted


def equirect_to_cubemap(img, face_size=1024):
    """Convert equirectangular image to 6 cubemap faces for better SfM."""
    h, w = img.shape[:2]
    faces = {}
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']

    # Create lookup maps for each face
    for face_idx, name in enumerate(face_names):
        face = np.zeros((face_size, face_size, 3), dtype=np.uint8)

        for y in range(face_size):
            for x in range(face_size):
                # Normalize to [-1, 1]
                nx = (2.0 * x / face_size - 1.0)
                ny = (2.0 * y / face_size - 1.0)

                # Convert to 3D direction based on face
                if face_idx == 0:    # front
                    vec = np.array([1, -ny, -nx])
                elif face_idx == 1:  # right
                    vec = np.array([nx, -ny, 1])
                elif face_idx == 2:  # back
                    vec = np.array([-1, -ny, nx])
                elif face_idx == 3:  # left
                    vec = np.array([-nx, -ny, -1])
                elif face_idx == 4:  # top
                    vec = np.array([nx, 1, ny])
                else:                # bottom
                    vec = np.array([nx, -1, -ny])

                # Convert to spherical
                vec = vec / np.linalg.norm(vec)
                theta = np.arctan2(vec[2], vec[0])
                phi = np.arcsin(np.clip(vec[1], -1, 1))

                # Map to equirect coords
                u = (theta / np.pi + 1) * 0.5 * w
                v = (-phi / (np.pi/2) + 1) * 0.5 * h

                u = np.clip(int(u), 0, w-1)
                v = np.clip(int(v), 0, h-1)

                face[y, x] = img[v, u]

        faces[name] = face

    return faces


def extract_perspective_views(frames_dir, perspective_dir, num_views=8):
    """Extract perspective crop views from equirectangular frames for better SfM matching."""
    print(f"\n[2/5] Extracting perspective views from equirectangular frames...")

    os.makedirs(perspective_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    total_views = 0

    # For each frame, extract several perspective crops
    # This is much faster than full cubemap conversion
    for fi, fname in enumerate(frame_files):
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue

        h, w = img.shape[:2]
        crop_w = w // 4  # ~90° horizontal FOV
        crop_h = h // 2

        # Extract views at different horizontal angles
        angles = np.linspace(0, w, num_views, endpoint=False, dtype=int)

        for vi, start_x in enumerate(angles):
            end_x = start_x + crop_w

            # Vertical center crop
            start_y = h // 4
            end_y = start_y + crop_h

            if end_x <= w:
                crop = img[start_y:end_y, start_x:end_x]
            else:
                # Wrap around
                part1 = img[start_y:end_y, start_x:w]
                part2 = img[start_y:end_y, 0:end_x-w]
                crop = np.hstack([part1, part2])

            # Resize to reasonable size
            crop = cv2.resize(crop, (1280, 720))

            out_name = f"view_{fi:04d}_{vi:02d}.jpg"
            cv2.imwrite(os.path.join(perspective_dir, out_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total_views += 1

        if (fi + 1) % 10 == 0:
            print(f"   Processed {fi+1}/{len(frame_files)} frames...")

    print(f"   Done: {total_views} perspective views extracted")
    return total_views


def run_colmap_sfm(image_dir, workspace_dir):
    """Run COLMAP Structure-from-Motion pipeline via pycolmap."""
    print(f"\n[3/5] Running Structure-from-Motion (COLMAP)...")

    import pycolmap

    database_path = os.path.join(workspace_dir, "database.db")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    # Feature extraction
    print("   Extracting features...")
    t0 = time.time()
    extraction_opts = pycolmap.FeatureExtractionOptions()
    extraction_opts.sift.max_num_features = 8192
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.AUTO,
        extraction_options=extraction_opts,
    )
    print(f"   Features extracted in {time.time()-t0:.1f}s")

    # Feature matching
    print("   Matching features...")
    t0 = time.time()
    pairing_opts = pycolmap.SequentialPairingOptions()
    pairing_opts.overlap = 10
    pairing_opts.loop_detection = True
    pycolmap.match_sequential(
        database_path=database_path,
        pairing_options=pairing_opts,
    )
    print(f"   Matching done in {time.time()-t0:.1f}s")

    # Incremental mapping
    print("   Running incremental mapper...")
    t0 = time.time()
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_dir,
    )
    print(f"   Mapping done in {time.time()-t0:.1f}s")

    if not reconstructions:
        raise RuntimeError("COLMAP failed to reconstruct any models. Try with more frames or different FPS.")

    # Use the largest reconstruction
    best_idx = max(reconstructions, key=lambda k: reconstructions[k].num_points3D())
    best_recon = reconstructions[best_idx]

    print(f"   Best reconstruction: {best_recon.num_images()} images, "
          f"{best_recon.num_points3D()} 3D points, "
          f"{best_recon.num_reg_images()} registered")

    # Save the reconstruction
    best_recon.write(sparse_dir)

    return best_recon, sparse_dir


def reconstruction_to_ply(reconstruction, output_path, frames_dir=None):
    """Convert COLMAP reconstruction to colored PLY point cloud."""
    print(f"\n[4/5] Generating PLY point cloud...")

    points3D = reconstruction.points3D

    num_points = len(points3D)
    print(f"   {num_points} 3D points from SfM")

    positions = np.zeros((num_points, 3), dtype=np.float64)
    colors = np.zeros((num_points, 3), dtype=np.uint8)

    for i, (pid, point) in enumerate(points3D.items()):
        positions[i] = point.xyz
        colors[i] = point.color  # RGB from COLMAP

    # Write PLY file
    write_ply(output_path, positions, colors)
    print(f"   Sparse PLY saved: {output_path} ({num_points} points)")

    return num_points


def densify_point_cloud(sparse_ply, frames_dir, workspace_dir, output_path):
    """Use Open3D to densify the point cloud with multi-view stereo estimation."""
    print(f"\n[5/5] Densifying point cloud with depth estimation...")

    import open3d as o3d

    # Load sparse point cloud
    pcd = o3d.io.read_point_cloud(sparse_ply)
    num_sparse = len(pcd.points)
    print(f"   Sparse cloud: {num_sparse} points")

    if num_sparse < 100:
        print("   Warning: Very few points. Trying to augment with depth estimation...")
        # Fall back to creating points from frames directly
        pcd = create_dense_from_frames(frames_dir, output_path)
    else:
        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Save final PLY
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
    final_count = len(pcd.points)
    print(f"   Final PLY saved: {output_path} ({final_count} points)")

    return final_count


def create_dense_from_frames(frames_dir, output_path):
    """Create a dense point cloud from frames using monocular depth estimation."""
    import open3d as o3d
    import torch

    print("   Using torch MiDaS for monocular depth estimation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Try to use MiDaS for depth estimation
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        midas.to(device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.small_transform
    except Exception as e:
        print(f"   MiDaS not available ({e}), creating point cloud from image colors...")
        return create_colored_pointcloud_from_frames(frames_dir)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    all_points = []
    all_colors = []

    # Process subset of frames
    step = max(1, len(frame_files) // 20)  # Use ~20 frames

    for fi in range(0, len(frame_files), step):
        fname = frame_files[fi]
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Predict depth
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            depth = midas(input_batch)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=img_rgb.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

        # Convert to point cloud
        h, w = depth.shape
        # Sample points (not all pixels - too many)
        sample_step = 4
        fx = fy = w * 0.5  # Approximate focal length
        cx, cy = w / 2, h / 2

        for y in range(0, h, sample_step):
            for x in range(0, w, sample_step):
                z = depth[y, x]
                if z > 0:
                    # Normalize depth
                    z_norm = z / depth.max() * 5.0
                    px = (x - cx) * z_norm / fx
                    py = (y - cy) * z_norm / fy

                    # Add frame offset for spatial separation
                    offset_x = fi * 0.5

                    all_points.append([px + offset_x, -py, z_norm])
                    all_colors.append(img_rgb[y, x] / 255.0)

        print(f"   Processed frame {fi+1}/{len(frame_files)}: {len(all_points)} total points")

    points_np = np.array(all_points, dtype=np.float64)
    colors_np = np.array(all_colors, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size=0.02)

    return pcd


def create_colored_pointcloud_from_frames(frames_dir):
    """Fallback: create a simple colored point cloud from equirectangular frames."""
    import open3d as o3d

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    all_points = []
    all_colors = []

    step = max(1, len(frame_files) // 15)

    for fi in range(0, len(frame_files), step):
        fname = frame_files[fi]
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Equirectangular → 3D sphere projection
        sample_step = 8
        radius = 2.0
        frame_angle_offset = (fi / len(frame_files)) * 0.3

        for y in range(0, h, sample_step):
            for x in range(0, w, sample_step):
                # Convert equirect coords to spherical
                theta = (x / w) * 2 * np.pi - np.pi  # longitude
                phi = (y / h) * np.pi - np.pi / 2     # latitude

                r = radius + frame_angle_offset
                px = r * np.cos(phi) * np.cos(theta)
                py = r * np.sin(phi)
                pz = r * np.cos(phi) * np.sin(theta)

                all_points.append([px, py, pz])
                all_colors.append(img_rgb[y, x] / 255.0)

    points_np = np.array(all_points, dtype=np.float64)
    colors_np = np.array(all_colors, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    return pcd


def write_ply(output_path, positions, colors):
    """Write PLY file with vertex positions and colors."""
    from plyfile import PlyData, PlyElement

    vertex_data = np.zeros(len(positions), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])

    vertex_data['x'] = positions[:, 0]
    vertex_data['y'] = positions[:, 1]
    vertex_data['z'] = positions[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=False).write(output_path)


def main():
    parser = argparse.ArgumentParser(description="Insta360 X5 Video → PLY Point Cloud")
    parser.add_argument("video", help="Path to 360° video file (.mp4, .insv)")
    parser.add_argument("--output", "-o", default=None, help="Output PLY file path")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate (default: 1)")
    parser.add_argument("--max-frames", type=int, default=150, help="Maximum frames to extract (default: 150)")
    parser.add_argument("--num-views", type=int, default=6, help="Perspective views per frame (default: 6)")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium",
                        help="Processing quality (default: medium)")
    parser.add_argument("--skip-sfm", action="store_true", help="Skip SfM, use depth estimation only")

    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Setup output
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = video_path.with_suffix('.ply')

    # Quality presets
    quality_settings = {
        "low": {"fps": 0.5, "max_frames": 60, "num_views": 4},
        "medium": {"fps": 1.0, "max_frames": 150, "num_views": 6},
        "high": {"fps": 2.0, "max_frames": 250, "num_views": 8},
    }

    qs = quality_settings[args.quality]
    fps = args.fps or qs["fps"]
    max_frames = args.max_frames or qs["max_frames"]
    num_views = args.num_views or qs["num_views"]

    # Workspace
    workspace = video_path.parent / f"_ply_workspace_{video_path.stem}"
    frames_dir = workspace / "frames"
    perspective_dir = workspace / "perspectives"
    colmap_dir = workspace / "colmap"
    sparse_ply = workspace / "sparse.ply"

    os.makedirs(workspace, exist_ok=True)
    os.makedirs(colmap_dir, exist_ok=True)

    print("=" * 60)
    print("  Insta360 X5 Video → PLY Point Cloud Pipeline")
    print("=" * 60)
    print(f"  Input:   {video_path}")
    print(f"  Output:  {output_path}")
    print(f"  Quality: {args.quality} (fps={fps}, max_frames={max_frames})")
    print("=" * 60)

    t_start = time.time()

    # Step 1: Extract frames
    num_frames = extract_frames(video_path, str(frames_dir), fps=fps, max_frames=max_frames)

    if num_frames < 3:
        print("Error: Not enough frames extracted. Video may be too short.")
        sys.exit(1)

    if not args.skip_sfm:
        # Step 2: Extract perspective views
        extract_perspective_views(str(frames_dir), str(perspective_dir), num_views=num_views)

        # Step 3: Run COLMAP SfM
        try:
            reconstruction, sparse_path = run_colmap_sfm(str(perspective_dir), str(colmap_dir))

            # Step 4: Export sparse PLY
            num_points = reconstruction_to_ply(reconstruction, str(sparse_ply), str(frames_dir))

            # Step 5: Densify
            if num_points > 50:
                final_count = densify_point_cloud(str(sparse_ply), str(frames_dir), str(workspace), str(output_path))
            else:
                print("   SfM produced few points, falling back to depth estimation...")
                args.skip_sfm = True

        except Exception as e:
            print(f"\n   SfM failed: {e}")
            print("   Falling back to depth-based reconstruction...")
            args.skip_sfm = True

    if args.skip_sfm:
        # Depth-only pipeline
        print("\n[3/5] Skipping SfM, using depth estimation...")
        import open3d as o3d

        pcd = create_dense_from_frames(str(frames_dir), str(output_path))

        # Clean up
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        final_count = len(pcd.points)
        print(f"\n   Final PLY: {output_path} ({final_count} points)")

    elapsed = time.time() - t_start
    file_size = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 60)
    print(f"  Done! Total time: {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Points: {final_count:,}")
    print("=" * 60)
    print(f"\n  View it at: https://mrfeixiang.github.io/insta360-ply-viewer/")
    print(f"  Or locally: open {output_path}")


if __name__ == "__main__":
    main()
