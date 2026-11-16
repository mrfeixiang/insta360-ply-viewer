#!/usr/bin/env python3
"""
Insta360 X5 .insv (dual fisheye) → PLY Point Cloud Pipeline

Processes raw .insv files with dual 3840x3840 fisheye streams.
Converts fisheye → equirectangular → 3D colored point cloud.

Usage:
    python make_ply_insv.py <file.insv> [--output scene.ply] [--high] [--web]
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path


def extract_dual_fisheye_frames(insv_path, output_dir, fps=2, max_frames=60):
    """Extract frames from both fisheye lens streams."""
    print(f"[1/5] Extracting dual fisheye frames at {fps} fps...")
    os.makedirs(output_dir, exist_ok=True)

    frames = {"lens0": [], "lens1": []}

    for stream_idx in range(2):
        cap = cv2.VideoCapture(insv_path)
        # Open specific stream
        cap = cv2.VideoCapture(insv_path, cv2.CAP_FFMPEG)

        if stream_idx == 1:
            # For second stream, use ffmpeg to extract
            import subprocess
            temp_dir = os.path.join(output_dir, f"lens{stream_idx}")
            os.makedirs(temp_dir, exist_ok=True)

            cmd = [
                "ffmpeg", "-y", "-i", insv_path,
                "-map", f"0:v:{stream_idx}",
                "-vf", f"fps={fps}",
                "-frames:v", str(max_frames),
                "-q:v", "2",
                os.path.join(temp_dir, "frame_%04d.jpg")
            ]
            subprocess.run(cmd, capture_output=True)

            frame_files = sorted([
                os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
                if f.endswith('.jpg')
            ])
            frames[f"lens{stream_idx}"] = frame_files
            print(f"   Lens {stream_idx}: {len(frame_files)} frames")
            continue

        # First stream via OpenCV
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 24
        interval = max(1, int(video_fps / fps))
        idx = 0
        extracted = 0
        lens_dir = os.path.join(output_dir, "lens0")
        os.makedirs(lens_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret or extracted >= max_frames:
                break
            if idx % interval == 0:
                path = os.path.join(lens_dir, f"frame_{extracted:04d}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frames["lens0"].append(path)
                extracted += 1
            idx += 1

        cap.release()
        print(f"   Lens 0: {extracted} frames")

    return frames


def fisheye_to_equirect(fisheye_img, out_w=2048, out_h=1024, fov_deg=200):
    """Convert a circular fisheye image to equirectangular projection."""
    h, w = fisheye_img.shape[:2]
    cx, cy = w / 2, h / 2
    radius = min(cx, cy) * 0.95  # Fisheye image circle radius

    fov = np.radians(fov_deg)

    # Create output map
    equirect = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Build lookup table
    u = np.arange(out_w)
    v = np.arange(out_h)
    uu, vv = np.meshgrid(u, v)

    # Equirect coords to spherical
    theta = (uu / out_w - 0.5) * np.pi  # longitude [-pi/2, pi/2] for half sphere
    phi = (0.5 - vv / out_h) * np.pi     # latitude [-pi/2, pi/2]

    # Spherical to 3D
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    # 3D to fisheye (equisolid projection)
    r3d = np.sqrt(x**2 + y**2)
    angle = np.arctan2(r3d, z)

    # Equisolid projection
    r_fish = 2 * np.sin(angle / 2) * radius / np.sin(fov / 4)

    # Direction in fisheye plane
    fish_angle = np.arctan2(y, x)

    # Fisheye pixel coords
    fx = (cx + r_fish * np.cos(fish_angle)).astype(np.float32)
    fy = (cy - r_fish * np.sin(fish_angle)).astype(np.float32)

    # Mask valid pixels
    mask = (fx >= 0) & (fx < w) & (fy >= 0) & (fy < h) & (angle < fov / 2)

    # Remap
    equirect = cv2.remap(fisheye_img, fx, fy, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Zero out invalid areas
    equirect[~mask] = 0

    return equirect


def stitch_dual_fisheye(lens0_img, lens1_img, out_w=4096, out_h=2048):
    """Stitch two fisheye images into a full equirectangular panorama."""
    half_w = out_w // 2

    # Convert each fisheye to half equirectangular
    equirect0 = fisheye_to_equirect(lens0_img, half_w, out_h, fov_deg=200)
    equirect1 = fisheye_to_equirect(lens1_img, half_w, out_h, fov_deg=200)

    # Flip lens1 horizontally (rear lens)
    equirect1 = cv2.flip(equirect1, 1)

    # Combine: front + back
    panorama = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    panorama[:, :half_w] = equirect0
    panorama[:, half_w:] = equirect1

    # Blend seam at the junction
    blend_w = min(out_w // 20, half_w // 4)
    for i in range(blend_w):
        alpha = i / blend_w
        # Left seam (where equirect1 ends meet equirect0 start)
        c_left = i
        c_right = half_w + half_w - blend_w + i
        if c_left < half_w and c_right < out_w:
            panorama[:, c_left] = cv2.addWeighted(
                equirect0[:, c_left], alpha,
                equirect1[:, half_w - blend_w + i], 1 - alpha, 0
            )
        # Center seam
        c = half_w - blend_w // 2 + i
        if 0 <= c < half_w and i < half_w:
            panorama[:, c] = cv2.addWeighted(
                equirect0[:, c], 1 - alpha,
                equirect1[:, i], alpha, 0
            )

    return panorama


def estimate_depth_multiscale(gray_img):
    """Estimate depth using multi-scale edge analysis."""
    h, w = gray_img.shape

    # Multi-scale Laplacian for depth cues
    depths = []
    for sigma in [1, 2, 4, 8]:
        blurred = cv2.GaussianBlur(gray_img, (0, 0), sigma)
        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        lap_var = cv2.GaussianBlur(np.abs(lap), (31, 31), 0)
        depths.append(lap_var)

    # Combine scales (fine detail = close, coarse = far)
    depth = depths[0] * 0.4 + depths[1] * 0.3 + depths[2] * 0.2 + depths[3] * 0.1

    # Normalize
    if depth.max() > depth.min():
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    # Invert: sharp = close (smaller radius in sphere projection)
    # Map to depth range [0.7, 1.3]
    depth = 0.7 + depth * 0.6

    return depth


def equirect_frames_to_pointcloud(frames_lens0, frames_lens1, sample_step=4, stitch=True):
    """Convert dual fisheye frames to 3D point cloud."""
    print(f"[3/5] Building 3D point cloud from {len(frames_lens0)} frame pairs...")

    all_points = []
    all_colors = []

    n_frames = min(len(frames_lens0), len(frames_lens1))
    frame_step = max(1, n_frames // 20)

    for fi in range(0, n_frames, frame_step):
        img0 = cv2.imread(frames_lens0[fi])
        img1 = cv2.imread(frames_lens1[fi]) if fi < len(frames_lens1) else None

        if img0 is None:
            continue

        if stitch and img1 is not None:
            # Stitch dual fisheye to equirectangular
            panorama = stitch_dual_fisheye(img0, img1, out_w=2048, out_h=1024)
        else:
            # Use single fisheye as half-sphere
            panorama = fisheye_to_equirect(img0, 2048, 1024, fov_deg=200)

        panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        h, w = panorama.shape[:2]

        # Depth estimation
        depth_map = estimate_depth_multiscale(gray)

        # Frame time for path offset
        t = fi / max(1, n_frames - 1)

        for y in range(0, h, sample_step):
            for x in range(0, w, sample_step):
                # Skip black areas (invalid projection)
                if panorama_rgb[y, x].sum() < 15:
                    continue

                # Equirectangular → spherical
                theta = (x / w) * 2 * np.pi - np.pi  # longitude
                phi = (0.5 - y / h) * np.pi            # latitude

                # Depth modulation
                local_depth = depth_map[y, x]
                radius = 2.0 * local_depth

                # Camera path offset
                path_x = t * 0.4
                path_z = t * 0.1

                # Spherical → Cartesian
                px = radius * np.cos(phi) * np.cos(theta) + path_x
                py = radius * np.sin(phi)
                pz = radius * np.cos(phi) * np.sin(theta) + path_z

                all_points.append([px, py, pz])
                all_colors.append(panorama_rgb[y, x])

        count = len(all_points)
        print(f"   Frame {fi}/{n_frames}: {count:,} points")

    return np.array(all_points, dtype=np.float64), np.array(all_colors, dtype=np.uint8)


def optimize_and_save(points, colors, output_path, voxel_size=0.015):
    """Optimize point cloud and save as PLY."""
    print(f"[4/5] Optimizing {len(points):,} points...")

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    # Voxel downsample
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"   After downsample: {len(pcd.points):,}")

    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"   After cleanup: {len(pcd.points):,}")

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    print(f"[5/5] Saving PLY...")
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   {output_path}: {len(pcd.points):,} points, {size_mb:.1f} MB")

    return len(pcd.points)


def create_web_version(full_ply_path, web_ply_path, target_mb=5.0):
    """Create a web-optimized smaller version."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(full_ply_path)
    n = len(pcd.points)

    # Estimate voxel size to hit target file size
    # ~15 bytes per point in binary PLY
    target_points = int(target_mb * 1024 * 1024 / 15)
    if n > target_points:
        ratio = n / target_points
        voxel = 0.015 * (ratio ** (1/3))
        pcd = pcd.voxel_down_sample(voxel_size=voxel)

    o3d.io.write_point_cloud(web_ply_path, pcd, write_ascii=False)
    size = os.path.getsize(web_ply_path) / (1024 * 1024)
    print(f"   Web version: {len(pcd.points):,} points, {size:.1f} MB")
    return len(pcd.points)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Insta360 X5 .insv → PLY")
    parser.add_argument("insv", help="Path to .insv file")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--high", action="store_true", help="High quality")
    parser.add_argument("--web", action="store_true", help="Also create web-optimized version")
    parser.add_argument("--web-only", action="store_true", help="Only create web version")
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--sample-step", type=int, default=0)
    parser.add_argument("--name", default=None, help="Scene name for web gallery")
    args = parser.parse_args()

    insv_path = os.path.abspath(args.insv)
    if not os.path.exists(insv_path):
        print(f"Error: {insv_path} not found")
        sys.exit(1)

    stem = Path(insv_path).stem
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_path = os.path.join(os.path.dirname(insv_path), f"{stem}.ply")

    # Quality settings
    if args.high:
        fps, sample_step, voxel = 3, 3, 0.012
    else:
        fps, sample_step, voxel = 2, 4, 0.015

    if args.fps:
        fps = args.fps
    if args.sample_step:
        sample_step = args.sample_step

    print("=" * 60)
    print("  Insta360 X5 .insv → PLY Pipeline")
    print("=" * 60)
    print(f"  Input:   {insv_path}")
    print(f"  Output:  {output_path}")
    print(f"  Quality: {'high' if args.high else 'standard'}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Extract frames from both lenses
    workspace = os.path.join(os.path.dirname(output_path), f"_workspace_{stem}")
    frames = extract_dual_fisheye_frames(insv_path, workspace, fps=fps, max_frames=80)

    print(f"\n[2/5] Stitching dual fisheye to equirectangular...")

    # Step 3: Build point cloud
    points, colors = equirect_frames_to_pointcloud(
        frames["lens0"], frames["lens1"],
        sample_step=sample_step, stitch=True
    )

    if len(points) == 0:
        print("Error: No points generated")
        sys.exit(1)

    # Step 4-5: Optimize and save
    n_points = optimize_and_save(points, colors, output_path, voxel_size=voxel)

    # Web version
    if args.web or args.web_only:
        web_path = output_path.replace('.ply', '_web.ply')
        create_web_version(output_path, web_path, target_mb=4.0)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed:.1f}s!")
    print(f"  Output: {output_path}")
    print(f"  Points: {n_points:,}")
    print(f"  View at: https://mrfeixiang.github.io/insta360-ply-viewer/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
