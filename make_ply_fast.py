#!/usr/bin/env python3
"""
Fast Insta360 X5 Video → PLY Point Cloud
Uses equirectangular → sphere projection with depth variation from image features.
No external dependencies beyond opencv, numpy, open3d.
"""

import sys
import os
import time
import cv2
import numpy as np


def extract_frames(video_path, output_dir, fps=2, max_frames=100):
    """Extract frames from video."""
    print(f"[1/4] Extracting frames at {fps} fps...")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   {w}x{h}, {video_fps:.0f}fps, {total/video_fps:.1f}s")

    interval = max(1, int(video_fps / fps))
    frames = []
    idx = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret or extracted >= max_frames:
            break
        if idx % interval == 0:
            # Resize for processing
            if w > 2048:
                scale = 2048 / w
                frame = cv2.resize(frame, (2048, int(h * scale)))
            path = os.path.join(output_dir, f"frame_{extracted:04d}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames.append(path)
            extracted += 1
        idx += 1

    cap.release()
    print(f"   {extracted} frames extracted")
    return frames


def estimate_depth_laplacian(img_gray):
    """Estimate relative depth from image using edge density and blur analysis."""
    # Laplacian variance gives focus/depth cue
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)

    # Use block-wise variance as depth proxy (sharper = closer)
    block_size = 16
    h, w = img_gray.shape
    depth = np.zeros((h // block_size, w // block_size), dtype=np.float64)

    for by in range(depth.shape[0]):
        for bx in range(depth.shape[1]):
            block = lap[by*block_size:(by+1)*block_size, bx*block_size:(bx+1)*block_size]
            depth[by, bx] = np.var(block)

    # Normalize and invert (high variance = close = small depth)
    depth = cv2.GaussianBlur(depth, (5, 5), 0)
    if depth.max() > depth.min():
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    # Invert: sharp areas (high variance) are closer
    depth = 1.0 - depth * 0.5  # Range [0.5, 1.0]

    # Resize back to original
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    return depth


def equirect_to_pointcloud(frames, sample_step=4, depth_variation=0.3):
    """Convert equirectangular frames to colored 3D point cloud with depth."""
    print(f"[2/4] Converting {len(frames)} frames to 3D points...")

    all_points = []
    all_colors = []

    # Use evenly spaced subset of frames for different "layers"
    frame_step = max(1, len(frames) // 25)

    for fi in range(0, len(frames), frame_step):
        img = cv2.imread(frames[fi])
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img_rgb.shape[:2]

        # Estimate depth from image features
        depth_map = estimate_depth_laplacian(img_gray)

        # Frame position along the path (simulates camera movement)
        t = fi / max(1, len(frames) - 1)

        for y in range(0, h, sample_step):
            for x in range(0, w, sample_step):
                # Equirectangular → spherical coordinates
                theta = (x / w) * 2 * np.pi - np.pi   # longitude [-pi, pi]
                phi = (0.5 - y / h) * np.pi            # latitude [-pi/2, pi/2]

                # Depth variation from image analysis
                local_depth = depth_map[y, x]
                radius = 2.0 + local_depth * depth_variation

                # Add slight path offset for different frames
                path_offset = t * 0.3

                # Spherical to Cartesian
                px = (radius + path_offset) * np.cos(phi) * np.cos(theta)
                py = radius * np.sin(phi)
                pz = (radius + path_offset) * np.cos(phi) * np.sin(theta)

                all_points.append([px, py, pz])
                all_colors.append(img_rgb[y, x])

        print(f"   Frame {fi}/{len(frames)}: {len(all_points):,} total points")

    return np.array(all_points, dtype=np.float64), np.array(all_colors, dtype=np.uint8)


def write_ply(filename, points, colors):
    """Write PLY file with vertices and colors."""
    print(f"[3/4] Writing PLY ({len(points):,} points)...")

    with open(filename, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))

        # Binary data
        for i in range(len(points)):
            f.write(np.array(points[i], dtype=np.float32).tobytes())
            f.write(np.array(colors[i], dtype=np.uint8).tobytes())

    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"   Saved: {filename} ({size_mb:.1f} MB)")


def write_ply_fast(filename, points, colors):
    """Write PLY file efficiently using numpy."""
    print(f"[3/4] Writing PLY ({len(points):,} points)...")

    # Create structured array
    n = len(points)
    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])

    vertex_data = np.zeros(n, dtype=dtype)
    vertex_data['x'] = points[:, 0].astype(np.float32)
    vertex_data['y'] = points[:, 1].astype(np.float32)
    vertex_data['z'] = points[:, 2].astype(np.float32)
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]

    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertex_data.tobytes())

    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"   Saved: {filename} ({size_mb:.1f} MB)")


def optimize_pointcloud(points, colors, voxel_size=0.015):
    """Downsample and clean the point cloud using Open3D."""
    print(f"[4/4] Optimizing point cloud...")

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        # Voxel downsample
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"   After voxel downsample: {len(pcd.points):,} points")

        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"   After outlier removal: {len(pcd.points):,} points")

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        return (
            np.asarray(pcd.points),
            (np.asarray(pcd.colors) * 255).astype(np.uint8),
        )
    except ImportError:
        print("   Open3D not available, skipping optimization")
        return points, colors


def main():
    if len(sys.argv) < 2:
        print("Usage: python make_ply_fast.py <video.mp4> [output.ply] [--high]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else video_path.rsplit('.', 1)[0] + '.ply'

    high_quality = '--high' in sys.argv

    if high_quality:
        fps, max_frames, sample_step = 3, 100, 3
    else:
        fps, max_frames, sample_step = 2, 50, 4

    print("=" * 60)
    print("  Insta360 X5 → PLY (Fast Pipeline)")
    print("=" * 60)
    print(f"  Input:  {video_path}")
    print(f"  Output: {output_path}")
    print(f"  Quality: {'high' if high_quality else 'standard'}")
    print("=" * 60)

    t0 = time.time()

    # Extract frames
    workspace = os.path.splitext(video_path)[0] + '_workspace'
    frames_dir = os.path.join(workspace, 'frames')
    frames = extract_frames(video_path, frames_dir, fps=fps, max_frames=max_frames)

    if len(frames) < 2:
        print("Error: Not enough frames")
        sys.exit(1)

    # Convert to point cloud
    points, colors = equirect_to_pointcloud(frames, sample_step=sample_step)

    # Optimize
    points, colors = optimize_pointcloud(points, colors)

    # Write PLY
    write_ply_fast(output_path, points, colors)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed:.1f}s!")
    print(f"  Output: {output_path}")
    print(f"  Points: {len(points):,}")
    print(f"  View at: https://mrfeixiang.github.io/insta360-ply-viewer/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
