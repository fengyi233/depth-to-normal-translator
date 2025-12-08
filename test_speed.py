"""
Speed comparison script for depth-to-normal conversion libraries.
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from d2nt import depth2normal
from d2nt.utils import get_cam_params, get_depth, get_normal_gt, vector_normalization, get_normal_vis

import open3d as o3d
import torch
import kornia


def method_kornia(depth_tensor, camera_matrix_tensor):
    """Kornia depth_to_normals function."""
    normal_tensor = kornia.geometry.depth_to_normals(depth_tensor, camera_matrix=camera_matrix_tensor)
    return normal_tensor


def method_open3d(o3d_depth, o3d_intrinsic):
    """Open3D depth to normals via point cloud."""
    depth_np = np.asarray(o3d_depth)
    height, width = depth_np.shape[:2]

    # Create point cloud from depth image (timing starts here)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth,
        o3d_intrinsic,
        depth_scale=1000.0,  # Convert from mm to meters (depth is in mm for uint16)
    )
    pcd.estimate_normals()

    # Orient normals consistently (pointing towards camera: [0, 0, 1])
    pcd.orient_normals_to_align_with_direction([0, 0, 1])

    # Get points and normals
    normals = np.asarray(pcd.normals, dtype=np.float32)
    normal_map = normals.reshape(height, width, 3)
    return normal_map


def main():
    """Main benchmarking function."""
    print("=" * 60)
    print("Depth-to-Normal Conversion Speed Comparison")
    print("=" * 60)

    # Load demo data
    normal_path = "./demo_data/normal.png"
    depth_path = "./demo_data/depth.bin"
    calib_path = "./demo_data/params.txt"

    cam_fx, cam_fy, u0, v0 = get_cam_params(calib_path)
    cam_intrinsic = np.array([[cam_fx, 0, u0], [0, cam_fy, v0], [0, 0, 1]], dtype=np.float32)

    normal_gt = get_normal_gt(normal_path)
    normal_gt = vector_normalization(normal_gt)
    h, w, _ = normal_gt.shape

    depth, mask = get_depth(depth_path, h, w)

    print(f"\nTest data: {h}x{w}")
    print("-" * 60)

    num_runs = 100

    # =====================================
    # Test d2nt_basic
    # =====================================
    for _ in range(3):  # Warmup
        _ = depth2normal(depth, cam_intrinsic, version='d2nt_basic')

    d2nt_times = []
    print('Starting d2nt_basic test...')
    for _ in range(num_runs):
        start = time.perf_counter()
        d2nt_normal = depth2normal(depth, cam_intrinsic, version='d2nt_basic')
        d2nt_times.append((time.perf_counter() - start) * 1000)
    print('Finished d2nt_basic test...')

    # =====================================
    # Test Open3D
    # =====================================
    import open3d as o3d
    # Convert depth from meters to millimeters for uint16 format
    # Open3D expects depth in millimeters when using uint16
    depth_mm = (depth * 1000.0).astype(np.uint16)

    # Convert depth to Open3D Image
    o3d_depth = o3d.geometry.Image(depth_mm)

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w, h, cam_fx, cam_fy, u0, v0
    )

    for _ in range(3):
        _ = method_open3d(o3d_depth, o3d_intrinsic)

    open3d_times = []
    print('Starting Open3D test...')
    for _ in range(num_runs):
        start = time.perf_counter()
        open3d_normal = method_open3d(o3d_depth, o3d_intrinsic)
        open3d_times.append((time.perf_counter() - start) * 1000)
    print('Finished Open3D test...')
    # =====================================
    # Test kornia
    # =====================================
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    camera_matrix_tensor = torch.from_numpy(cam_intrinsic).unsqueeze(0).float()

    # Warmup
    for _ in range(3):
        _ = method_kornia(depth_tensor, camera_matrix_tensor)

    kornia_times = []
    print('Starting Kornia test...')
    for _ in range(num_runs):
        start = time.perf_counter()
        normal_tensor = method_kornia(depth_tensor, camera_matrix_tensor)
        end = time.perf_counter()
        kornia_times.append((end - start) * 1000)
    print('Finished Kornia test...')
    # Kornia returns (B, 3, H, W), convert to (H, W, 3)
    kornia_normal = normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    kornia_normal = vector_normalization(kornia_normal)

    # =====================================
    # Summary of results
    # =====================================
    d2nt_avg_ms = np.mean(d2nt_times)
    d2nt_std_ms = np.std(d2nt_times)
    d2nt_fps = 1000.0 / d2nt_avg_ms
    d2nt_fps_std = d2nt_std_ms * d2nt_fps / d2nt_avg_ms  # Approximate FPS std
    print(f"  d2nt_basic                    : {d2nt_fps:6.2f} ± {d2nt_fps_std:5.2f} FPS ({d2nt_avg_ms:6.2f} ± {d2nt_std_ms:5.2f} ms)")

    open3d_avg_ms = np.mean(open3d_times)
    open3d_std_ms = np.std(open3d_times)
    open3d_fps = 1000.0 / open3d_avg_ms
    open3d_fps_std = open3d_std_ms * open3d_fps / open3d_avg_ms
    print(f"  Open3D (point cloud)          : {open3d_fps:6.2f} ± {open3d_fps_std:5.2f} FPS ({open3d_avg_ms:6.2f} ± {open3d_std_ms:5.2f} ms)")
    
    kornia_avg_ms = np.mean(kornia_times)
    kornia_std_ms = np.std(kornia_times)
    kornia_fps = 1000.0 / kornia_avg_ms
    kornia_fps_std = kornia_std_ms * kornia_fps / kornia_avg_ms
    print(f"  Kornia depth_to_normals       : {kornia_fps:6.2f} ± {kornia_fps_std:5.2f} FPS ({kornia_avg_ms:6.2f} ± {kornia_std_ms:5.2f} ms)")

    speedup_open3d = open3d_avg_ms / d2nt_avg_ms
    print(f"  d2nt_basic is {speedup_open3d:.2f}x faster than Open3D")
    speedup_kornia = kornia_avg_ms / d2nt_avg_ms
    print(f"  d2nt_basic is {speedup_kornia:.2f}x faster than Kornia")

    # =====================================
    # Plot results
    # =====================================
    # Get normal visualizations
    gt_normal_vis = get_normal_vis(normal_gt, valid_mask=mask)
    d2nt_normal_vis = get_normal_vis(d2nt_normal, valid_mask=mask)
    open3d_normal_vis = get_normal_vis(-1 * open3d_normal, valid_mask=mask)
    kornia_normal_vis = get_normal_vis(-1 * kornia_normal, valid_mask=mask)

    # Create 1x4 subplot layout
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Ground truth normal
    axes[0].imshow(gt_normal_vis)
    axes[0].axis("off")
    axes[0].set_title("Ground Truth Normal", fontsize=12)

    # d2nt_basic normal
    axes[1].imshow(d2nt_normal_vis)
    axes[1].axis("off")
    axes[1].set_title(f"d2nt_basic ({d2nt_fps:.1f} FPS)", fontsize=12)

    # Open3D normal
    axes[2].imshow(open3d_normal_vis)
    axes[2].axis("off")
    axes[2].set_title(f"Open3D ({open3d_fps:.1f} FPS)", fontsize=12)

    # Kornia normal
    axes[3].imshow(kornia_normal_vis)
    axes[3].axis("off")
    axes[3].set_title(f"Kornia ({kornia_fps:.1f} FPS)", fontsize=12)

    plt.tight_layout()

    os.makedirs("./demo_results", exist_ok=True)
    plt.savefig("./demo_results/speed_comparison.png", dpi=150, bbox_inches='tight')
    print("\nVisualization saved to ./demo_results/speed_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
