"""
D2NT Complete Demo Script

This script demonstrates how to use the d2nt package for depth-to-normal conversion
with visualization capabilities. Requires matplotlib for visualization.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from d2nt import depth2normal
from d2nt.utils import (
    get_cam_params,
    get_depth,
    get_normal_gt,
    vector_normalization,
    evaluation,
    get_normal_vis,
    get_normal_vis_reference,
)

# Version choices: ['d2nt_basic', 'd2nt_v2', 'd2nt_v3']
VERSION = "d2nt_v3"

if __name__ == "__main__":
    # Data paths
    normal_path = "./demo_data/normal.png"
    depth_path = "./demo_data/depth.bin"
    calib_path = "./demo_data/params.txt"

    # Get camera parameters
    cam_fx, cam_fy, u0, v0 = get_cam_params(calib_path)

    # Build camera intrinsic matrix
    cam_intrinsic = np.array([[cam_fx, 0, u0], [0, cam_fy, v0], [0, 0, 1]])

    # Get ground truth normal map [-1,1]
    normal_gt = get_normal_gt(normal_path)
    normal_gt = vector_normalization(normal_gt)
    h, w, _ = normal_gt.shape

    # Get depth map
    depth, valid_mask = get_depth(depth_path, h, w)

    # Convert depth to normal using d2nt package
    est_normal = depth2normal(depth, cam_intrinsic, version=VERSION)

    # Compute error and evaluate the model
    error_map, ea = evaluation(normal_gt, est_normal, valid_mask)
    error_map[valid_mask == 0] = np.nan

    # Prepare depth visualization data
    depth_masked = depth.copy()
    depth_masked[valid_mask == 0] = np.nan
    depth_min = np.nanmin(depth_masked)
    depth_max = np.nanmax(depth_masked)

    # Get normal visualizations
    gt_normal_vis = get_normal_vis(normal_gt, valid_mask=valid_mask)
    est_normal_vis = get_normal_vis(est_normal, valid_mask=valid_mask)

    # Prepare error map visualization data
    error_map_masked = error_map.copy()
    error_map_masked[valid_mask == 0] = np.nan
    error_min = np.nanmin(error_map_masked)
    error_max = np.nanmax(error_map_masked)

    # Create 1x4 subplot layout with colorbars
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Depth map with colorbar
    im1 = axes[0].imshow(depth_masked, cmap="Greys", vmin=depth_min, vmax=depth_max)
    axes[0].axis("off")
    axes[0].set_title("Input Depth", fontsize=12)
    cbar1 = plt.colorbar(im1, ax=axes[0], orientation='horizontal', location='bottom', label="Depth", fraction=0.046,
                         pad=0.04)

    # GT Normal with colorbar (using normal visualization)
    im2 = axes[1].imshow(gt_normal_vis)
    axes[1].axis("off")
    axes[1].set_title("GT Normal", fontsize=12)
    # Normal map doesn't need a traditional colorbar, but we can add a note
    # or create a custom colorbar showing the normal vector range

    # Estimated Normal with colorbar
    im3 = axes[2].imshow(est_normal_vis)
    axes[2].axis("off")
    axes[2].set_title(f"{VERSION} Normal", fontsize=12)

    # Error map with colorbar
    im4 = axes[3].imshow(error_map_masked, cmap="pink", vmin=error_min, vmax=error_max)
    axes[3].axis("off")
    axes[3].set_title(f"Error Map of {VERSION} (MAE: {ea:.2f}°)", fontsize=12)
    cbar4 = plt.colorbar(im4, ax=axes[3], orientation='horizontal', location='bottom', label="Angular Error (degrees)",
                         fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    os.makedirs("./demo_results", exist_ok=True)
    plt.savefig(f"./demo_results/vis_{VERSION}.png")
    plt.show()

    print(f"Mean Angular Error of [{VERSION}]: {ea:.2f} degrees")

    # Save normal visualization reference
    plt.imsave("./demo_results/normal_vis_reference.png", get_normal_vis_reference())