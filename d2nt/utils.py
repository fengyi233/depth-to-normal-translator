"""
Utility functions for depth and normal processing.
"""

import struct

import cv2
import numpy as np


def get_cam_params(calib_path):
    """
    Read camera parameters from a text file.

    Parameters
    ----------
    calib_path : str
        Path to the calibration file containing camera parameters.

    Returns
    -------
    tuple
        (cam_fx, cam_fy, u0, v0) - Camera focal lengths and principal point.
    """
    with open(calib_path, "r") as f:
        data = f.read()
        params = list(map(int, (data.split())))[:-1]
    return tuple(params)


def get_normal_gt(normal_path):
    """
    Read ground truth normal map from an image file.

    Parameters
    ----------
    normal_path : str
        Path to the normal map image file.

    Returns
    -------
    numpy.ndarray
        Normal map with values in range [-1, 1], shape (H, W, 3).
    """
    normal_gt = cv2.imread(normal_path, -1)
    normal_gt = normal_gt[:, :, ::-1]
    normal_gt = 1.0 - normal_gt / 65535.0 * 2
    return normal_gt.astype(np.float32)


def get_depth(depth_path, height, width):
    """
    Read depth map from a binary file.

    Parameters
    ----------
    depth_path : str
        Path to the binary depth file.
    height : int
        Height of the depth map.
    width : int
        Width of the depth map.

    Returns
    -------
    tuple
        (depth, mask) - Depth map and foreground mask (1 for foreground, 0 for background).
    """
    with open(depth_path, "rb") as f:
        data_raw = struct.unpack("f" * width * height, f.read(4 * width * height))
        z = np.array(data_raw).reshape(height, width)

    # create mask, 1 for foreground, 0 for background
    mask = np.ones_like(z)
    mask[z == 1] = 0

    return z.astype(np.float32), mask.astype(np.float32)


def vector_normalization(normal, eps=1e-8):
    """
    Normalize normal vectors to unit length.

    Parameters
    ----------
    normal : numpy.ndarray
        Normal map with shape (H, W, 3).
    eps : float, optional
        Small epsilon value to avoid division by zero (default: 1e-8).

    Returns
    -------
    numpy.ndarray
        Normalized normal map.
    """
    mag = np.linalg.norm(normal, axis=2, keepdims=True) + eps
    normal /= mag
    return normal


def get_normal_vis(normal, valid_mask=None):
    """
    Get normal visualization map using the following mapping:
    normal_vis = (1 - normal) / 2

    Parameters
    ----------
    normal : numpy.ndarray
        Normal map with shape (H, W, 3), values in range [-1, 1].
    valid_mask : numpy.ndarray, optional
        Foreground mask with shape (H, W). If None, all pixels are considered valid.

    Returns
    -------
    numpy.ndarray
        Visualization map with values in range [0, 1], shape (H, W, 3).
        Can be directly displayed using matplotlib or saved as image.
    """
    normal = np.asarray(normal, dtype=np.float32)

    # convert [-1,1] → [0,1]
    normal_vis = (1.0 - normal) * 0.5

    if valid_mask is None:
        return np.clip(normal_vis, 0, 1).astype(np.float32)

    # validate and expand mask
    valid_mask = np.asarray(valid_mask, dtype=np.float32)
    valid_mask = np.repeat(valid_mask[..., None], 3, axis=2)

    # invalid → white
    normal_vis[valid_mask == 0] = 1.0

    return np.clip(normal_vis, 0, 1).astype(np.float32)


def get_normal_vis_reference(size=256):
    """
    Generate a circular colorbar reference for normal map visualization.
    
    This function creates a circular image where each point's color represents
    the normal vector direction corresponding to that point on a 3D unit sphere.
    Only the points inside the circle are valid normals; outside is filled with white.
    
    Parameters
    ----------
    size : int, optional
        The width and height of the output image. Must be >= 2. Default is 256.
    
    Returns
    -------
    numpy.ndarray
        Normal visualization reference map with shape (size, size, 3),
        values in range [0, 1]. Can be displayed or saved as an image.
    """
    if size < 10:
        raise ValueError(f"size must be >= 2, got {size}")
    
    H = W = size

    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    radius = min(cx, cy)

    x = (u - cx) / radius
    y = (v - cy) / radius

    rr = x ** 2 + y ** 2
    valid_mask = rr <= 1.0

    normal = np.zeros((H, W, 3), dtype=np.float32)
    normal[..., 0] = x
    normal[..., 1] = y
    normal[..., 2] = -np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))

    normal_vis = (1.0 - normal) * 0.5

    # Set outside the circle to white
    normal_vis[~valid_mask] = 1.0
    return np.clip(normal_vis, 0, 1).astype(np.float32)


def angle_normalization(err_map):
    """
    Normalize angles to [0, π/2].

    Parameters
    ----------
    err_map : numpy.ndarray
        Error map in radians.

    Returns
    -------
    numpy.ndarray
        Normalized error map.
    """
    err_map[err_map > np.pi / 2] = np.pi - err_map[err_map > np.pi / 2]
    return err_map


def evaluation(n_gt, n_est, mask):
    """
    Evaluate estimated normal map against ground truth.

    Parameters
    ----------
    n_gt : numpy.ndarray
        Ground truth normal map with shape (H, W, 3).
    n_est : numpy.ndarray
        Estimated normal map with shape (H, W, 3).
    mask : numpy.ndarray
        Foreground mask with shape (H, W).

    Returns
    -------
    tuple
        (error_map, ea) - Error map in degrees and mean angular error.
    """
    scale = np.pi / 180
    # Compute dot product and clip to [-1, 1] to avoid numerical errors in arccos
    dot_product = np.sum(n_gt * n_est, axis=2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    error_map = np.arccos(dot_product)
    error_map = angle_normalization(error_map) / scale
    error_map *= mask
    ea = error_map.sum() / mask.sum()
    return error_map, ea
