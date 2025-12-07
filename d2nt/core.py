"""
Core functionality for depth-to-normal translation.
"""

import numpy as np
from .utils import vector_normalization
from .filters import get_filter, get_DAG_filter, MRF_optim


def depth2normal(
    depth,
    cam_intrinsic,
    version="d2nt_basic",
):
    """
    Convert depth map to surface normal map.

    Parameters
    ----------
    depth : numpy.ndarray
        Depth map with shape (H, W). Depth values should be in the same units as
        the camera focal length.
    cam_intrinsic : numpy.ndarray
        Camera intrinsic matrix with shape (3, 3). The matrix should be in the format:
        [[fx,  0, u0],
         [ 0, fy, v0],
         [ 0,  0,  1]]
        where fx, fy are focal lengths and (u0, v0) is the principal point.
    version : str, optional
        Version of the algorithm to use. Options:
        - 'd2nt_basic': Basic version without optimization
        - 'd2nt_v2': With Discontinuity-Aware Gradient (DAG) filter
        - 'd2nt_v3': With DAG filter and MRF-based Normal Refinement (default)

    Returns
    -------
    normal : numpy.ndarray
        Surface normal map with shape (H, W, 3). Normal vectors are normalized
        and in the range [-1, 1]. The three channels represent (nx, ny, nz).

    Examples
    --------
    >>> import numpy as np
    >>> from d2nt import depth2normal
    >>> depth = np.random.rand(480, 640) * 10.0  # Example depth map
    >>> # Camera intrinsic matrix
    >>> K = np.array([[525.0, 0, 320.0],
    ...               [0, 525.0, 240.0],
    ...               [0, 0, 1]])
    >>> normal = depth2normal(depth, K)
    >>> print(normal.shape)  # (480, 640, 3)
    """
    if version not in ["d2nt_basic", "d2nt_v2", "d2nt_v3"]:
        raise ValueError(f"Unsupported version: {version}. Must be one of: 'd2nt_basic', 'd2nt_v2', 'd2nt_v3'")
    # Extract camera parameters from intrinsic matrix
    cam_intrinsic = np.asarray(cam_intrinsic)
    if cam_intrinsic.shape != (3, 3):
        raise ValueError(
            f"cam_intrinsic must be a 3x3 matrix, got shape {cam_intrinsic.shape}"
        )
    
    cam_fx = cam_intrinsic[0, 0]
    cam_fy = cam_intrinsic[1, 1]
    u0 = cam_intrinsic[0, 2]
    v0 = cam_intrinsic[1, 2]

    h, w = depth.shape
    depth = np.asarray(depth, dtype=np.float32)

    # Create coordinate maps
    u_map = (np.ones((h, 1), dtype=np.float32) * np.arange(1, w + 1, dtype=np.float32) - u0).astype(np.float32)  # u-u0
    v_map = (np.arange(1, h + 1, dtype=np.float32).reshape(h, 1) * np.ones((1, w), dtype=np.float32) - v0).astype(np.float32)  # v-v0

    # Get depth gradients
    if version == "d2nt_basic":
        Gu, Gv = get_filter(depth)
    else:
        Gu, Gv = get_DAG_filter(depth)

    # Depth to Normal Translation
    est_nx = (Gu * cam_fx).astype(np.float32)
    est_ny = (Gv * cam_fy).astype(np.float32)
    est_nz = (-(depth + v_map * Gv + u_map * Gu)).astype(np.float32)
    
    # Stack arrays along the last axis to create (H, W, 3) array
    est_normal = np.stack((est_nx, est_ny, est_nz), axis=-1).astype(np.float32)

    # Vector normalization
    est_normal = vector_normalization(est_normal)

    # MRF-based Normal Refinement
    if version == "d2nt_v3":
        est_normal = MRF_optim(depth, est_normal)

    return est_normal

