from copy import copy
import matplotlib.pyplot as plt
from utils import *

# version choices: ['d2nt_basic', 'd2nt_v2', 'd2nt_v3']
VERSION = 'd2nt_v3'
if __name__ == '__main__':
    normal_path = './figs/normal.png'
    depth_path = './figs/depth.bin'
    calib_path = './figs/params.txt'

    # get camera parameters
    cam_fx, cam_fy, u0, v0 = get_cam_params(calib_path)

    # get ground truth normal [-1,1]
    normal_gt = get_normal_gt(normal_path)
    normal_gt = vector_normalization(normal_gt)
    h, w, _ = normal_gt.shape

    # get depth
    depth, mask = get_depth(depth_path, h, w)
    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - u0  # u-u0
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - v0  # v-v0

    # get depth gradients
    if VERSION == 'd2nt_basic':
        Gu, Gv = get_filter(depth)
    else:
        Gu, Gv = get_DAG_filter(depth)

    # Depth to Normal Translation
    est_nx = Gu * cam_fx
    est_ny = Gv * cam_fy
    est_nz = -(depth + v_map * Gv + u_map * Gu)
    est_normal = cv2.merge((est_nx, est_ny, est_nz))

    # vector normalization
    est_normal = vector_normalization(est_normal)

    # MRF-based Normal Refinement
    if VERSION == 'd2nt_v3':
        est_normal = MRF_optim(depth, est_normal)

    # show the ground truth normal
    gt_vis = visualization_map_creation(normal_gt, mask)
    plt.figure('gt')
    plt.imshow(gt_vis)

    # show the computed normal
    n_vis = visualization_map_creation(est_normal, mask)
    plt.figure(f'{VERSION}')
    plt.imshow(n_vis)

    # compute error and evaluate the model
    error_map, ea = evaluation(normal_gt, est_normal, mask)
    error_map[mask == 0] = np.nan
    platte = copy(plt.cm.pink)
    platte.set_bad('black', 1.0)
    plt.figure(f'{VERSION} error map')
    fig = plt.imshow(error_map, cmap=platte)
    plt.colorbar(fig)
    print(f"ea of [{VERSION}]:", ea)
