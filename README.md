# D2NT: A High-Performing Depth-to-Normal Translator

This repo is the official implementation of the paper:
> ["D2NT: A High-Performing Depth-to-Normal Translator"](https://arxiv.org/abs/2304.12031)

> [[arXiv]](https://arxiv.org/abs/2304.12031)
> [[homepage]](https://mias.group/D2NT)
> [[video]](https://www.bilibili.com/video/BV1GX4y1m7jF/)


<p align="center">
  <img src="https://raw.githubusercontent.com/fengyi233/depth-to-normal-translator/main/assets/compare.gif" width="500" alt="compare"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/fengyi233/depth-to-normal-translator/main/assets/tradeoff.png" width="500" alt="trade-off"/>
</p>


# Introduction
D2NT is a high-performance Python library for converting depth maps directly to surface normal maps. Unlike prevalent methods or computational libraries (e.g., [Open3D](https://github.com/isl-org/Open3D), [Kornia](https://github.com/kornia/kornia)) that typically require projecting depth images to 3D point clouds and then estimating normals through local plane fitting, D2NT explicitly constructs the mathematical relationship between depth maps and normal maps, enabling end-to-end normal estimation. On 640×480 images, D2NT achieves a computational speed **28× faster than Open3D** and **1.8× faster than Kornia** (even though it's implemented with cuda), significantly outperforming traditional point-cloud-based approaches in both speed and accuracy.

D2NT provides three algorithm versions with increasing accuracy: a fast basic version, an optimized version with Discontinuity-Aware Gradient (DAG) filter, and a refined version with DAG filter and MRF-based Normal Refinement. The library is designed for efficiency, accuracy, and ease of use in computer vision and robotics applications.

# Installation

## Install from PyPI (Recommended)

```bash
pip install d2nt
```

## Install from Source

```bash
# Clone the repository
git clone https://github.com/fengyi233/depth-to-normal-translator.git
cd depth-to-normal-translator

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

## Prerequisites

+ Python >= 3.7
+ numpy >= 1.20.0
+ opencv-python >= 4.0.0
+ matplotlib >= 3.5.0 (optional, for visualization)

# Dataset Preparation

Public real-world datasets generally obtain surface normals by local plane fitting,
which makes the surface normal ground truth unreliable. Therefore, we use the synthesis **3F2N dataset** provided
in this [paper](https://ieeexplore.ieee.org/document/9381580) to evaluate estimation performance.

The 3F2N dataset can be downloaded from: \
[GoogleDrive](https://drive.google.com/drive/folders/1TLj033oV0aplLE6xcQaSRcZpggDVTCHY) \
The dataset is organized as follows:

```
3F2N
 |-- Easy
 |  |-- android
 |  |  |-- depth
 |  |  |-- normal
 |  |  |-- params.txt
 |  |  |-- pose.txt
 |  |-- cube
 |  |-- ...
 |  |-- torusknot
 |-- Medium
 |  |-- ...
 |-- Hard
 |  |-- ...
```

# Usage

## Python Package Usage

After installation, you can use the `depth2normal()` function directly:

```python
import numpy as np
from d2nt import depth2normal

# Prepare depth map (example)
depth = np.random.rand(480, 640) * 10.0

cam_intrinsic = np.array([
    [525.0, 0, 320.0],  # fx=525.0, u0=320.0
    [0, 525.0, 240.0],  # fy=525.0, v0=240.0
    [0, 0, 1]
])

# Convert depth to normal
normal = depth2normal(depth, cam_intrinsic, version='d2nt_v3')

print(f"Normal map shape: {normal.shape}")  # (480, 640, 3)
```

### Algorithm Versions

- **`d2nt_basic`**: Basic version without any optimization method
- **`d2nt_v2`**: With Discontinuity-Aware Gradient (DAG) filter
- **`d2nt_v3`**: With DAG filter and MRF-based Normal Refinement (MNR) module (recommended)

## Performance

We benchmarked D2NT against popular depth-to-normal conversion libraries on 640×480 images:

| Method | FPS | Speedup vs d2nt_basic |
|--------|-----|----------------------|
| **d2nt_basic** | **65.5** | 1.0× (baseline) |
| Kornia | 36.9 | 0.56× (1.8× slower) |
| Open3D | 2.3 | 0.04× (28× slower) |

**Note**: Performance was measured on a standard CPU. Results may vary depending on hardware configuration. Run `python test_speed.py` to benchmark on your system.

## Python Demo

Run the demo script in the root directory to see visualization results and error maps:

```bash
python demo.py
```

This will display:
- Ground truth normal map
- Estimated normal map
- Error map (in degrees) with mean angular error

The demo uses test data from `demo_data/` directory. The results will be saved in `demo_results/` directory.

You can change the `VERSION` parameter in `demo.py` to select different D2NT versions:
- **`d2nt_basic`**: Basic version without any optimization method
- **`d2nt_v2`**: With Discontinuity-Aware Gradient (DAG) filter
- **`d2nt_v3`**: With DAG filter and MRF-based Normal Refinement (MNR) module (recommended)

## Coordinate System and Normal Visualization

### Coordinate System

The normal vectors returned by `depth2normal()` are defined in the **camera coordinate system**:
- **X-axis (nx)**: Points to the right in the image
- **Y-axis (ny)**: Points downward in the image  
- **Z-axis (nz)**: Points forward into the image plane

The normal vectors are normalized unit vectors with values in the range `[-1, 1]` for each component.

### Normal Visualization

This package provides visualization functions that convert normal vectors to RGB images for display:

```python
from d2nt import get_normal_vis, get_normal_vis_reference
import matplotlib.pyplot as plt

# Visualize normal map
normal_vis = get_normal_vis(normal, valid_mask=mask)
plt.imshow(normal_vis)

# Generate normal visualization reference
plt.imsave("normal_vis_reference.png", get_normal_vis_reference())

```

**Visualization Formula**: The visualization uses the formula:
```
normal_img = (1 - normal) / 2
```

This formula maps normal vectors from the range `[-1, 1]` to `[0, 1]` for RGB display. This convention is chosen to match the **3F2N dataset** ground truth encoding, where normal maps are stored with this specific encoding scheme.

**Note on Different Conventions**: Other projects (e.g., [DSINE](https://github.com/baegwangbin/DSINE/blob/main/utils/visualize.py)) may use the alternative formula `(normal + 1) / 2` for visualization, which is also valid. However, when comparing results or using different datasets, it is important to be aware of which convention is being used, as the color mapping will be inverted. The `get_normal_vis()` function in this package uses `(1 - normal) / 2` to maintain consistency with the 3F2N dataset format.




# Cite
If you find our work useful in your research, please consider citing our paper:

```
@inproceedings{feng2023d2nt,
	author      = {{Yi Feng, Bohuan Xue, Ming Liu, Qijun Chen, and Rui Fan}},
	title       = {{D2NT: A High-Performing Depth-to-Normal Translator}},
	booktitle   = {{IEEE International Conference on Robotics and Automation (ICRA)}},
	year        = {{2023}}
}
```