# D2NT: A High-Performing Depth-to-Normal Translator

This repo is the official implementation of the paper:
> ["D2NT: A High-Performing Depth-to-Normal Translator"](https://arxiv.org/pdf/.pdf) \
> by [Yi Feng](https://xxx.com), [Bohuan Xue](https://mi.eng.cam.ac.uk/~ib255/),
> and [Rui Fan](https://ruirangerfan.com/).

> [[arXiv]](https://arxiv.org/abs/) [[youtube]](https://youtu.be/mTy85tJ2oAQ) [[bilibili]](https://youtu.be/mTy85tJ2oAQ)

<p align="center">
  <img src="assets/tradeoff.png" alt="algos comparison"/>
</p>


# Prerequisites

+ matplotlib==3.5.1
+ numpy==1.24.2
+ opencv-python==4.5.1.48

# Dataset Preparation

Public real-world datasets generally obtain surface normals by local plane fitting,
which makes the surface normal ground truth unreliable. Therefore, we use the synthesis **3F2N dataset** provided
in this [paper]() to evaluate estimation performance.

The 3F2N dataset can be downloaded from: \
[BaiduDisk]() \
[GoogleDrive]() \
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

# Demo

It is recommended to run '**demo.py**' in your Python IDE instead of the Terminal for the sake of visualization. 

You can change the parameter 'VERSION' to select the D2NT version.\
'**d2nt_basic**' represents for the depth-to-normal translator without any optimization method.\
'**d2nt_v2**' represents for the D2NT with Discontinuity-Aware Gradient (DAG) filter.\
'**d2nt_v3**' represents for the D2NT with DAG filter and MRF-based Normal Refinement (MNR) module.



### 5.1  Python code
Navigate to [python]() directory and run `demo.py`, a result and the corresponding error map (degrees) will be displayed.
We also implement [3F2N SNE](https://ieeexplore.ieee.org/document/9381580) in python. The matlab and c++ implementation
can be found in this [repository](https://github.com/ruirangerfan/Three-Filters-to-Normal). 

### 5.2  Matlab code
Navigate to [matlab]() directory and run `demo.m`, a result and the corresponding error map (degrees) will be displayed.

### 5.3  C++ code
Navigate to [cpp]() directory and run `demo.cpp`, a result and the corresponding error map (degrees) will be displayed.


# Cite
This code is for non-commercial use; please see the license file for terms.

If you find our work useful in your research please consider citing our paper:

```
@article{D2NT,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
```