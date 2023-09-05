# UOP : Unseen Object Placement with Large Scale Simulation

![Main Image](figure/main_image.png)

This repository contains official implementation of following paper:
> **UOP-Net: Learning to Place Objects Stably using a Large-scale Simulation**<br>
> **Author:** *Anonymous*<br>
> **Abstract:** *Object placement is a fundamental task for robots, yet it remains challenging for partially observed objects. Existing methods for object placement have limitations, such as the requirement for a complete 3D model of the object or the inability to handle complex shapes and novel objects that restrict the applicability of robots in the real world. Herein, we focus on addressing the Unseen Object Placement(UOP) problem. We tackled the UOP problem using two methods: (1) UOP-Sim, a large-scale dataset to accommodate various shapes and novel objects, and (2) UOP-Net, a point cloud segmentation-based approach that directly detects the most stable plane from partial point clouds. Our UOP approach enables robots to place objects stably, even when the object's shape and properties are not fully known, thus providing a promising solution for object placement in various environments. We verify our approach through simulation and real-world robot experiments, demonstrating state-of-the-art performance for placing single-view and partial objects.*<br>
> [Click here for website and paper.](https://gistailab.github.io/uop/)

<br>
<br>

---

### ToDo

- [ ] Sample Data Download link
- [ ] Release demo and test code
- [ ] Environment Setting ... Add pkg requirements for training UOP-Net

---

## Table of Contents


[**0. Environment Setting for UOP-Sim**](#environment-setting)\
[**0. Environment Setting for Training UOP-Net**](#environment-setting)

[**1. Data Generation (UOP-Sim)**](#data-generation)
  * [**1-1. Download Public 3D models**](#1-download-public-3d-models)
  * [**1-2. Preprocess 3D models**](#2-preprocess-3d-models)
  * [**1-3. Run Data Generation**](#3-run-datagenerator)

[**2. Train UOP-Net (UOP-Net)**](#train-uop-net)

[**3. Test and Inference**](#test-and-inference)

[**4. Reference**](#4-references)
[**5. License**](#5-license)
[**6. License**](#6-citation)

<br>

---

## 0. Environment Setting for UOP-Sim

  * Our code implemented on Ubuntu 20.04 and Conda virtual environment.
  * Please follow below instruction.

```
conda create -n uop_sim python=3.8
conda activate uop_sim

```

<br>

## 0. Environment Setting for Training UOP-Net

  * Our code implemented on Ubuntu 20.04 and Conda virtual environment.
  * Please follow below instruction.

```
conda create -n uop_net python=3.8
conda activate uop_net

```

### 0-1. Simulation

  * uop data generation code built in CoppeliaSim and PyRep. Please install CoppeliaSim and pyrep first [PyRep github](https://github.com/stepjam/PyRep)
  * after install CoppeliaSim, you must open CoppeliaSim program at once. (This step )

### 0-2. Python Requirements

* After Install pyrep in your environment install requirments packages

```
conda activate uop_sim

pip install open3d numpy natsort tqdm trimesh pyfastnoisesimd opencv-python point_cloud_utils

```

<br>

## 1. Data Generation

### 1-1. Download public 3D models
- [YCB Dataset](https://www.ycbbenchmarks.com/object-models/)
- [3DNet](https://strands.readthedocs.io/en/latest/datasets/three_d_net.html)
- [ShapeNet](https://shapenet.org/)

After download public data your data folder looks like below
```shell
UOPROOT # path to generate uopdata
├──public_data
│  ├──ycb-tools/models/ycb # ycb data root
│  │  ├── 001_chips_can
│  │  ├── 002_master_chef_can
│  │  ├── ...
│  │  ├── 076_timer
│  │  └── 077_rubiks_cube
│  ├──ShapeNetCore.v1 # shapenet data root
│  │  ├── 02691156
│  │  ├── 02691156.csv
│  │  ├── 02691156.zip
│  │  ├── ...
│  │  ├── count-models.sh
│  │  ├── get-metadata.sh
│  │  ├── jq
│  │  ├── README.txt
│  │  └── taxonomy.json
│  └──3DNet_raw/ # 3dnet data_root
│     ├── Cat10_ModelDatabase
│     ├── Cat10_TestDatabase
│     ├── Cat200_ModelDatabase
│     ├── Cat60_ModelDatabase
│     └── Cat60_TestDatabase
└──uop_data # the path uop data generated
```
Add following term to your ~/.bashrc file
```
export UOPROOT=PATH/TO/GENRATE/UOP/DATA # # uop data root
```

### 1-2. Preprocess 3D models

First build Manifold to watertight 3D models following the instruction in [Manifold](https://github.com/hjwdzh/Manifold)

Add following term to your ~/.bashrc file
```
export MANIFOLD=PATH/TO/MANIFOLD/BUILD/DIR # ex.) ~/Manifold/build
```

And run preprocess.py

```
python preprocess.py --data_type ycb # ycb, shapenet, 3dnet, ycb-texture
```

### 1-3. Run DataGenerator
```shell
python data_generator.py --data_type ycb --inspect
```

<br>

## 2. Train UOP-Net

befor training UOP-Net, set path and the hyper-parameraters in /config/uop_net.json

```shell
conda activate uop_net
# install required packages in conda environment

cd uop_net
python train.py --config_path 
```

<br>

## 3. Test and Demo
*The code for Test and Demo will be released*

<br>

## 4. References

### 3D Model - YCB
```
[1] Berk Calli, Aaron Walsman, Arjun Singh, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M. Dollar, Benchmarking in Manipulation Research: The YCB Object and Model Set and Benchmarking Protocols, IEEE Robotics and Automation Magazine, pp. 36 – 52, Sept. 2015.

[2] Berk Calli, Arjun Singh, James Bruce, Aaron Walsman, Kurt Konolige, Siddhartha Srinivasa, Pieter Abbeel, Aaron M Dollar, Yale-CMU-Berkeley dataset for robotic manipulation research, The International Journal of Robotics Research, vol. 36, Issue 3, pp. 261 – 268, April 2017.

[3] Berk Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M. Dollar, The YCB Object and Model Set: Towards Common Benchmarks for Manipulation Research, proceedings of the 2015 IEEE International Conference on Advanced Robotics (ICAR), Istanbul, Turkey, 2015.
```
### 3D Model - 3DNet
```
@inproceedings{wohlkinger20123dnet,
  title={3dnet: Large-scale object class recognition from cad models},
  author={Wohlkinger, Walter and Aldoma, Aitor and Rusu, Radu B and Vincze, Markus},
  booktitle={2012 IEEE international conference on robotics and automation},
  pages={5384--5391},
  year={2012},
  organization={IEEE}
}
```

### 3D Model - ShapeNet
```
@techreport{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel X. and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  number      = {arXiv:1512.03012 [cs.GR]},
  institution = {Stanford University --- Princeton University --- Toyota Technological Institute at Chicago},
  year        = {2015}
}
```
### 3D Model - Watertight Method
```
@article{huang2018robust,
  title={Robust Watertight Manifold Surface Generation Method for ShapeNet Models},
  author={Huang, Jingwei and Su, Hao and Guibas, Leonidas},
  journal={arXiv preprint arXiv:1802.01698},
  year={2018}
}
```
### Simulation - PyRep
```
@article{james2019pyrep,
  title={PyRep: Bringing V-REP to Deep Robot Learning},
  author={James, Stephen and Freese, Marc and Davison, Andrew J.},
  journal={arXiv preprint arXiv:1906.11176},
  year={2019}
}
```

<br>

## 5. License
See [LICENSE](LICENSE)

<br>

## 6. Citation
```
@article{noh2023learning,
  title={Learning to Place Unseen Objects Stably using a Large-scale Simulation},
  author={Noh, Sangjun and Kang, Raeyoung and Kim, Taewon and Back, Seunghyeok and Bak, Seongho and Lee, Kyoobin},
  journal={arXiv preprint arXiv:2303.08387},
  year={2023}
}
```

