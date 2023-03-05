## UOP : Unseen Object Placement with Large Scale Simulation

![Main Image]()
**Picture:** Detail about main image

This repository contains official implementation of following paper:
> **UOP-Net: Learning to Place Objects Stably using a Large-scale Simulation**<br>
> **Author:** *Anonymous*<br>
> **Abstract:** *Object placement is critical for robots operating in unstructured environments because it allows them to manipulate and arrange objects in a safe and efficient manner. However, existing methods for object placement have limitations such as the need for a complete 3D model of the object or the inability to handle complex object shapes; these limitations restrict the robot's applicability. In this paper, we present an unseen object placement (UOP) method that directly detects the stable planes of an unseen object from a single view and partial point cloud. Our approach is useful for scenarios where the shape and properties of the object are not fully known because it allow the robot to adapt and place the object accurately. We train our model on large-scale simulation data for generalizing over the shape and properties of the stable plane with a 3D point cloud. Our approach is verified through simulation and real-world robot experiments; the proposed approach achieves a state-of-the-art performance for placing single-view and partial objects.*
> [Click here for website and paper.](https://sites.google.com/view/uop-net-iccv-submission/home)


## Table of Contents

[**Environment Setting**](#environment-setting)

[**Data Generation**](#data-generation)
  * [**1 - Download Public 3D models**](#1-download-public-3d-models)
  * [**2 - Preprocess 3D models**](#2-preprocess-3d-models)
  * [**4 - Run Data Generation**](#3-run-datagenerator)
  

[**Train UOP-Net**](#train-uop-net)

[**Test and Inference**](#test-and-inference)

**Additional Info**
  * [**Reference**](#references)
  * [**License**](#license)


## Environment Setting

  * Our code implemented on Ubuntu 20.04 and Conda virtual environment.
  * Please follow below instruction.

```
conda create -n uop python=3.8
conda activate uop

```


### 1. Simulation

  * uop data generation code built in PyRep and CoppeliaSim. Please install pyrep first [PyRep github](https://github.com/stepjam/PyRep)

### 2. Python Requirements

  * After Install pyrep in your environment install requirments packages

```
conda activate pyrep

pip install open3d numpy natsort tqdm trimesh



```

## Data Generation

### 1. Download public 3D models
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

### 2. Preprocess 3D models

First build Manifold to watertight 3D models following the instruction in [Manifold](https://github.com/hjwdzh/Manifold)

Add following term to your ~/.bashrc file
```
export MANIFOLD=PATH/TO/MANIFOLD/BUILD/DIR # ex.) ~/Manifold/build
```

And run preprocess.py

```
python preprocess.py --data_type ycb # ycb, shapenet, 3dnet, ycb-texture
```

### 3. Run DataGenerator
```shell
python data_generator.py --data_type ycb --inspect
```











## Train UOP-Net



## Test and Inference



## References


### YCB
```
[1] Berk Calli, Aaron Walsman, Arjun Singh, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M. Dollar, Benchmarking in Manipulation Research: The YCB Object and Model Set and Benchmarking Protocols, IEEE Robotics and Automation Magazine, pp. 36 – 52, Sept. 2015.

[2] Berk Calli, Arjun Singh, James Bruce, Aaron Walsman, Kurt Konolige, Siddhartha Srinivasa, Pieter Abbeel, Aaron M Dollar, Yale-CMU-Berkeley dataset for robotic manipulation research, The International Journal of Robotics Research, vol. 36, Issue 3, pp. 261 – 268, April 2017.

[3] Berk Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M. Dollar, The YCB Object and Model Set: Towards Common Benchmarks for Manipulation Research, proceedings of the 2015 IEEE International Conference on Advanced Robotics (ICAR), Istanbul, Turkey, 2015.
```
### 3DNet


### ShapeNet
```
@techreport{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel X. and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  number      = {arXiv:1512.03012 [cs.GR]},
  institution = {Stanford University --- Princeton University --- Toyota Technological Institute at Chicago},
  year        = {2015}
}
```


### Simulation
```
@article{james2019pyrep,
  title={PyRep: Bringing V-REP to Deep Robot Learning},
  author={James, Stephen and Freese, Marc and Davison, Andrew J.},
  journal={arXiv preprint arXiv:1906.11176},
  year={2019}
}
```
```
@article{huang2018robust,
  title={Robust Watertight Manifold Surface Generation Method for ShapeNet Models},
  author={Huang, Jingwei and Su, Hao and Guibas, Leonidas},
  journal={arXiv preprint arXiv:1802.01698},
  year={2018}
}
```



## License
See [LICENSE](LICENSE)



## Citation
```
```


