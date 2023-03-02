## UOP : Unseen Object Placement with Large Scale Simulation

![Main Image]()
**Picture:** Detail about main image

This repository contains official implementation of following paper:
> **UOP-Net: Learning to Place Objects Stably using a Large-scale Simulation**<br>
> **Author**
> **Abstract:** *Contents of Abstract*
> [Click here for website and paper.](https://sites.google.com/view/uop-net-iccv-submission/home)


## Table of Contents

[**Environment Setting**](#environment-setting)

[**Data Generation**](#data-generation)
  * [**1 - Download Public 3D models**](#1-download-public-3d-models)
  * [**2 - Edit Config file**](#2-edit-path-of-data-in-config-file-ex-defaultyaml)
  * [**3 - Run Data Generation**](#3-run-data_generatorpy)

[**Train UOP-Net**](#train-uop-net)

[**Test and Inference**](#test-and-inference)

**Additional Info**
  * [**Reference**](#references)
  * [**License**](#license)


## Environment Setting


### 1. Simulation

  * uop data generation code built in PyRep and CoppeliaSim. Please install pyrep first [PyRep github](https://github.com/stepjam/PyRep)

### 2. Python Requirements

```
conda activate pyrep

pip install open3d



```



## Data Generation

### 1. Download public 3D models
- [YCB Dataset](https://www.ycbbenchmarks.com/object-models/)
- [3DNet](https://strands.readthedocs.io/en/latest/datasets/three_d_net.html)
- [ShapeNet](https://shapenet.org/)
```shell
ycb-tools/models/ycb # ycb data root
├── 001_chips_can
├── 002_master_chef_can
├── ...
├── 076_timer
└── 077_rubiks_cube
ShapeNetCore.v1 # shapenet data root
├── 02691156
├── 02691156.csv
├── 02691156.zip
├── ...
├── count-models.sh
├── get-metadata.sh
├── jq
├── README.txt
└── taxonomy.json
3DNet_raw/ # 3dnet data_root
├── Cat10_ModelDatabase
├── Cat10_TestDatabase
├── Cat200_ModelDatabase
├── Cat60_ModelDatabase
└── Cat60_TestDatabase
```


### 2. Edit path of data in config file (ex. default.yaml)
```yaml
########## TODO: EDIT BELOW SETTINGS ###########
data_type: 'ycb' # raw data type ['ycb', 'shapenet', '3dnet']
data_root: 'PATH TO RAW MESH DATA' # root of raw mesh data
save_root: 'PATH TO SAVE GENERATED DATA' # root of save generated data

########### FIX BELOW SETTINGS ###########
bbox_size: 0.2 # scene model size
number_of_points: 5000 # sampling point number
sampling_method: 'uniform-poisson' # [uniform, poisson, uniform-poisson]
orientation_grid_size: 8 # checking orientation per each axis

# simulation setting for sampling pose
table_grid_size: 8 # table num of each axis 
time_step: 0.005 # simulation time step
tolerance: 0.001 # threshold of check object stoppness
max_step: 1000 # max step of check object stoppness

# simulation setting for inspection env
inspection_tolerance: 18 # threshold of inspectiion of stable pose
tilt: 10
min_step: 200 # min step of check stability of pose
```


### 3. Run `data_generator.py`
```shell
python data_generator.py --config config_file/raeyo_shape.yaml --inspect --split 9
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



## License
See [LICENSE][LICENSE]



## Citation
```
```


