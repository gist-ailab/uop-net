
<br>

## Table of Contents

[**0. Environment Setting**](#0-environment-setting)
* [**0-1. Create conda env & python requirements**](#0-1-create-conda-env-and-python-requirements)
* [**0-2. Build Manifold**](#0-2-build-manifold)
* [**0-3. Simulation**](#0-3-simulation)
* [**0-4. Sample code**](#0-4-sample-code)

[**1. Prepare Dataset (UOP-Sim)**](#1-prepare-dataset-uop-sim)
* [**1-1. Download pre-built UOP-Sim**](#1-1-download-pre-built-uop-sim)
* [**1-2. Build UOP-Sim**](#1-2-build-uop-sim)
  * [**1-2-1. Download Public 3D models**](#1-2-1-download-public-3d-models)
  * [**1-2-2. Preprocess 3D models**](#1-2-2-preprocess-3d-models)
  * [**1-2-3. Run Data Generator**](#1-2-3-run-data-generator)

[**2. Train UOP-Net (UOP-Net)**](#2-train-uop-net)

[**3. Test and Inference**](#3-test-and-inference)

[**4. Reference**](#4-references)

[**5. License**](#5-license)

<br>
<br>

---

<br>
<br>

## 0. Environment Setting
<!-- checked -->
* Our code implemented on ```Ubuntu 20.04```, Nvidia-gpu environments and ```Conda``` virtual environment.
  * etc. ```cmake```, ```anaconda```, ```cosimppeliasim``` & ```pyrep``` 
  * tested on 
    - ubuntu >= 18.04, <= 22.04
    - nvidia-driver ver. > 52x.xx
    - cuda > 11.3 , <=12.1
    - python >= 3.7, <= 3.8 (conda)
* Please follow below instruction.

```shell
# 0. Download UOP & thirdparty repository
git clone --recursive https://github.com/gist-ailab/uop-net.git
cd ./uop-net

# 1. check device
sh ./setups/1_check_env.sh

# 2. prepare setups
sh ./setups/2_prepare_env.sh
```

* if you don't know how to install ```cmake```, ```anaconda```, ```cosimppeliasim``` & ```pyrep``` then, follow the [2_prepare_env.sh](2_prepare_env.sh) bash file
* This project can't setup on ```SSH-termianl```/```Windows-WSL-terminal``` (due to simulations/packages run with GPU driver)

<br>

### 0-1. Create conda env and python requirements
<!-- checked -->
```shell
# 
conda create -n uop_net python=3.8
conda activate uop_net

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio                    # : torch 2, cuda 12
pip install trimesh plotly pyyaml open3d point_cloud_utils natsort tqdm
# pip install -r requirements.txt 
```
<!-- org: 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch 
-->
<!-- TODO: make packages pip install -r requirement.txt -->

<br>

### 0-2. Build Manifold
<!-- checked -->
```shell
# start from uop-net folder
cd setups/thirdparty/Manifold
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
<!-- TODO: make bash script -->

<br>

### 0-3. Simulation
<!-- checked -->
* uop data generation code built in CoppeliaSim and PyRep. Please install CoppeliaSim and pyrep first [PyRep github](https://github.com/stepjam/PyRep)
* after install CoppeliaSim, you must open CoppeliaSim program at once. (This step )
* Also follow [CoppeliaSim video compression library issue to solve them](https://github.com/stepjam/PyRep/issues/142)

```shell
# start from uop-net folder
cd setups/thirdparty/VideoRecorder/vvcl
mkdir build
cd build
cmake ..
make

# apply to CoppeliaSim
cp ./libvvcl.so $COPPELIASIM_ROOT
```

<br>

### 0-4. Sample code

* We provide sample script for check settings to follow whole pipeline of our project (data generation, inference, test on simulation) with YCB dataset

<!-- TODO : fill in/out & final result files + how to view -->
```shell
# 1. download YCB dataset
python models/download_ycb_dataset.py
```
* ```--object-dir``` : root to save ycb data, default: "```models/ycb```"
* out: 3D object model files in models/ycb/

```shell
# 2. data generation
python sample_data_generation.py --object 002_master_chef_can
```
* ```--object``` : ycb object name 
* --object : 
* in: mesh_input.obj
* out: mesh.ply, visualize_mesh.html, 

```shell
# 3. inference object
python sample_inference.py --object 002_master_chef_can
# in: , out:
```

```shell
# 4. test on simulation
python sample_test.py --object 002_master_chef_can
# in: , out:

```

<br>
<br>

## 1. Prepare Dataset (UOP-Sim)
* You can simply use UOP-Sim dataset for evaluate UOP-Net, follow [1-1. Download pre-built UOP-Sim](#1-1-download-pre-built-uop-sim) instruction.
* If you want to simply download dataset(UOP-Sim), 

<br>

### 1-1. Download pre-built UOP-Sim
<!-- TODO: seperate dataset build & download built dataset -->
* Download dataset with google drive link 
  * [uop_data.zip - 28GB](https://drive.google.com/file/d/11yvzrLgIbv8e3Yy2gyCG0k2QMHepeGa0/view?usp=drive_link) (this can't be download using bash or python)
    * includes : 
    * file tree :
  * [uop_data_for_evaluation.zip - 171MB](https://drive.google.com/file/d/19mmLYNT_2reMV7C7Z8pEWwgjBulVobCG/view?usp=drive_link)
    * includes :
    * file tree :
      ```shell
      .
      ├── uop_data_tree.txt
      └── ycb
          ├── 002_master_chef_can
          │   ├── inspected_zaxis.pkl
          │   ├── mesh_watertight.ply
          │   ├── model.ttm
          │   └── partial
          │       ├── 0.pkl
          │       ├── 1.pkl
          │       ├── ...
          │       └── 99.pkl
          ├── 003_cracker_box
          ├── ...
          └── 077_rubiks_cube

      127 directories, 6489 files, 171.1MB
      ```
```shell
# codes will here
```

<br>

### 1-2. Build UOP-Sim

#### 1-2-1. Download Public 3D models
<!-- checked -->
- [YCB Dataset](https://www.ycbbenchmarks.com/) (you can download from web or run script below.)
  ```shell
  python download_ycb_dataset.py
  ```

- [3DNet](https://www.acin.tuwien.ac.at/vision-for-robotics/software-tools/3dnet-dataset/) (you can download from web or run script below.)
  ```shell
  mkdir -p models/3DNet

  # 3DNet Cat10 model/test
  wget https://data.acin.tuwien.ac.at/index.php/s/7GrnwJlDQrajFTo/download -O models/3dnet/Cat10_ModelDatabase.zip --no-check-certificate
  wget https://data.acin.tuwien.ac.at/index.php/s/6XkgKCAmpUyznnP/download -O models/3dnet/Cat10_TestDatabase.zip --no-check-certificate

  # 3DNet Cat60 model/test
  wget https://data.acin.tuwien.ac.at/index.php/s/uGarwrExmIjlwwP/download -O models/3dnet/Cat60_ModelDatabase.zip --no-check-certificate
  wget https://data.acin.tuwien.ac.at/index.php/s/GYAseQVDSslZ82A/download -O models/3dnet/Cat60_TestDatabase.zip --no-check-certificate

  # 3DNet Cat200 model
  wget https://data.acin.tuwien.ac.at/index.php/s/9ngYbr00uFvEThj/download -O models/3dnet/Cat200_ModelDatabase.zip --no-check-certificate
  ```

  - After download public data your data folder looks like below
  - Also use [unzip script](../models/unzip_data.sh)

- [ShapeNet](https://shapenet.org/) (you can download from web only.)
  - [ShapeNet alternated](https://huggingface.co/ShapeNet) (original web download link has been crashed. use alternated one.)
  <!-- ```shell
  wget  --no-check-certificate
  ``` -->  

  - After download public data your data folder looks like below
  - Also use [unzip script](../models/unzip_data.sh)

- Data path
```shell
UOPROOT # path to generate UOP-Sim data
├──models/ycb             # YCB data root
│  ├── 001_chips_can
│  ├── 002_master_chef_can
│  ├── ...
│  ├── 076_timer
│  └── 077_rubiks_cube
├──models/ShapeNetCore.v1 # ShapeNet data root
│  ├── 02691156
│  ├── 02691156.csv
│  ├── 02691156.zip
│  ├── ...
│  ├── count-models.sh
│  ├── get-metadata.sh
│  ├── jq
│  ├── README.txt
│  └── taxonomy.json
└──models/3DNet/          # 3DNet data_root
│  ├── Cat10_ModelDatabase
│  ├── Cat10_TestDatabase
│  ├── Cat200_ModelDatabase
│  ├── Cat60_ModelDatabase
│  └── Cat60_TestDatabase
└──models/uop_data        # the path uop data generated
```

<br>

#### 1-2-2. Preprocess 3D models

```
python uop_sim/preprocess.py --root <UOPROOT> --data_type ycb # ycb, shapenet, 3dnet
```

<br>

#### 1-2-3. Run Data Generator

```shell
python data_generator.py --data_type ycb --inspect
```

<br>
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
<br>

## 3. Test and Inference

*The code for Test and Demo will be released*

<br>
<br>

## 4. References

### 3D Object Model - YCB

```
[1] Berk Calli, Aaron Walsman, Arjun Singh, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M. Dollar, Benchmarking in Manipulation Research: The YCB Object and Model Set and Benchmarking Protocols, IEEE Robotics and Automation Magazine, pp. 36 – 52, Sept. 2015.

[2] Berk Calli, Arjun Singh, James Bruce, Aaron Walsman, Kurt Konolige, Siddhartha Srinivasa, Pieter Abbeel, Aaron M Dollar, Yale-CMU-Berkeley dataset for robotic manipulation research, The International Journal of Robotics Research, vol. 36, Issue 3, pp. 261 – 268, April 2017.

[3] Berk Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M. Dollar, The YCB Object and Model Set: Towards Common Benchmarks for Manipulation Research, proceedings of the 2015 IEEE International Conference on Advanced Robotics (ICAR), Istanbul, Turkey, 2015.
```

### 3D Object Model - 3DNet

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

### 3D Object Model - ShapeNet

```
@techreport{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel X. and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  number      = {arXiv:1512.03012 [cs.GR]},
  institution = {Stanford University --- Princeton University --- Toyota Technological Institute at Chicago},
  year        = {2015}
}
```

### Processing 3D Object Model - Watertight Method

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

See [LICENSE](../LICENSE)

<br>