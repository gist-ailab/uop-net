## Data Generation

### Download public 3D models

#### YCB Dataset

you can download from [web](https://www.ycbbenchmarks.com/) or run script below.

```shell
python download_ycb_dataset.py
```

#### 3DNet

you can download from [web](https://www.acin.tuwien.ac.at/vision-for-robotics/software-tools/3dnet-dataset/) or run script below.


```shell
# 3DNet Cat10 model/test
wget https://data.acin.tuwien.ac.at/index.php/s/7GrnwJlDQrajFTo/download -O models/3dnet/Cat10_ModelDatabase.zip --no-check-certificate
wget https://data.acin.tuwien.ac.at/index.php/s/6XkgKCAmpUyznnP/download -O models/3dnet/Cat10_TestDatabase.zip --no-check-certificate

# 3DNet Cat60 model/test
wget https://data.acin.tuwien.ac.at/index.php/s/uGarwrExmIjlwwP/download -O models/3dnet/Cat60_ModelDatabase.zip --no-check-certificate
wget https://data.acin.tuwien.ac.at/index.php/s/GYAseQVDSslZ82A/download -O models/3dnet/Cat60_TestDatabase.zip --no-check-certificate

# 3DNet Cat200 model
wget https://data.acin.tuwien.ac.at/index.php/s/9ngYbr00uFvEThj/download -O models/3dnet/Cat200_ModelDatabase.zip --no-check-certificate
```



#### ShapeNet

you can download from [web](https://shapenet.org/) only.

- [ShapeNet alternated](https://huggingface.co/ShapeNet) (original web download link has been crashed. use alternated one.)
  

#### After donwload public data your data folder should looks like below
```shell
UOPROOT # path to generate UOP-Sim data
├──models/ycb      # YCB data root
│  ├── 001_chips_can
│  ├── 002_master_chef_can
│  ├── ...
│  ├── 076_timer
│  └── 077_rubiks_cube
├──ShapeNetCore.v1 # ShapeNet data root
│  ├── 02691156
│  ├── 02691156.csv
│  ├── 02691156.zip
│  ├── ...
│  ├── count-models.sh
│  ├── get-metadata.sh
│  ├── jq
│  ├── README.txt
│  └── taxonomy.json
└──3DNet           # 3DNet data_root
│  ├── Cat10_ModelDatabase
│  ├── Cat10_TestDatabase
│  ├── Cat200_ModelDatabase
│  ├── Cat60_ModelDatabase
│  └── Cat60_TestDatabase
└──uop_data        # the path uop data generated
```

### Preprocess 3D models

```shell
python uop_sim/preprocess.py --root <UOPROOT> --data_type ycb 
```
- ```--root``` : root of uop data, all public data should be under UOPROOT

- ```--data_type```: name of public data; ycb, 3dnet, shapenet


### Run DataGenerator

```shell
python data_generator.py --data_type ycb --inspect
```

- ```--data_type```: name of public data; ycb, 3dnet, shapenet

- ```--inspect```: if true -> inspect the sampled label with tilted table


### References

Below are references of public dataset and thirdparties for generate UOP-Sim dataset.

```
# YCB
[1] Berk Calli, Aaron Walsman, Arjun Singh, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M. Dollar, Benchmarking in Manipulation Research: The YCB Object and Model Set and Benchmarking Protocols, IEEE Robotics and Automation Magazine, pp. 36 – 52, Sept. 2015.

[2] Berk Calli, Arjun Singh, James Bruce, Aaron Walsman, Kurt Konolige, Siddhartha Srinivasa, Pieter Abbeel, Aaron M Dollar, Yale-CMU-Berkeley dataset for robotic manipulation research, The International Journal of Robotics Research, vol. 36, Issue 3, pp. 261 – 268, April 2017.

[3] Berk Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M. Dollar, The YCB Object and Model Set: Towards Common Benchmarks for Manipulation Research, proceedings of the 2015 IEEE International Conference on Advanced Robotics (ICAR), Istanbul, Turkey, 2015.

# 3DNet
@inproceedings{wohlkinger20123dnet,
  title={3dnet: Large-scale object class recognition from cad models},
  author={Wohlkinger, Walter and Aldoma, Aitor and Rusu, Radu B and Vincze, Markus},
  booktitle={2012 IEEE international conference on robotics and automation},
  pages={5384--5391},
  year={2012},
  organization={IEEE}
}

# shapenet
@techreport{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel X. and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  number      = {arXiv:1512.03012 [cs.GR]},
  institution = {Stanford University --- Princeton University --- Toyota Technological Institute at Chicago},
  year        = {2015}
}

# manifold
@article{huang2018robust,
  title={Robust Watertight Manifold Surface Generation Method for ShapeNet Models},
  author={Huang, Jingwei and Su, Hao and Guibas, Leonidas},
  journal={arXiv preprint arXiv:1802.01698},
  year={2018}
}

# pyrep
@article{james2019pyrep,
  title={PyRep: Bringing V-REP to Deep Robot Learning},
  author={James, Stephen and Freese, Marc and Davison, Andrew J.},
  journal={arXiv preprint arXiv:1906.11176},
  year={2019}
}
```


