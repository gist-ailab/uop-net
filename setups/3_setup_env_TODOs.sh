#!/bin/bash




echo "-->> UOP-setup bash : start bash code"
# TODO : make flag operate
error_flag = 0


# conda create env 
## create env
#TODO : check env name is exist?
if conda info --envs | grep -q uop_net; 
then echo "... uop_net already exists"; 
else conda create -y -n uop_net python=3.8; 
fi
## activate
conda activate uop_net
pip install --upgrade pip setuptools wheel

# test
#TODO : check current env name




# ---- ---- ---- ----      ---- ---- ---- ----




# sim install pyrep

## clone git
#TODO : check is git installed ?
cd ..
git clone https://github.com/stepjam/PyRep.git

## add envrionment variable
UOP_COPPELIASIM_PATH=$(realpath CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/)
echo 'export COPPELIASIM_ROOT='\"$UOP_COPPELIASIM_PATH\" >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT' >> ~/.bashrc
echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT' >> ~/.bashrc


## install pyrep
cd PyRep 
#TODO : check, is env name uop env?
# conda activate uop # check duplicated
pip install -r requirements.txt

source ~/.bashrc
conda activate uop_net

pip install .
cd ..
cd ..

# test
python -c "import pyrep"




# ---- ---- ---- ----      ---- ---- ---- ----




# conda install package
#TODO : check, is env name uop env?
pip install torch torchvision torchaudio    # cuda 12.1 torch 2.1.0 py38 / 2023.11.13.
pip install numpy open3d natsort tqdm trimesh pyfastnoisesimd opencv-python==4.4.0.44 point_cloud_utils
pip install pycollada pyglet plotly networkx
#TODO : make requirements.txt
# pip install -r requirements.txt

# test
#TODO : test run code

cd ..
cd ..


# ---- ---- ---- ----      ---- ---- ---- ----




# download dataset

# check YCB dataset
## read dataset
## visualize dataset

# check 3DNet dataset
## read dataset
## visualize dataset

# check ShapeNet dataset
## read dataset
## visualize dataset

# check UOP-Sim dataset
## read dataset
## visualize dataset




# ---- ---- ---- ----      ---- ---- ---- ----




# 



