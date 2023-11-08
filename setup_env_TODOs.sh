#!/bin/bash




echo "-->> UOP-setup bash : start bash code"
# TODO : make flag
error_flag = 0




# ---- ---- ---- ----      ---- ---- ---- ----




# = check system
#TODO : check gpu
#TODO : check file storage
#

# check linux 20.04
#TODO : check ubuntu
echo "-->> UOP-setup bash : system check"

mkdir uop-download
cd uop-download
# check nvidia driver ver
# check cuda ver




# ---- ---- ---- ----      ---- ---- ---- ----




# = setup conda environment

# conda install
## download
#TODO : check storage space
#TODO : check linux ubuntu ver
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
# https://www.anaconda.com/
# https://www.anaconda.com/download#downloads

## install
#TODO : check file exist
#TODO : check permission ? / chmod +777 ?
sh ./Anaconda3-2023.09-0-Linux-x86_64.sh
# sh ./Anaconda3-2023.09-0-Linux-x86_64.sh -b


# conda create env 
## create env
#TODO : check env name is exist?
conda create -n uop
## activate
conda activate uop

# test
#TODO : check current env name




# ---- ---- ---- ----      ---- ---- ---- ----




# sim install coppelia sim
## download
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
# https://www.coppeliarobotics.com/
# https://www.coppeliarobotics.com/downloads
# https://www.coppeliarobotics.com/previousVersions
# 

## unzip
tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

## turn on once.
#TODO : check uncompressed file 
cd CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/
sh ./coppeliaSim.sh

# sim install pyrep

## clone git
#TODO : check is git installed ?
cd ..
git clone https://github.com/stepjam/PyRep.git

## add envrionment variable
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
#TODO : change ROOT abs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

## install pyrep
cd PyRep 
#TODO : check, is env name uop env?
conda activate uop # check duplicated
pip install -r requirements.txt
pip install .

# test
python -m import pyrep # is this right to check?




# ---- ---- ---- ----      ---- ---- ---- ----




# conda install package
#TODO : check, is env name uop env?
pip install torch
# pip install numpy open3d natsort tqdm trimesh pyfastnoisesimd opencv-python==4.4.0.44 point_cloud_utils
#TODO : make requirements.txt
# pip install -r requirements.txt

# test
#TODO : test run code




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



