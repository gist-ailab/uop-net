#!/bin/bash




echo "-->> UOP-setup bash : start bash code"
# TODO : make flag operate
error_flag = 0
UOP_PRE_FLAG = 0



# ---- ---- ---- ----      ---- ---- ---- ----




# go to dir
cd ./setups
mkdir ./uop-download
cd ./uop-download

# = setup conda environment

# conda install
## download
#TODO : check storage space
#TODO : check linux ubuntu ver

# if ! command -v conda &> /dev/null
if [ ! command -v conda &> /dev/null ]
then
    echo "... conda could not be found"

    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh --no-check-certificate
    # https://www.anaconda.com/
    # https://www.anaconda.com/download#downloads

    ## install
    #TODO : check file exist
    #TODO : check permission ? / chmod +777 ?
    # sh ./Anaconda3-2023.09-0-Linux-x86_64.sh
    sh ./Anaconda3-2023.09-0-Linux-x86_64.sh -b
    eval "$(/home/ailab-ur5/anaconda3/bin/conda shell.bash hook)"
    conda init
else
    echo "... conda found"
    echo "... check env"
    if conda info --envs | grep -q uop_net
    then 
        echo "... uop_net - conda env already exists"
    else 
        echo "... uop_net - conda doesn't exists, follow 3_setup_env.sh"
        UOP_PRE_FLAG = 1
    fi
fi




# ---- ---- ---- ----      ---- ---- ---- ----



# sim install coppelia sim
if [ ! -d "./CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" ]
then
    echo "... there is no CoppeliaSim, download file. "
    ## download
    wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz --no-check-certificate
    # https://www.coppeliarobotics.com/
    # https://www.coppeliarobotics.com/downloads
    # https://www.coppeliarobotics.com/previousVersions
    # 

    ## unzip
    tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
fi
## turn on once.
#TODO : check uncompressed file 
cd CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/
./coppeliaSim.sh -h -s10000 -q # headless & quit after 10 sec
# ./coppeliaSim.sh
# at first, you must run with GUI monitor setup
cd ..




# ---- ---- ---- ----      ---- ---- ---- ----

