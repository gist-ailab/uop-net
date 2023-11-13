#!/bin/bash




echo "-->> UOP-setup bash : start bash code"
# TODO : make flag operate
error_flag = 0




# ---- ---- ---- ----      ---- ---- ---- ----




mkdir ./setups/uop-download
cd ./setups/uop-download


# = setup conda environment

# conda install
## download
#TODO : check storage space
#TODO : check linux ubuntu ver
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh --no-check-certificate
# https://www.anaconda.com/
# https://www.anaconda.com/download#downloads

## install
#TODO : check file exist
#TODO : check permission ? / chmod +777 ?
sh ./Anaconda3-2023.09-0-Linux-x86_64.sh
# sh ./Anaconda3-2023.09-0-Linux-x86_64.sh -b




# ---- ---- ---- ----      ---- ---- ---- ----

