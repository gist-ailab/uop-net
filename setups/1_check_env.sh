#!/bin/bash




echo "-->> UOP-setup bash : start system check bash code"
# TODO : make flag operate
UOP_ERROR_FLAG=0




# ---- ---- ---- ----      ---- ---- ---- ----




# = check system
UOP_GPU_STRING=$(lspci | grep asdf | cut -d" " -f 1)
if [ -z $UOP_GPU_STRING ]
then 
    echo "... [ERROR] GPU is Not Detected."
    UOP_ERROR_FLAG=1
    echo "-->> UOP-setup bash : End system check bash code with Error : $UOP_ERROR_FLAG"
    exit $UOP_ERROR_FLAG
else
    echo "... [CHECK] GPU Detected. continue ... "
fi

#TODO : check file storage


# check linux 20.04
#TODO : check ubuntu


# check nvidia driver ver
# check cuda ver
# check is there conda




# ---- ---- ---- ----      ---- ---- ---- ----




echo "-->> UOP-setup bash : system check"



