#!/bin/bash




echo "-->> UOP-setup bash : start bash code"
# TODO : make flag operate
UOP_ERROR_FLAG=0




# ---- ---- ---- ----      ---- ---- ---- ----




# = check system
#TODO : check gpu
UOP_GPU_STRING=$(lspci | grep VGA | cut -d" " -f 1)
if [ -z $UOP_GPU_STRING ]
then 
    echo "... [ERROR] GPU is Not Detected."
    UOP_ERROR_FLAG=1
    exit $UOP_ERROR_FLAG
else
    echo "... [CHECK] GPU Detected. continue ... "
fi
#TODO : check file storage


# check linux 20.04
#TODO : check ubuntu
echo "-->> UOP-setup bash : system check"

# check nvidia driver ver
# check cuda ver
# check is there conda




# ---- ---- ---- ----      ---- ---- ---- ----



