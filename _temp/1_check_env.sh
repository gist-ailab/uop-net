#!/bin/bash




echo "-->> UOP-setup bash : start system check bash code"
# TODO : make flag operate
UOP_ERROR_FLAG=0




# ---- ---- ---- ----      ---- ---- ---- ----




# = check system


# check linux 20.04
#TODO : check ubuntu


## check GPU
UOP_GPU_STRING=$(lspci | grep VGA | cut -d" " -f 1)
if [ -z $UOP_GPU_STRING ]
then 
    echo "... [ERROR] GPU is Not Detected."
    UOP_ERROR_FLAG=1
    echo "-->> UOP-setup bash : End system check bash code with Error : $UOP_ERROR_FLAG"
    exit $UOP_ERROR_FLAG
else
    echo "... [CHECK] GPU Detected. continue ... "
fi


## check Stroage
UOP_STORAGE_STRING=$(df -PBG $PWD | tail -1 | awk '{print $4}' | tr "G" " ")
if [ 2 -gt $UOP_STORAGE_STRING ];
then 
    echo "... [ERROR] Storage is Not Sufficient"
    UOP_ERROR_FLAG=1
    echo "-->> UOP-setup bash : End system check bash code with Error : $UOP_ERROR_FLAG"
    exit $UOP_ERROR_FLAG
else
    echo "... [CHECK] Storage Checked. continue ... "
fi

## check nvidia driver ver

## check cuda ver
# UOP_CUDA_STRING=$(nvcc -V | tail -1 | awk '{print $2}' | tr "cuda_" " ")
UOP_CUDA_STRING_1=$(nvcc -V | tail -1 | awk '{print $2}' | cut -d "/" -f 1 | tr -d "cuda_" | cut -d "." -f 1)
UOP_CUDA_STRING_2=$(nvcc -V | tail -1 | awk '{print $2}' | cut -d "/" -f 1 | tr -d "cuda_" | cut -d "." -f 2)
if [ -z $UOP_CUDA_STRING_1 ]
then 
    echo "... [ERROR] CUDA is Not Detected."
    UOP_ERROR_FLAG=1
    echo "-->> UOP-setup bash : End system check bash code with Error : $UOP_ERROR_FLAG"
    exit $UOP_ERROR_FLAG
else
    echo "... [CHECK] CUDA Detected. continue ... "
fi
if [ 10 -gt $UOP_CUDA_STRING_1 ];
then 
    echo "... [ERROR] CUDA is Not Sufficient"
    UOP_ERROR_FLAG=1
    echo "-->> UOP-setup bash : End system check bash code with Error : $UOP_ERROR_FLAG"
    exit $UOP_ERROR_FLAG
else
    echo "... [CHECK] CUDA Checked. continue ... "
fi

## check is there conda




# ---- ---- ---- ----      ---- ---- ---- ----




echo "-->> UOP-setup bash : system check end."



