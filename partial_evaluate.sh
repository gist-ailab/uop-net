#!/bin/bash

# set ROOT to the root directory of the uop_data
ROOT=$1
PS_NUM=$2

echo "Start to sampling evaluation data"
for data_name in 'ycb'
    do
        python evaluate_script/1.data_sampling.py --root $ROOT --name $data_name --partial --trial 100
    done
echo "End to sampling evaluation data"

for mod in 'trimesh' 'primitive'
    do
        echo "Start inference" $mod
        python evaluate_script/2.inference.py --root $ROOT --name $data_name --partial --trial 100 --module $mod --maxprocess $PS_NUM
    done

for mod in 'ransac' 'uop'
    do
        echo "Start inference" $mod
        python evaluate_script/2.inference.py --root $ROOT --name $data_name --partial --trial 100 --module $mod
    done

for mod in 'trimesh' 'primitive' 'ransac' 'uop'
    do
        echo "Start evaluate" $mod
        python evaluate_script/3.evaluate.py --root $ROOT --name $data_name --partial --trial 100 --module $mod --maxprocess $PS_NUM
    done

echo "Start metric"
python evaluate_script/4.metric.py --root $ROOT --name 'ycb' --partial --trial 100
