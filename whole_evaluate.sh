#!/bin/bash

echo "Start to sampling evaluation data"
for data_name in 'ycb' '3dnet' 'shapenet'
    do
        python evaluate_script/sampling_data.py --dtype whole --sampling random --trial 100 --data $data_name
    done
echo "End to sampling evaluation data"

for mod in 'trimesh' 'primitive'
    do
        echo "Start inference" $mod
        python evaluate_script/inference_data.py --dtype whole  --trial 100 --module $mod --maxprocess 5
        
    done

for mod in 'ransac' 'sop-whole'
# for mod in 'ransac'
    do
        echo "Start inference" $mod
        python evaluate_script/inference_data.py --dtype whole  --trial 100 --module $mod --maxprocess 1
    done

for mod in 'trimesh' 'primitive' 'ransac' 'sop-whole'
    do
        echo "Start evaluate" $mod
        python evaluate_script/evaluate_data.py --dtype whole  --trial 100 --module $mod --maxprocess 5
    done

for mod in 'trimesh' 'primitive' 'ransac' 'sop-whole'
    do
        echo "Start metric" $mod
        python evaluate_script/metric_data.py --dtype whole  --trial 100 --module $mod
    done
