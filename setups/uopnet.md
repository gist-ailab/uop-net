## Training
### Training UOP-Net 

```shell
python uop_net/run.py --shapenet_root $ROOT --threednet_root $ROOT --ckp_root $ROOT
```
- ```--shapenet_root``` : directory of UOP-Sim (Shapenet)
- ```--threednet_root``` : directory of UOP-Sim (3DNet)
- ```--ckp_root``` : root for saving UOP-Net checkpoint

- other arguments
  - points_shape : which shape wants to train (whole or partial)
  - stab_scale & plane scale : Loss scaling factor (default: 10, 1)
  - detla_d & delta_v : parameters for Plane Loss (default: 1.5, 0.5)
  - epochs, batch_size, num_workers : hyper paramerters for training UOP-Net (default: 1,000, 16, 4)



## Inference

### Sampling evaluation data

```shell
python evaluate_script/1.data_sampling.py --root $ROOT --name $data_name --partial --trial 100
```
- ```--root``` : directory of uop data, end with ..../uop_data
- ```--name``` : dataset name (ycb, 3dnet, shapenet)
- ```--partial``` : partial or whole
- ```--sampling``` : points sampling method
- ```--trial``` : # of partial cloud sampling for each object

### Inference each placement module

```shell
python evaluate_script/2.inference.py --root $ROOT --name $data_name --partial --trial 100 --module $mod --maxprocess $PS_NUM
```
- ```--root``` : directory of uop data, end with ..../uop_data
- ```--name``` : dataset name (ycb, 3dnet, shapenet)
- ```--partial``` : partial or whole
- ```--trial``` : # of partial cloud sampling for each object
- ```--module``` : module name of each placement module
  - uop : (ours)
  - trimesh : Convex Hull Stability Analysis(CHSA)
  - primitive : Bonding Box Fitting(BBF)
  - ransac : Ransac Plane Fitting(RPF)
- ```--maxprocess```: # of process(uop and ransac module -> only one process available)

### Evaluate each inference result on simulation

```shell
python evaluate_script/3.evaluate.py --root $ROOT --name $data_name --partial --trial 100 --module $mod --maxprocess $PS_NUM
```
- ```--root``` : directory of uop data, end with ..../uop_data
- ```--name``` : dataset name (ycb, 3dnet, shapenet)
- ```--partial``` : partial or whole
- ```--trial``` : # of partial cloud sampling for each object
- ```--module``` : module name of each placement module
  - uop : (ours)
  - trimesh : Convex Hull Stability Analysis(CHSA)
  - primitive : Bonding Box Fitting(BBF)
  - ransac : Ransac Plane Fitting(RPF)
- ```--maxprocess```: # of process(uop and ransac module -> only one process available)
- ```--tilt```: tilt evaluation table (default is 0)

### Calculate metric for evaluation result


```shell
python evaluate_script/4.metric.py --root $ROOT --name 'ycb' --partial --trial 100
```
- ```--root``` : directory of uop data, end with ..../uop_data
- ```--name``` : dataset name (ycb, 3dnet, shapenet)
- ```--partial``` : partial or whole
- ```--trial``` : # of partial cloud sampling for each object
- ```--tilt```: tilt evaluation table (default is 0)
