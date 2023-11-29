<!-- # UOP : Unseen Object Placement with Large Scale Simulation -->

<br>
<br>

<!-- TODO : add icons -->
<!-- add icons - projectpage / youtube / code / dataset link / AILAB hompage -->
<!-- TODO : add main concept image -->

<p align="center">
  <img src="resources/uop_figure/UOP-Sim.gif" align="center" width="100%">
  <h1 align="center">
    <strong>Learning to Place Unseen Objects Stably using a Large-scale Simulation</strong>
  </h1>
</p>
<center>
  <br>
  <a href="mailto:sangjun7@gm.gist.ac.kr">Sangjun Noh</a>
  <sup>*</sup> &nbsp; &nbsp; &nbsp; &nbsp;
  <a href="mailto:raeyo@gm.gist.ac.kr">Raeyoung Kang</a>
  <sup>*</sup> &nbsp; &nbsp; &nbsp; &nbsp;
  <a href="mailto:ailab.ktw@gm.gist.ac.kr">Taewon Kim</a>
  <sup>*</sup> &nbsp; &nbsp;
  <br>
  <a href="mailto:shback@gm.gist.ac.kr">Seunghyeok Back</a> &nbsp; &nbsp;
  <a href="mailto:bakseongho@gm.gist.ac.kr">Seongho Bak</a> &nbsp; &nbsp;
  <a href="mailto:kyoobinlee@gist.ac.kr">Kyoobin Lee</a> &nbsp; &nbsp;
  <sup>†</sup>
  <br>
  <br>
  GIST AILAB
  <br>
  <sup>*</sup>These authors contributed equally to the paper
  <sup>†</sup>Corresponding author
</center>

<br>

<center>
  <a href="https://arxiv.org/abs/2303.08387" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-arxiv.org-black">
  </a> &nbsp; &nbsp; &nbsp; &nbsp;
  <a href="https://gistailab.github.io/uop/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-github.io-white">
  </a> &nbsp; &nbsp; &nbsp; &nbsp;
  <a href="" target='_blank'>
    <img src="https://youtu.be/08F4jxSEL7A">
  </a>
</center>

<br>
<br>

This repository contains official implementation of following paper:

> **Learning to Place Objects Stably using a Large-scale Simulation** <br>
> 
> *We introduce the Unseen Object Placement (UOP) approach, combining UOP-Sim, a diverse dataset for various object shapes, with UOP-Net, a point cloud segmentation method for detecting stable planes from partial observation.* <br>
[Click here for website and paper.](https://gistailab.github.io/uop/)
<!-- This approach, validated through simulations and real-world tests, shows superior performance in placing objects with partial observation, enhancing the real-world robot applicability. -->
<!--  -->

<br>
<br>

---

## Environment Setting
* Please follow instruction in [setups/install.md](setups/install.md)

<br>

## Fast view our overall pipeline
* Please follow instruction in [setups/example.md](setups/example.md)
<p align="center">
  <img src="resources/uop_figure/UOP-pipeline.png" align="center" width="100%">
</p>

<br>
<br>

## Download Data

#### Evaluation Data

- The *UOP-Sim* contains 63 YCB object datas for evaluation with 100 partial sampled points on each objects. these evaluation set was used for test and evaluate.
- You can run inference and evaluate code after download this data
- *UOP-Sim* Evaluation data can be download this google drive [link](https://drive.google.com/file/d/19mmLYNT_2reMV7C7Z8pEWwgjBulVobCG/view?usp=drive_link) or run the [0.download_uop_sim_dataset.sh](./example/0.download_uop_sim_dataset.sh) command.
```shell
sh ./example/0.download_uop_sim_dataset.sh
# output : uop_data_for_evaluation.zip 
```

##### Evaluation Data File tree

```shell
└── uop_data
    └── ycb
        ├── 002_master_chef_can
        │   ├── inspected_zaxis.pkl   # uopsim label(axis of placement)
        │   ├── mesh_watertight.ply   # watertight mesh
        │   ├── model.ttm             # scene model to evaluate in simulation
        │   └── partial
        │       ├── 0.pkl             # partial sampled point cloud
        │       ├── 1.pkl
        │       ├── ...
        │       └── 99.pkl
        ├── 003_cracker_box
        ├── ...
        └── 077_rubiks_cube
127 directories, 6489 files, 171.1MB
```

#### Whole Data
You can download whole *UOP-Sim* dataset [here](https://drive.google.com/file/d/11yvzrLgIbv8e3Yy2gyCG0k2QMHepeGa0/view?usp=drive_link)

<br>
<br>

## Generate Data (optional)

If you want to generate *UOP-Sim* data yourself. Please follow the instruction in [setups/data_generation.md](./setups/data_generation.md)

<br>
<br>

## Inference and Evaluate

To place objects with placement modules; UOP(ours), RPF, CHSA, BBF

you should follow step by step instruction in [setups/uopnet.md](./setups/uopnet.md#inference)

or run the combined script below. 

```shell
sh ./partial_evaluate.sh 'path/to/uop_data(ex.~/uop_data)' process_num(ex.16)
```

#### Result

After all processes, you can get below table.

All metrics are the results of measuring the object's movement until it stops after placeing it at the table (in simulation).

```shell
-----------------------------------------------------------------
Module           | UOP   | RPF   | CHSA  | BBF  
rotation(deg)    | 6.93  | 24.26 | 37.56 | 44.02
translation(cm)  | 0.58  | 2.84  | 5.45  | 6.21 
l2norm           | 0.19  | 0.63  | 0.97  | 1.13 
Success(<10deg)  | 69.46 | 60.87 | 42.29 | 30.40
Success(/infer)  | 85.05 | 60.87 | 42.29 | 30.40
-----------------------------------------------------------------
```

- ```rotation``` : rotation of object
- ```translation``` : translation of object
- ```l2norm``` : transform matrix differnce(l2norm)
- ```Success(<10deg)```: success rate of placement for all trial, rotation error lower than 10 deg.
- ```Success(/infer)```: success rate of placement for inferenced trial, rotation error lower than 10 deg.

#### Visualize inference results

You can visualize inference result of each module with matplotlib

```shell
python example/visualize_inference_result.py --exp_file path/to/inference_result/endwith.pkl --module uop
```
- ```--exp_file``` : pkl file of inference result of each module, after inference the result saved at each object directory(ex. ~/uop_data/ycb/002_master_chef_can/partial_eval/uop/0.pkl)
- ```--module``` : module name of each placement module
  - uop : (ours)
  - trimesh : Convex Hull Stability Analysis(CHSA)
  - primitive : Bonding Box Fitting(BBF)
  - ransac : Ransac Plane Fitting(RPF)

<br>
<br>

## Training (optional)

We propose pretrained model weight inside our repository.

If you want to training yourself please follow the instruction [here](./setups/uopnet.md#training)

<br>
<br>

## Inferences Result
- This is the UOP-Net and other methods Inferences images. Partial points are observed like follow [Partial View Generation](#partial-view-generation)

<!-- Prototype Table #1 -->
<p align="center">
  <table align="center">
    <thead>
      <tr>
        <th colspan="5"><b><center>Sample Inferences Result Visualization</center></b></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <!-- 1 -->
        <th colspan="2" align="center"><b><center>　Object　</center></b></th>
        <th colspan="1" align="center"><b><center>Whole Points</center></b></th>
        <th colspan="1" align="center"><b><center>Stable Label</center></b></th>
        <th colspan="1" align="center"><b><center>Partial Points</center></b></th>
      </tr>
      <tr>
        <!-- 2 -->
        <td colspan="2" align="center" ><img src="resources/inference/002_master_chef_can.png" align="center" width="100%"></td>
        <td colspan="3" align="center"><img src="resources/inference/002_master_chef_can.gif" width="100%"></td>
        <!-- <td><img src="resources/demo/whole.gif" width="80%"></td>
        <td><img src="resources/demo/label.gif" align="center" align="center" width="80%"></td>
        <td><img src="resources/demo/partial.gif" align="center" width="80%"></td> -->
      </tr>
      <tr>
        <!-- 3 -->
        <th colspan="5"><b>Inference Result</b></th>
      </tr>
      <tr>
        <!-- 4 -->
        <td colspan="5"><img src="resources/inference/partial_inference.png" align="center" width="100%">　　</td>
      </tr>
      <tr>
        <td colspan="1" align="center"><b><center>Partial Points</center></b></td>
        <td colspan="1" align="center"><b><center>　UOP-Net　</center></b></td>
        <td colspan="1" align="center"><b><center>RPF</center></b></td>
        <td colspan="1" align="center"><b><center>CHSA</center></b></td>
        <td colspan="1" align="center"><b><center>BBF</center></b></td>
      </tr>
    </tbody>
  </table>
</p>

<br>

### Partial View Generation
- This is visualization of our partial points generation sequence(gif image) and inference results at each partial points

<br>

<center>
  <p align="center">
    <table align="center">
      <thead>
        <tr>
          <th colspan="5" align="center"><b><center>Partial View Points & Inference Result</center></b></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th colspan="5" align="center"><img src="resources/inference/partial_2sec.gif" width="100%"></th>
        </tr>
        <tr>
          <th colspan="1" align="center"><b><center>Partial</center></b></th>
          <th colspan="1" align="center"><b><center>UOP-Net</center></b></th>
          <th colspan="1" align="center"><b><center>　RPF　</center></b></th>
          <th colspan="1" align="center"><b><center>　CHSA　</center></b></th>
          <th colspan="1" align="center"><b><center>　BBF　</center></b></th>
        </tr>
      </tbody>
    </table>
  </p>
</center>

<br>
<br>


---


## Citation

```
@article{noh2023learning,
  title={Learning to Place Unseen Objects Stably using a Large-scale Simulation},
  author={Noh, Sangjun and Kang, Raeyoung and Kim, Taewon and Back, Seunghyeok and Bak, Seongho and Lee, Kyoobin},
  journal={arXiv preprint arXiv:2303.08387},
  year={2023}
}
```

## License

See [LICENSE](LICENSE)
