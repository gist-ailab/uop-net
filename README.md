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
  <img src="resources/uop_figure/UOP-pipline.png" align="center" width="100%">
</p>

<br>
<br>

## Download Data

#### Evaluation Data

You can download evaluation data [here](https://drive.google.com/file/d/19mmLYNT_2reMV7C7Z8pEWwgjBulVobCG/view?usp=drive_link)

- 63 YCB data with 100 partial sampled point cloud which was used for test and evaluate
- You can inference and evaluate code after download this data

##### File tree

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
You can download UOP-Sim dataset [here](https://drive.google.com/file/d/11yvzrLgIbv8e3Yy2gyCG0k2QMHepeGa0/view?usp=drive_link)

## Generate Data (optional)

If you want to generate UOP-Sim data yourself. Please follow the instruction in [setups/data_generation.md](./setups/data_generation.md)


## Inference and Evaluate

To place objects with placement modules; UOP(ours), RPF, CHSA, BBF

you should follow step by step instruction in [setups/uopnet.md](./setups/uopnet.md#inference)

or run the combined script below (you have to change path of uop data inside [partial_evaluate.sh](./partial_evaluate.sh))

```shell
sh ./partial_evaluate.sh
```

## Training (optional)

We propose pretrained model weight inside our repository.

If you want to training yourself please follow the instruction [here](./setups/uopnet.md#training)


### ToDo

- [ ] update Experiment Tables
- [ ] update Experiment Inferences
- [ ] update Experiment Demo


<br>
<br>

## Experiment Tables
- experiment table will be upload.

<br>
<br>

## Experiment Inferences
- experiment Inferences images will be upload.

<br>

### Partial View Generation
- experiment Inferences images will be upload.

<br>

<!-- Prototype Table #1 -->
<table>
  <thead>
    <tr>
      <th colspan="4"><b>Prototype Table #1</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <!-- 1 -->
      <td><b>Object</b></td>
      <td><b>Whole Points</b></td>
      <td><b>Stable Label</b></td>
      <td><b>Partial Points</b></td>
    </tr>
    <tr>
      <!-- 2 -->
      <td align="center" ><img src="-.png" align="center" width="80%"></td>
      <th colspan="3" align="center"><img src="resources/demo/002_master_chef_can.gif" width="100%"></th>
      <!-- <td><img src="resources/demo/whole.gif" width="80%"></td>
      <td><img src="resources/demo/label.gif" align="center" align="center" width="80%"></td>
      <td><img src="resources/demo/partial.gif" align="center" width="80%"></td> -->
    </tr>
    <tr>
      <!-- 3 -->
      <th colspan="4"><b>Inference Result</b></th>
    </tr>
    <tr>
      <!-- 4 -->
      <th colspan="4"><img src="-.png" align="center" width="40%"></th>
    </tr>
  </tbody>
</table>

<br>

### Partial View Generation
- This is visualization of our partial points generation sequence(gif image).

<br>

| [**Object**](-) |
| :-:             |
| <img width="128" src="-.png"> |


<br>
<br>

## Experiment Demo
- experiment demo images will be upload.

<br>
<br>
<br>
<br>
<br>
<br>
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
