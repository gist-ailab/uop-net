<!-- # UOP : Unseen Object Placement with Large Scale Simulation -->

<br>
<br>

<!-- TODO : add icons -->
<!-- add icons - projectpage / youtube / code / dataset link / AILAB hompage -->
<!-- TODO : add main concept image -->


<p align="center">
  <!-- <img src="resources/logo.png" align="center" width="25%"> -->
  <h1 align="center">
    <strong>UOP : Unseen Object Placement with Large Scale Simulation</strong>
  </h1>
</p>

<br>
<br>

<br>
<br>

![Main Image](resources/uop_figure/main_image.png)

This repository contains official implementation of following paper:

> **UOP-Net: Learning to Place Objects Stably using a Large-scale Simulation** <br>
> **Author:** *Anonymous* <br>
> **Abstract:** *Object placement is a fundamental task for robots, yet it remains challenging for partially observed objects. Existing methods for object placement have limitations, such as the requirement for a complete 3D model of the object or the inability to handle complex shapes and novel objects that restrict the applicability of robots in the real world. Herein, we focus on addressing the Unseen Object Placement(UOP) problem. We tackled the UOP problem using two methods: (1) UOP-Sim, a large-scale dataset to accommodate various shapes and novel objects, and (2) UOP-Net, a point cloud segmentation-based approach that directly detects the most stable plane from partial point clouds. Our UOP approach enables robots to place objects stably, even when the object's shape and properties are not fully known, thus providing a promising solution for object placement in various environments. We verify our approach through simulation and real-world robot experiments, demonstrating state-of-the-art performance for placing single-view and partial objects.* <br>
> [Click here for website and paper.](https://gistailab.github.io/uop/)

<br>
<br>

---

## Environment Setting
<!-- checked -->
* Please follow instruction in [setups/install.md](setups/install.md)

## Fast view our overall pipeline

* Please follow instruction in [setups/example.md](setups/example.md)

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
