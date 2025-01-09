# HIT


This project is the official implementation of our paper 
[Holistic Interaction Transformer Network for Action Detection](https://arxiv.org/abs/2210.12686) (**WACV 2023**), authored
by Gueter Josmy Faure, Min-Hung Chen and Shang-Hong Lai. 

### Updates
- (03/06/2023) We have added the code to train/test on AVA [here](https://github.com/joslefaure/HIT_ava). Any issues about AVA, please open them from the other repo.


## Demo Video

![output1](https://user-images.githubusercontent.com/84136752/213919371-4a124959-2c2f-4d4c-8b9d-837417b584fc.gif) &nbsp; ![output2](https://user-images.githubusercontent.com/84136752/213919382-f7eb8347-afc0-4e38-adc0-faef8e13edc0.gif) &nbsp; ![output3](https://user-images.githubusercontent.com/84136752/213919453-78c48c77-2fb1-4c96-85e1-06a2fe51e6d6.gif)

## Installation


You need first to install this project, please check [INSTALL.md](INSTALL.md)

## Data Preparation

To do training or inference on J-HMDB, please check [DATA.md](DATA.md)
for data preparation instructions. Instructions for other datasets coming soon.

## Model Zoo

Please see [MODEL_ZOO.md](MODEL_ZOO.md) for downloading models.

## Training and Inference

To do training or inference with HIT, please refer to [GETTING_STARTED.md](GETTING_STARTED.md).


## Citation

If this project helps you in your research or project, please cite
this paper:

```
@InProceedings{Faure_2023_WACV,
    author    = {Faure, Gueter Josmy and Chen, Min-Hung and Lai, Shang-Hong},
    title     = {Holistic Interaction Transformer Network for Action Detection},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {3340-3350}
}
```

## LICENCE
MIT

## Acknowledgement
We are very grateful to the authors of [AlphAction](https://github.com/MVIG-SJTU/AlphAction) for open-sourcing their code from which this repository is heavily sourced. If your find this research useful, please consider citing their paper as well.

```
@inproceedings{tang2020asynchronous,
  title={Asynchronous Interaction Aggregation for Action Detection},
  author={Tang, Jiajun and Xia, Jin and Mu, Xinzhi and Pang, Bo and Lu, Cewu},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  year={2020}
}
```
