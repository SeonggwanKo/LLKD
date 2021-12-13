# Learning Lightweight Low-Light Enhancement Network using Pseudo Well-Exposed Images


This repository contains the implementation of the following paper:
>**Learning Lightweight Low-Light Enhancement Network using Pseudo Well-Exposed Images**<br>
>Seonggwan Ko*, Jinsun Park*, Byungjoo Chae and Donghyeon Cho<br>
>Signal Processing Letters<br>
 
# Overview
<p align="center">
<img src = "./assets/model_git.png" width="70%">
<p>


## Visual results
<p align="center">
<img src = "./assets/image_git.png" width="80%">
<p>

## Requirements
The following packages must be installed to perform the proposed model:
- PyTorch 1.7.1
- torchvision 0.8.2
- Pillow 8.2.0
- TensorBoardX 2.2
- tqdm


## Test 
Test datasets should be arranged as the following folder `TEST/testset`.
```bash
dataset
│   ├── test
│   │   ├── LIME
│   │   ├── LOL
│   │   ├── DICM
│   │   └── ...
└── ...
```
If you set up the folder, you can make it run.
```
python test.py
```

## Train
To train the proposed model, the following options are required:
```
python train.py --lowlight_images_path 'your_dataset_path' --gt_images_path 'your_GT_dataset_path' --pretrain_dir  'your_pretrain_path'
```

`lowlight_images_path` is your low-light images

`gt_images_path` is your ground truth images

`pretrain_dir` is your pretrained teacher model


# Dataset
<p align="center">
<img src = "./assets/1.JPG" width="80%">
<p>
We provide 10,000 training pairs and 387 test images.

Please click [here]() if you want to download our dataset.

## Dataset Creation
- We collected 25,967 low-light images from BDD100k(4,830 images) and Dark Zurich(5,336 images), LoLi-Phone(6,442 images), ExDark(7,263 images), SICE(1,611), LOL(485 images).
- Then, we generate pseudo well-exposed images using the pretrained EnlightenGAN, and additionally reduce noise using DnCNN.



## Citation
```
 @ARTICLE{,
  author={S. {Ko} and J. {Park} and B. {Chae} and D. {Cho}},
  journal={IEEE Signal Processing Letters}, 
  title={Learning Lightweight Low-Light Enhancement Network using Pseudo Well-Exposed Images}, 
  year={2021}
}
```


## License and Acknowledgement
The code framework is mainly modified from [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE), [AdaBelief](https://github.com/juntang-zhuang/Adabelief-Optimizer) and [SPKD](https://github.com/DongGeun-Yoon/SPKD). Please refer to the original repo for more usage and documents.
Thanks to authors for sharing the codes!




