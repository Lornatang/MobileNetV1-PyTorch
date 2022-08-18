# Xception-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357v3.pdf).

## Table of contents

- [Xception-PyTorch](#xception-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Xception: Deep Learning with Depthwise Separable Convolutions](#xception-deep-learning-with-depthwise-separable-convolutions)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `xception`.
- line 31: `model_mean_parameters` change to `[0.5, 0.5, 0.5]`.
- line 32: `model_std_parameters` change to `[0.5, 0.5, 0.5]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 91: `model_weights_path` change to `./results/pretrained_models/xception`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `xception`.
- line 31: `model_mean_parameters` change to `[0.5, 0.5, 0.5]`.
- line 32: `model_std_parameters` change to `[0.5, 0.5, 0.5]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change to `./results/pretrained_models/Xception-ImageNet_1K-a0b40234.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `xception`.
- line 31: `model_mean_parameters` change to `[0.5, 0.5, 0.5]`.
- line 32: `model_std_parameters` change to `[0.5, 0.5, 0.5]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/xception-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1610.02357v3.pdf](https://arxiv.org/pdf/1610.02357v3.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|  Model   |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:--------:|:-----------:|:-----------------:|:-----------------:|
| xception | ImageNet_1K | 21.0%(**21.2%**)  |  5.5%(**5.6%**)   |

```bash
# Download `Xception-ImageNet_1K-a0b40234.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `xception` model successfully.
Load `xception` model weights `/Xception-PyTorch/results/pretrained_models/Xception-ImageNet_1K-a0b40234.pth.tar` successfully.
tench, Tinca tinca                                                          (81.23%)
barracouta, snoek                                                           (1.41%)
armadillo                                                                   (0.80%)
dugong, Dugong dugon                                                        (0.11%)
rock beauty, Holocanthus tricolor                                           (0.11%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Xception: Deep Learning with Depthwise Separable Convolutions

*François Chollet*

##### Abstract

We present an interpretation of Inception modules in convolutional neural networks as being an intermediate step
in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a
pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a
maximally large number of towers. This observation leads us to propose a novel deep convolutional neural network
architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions. We
show that this architecture, dubbed Xception, slightly outperforms Inception V3 on the ImageNet dataset (which Inception
V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350
million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3,
the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.

[[Paper]](https://arxiv.org/pdf/1610.02357v3.pdf)

```bibtex
@inproceedings{chollet2017xception,
  title={Xception: Deep learning with depthwise separable convolutions},
  author={Chollet, Fran{\c{c}}ois},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1251--1258},
  year={2017}
}
```