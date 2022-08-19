# MobileNetV1-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861v1.pdf)
.

## Table of contents

- [MobileNetV1-PyTorch](#mobilenetv1-pytorch)
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
        - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](#mobilenets-efficient-convolutional-neural-networks-for-mobile-vision-applications)

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

- line 29: `model_arch_name` change to `mobilenet_v1`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/MobileNetV1-ImageNet_1K.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `mobilenet_v1`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change to `./results/pretrained_models/MobileNetV1-ImageNet_1K.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `mobilenet_v1`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/mobilenet_v1-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1704.04861v1.pdf](https://arxiv.org/pdf/1704.04861v1.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|    Model     |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:------------:|:-----------:|:-----------------:|:-----------------:|
| mobilenet_v1 | ImageNet_1K |   29.4%(**-**)    |     -(**-**)      |

```bash
# Download `mobilenet_v1.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `mobilenet_v1` model successfully.
Load `mobilenet_v1` model weights `/MobileNetV1-PyTorch/results/pretrained_models/mobilenet_v1.pth.tar` successfully.
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

#### MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

*Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig
Adam*

##### Abstract

We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are
based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural
networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These
hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints
of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared
to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide
range of applications and use cases including object detection, finegrain classification, face attributes and large
scale geo-localization.

[[Paper]](https://arxiv.org/pdf/1704.04861v1.pdf)

```bibtex
@article{howard2017mobilenets,
  title={Mobilenets: Efficient convolutional neural networks for mobile vision applications},
  author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  journal={arXiv preprint arXiv:1704.04861},
  year={2017}
}
```