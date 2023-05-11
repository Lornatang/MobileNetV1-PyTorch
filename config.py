# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0) ##initializes the seed for PyTorch's random number generator with value 0, if seed is 0, RNG will always generate the same sequence of pseudo-random numbers, regardless of the number of times you run your code
torch.manual_seed(0) ##initialize the seed for Python's built-in random number generator with value 0
np.random.seed(0) ##initialize the seed for the NumPy random number generator with value 0
# Use GPU for training by default
device = torch.device("cuda", 0) ##creates a PyTorch device object that represents a specific CUDA device (GPU) with index 0
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True ##turn on the cudnn.benchmark to optimize the performace of convolutional neural networks on NVIDIA GPUs. When cudnn.benchmark is set to True, PyTorch will use cuDNN's auto-tuner to dynamically choose the most efficient convolution algorithms based on the size and shape of the input data and the available GPU memory
# Model arch name
model_arch_name = "mobilenet_v1" ##initialise the model_arch_name name with the value mobilenet_v1
# Model normalization parameters
model_mean_parameters = [0.485, 0.456, 0.406] ##initialize model_mean_parameters with the mean values [0.485, 0.456, 0.406]
model_std_parameters = [0.229, 0.224, 0.225] ##initialize model_std_parameters with the standard deviation values [0.229, 0.224, 0.225]
#the above parameters are computed from the training data of the ImageNet dataset and are used to normalize the pixel values of the input images before they are fed into the model.
# Model number class
model_num_classes = 1000 ##initialize the number of model classes with the value 1000
# Current configuration parameter method
mode = "train" ##initialize mode with the value "train"
# Experiment name, easy to save weights and log files
exp_name = f"{model_arch_name}-ImageNet_1K" ##model_arch_name refers to the name of the neural network architecture being used, ImageNet_1K is the name of the dataset being used for training

if mode == "train":
    # Dataset address
    train_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_train"
    valid_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"

    #initialise some parameters for the model training process.
    image_size = 224 ##initialize the image size with 224 value
    batch_size = 128 ##initialize batch_size with 128 value
    num_workers = 4 ##initialize num_workers with value 4

    # The address to load the pretrained model
    pretrained_model_weights_path = "" ##initialize pretrained_model_weights_path with value "", this is used to store the path to a pretrained model.

    # Incremental training and migration training
    resume = "" ##initialize resume with value ""

    # Total num epochs
    epochs = 600 ##initialize epochs with value 600 which is the number of times the model will be trained on the entire dataset.

    # Loss parameters
    # Loss parameters are used to define the type and strength of the loss function used in a machine learning model during training.
    # The loss function is a measure of how well the model is able to predict the correct output for a given input. During training, the model tries to minimize this loss function by adjusting its parameters
    loss_label_smoothing = 0.1 ##initialize loss_label_smoothing with value 0.1
    loss_weights = 1.0 ##initialize loss_weights with value 1.0

    # Optimizer parameter
    # The optimizer is an algorithm that updates the parameters (weights and biases) of the model during training to minimize the loss function.
    model_lr = 0.1 ##initialize learning rate for the optimizer with value 0.1. It determines the step size at each iteration while moving toward a minimum of a loss function.
    model_momentum = 0.9 ##initialize model_momentum with value 0.9.  momentum helps the optimizer to remember the direction it was previously moving and use that information to accelerate the update in the same direction.
    model_weight_decay = 2e-05 ##initialize  the weight decay with value 2e-05. It is a regularization technique that adds a penalty term to the loss function to prevent overfitting.
    model_ema_decay = 0.99998 ##initialize model_ema_decay with value 0.99998. This is the exponential moving average decay for the optimizer. It is used to smooth out the update process for the parameters.

    # Learning rate scheduler parameter
    # The learning rate scheduler is a technique that adjusts the learning rate during training.
    lr_scheduler_T_0 = epochs // 4 ##initialize the initial learning rate decay steps for the learning rate scheduler, with the number of epochs after which the learning rate will be decayed.
    lr_scheduler_T_mult = 1 ##initialize the learning rate decay multiplication factor for the learning rate scheduler. It determines how much the learning rate will be decayed after each lr_scheduler_T_0 epochs.
    lr_scheduler_eta_min = 5e-5 ##initialize the minimum learning rate for the learning rate scheduler. It determines the minimum value that the learning rate can reach during training.

    # How many iterations to print the training/validate result
    train_print_frequency = 200 ##initialize with 200 which represents how often to print the training results during training
    valid_print_frequency = 20 ##initialize with 20 which represents how often to print the validation results during training, i.e., after how many iterations.

if mode == "test": ##check if mode has the value "test"
    # Test data address
    test_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val" ##initialize test_image_dir with the value of the given path

    # Test dataloader parameters
    image_size = 224 ##initialize image_size with the value 224
    batch_size = 256 ##initialize batch_size with the value 256
    num_workers = 4 ##initialize num_workers with value 4

    # How many iterations to print the testing result
    test_print_frequency = 20 ##initialize test_print_frequency with value 20 => every 20 iterations, the testing result will be printed.

    model_weights_path = "results/pretrained_models/Xception-ImageNet_1K-a0b40234.pth.tar" ##initialize model_weights_path with the given path, which is a pretrained model on the ImageNet 1K dataset
