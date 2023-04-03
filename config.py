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
random.seed(0) ##undefined
torch.manual_seed(0) ##undefined
np.random.seed(0) ##undefined
# Use GPU for training by default
device = torch.device("cuda", 0) ##undefined
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True ##undefined
# Model arch name
model_arch_name = "mobilenet_v1" ##undefined
# Model normalization parameters
model_mean_parameters = [0.485, 0.456, 0.406] ##undefined
model_std_parameters = [0.229, 0.224, 0.225] ##undefined
# Model number class
model_num_classes = 1000 ##undefined
# Current configuration parameter method
mode = "train" ##undefined
# Experiment name, easy to save weights and log files
exp_name = f"{model_arch_name}-ImageNet_1K" ##undefined

if mode == "train":
    # Dataset address
    train_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_train"
    valid_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"

    image_size = 224 ##undefined
    batch_size = 128 ##undefined
    num_workers = 4 ##undefined

    # The address to load the pretrained model
    pretrained_model_weights_path = "" ##undefined

    # Incremental training and migration training
    resume = "" ##undefined

    # Total num epochs
    epochs = 600 ##undefined

    # Loss parameters
    loss_label_smoothing = 0.1 ##undefined
    loss_weights = 1.0 ##undefined

    # Optimizer parameter
    model_lr = 0.1 ##undefined
    model_momentum = 0.9 ##undefined
    model_weight_decay = 2e-05 ##undefined
    model_ema_decay = 0.99998 ##undefined

    # Learning rate scheduler parameter
    lr_scheduler_T_0 = epochs // 4 ##undefined
    lr_scheduler_T_mult = 1 ##undefined
    lr_scheduler_eta_min = 5e-5 ##undefined

    # How many iterations to print the training/validate result
    train_print_frequency = 200 ##undefined
    valid_print_frequency = 20 ##undefined

if mode == "test": ##undefined
    # Test data address
    test_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val" ##undefined

    # Test dataloader parameters
    image_size = 224 ##undefined
    batch_size = 256 ##undefined
    num_workers = 4 ##undefined

    # How many iterations to print the testing result
    test_print_frequency = 20 ##undefined

    model_weights_path = "results/pretrained_models/Xception-ImageNet_1K-a0b40234.pth.tar" ##undefined
