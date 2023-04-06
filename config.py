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
random.seed(0) #sets a predictable random seed that can be reproduced
torch.manual_seed(0) #sets the seed for generating random numbers
np.random.seed(0) #sets the seed for the numpy library
# Use GPU for training by default
device = torch.device("cuda", 0) #sets the default device for pytorch to the first available GPU
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True #enables the cudnn benchmarking mode
# Model arch name
model_arch_name = "mobilenet_v1" #stores the name of the model architecture
# Model normalization parameters
model_mean_parameters = [0.485, 0.456, 0.406] #parameters stored to normalize the input data
model_std_parameters = [0.229, 0.224, 0.225] #parameters stored to normalize the input data
# Model number class
model_num_classes = 1000 #a variable for the number of classes being stored
# Current configuration parameter method
mode = "train" #stores the current configuration method
# Experiment name, easy to save weights and log files
exp_name = f"{model_arch_name}-ImageNet_1K" #stores the name of the current experiment

if mode == "train":
    # Dataset address
    train_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_train"
    valid_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"

    image_size = 224 #sets image size
    batch_size = 128 #sets batch size
    num_workers = 4 #and number of workers for the training and validation

    # The address to load the pretrained model
    pretrained_model_weights_path = "" #sets the path to the pretrained model weights

    # Incremental training and migration training
    resume = "" #sets the path to the checkpoint for incremental/migration training

    # Total num epochs
    epochs = 600 #total number of epochs to train for

    # Loss parameters
    loss_label_smoothing = 0.1 #label smoothing parameter
    loss_weights = 1.0 #loss weight parameter

    # Optimizer parameter
    model_lr = 0.1 # learningrate parameter
    model_momentum = 0.9 #momentum parameter
    model_weight_decay = 2e-05 #weight decay parameter
    model_ema_decay = 0.99998 #exponential moving average decay

    # Learning rate scheduler parameter
    lr_scheduler_T_0 = epochs // 4 #nr of epochs before the first restart
    lr_scheduler_T_mult = 1 #multiplication factor after each restart
    lr_scheduler_eta_min = 5e-5 #minimum learning rate

    # How many iterations to print the training/validate result
    train_print_frequency = 200 #frequency of training
    valid_print_frequency = 20 #frequency of printing the result

if mode == "test": #checks for test mode
    # Test data address
    test_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val" #set the directory for testing img dataset

    # Test dataloader parameters
    image_size = 224 #set image size
    batch_size = 256 #set batch size
    num_workers = 4 #set num of workers

    # How many iterations to print the testing result
    test_print_frequency = 20 #set frequency of printing the result

    model_weights_path = "results/pretrained_models/Xception-ImageNet_1K-a0b40234.pth.tar" #path to the model weights for testing
