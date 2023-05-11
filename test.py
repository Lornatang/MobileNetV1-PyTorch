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
import os
import time

import config
import model
import torch
from dataset import CUDAPrefetcher, ImageDataset
from torch import nn
from torch.utils.data import DataLoader
from utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter##imports load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter from utils module

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def build_model() -> nn.Module:##build a model
    mobilenet_v1_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)##model.__dict__ is a dictionary object that contains the attributes and methods of the model module. It allows us to access the desired model architecture by name, using the model_arch_name parameter
    mobilenet_v1_model = mobilenet_v1_model.to(device=config.device, memory_format=torch.channels_last)##mobilenet_v1_model.to(device=device, memory_format=torch.channels_last) is a method call to move the model to a specified device and memory format, device is a torch.device object representing the device (e.g. CPU or GPU) on which the model will be stored and operated, torch.channels_last specifies that the memory layout will be such that the channels (i.e. the color channels) will be stored in the last dimension of the tensor
    #MobileNetV1 is a type of deep convolutional neural network architecture designed for mobile and embedded vision applications
    return mobilenet_v1_model

#CUDAPrefetcher is a custom class that is used to prefetch data onto the GPU in batches to speed up training or inference in a PyTorch model.
def load_dataset() -> CUDAPrefetcher:## that loads a test dataset and returns a CUDAPrefetcher object.
    test_dataset = ImageDataset(config.test_image_dir,##he directory path where the test images are located.
                                config.image_size,##image size
                                config.model_mean_parameters,##list of values that should be used to normalize the input images. This list represents the mean values for each color channel of the images.
                                config.model_std_parameters,##list of values that should be used to normalize the input images. This list represents the standard deviation values for each color channel of the images.
                                "Test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return test_prefetcher


def main() -> None:
    # Initialize the model
    mobilenet_v1_model = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")

    # Load model weights
    mobilenet_v1_model, _, _, _, _, _ = load_state_dict(mobilenet_v1_model, config.model_weights_path)##loads the pre-trained weights for the MobileNetV1 model from the specified file path and assigns the loaded model to the variable mobilenet_v1_model
    print(f"Load `{config.model_arch_name}` "
          f"model weights `{os.path.abspath(config.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    mobilenet_v1_model.eval()

    # Load test dataloader
    test_prefetcher = load_dataset()

    # Calculate how many batches of data are in each Epoch
    batches = len(test_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)##the name of the metric is "Time", the format string is ":6.3f", which means that the metric value should be displayed as a floating-point number with 6 digits and 3 decimal places, and the type of summary is Summary.NONE, which means that no summary is computed over the metric values
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)##the name of the metric is "Acc@1", the format string is ":6.2f", which means that the metric value should be displayed as a floating-point number with 6 digits and 2 decimal places, and the type of summary is Summary.AVERAGE, the average of the metric value will be calculated over all the samples in the dataset
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)##the name of the metric is "Acc@5", the format string is ":6.2f", which means that the metric value should be displayed as a floating-point number with 6 digits and 2 decimal places, and the type of summary is Summary.AVERAGE, the average of the metric value will be calculated over all the samples in the dataset
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"Test: ")
#acc1 and acc5 are accuracy metrics

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, non_blocking=True)##moves the tensor batch_data["image"] to a specified device (e.g., a GPU device) and enables non-blocking data transfer which means that the transfer is done asyncronously
            target = batch_data["target"].to(device=config.device, non_blocking=True)####moves the tensor batch_data["targer"] to a specified device (e.g., a GPU device) and enables non-blocking data transfer which means that the transfer is done asyncronously

            # Get batch size
            batch_size = images.size(0)##images is a tensor of shape (batch_size, channels, height, width)

            # Inference
            output = mobilenet_v1_model(images)##passes the images batch through the mobilenet_v1_model object to obtain the model's output predictions

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))##topk argument is used to specify the number of top predictions to consider, which is 1 and 5 in this case, accuracy() is a function that calculates the top-k accuracy for a given output and target, The function returns two values, top1 and top5, which represent the top-1 and top-5 accuracy respectively
            acc1.update(top1[0].item(), batch_size)##updates acc1 with the top accuracy for the current batch by adding it to the running average, weighted by the batch size
            acc5.update(top5[0].item(), batch_size)##updates acc5 with the top accuracy for the current batch by adding it to the running average, weighted by the batch size

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)##update the batch_time
            end = time.time()##end time

            # Write the data during training to the training log file
            if batch_index % config.test_print_frequency == 0:##checks if the batch_index is a multiple of the config.test_print_frequency
                progress.display(batch_index + 1)##pdates the progress bar with the current batch index, which is the number of the current batch being processed in the loop

            # Preload the next batch of data
            batch_data = test_prefetcher.next()##Preload the next batch of data

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1##increment batch_index

    # print metrics
    progress.display_summary()##print metrics


if __name__ == "__main__":
    main()
