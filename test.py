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
from utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter##undefined

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def build_model() -> nn.Module:##undefined
    mobilenet_v1_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)##undefined
    mobilenet_v1_model = mobilenet_v1_model.to(device=config.device, memory_format=torch.channels_last)##undefined

    return mobilenet_v1_model


def load_dataset() -> CUDAPrefetcher:##undefined
    test_dataset = ImageDataset(config.test_image_dir,##undefined
                                config.image_size,##undefined
                                config.model_mean_parameters,##undefined
                                config.model_std_parameters,##undefined
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
    mobilenet_v1_model, _, _, _, _, _ = load_state_dict(mobilenet_v1_model, config.model_weights_path)##undefined
    print(f"Load `{config.model_arch_name}` "
          f"model weights `{os.path.abspath(config.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    mobilenet_v1_model.eval()

    # Load test dataloader
    test_prefetcher = load_dataset()

    # Calculate how many batches of data are in each Epoch
    batches = len(test_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)##undefined
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)##undefined
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)##undefined
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"Test: ")

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
            images = batch_data["image"].to(device=config.device, non_blocking=True)##undefined
            target = batch_data["target"].to(device=config.device, non_blocking=True)##undefined

            # Get batch size
            batch_size = images.size(0)##undefined

            # Inference
            output = mobilenet_v1_model(images)##undefined

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))##undefined
            acc1.update(top1[0].item(), batch_size)##undefined
            acc5.update(top5[0].item(), batch_size)##undefined

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)##undefined
            end = time.time()##undefined

            # Write the data during training to the training log file
            if batch_index % config.test_print_frequency == 0:##undefined
                progress.display(batch_index + 1)##undefined

            # Preload the next batch of data
            batch_data = test_prefetcher.next()##undefined

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1##undefined

    # print metrics
    progress.display_summary()##undefined


if __name__ == "__main__":
    main()
