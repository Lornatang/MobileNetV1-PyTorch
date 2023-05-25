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
import os ##operating system interfaces
import time ##time access and conversions

import torch ##pytorch library
from torch import nn ##neural networks
from torch import optim ##optimization module
from torch.cuda import amp ##automatic mixed precision module from cuda
from torch.optim import lr_scheduler ##learning rate scheduler
from torch.optim.swa_utils import AveragedModel ##Stochastic Weight Averaging (SWA) technique during training
from torch.utils.data import DataLoader ##dataloader for loading data
from torch.utils.tensorboard import SummaryWriter ##logs/visualisation

import config ##import the config from project
import model ##import the model from project
from dataset import CUDAPrefetcher, ImageDataset #import the dataset 
from utils import accuracy, load_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter ##dictionary and summary utilities

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name])) #sort all the model names


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_acc1 = 0.0

    train_prefetcher, valid_prefetcher = load_dataset() #prefetch the dataset (load)
    print(f"Load `{config.model_arch_name}` datasets successfully.")

    mobilenet_v1_model, ema_mobilenet_v1_model = build_model() #build the mobilenet model
    print(f"Build `{config.model_arch_name}` model successfully.")

    pixel_criterion = define_loss() #define loss functions
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(mobilenet_v1_model) #optimize as you can
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer) #define all the schedulers
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path: #if there are pretrained weights
        mobilenet_v1_model, ema_mobilenet_v1_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict( #load from dictionary
            mobilenet_v1_model, #the model
            config.pretrained_model_weights_path, #the config weights
            ema_mobilenet_v1_model, #the model
            start_epoch, #the start epoch
            best_acc1, #the best accuracy
            optimizer, #the optimizer
            scheduler) #and the scheduler
        print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if config.resume: #if it could restore the model
        mobilenet_v1_model, ema_mobilenet_v1_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict( #load the weights in
            mobilenet_v1_model,
            config.pretrained_model_weights_path,
            ema_mobilenet_v1_model,
            start_epoch,
            best_acc1,
            optimizer,
            scheduler,
            "resume")
        print("Loaded pretrained generator model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name) #set the sample directory
    results_dir = os.path.join("results", config.exp_name) #and the result directory
    make_directory(samples_dir) #create them
    make_directory(results_dir) #create them

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name)) #create the writer

    # Initialize the gradient scaler
    scaler = amp.GradScaler() #initialize the gradient scaler

    for epoch in range(start_epoch, config.epochs):
        train(mobilenet_v1_model, ema_mobilenet_v1_model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, writer) #train te model
        acc1 = validate(ema_mobilenet_v1_model, valid_prefetcher, epoch, writer, "Valid") #define accuracy
        print("\n")

        # Update LR
        scheduler.step()  #update the scheduler

        # Automatically save the model with the highest index
        is_best = acc1 > best_acc1  #set the new best
        is_last = (epoch + 1) == config.epochs  #set the last 
        best_acc1 = max(acc1, best_acc1)  #the best accuracy is the higher value
        save_checkpoint({"epoch": epoch + 1,  #save checkpoint, incrementing epoch
                         "best_acc1": best_acc1,  #set best accuracy
                         "state_dict": mobilenet_v1_model.state_dict(),  #set state
                         "ema_state_dict": ema_mobilenet_v1_model.state_dict(),  #set state
                         "optimizer": optimizer.state_dict(),  #set optimizer
                         "scheduler": scheduler.state_dict()},  ##set scheduler
                        f"epoch_{epoch + 1}.pth.tar",  ##set epoch
                        samples_dir,  #set samples dir
                        results_dir,  #set results dir
                        is_best,  #set is_best
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:  #dataset load function 
    # Load train, test and valid datasets
    train_dataset = ImageDataset(config.train_image_dir,  #load directory from config
                                 config.image_size, #load image size
                                 config.model_mean_parameters, #load mean parameters
                                 config.model_std_parameters, #load std parameters
                                 "Train") #and train
    valid_dataset = ImageDataset(config.valid_image_dir, #load img dir for valid dataset
                                 config.image_size,
                                 config.model_mean_parameters,
                                 config.model_std_parameters,
                                 "Valid")

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset, #load train dataset
                                  batch_size=config.batch_size, #read batch size
                                  shuffle=True, #enable shuffle for randomness
                                  num_workers=config.num_workers, #set number of workers
                                  pin_memory=True, #set pin memory to true
                                  drop_last=True, #set drop last to true
                                  persistent_workers=True) #set persistent workers to true
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device) #place in prefetcher
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device) #place in prefetcher

    return train_prefetcher, valid_prefetcher


def build_model() -> [nn.Module, nn.Module]:  #build model function
    mobilenet_v1_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)  #load architecture and num classes
    mobilenet_v1_model = mobilenet_v1_model.to(device=config.device, memory_format=torch.channels_last) #set device and memory

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter #calculate the average
    ema_mobilenet_v1_model = AveragedModel(mobilenet_v1_model, avg_fn=ema_avg) #set the avg model

    return mobilenet_v1_model, ema_mobilenet_v1_model #return it


def define_loss() -> nn.CrossEntropyLoss: #define loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing) #crossentropy criterion for loss
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last) #add the criterion to device

    return criterion #return it


def *define_optimizer(model) -> optim.SGD: #optimizer definition
    optimizer = optim.SGD(model.parameters(), #load parameters
                          lr=config.model_lr, #load lr
                          momentum=config.model_momentum, #load momentum 
                          weight_decay=config.model_weight_decay) #load weight decay

    return optimizer


def define_scheduler(optimizer: optim.SGD) -> lr_scheduler.CosineAnnealingWarmRestarts: #schduler definition
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, #loads the optimizer
                                                         config.lr_scheduler_T_0, #loads  t0
                                                         config.lr_scheduler_T_mult, ##loads tmult
                                                         config.lr_scheduler_eta_min) ##loads eta min

    return scheduler


def train( #train function
        model: nn.Module, #load neural network module
        ema_model: nn.Module, #exp moving average model
        train_prefetcher: CUDAPrefetcher, #data prefetcher
        criterion: nn.CrossEntropyLoss, #loss function
        optimizer: optim.Adam, #optimization algorithm
        epoch: int, #epoch number of training
        scaler: amp.GradScaler, #gradient scaler for mixed precision
        writer: SummaryWriter #writer for exporting
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher) #batches is length of the prefetcher
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f") #prints batch time 
    data_time = AverageMeter("Data", ":6.3f") #data time
    losses = AverageMeter("Loss", ":6.6f") #losses
    acc1 = AverageMeter("Acc@1", ":6.2f") #accuracy @ 1
    acc5 = AverageMeter("Acc@5", ":6.2f") #accuracy @ 5
    progress = ProgressMeter(batches, #progress print dependent on batches
                             [batch_time, data_time, losses, acc1, acc5],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train() #train the model

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0 #initialize batch index with 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset() #reset the prefetcher
    batch_data = train_prefetcher.next() #load the data

    # Get the initialization training time
    end = time.time()

    while batch_data is not None: #while there is data to run on
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True) #load image
        target = batch_data["target"].to(device=config.device, non_blocking=True)  #set target device

        # Get batch size
        batch_size = images.size(0)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True) #initialize gradient

        # Mixed precision training
        with amp.autocast(): #automatic mixed precision
            output = model(images) #compute output of model
            loss = config.loss_weights * criterion(output, target) #compute loss

        # Backpropagation
        scaler.scale(loss).backward() #scale the loss backwards
        # update generator weights
        scaler.step(optimizer) #optimize
        scaler.update() #update scale

        # Update EMA
        ema_model.update_parameters(model) #exp moving avg model update

        # measure accuracy and record loss
        top1, top5 = accuracy(output, target, topk=(1, 5)) #measury top1,5 with topk function
        losses.update(loss.item(), batch_size) #update the losses
        acc1.update(top1[0].item(), batch_size) ##update accuracy @1
        acc5.update(top5[0].item(), batch_size) #update accuracy @5

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end) #update batch time
        end = time.time() #set end time

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1) #write data in log
            progress.display(batch_index + 1) #display on progress the current batch

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        ema_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        mode: str
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"{mode}: ")

    # Put the exponential moving average model in the verification mode
    ema_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad(): ##Context manager to disable gradient calculation, used during inference/testing to save memory and speed up computation
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            target = batch_data["target"].to(device=config.device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Inference
            output = ema_model(images)

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0].item(), batch_size)
            acc5.update(top5[0].item(), batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/Acc@1", acc1.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return acc1.avg


if __name__ == "__main__": #runner
    main() #calls main
