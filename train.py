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
import os ##import os module
import time ##import time module

import torch ##import torch module
from torch import nn ##import nn class from torch module. torch.nn module provides a set of classes and functions for defining and training neural network models
from torch import optim ##import optim class from torch module. Provides a set of optimization algorithms that can be used to train neural network models
from torch.cuda import amp ##import amp class from torch.cuda. amp stands for Automatic Mixed Precision and is a feature in PyTorch that enables faster and more memory-efficient training of neural networks by automatically mixing precision of the tensor calculations
from torch.optim import lr_scheduler ##import lr_scheduler class from torch.optim. lr_scheduler module provides a number of different strategies for learning rate scheduling
from torch.optim.swa_utils import AveragedModel ##import AveragedModel class from torch.optim.swa_utils
from torch.utils.data import DataLoader ##import DataLoader class from torch.utils.data module
from torch.utils.tensorboard import SummaryWriter ##import SummaryWriter class from torch.utils.tensorboard module

import config ##import config module
import model ##import model module
from dataset import CUDAPrefetcher, ImageDataset ##import CUDAPrefetcher, ImageDataset classes from dataset
from utils import accuracy, load_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter ##import accuracy, load_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter classes form utils modules

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name])) ##for each name in dictionary check if is lower case and not start with __ and if that name can be called


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_acc1 = 0.0

    train_prefetcher, valid_prefetcher = load_dataset() ##loads the training and validation dataset
    print(f"Load `{config.model_arch_name}` datasets successfully.")

    mobilenet_v1_model, ema_mobilenet_v1_model = build_model() ##build_model() is a function that constructs two instances of a MobileNet V1 neural network model, mobilenet_v1_model and ema_mobilenet_v1_model.  MobileNet V1 is a type of convolutional neural network. ema_mobilenet_v1_model likely refers to an instance of the MobileNet V1 neural network model in PyTorch that is used to maintain an exponential moving average of the weights of another model
    print(f"Build `{config.model_arch_name}` model successfully.")

    pixel_criterion = define_loss() ##used to measure the difference between the predicted output of a model and the expected output. During training, the goal is to minimize this loss function to improve the accuracy of the model
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(mobilenet_v1_model) ##stores an instance of an optimization algorithm that is used to update the weights of a neural network during training.
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer) ##creates and returns an instance of the appropriate learning rate scheduler based on the optimizer being used and the learning rate schedule chosen. The scheduler object is responsible for adjusting the learning rate of the optimizer during training according to a predefined schedule. The learning rate schedule determines how the learning rate changes over time during training.
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path: ## checks whether a path to a pre-trained model weights file is specified in the config object.
        mobilenet_v1_model, ema_mobilenet_v1_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict( ## to load a saved state dictionary from a previously trained model.
            mobilenet_v1_model, ##the main model that will be trained.
            config.pretrained_model_weights_path, ##Pretrained model weights
            ema_mobilenet_v1_model, ##the exponential moving average (EMA) model, which is used for model checkpointing and evaluation
            start_epoch, ##represents the starting epoch number for training the mode
            best_acc1, ##the highest validation accuracy achieved so far during training.
            optimizer, ##the optimizer object that is used to optimize the model weights during training.
            scheduler) ##the learning rate scheduler object that is used to adjust the learning rate during training.
        print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")
#an epoch refers to a single pass through the entire training dataset during the training of a machine learning model. During one epoch, the model takes in all the training examples, makes predictions on them, calculates the loss based on the difference between the predicted and true values, and updates the model weights to minimize the loss. One epoch consists of several batches of training examples, and the number of batches depends on the batch size of the training process. After an epoch, the model parameters may be updated and the training process may continue with the next epoch.
    print("Check whether the pretrained model is restored...")
    if config.resume: ##checks whether the resume flag is set in the config object. If it is set to True, then some previously saved state of the model and the training process is loaded to resume the training from where it was left off. If it is set to False or not specified, then the training is started from scratch
        mobilenet_v1_model, ema_mobilenet_v1_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict( ## to load a saved state dictionary from a previously trained model.
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

    # Create an experiment results
    samples_dir = os.path.join("samples", config.exp_name) ##is creating a directory path by joining two strings using the os.path.join() method. The first string is "samples" and the second string is config.exp_name, which is a variable that holds the name of the experiment.
    results_dir = os.path.join("results", config.exp_name) ##is creating a directory path by joining two strings using the os.path.join() method. The first string is "results" and the second string is config.exp_name, which is a variable that holds the name of the experiment.
    make_directory(samples_dir) ##create sample_dir directory
    make_directory(results_dir) ##create results_dir driectory

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name)) ##Create training process log file

    # Initialize the gradient scaler
    scaler = amp.GradScaler() ##Initialize the gradient scaler

    for epoch in range(start_epoch, config.epochs):
        train(mobilenet_v1_model, ema_mobilenet_v1_model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, writer) ##trains a model, mobilenet_v1_model: This is the model being trained. It is likely an instance of the MobileNet v1 architecture, which is a type of convolutional neural network commonly used for image classification and recognition tasks, ema_mobilenet_v1_model: This is an exponential moving average (EMA) version of the mobilenet_v1_model, which is used to stabilize the training process and improve the accuracy of the model. The EMA version of the model is updated during each training step and is used for evaluation and inference,
        #train_prefetcher: This is likely a data loader or iterator that provides batches of training data to the model during each training epoch.
        #pixel_criterion: This is the loss function used to evaluate the difference between the predicted outputs of the model and the true labels of the training data
        #optimizer: This is the optimization algorithm used to update the model parameters during training
        #epoch: This is the current epoch of training. An epoch refers to a full pass through the training dataset
        #scala: used to perform automatic mixed precision training to speed up the training process and reduce memory usage
        acc1 = validate(ema_mobilenet_v1_model, valid_prefetcher, epoch, writer, "Valid") ##The validate function likely evaluates the ema_mobilenet_v1_model on the validation dataset provided by valid_prefetcher, computes some performance metrics such as accuracy, and logs the results using writer. The function then returns the accuracy value, which is assigned to the variable acc1 for further use.
        print("\n")

        # Update LR
        scheduler.step()  ##updates the learning rate according to the policy defined by the scheduler

        # Automatically save the model with the highest index
        is_best = acc1 > best_acc1  ##check if is the best acc
        is_last = (epoch + 1) == config.epochs  ##check if it is the last
        best_acc1 = max(acc1, best_acc1)  ##extract the max acc
        save_checkpoint({"epoch": epoch + 1,  ##save the epoch
                         "best_acc1": best_acc1,  ##the best acc
                         "state_dict": mobilenet_v1_model.state_dict(),  ##the dictionary state
                         "ema_state_dict": ema_mobilenet_v1_model.state_dict(),  ##the ema_mobilenet_v1_model.state_dict()
                         "optimizer": optimizer.state_dict(),  ##the optimizer
                         "scheduler": scheduler.state_dict()},  ##the sceduler
                        f"epoch_{epoch + 1}.pth.tar",  ##creates a string that contains the current epoch number plus 1 as a part of the string
                        samples_dir,  ##samples directory
                        results_dir,  ##results directory
                        is_best,  ##if the acc is the best
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:  ## function returns a list of two objects, both of which are instances of the CUDAPrefetcher class
    # Load train, test and valid datasets
    train_dataset = ImageDataset(config.train_image_dir,  ##train image directory
                                 config.image_size, ##image size
                                 config.model_mean_parameters, ##model mean parameters
                                 config.model_std_parameters, ##model standard deviation parameters
                                 "Train") ##type to be Train
    valid_dataset = ImageDataset(config.valid_image_dir, ##train image directory
                                 config.image_size,
                                 config.model_mean_parameters,
                                 config.model_std_parameters,
                                 "Valid")

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset, ##the train dataset
                                  batch_size=config.batch_size, ##initialize the batch size
                                  shuffle=True, ##do shuffle
                                  num_workers=config.num_workers, ##the number of worker processes to use for data loading in a PyTorch or TensorFlow DataLoader
                                  pin_memory=True, ## is a useful option to optimize data transfer between CPU and GPU during training, especially for large datasets or complex preprocessing steps
                                  drop_last=True, ##tells PyTorch to drop the last incomplete batch if the dataset size is not evenly divisible by the batch size.
                                  persistent_workers=True) ##argument tells PyTorch to keep the worker processes running even after a data loading operation has completed.
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device) ##creates a CUDAPrefetcher object to prefetch batches of data from the train_dataloader PyTorch DataLoader. The config.device argument specifies the device to use for training, which is typically a GPU
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device) ##valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device) creates a CUDAPrefetcher object to prefetch batches of data from the valid_dataloader PyTorch DataLoader. The config.device argument specifies the device to use for validation, which is typically a GPU

    return train_prefetcher, valid_prefetcher


def build_model() -> [nn.Module, nn.Module]:  ##is used to create and configure the neural network model(s) that will be used for training or validation. The function returns two nn.Module objects: one for the main model and one for the auxiliary model
    mobilenet_v1_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)  ##creates an instance of a MobileNet_v1 neural network model.
    mobilenet_v1_model = mobilenet_v1_model.to(device=config.device, memory_format=torch.channels_last) ##moves the mobilenet_v1_model instance to a specified device (e.g. CPU or GPU) for computation and changes the memory layout format to torch.channels_last.

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter ##efines a lambda function that calculates an exponentially weighted moving average (EMA) of two sets of model parameters
    ema_mobilenet_v1_model = AveragedModel(mobilenet_v1_model, avg_fn=ema_avg) ##creates a new model instance called ema_mobilenet_v1_model by wrapping an existing model mobilenet_v1_model with an averaged model

    return mobilenet_v1_model, ema_mobilenet_v1_model ##return those 2 models


def define_loss() -> nn.CrossEntropyLoss: ##used to define the loss function that will be used during training to optimize the model parameters. The returned nn.CrossEntropyLoss instance will be used to compute the loss between the model's predicted outputs and the ground truth labels during each training iteration.
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing) ##label_smoothing parameter specifies the amount of label smoothing to apply, which is a hyperparameter that controls the amount of noise added to the ground truth labels, criterion object is used as the loss function during training to optimize the model parameters
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last) ##moves the criterion object to a specific device and sets the memory format to torch.channels_last, memory_format parameter specifies the layout of the tensor's memory. In this case, torch.channels_last specifies that the memory layout should be optimized for data with multiple channels, where the channels are stored in the last dimension of the tensor

    return criterion ##return the criterion


def define_optimizer(model) -> optim.SGD: ##SGD = stochastic gradient descent (SGD) optimization algorithm.
    optimizer = optim.SGD(model.parameters(), ##the list of parameters that need to be updated by the optimizer
                          lr=config.model_lr, ## the learning rate for the optimizer. It determines the size of the step taken for each update.
                          momentum=config.model_momentum, ##the momentum factor for the optimizer. It determines how much the optimizer relies on the previous update direction when computing the current update direction
                          weight_decay=config.model_weight_decay) ## It penalizes large weights in the model by adding a regularization term to the loss function

    return optimizer


def define_scheduler(optimizer: optim.SGD) -> lr_scheduler.CosineAnnealingWarmRestarts: ##takes the optimizer as an argument and returns a cosine annealing warm restarts scheduler, which is a commonly used scheduler.  scheduler in deep learning is to adjust the learning rate of the optimizer during training
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, ##optimizer object whose learning rate is being scheduled
                                                         config.lr_scheduler_T_0, ##number of iterations before the learning rate is restarted
                                                         config.lr_scheduler_T_mult, ## A factor by which the number of iterations is multiplied after each restart
                                                         config.lr_scheduler_eta_min) ##minimum value of the learning rate after the restart.

    return scheduler


def train( ## trains a given neural network model on a training dataset for one epoch
        model: nn.Module, ##the neural network model to train
        ema_model: nn.Module, ## the exponential moving average (EMA) of the model weights
        train_prefetcher: CUDAPrefetcher, ##the data loader for the training dataset
        criterion: nn.CrossEntropyLoss, ##the loss function to optimize
        optimizer: optim.Adam, ## the optimization algorithm used to update the model parameters
        epoch: int, ## the current epoch number, In machine learning, an epoch refers to one complete iteration over the entire dataset used for training a mode
        scaler: amp.GradScaler, ## the AMP GradScaler object for mixed precision training
        writer: SummaryWriter ##the tensorboard SummaryWriter object for logging
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher) ##extract the length of the train_prefetcher
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f") ##an average meter that measures the time it takes to process each batch during training
    data_time = AverageMeter("Data", ":6.3f") ##n average meter that measures the time it takes to load each batch of data during training
    losses = AverageMeter("Loss", ":6.6f") ##an average meter that keeps track of the average loss over the course of the epoch
    acc1 = AverageMeter("Acc@1", ":6.2f") ##an average meter that keeps track of the top-1 accuracy over the course of the epoch
    acc5 = AverageMeter("Acc@5", ":6.2f") ## average meter that keeps track of the top-5 accuracy over the course of the epoch
    progress = ProgressMeter(batches, ## a utility class that displays the progress of the training process in a user-friendly way, by showing the epoch number, batch number, loss, and accuracy
                             [batch_time, data_time, losses, acc1, acc5],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train() ##Put the generative network model in training mode

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0 ##Initialize the number of data batches to print logs on the terminal

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset() ##reset the prefetcher so that it starts reading batches from the beginning of the training data
    batch_data = train_prefetcher.next() ##returns the next batch of data from the training data that was previously loaded into memory by the prefetcher

    # Get the initialization training time
    end = time.time()

    while batch_data is not None: ##nsures that the loop continues until all batches of data have been processed
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True) ##oves the input images from the batch data dictionary to the specified device (e.g., GPU), while also specifying that the tensor should use the memory format torch.channels_last, which can potentially improve performance. The non_blocking parameter is set to True to allow for asynchronous data loading.
        target = batch_data["target"].to(device=config.device, non_blocking=True)  ##batch_data["target"] returns a tensor object representing the target labels for this batch. The to() method in PyTorch is used to move this tensor object to a specific device (in this case, config.device) and is made non-blocking by setting non_blocking=True. This method is used to enable efficient data transfer to a GPU device and can help to improve the training performance

        # Get batch size
        batch_size = images.size(0)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True) ##sets all gradients to None. gradients are used to optimize the weights of a neural network during the training process. Gradients represent the slope of the loss function with respect to the model parameters, which indicates the direction in which the weights should be adjusted to minimize the loss. The optimizer uses the gradients to update the weights at each iteration of the training process, moving the model closer to the optimal solution.

        # Mixed precision training
        with amp.autocast(): ##allows for automatic mixed precision (AMP) training. It uses a combination of single-precision and half-precision floating-point numbers to speed up the training process while maintaining the desired level of numerical precision
            output = model(images) ##enerate output for the given input images
            loss = config.loss_weights * criterion(output, target) ##output is then compared to the target using the specified loss function criterion. The computed loss is then multiplied by the loss weight specified in the configuration file config.loss_weights

        # Backpropagation
        scaler.scale(loss).backward() ##scaler.scale method is used to scale the loss value, and the .backward() method is called on the scaled loss to compute the gradients. The gradients will be accumulated in the model's parameter tensors.
        # update generator weights
        scaler.step(optimizer) ##performs the optimizer step using the scaled gradients computed during the forward-backward pass
        scaler.update() ##updates the loss scale for the next iteration based on whether the gradients overflowed or underflowed the available precision, so that the gradients don't become too small or too large and lead to numerical instability

        # Update EMA
        ema_model.update_parameters(model) ##updates the parameters of the Exponential Moving Average (EMA) model with the parameters of the current model

        # measure accuracy and record loss
        top1, top5 = accuracy(output, target, topk=(1, 5)) ##calculates both top-1 and top-5 accuracy given the model's output and the ground truth target labels. The output tensor contains the predicted probabilities for each class, while the target tensor contains the true class labels encoded as integers. The topk argument is a tuple indicating the values of k to use for the top-k accuracy calculation. For example, topk=(1, 5) means that the function should return both top-1 and top-5 accuracy.
        losses.update(loss.item(), batch_size) ##updates the loss object with the current batch's loss value (obtained by calling loss.item()) and the batch size. The batch size is used to weight the average loss calculation because the loss is calculated as a mean over the batch size
        acc1.update(top1[0].item(), batch_size) ##update the average accuracy with the current batch's top1 accuracy
        acc5.update(top5[0].item(), batch_size) ##update the average accuracy with the current batch's top5 accuracy
    # a batch is a set of samples that are processed together in a single forward and backward pass during training, An epoch, on the other hand, is a complete pass through the entire training dataset. It is composed of multiple iterations over batches of data, where the model is trained to minimize the loss on the training dataset.

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end) ##calculate and store the time it takes to process a batch of data.
        end = time.time() ##get the end time

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1) ## adds the value of the current training loss to the TensorBoard summary writer, which is used for visualization purposes. first argument of the function specifies the name of the scalar that will be displayed in the TensorBoard. In this case, it is "Train/Loss", which means that it represents the training loss. second argument is the actual value of the loss that is computed during the current training iteration. 3rd argument used to define the global step number, which is used by TensorBoard to keep track of the training progress over time
            progress.display(batch_index + 1) ##a method to display the progress of the current epoch during training

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

    with torch.no_grad(): ## is a context manager that disables gradient computation
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


if __name__ == "__main__": ##the script is executed directly as the main program
    main() ##execute the main program
