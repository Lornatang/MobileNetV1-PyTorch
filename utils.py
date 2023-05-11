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
import shutil
from enum import Enum
from typing import Any, Dict, TypeVar

import torch
from torch import nn

__all__ = [
    "accuracy", "load_state_dict", "make_directory", "ovewrite_named_param", "save_checkpoint",
    "Summary", "AverageMeter", "ProgressMeter"
]

V = TypeVar("V")


def accuracy(output, target, topk=(1,)): ##define the accuracy method, output: The output predictions of the neural network for a batch of data. This is a tensor of size (batch_size, num_classes). topk: A tuple of integers specifying the top k accuracy values to calculate. topk=(1,) will calculate the top 1 accuracy
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():##this is a context manager that tells PyTorch to disable gradient calculation. This is because we are not interested in computing gradients for this function; we only want to compute the accuracy of the model's output.
        maxk = max(topk)##return the largest value of k in topk
        batch_size = target.size(0)##retrieves the size of the first dimension of the target tensor, which corresponds to the batch size.

        _, pred = output.topk(maxk, 1, True, True)##calculates the top-k predictions of the model's output tensor output. The topk method returns a tuple of two tensors: the first tensor contains the top-k values themselves, while the second tensor contains the indices of the corresponding elements in the input tensor. the last parameter being True means that the returned result will be transposed. Since the third parameter is also True, the returned tensors will be sorted in descending order
        pred = pred.t()##pred tensor is transposed, so that the k top predictions for each sample are in separate rows
        correct = pred.eq(target.view(1, -1).expand_as(pred))##check between two tensors and returns a boolean tensor of the same shape. target.view(1, -1) reshapes the target tensor to a row tensor, and expand_as(pred) broadcasts the target tensor to the same shape as pred. This allows for element-wise comparison between the predicted tensor and the ground truth tensor.

        results = []
        for k in topk:##iterates over each value of k in topk.
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)##correct is a binary tensor of shape (k, batch_size) where each element represents whether the corresponding example in the batch is correctly classified in the top-k predictions or not. We take the first k rows of this tensor using correct[:k] and reshape it into a 1D tensor of shape (k * batch_size) using reshape(-1). The sum(0, keepdim=True) function call then computes the sum of all elements in this 1D tensor, returning a tensor of shape (1, 1) representing the total number of correctly classified examples in the top-k predictions.
            results.append(correct_k.mul_(100.0 / batch_size))##this count is converted to a percentage using mul_(100.0 / batch_size) and stored in the results list
        return results##return the results


def load_state_dict(##define the method used to load a saved state dictionary into a model.
        model: nn.Module,
        model_weights_path: str,
        ema_model: nn.Module = None,
        start_epoch: int = None,
        best_acc1: float = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        load_mode: str = None,
) -> [nn.Module, nn.Module, str, int, float, torch.optim.Optimizer, torch.optim.lr_scheduler]:
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)##checkpoint is loaded from a previously saved model checkpoint using torch.load(), where model_weights_path is the path to the saved model. The map_location parameter specifies where the loaded data should be placed. In this case, map_location=lambda storage, loc: storage means that the storage for the loaded model should be placed in the default device, which is usually the GPU

    if load_mode == "resume":##checks if load_mode is resume
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()##returns a dictionary containing the model's weights and biases
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}##creates a new dictionary state_dict by iterating over the items of checkpoint["state_dict"] and keeping only the keys that are also present in the keys of the model_state_dict
        # Overwrite the model weights to the current model (base model)
        model_state_dict.update(state_dict)##update model_state_dict
        model.load_state_dict(model_state_dict)##used to load the state dictionary of a pre-trained model into a new model.
        # Load ema model state dict. Extract the fitted model weights
        ema_model_state_dict = ema_model.state_dict()
        ema_state_dict = {k: v for k, v in checkpoint["ema_state_dict"].items() if k in ema_model_state_dict.keys()}
        # Overwrite the model weights to the current model (ema model)
        ema_model_state_dict.update(ema_state_dict)
        ema_model.load_state_dict(ema_model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()##returns a dictionary containing the current state (values of all the parameters) of the PyTorch model
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()} ##creates a new dictionary state_dict by iterating through the items of the checkpoint["state_dict"] dictionary and selecting only those items whose keys exist in model_state_dict keys and have the same size as the corresponding tensors in model_state_dict
        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict)##update model state
        model.load_state_dict(model_state_dict)##used to load the state dictionary of a pre-trained model into a new model.

    return model, ema_model, start_epoch, best_acc1, optimizer, scheduler


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def save_checkpoint(
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        is_best: bool = False,
        is_last: bool = False,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name)
    torch.save(state_dict, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, "best.pth.tar"))
    if is_last:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, "last.pth.tar"))


class Summary(Enum):##define an enum
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):##define AverageMeter class which takes an object and calculate the average
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):##a method which restes the values
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):##methos which updates the values, renitialise the value, and do the average
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"## is creating a format string that includes placeholders for variables called "name," "val," and "avg."
        return fmtstr.format(**self.__dict__)

    def summary(self):##define a method wihch returns a formatting string
        if self.summary_type is Summary.NONE:##check if summary_type is NONE
            fmtstr = ""##initialize formatstring with empty
        elif self.summary_type is Summary.AVERAGE:##check if summary_type is AVERAGE
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:##check if summary_type is SUM
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:##check if summary_type is COUNT
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)

#a "meter" generally refers to an object or function that measures and reports on the performance of a model or system.
class ProgressMeter(object):##define ProgressMeter class which get the progress of a meter
    def __init__(self, num_batches, meters, prefix=""):##define the constructor
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)##By setting the "batch_fmtstr" attribute using the "_get_batch_fmtstr" method, the instance will have a pre-defined format string that can be used to display progress information for each batch during the iterative process
        self.meters = meters##initialize the meters
        self.prefix = prefix##initialize the prefix

    def display(self, batch):##define display method
        entries = [self.prefix + self.batch_fmtstr.format(batch)]##initialise the entries
        entries += [str(meter) for meter in self.meters]##create for each meter the string format which is concatenated to the entries
        print("\t".join(entries))

    def display_summary(self):##defin edisplay summary
        entries = [" *"]##initialise the entries
        entries += [meter.summary() for meter in self.meters]##create the entries by concatenating the summary for each meter
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
