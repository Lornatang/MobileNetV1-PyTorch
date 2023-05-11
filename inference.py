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
import argparse##imports argparse moduel  which will help to parse command line arguments
import json##imports json module which helps to do encode object to a JSON format or to decode JSON into an object
import os

import cv2
import torch
from PIL import Image##imports Image class from the PIL module in order to manipulate images
from torch import nn
from torchvision.transforms import Resize, ConvertImageDtype, Normalize

import imgproc
import model
from utils import load_state_dict

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def load_class_label(class_label_file: str, num_classes: int) -> list:##definition of a function which takes 2 parameters: a string class_label_file and an int num_classes and which returns a list
    class_label = json.load(open(class_label_file))##deserialize a json string using json.load, from the class_label_file which is oppend using open function
    class_label_list = [class_label[str(i)] for i in range(num_classes)]##for each i in the range [0, num_classes) creates a list based on the index i which is converted to a string (str(i) with which is obtained the value from the class_label directory class_label[str(i) and based on those values is created a list which is sotred in the class_label_list

    return class_label_list


def choice_device(device_type: str) -> torch.device:##is a function which has as argument as string and which returns  a specific device on which tensor computation will be performed
    # Select model processing equipment type
    if device_type == "cuda":##chef if the device_type is cuda
        device = torch.device("cuda", 0)##if yes creates a torch.device obj representing the first available GPU
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, model_num_classes: int, device: torch.device) -> [nn.Module, nn.Module]:##a method which takes 3 argumetns and  returns a tuple of two PyTorch objects that represent the built model. The first object is the actual model, and the second object is a reference to a potential loss function to be used with the model.
    mobilenet_v1_model = model.__dict__[model_arch_name](num_classes=model_num_classes)##model.__dict__ is a dictionary object that contains the attributes and methods of the model module. It allows us to access the desired model architecture by name, using the model_arch_name parameter
    mobilenet_v1_model = mobilenet_v1_model.to(device=device, memory_format=torch.channels_last)##mobilenet_v1_model.to(device=device, memory_format=torch.channels_last) is a method call to move the model to a specified device and memory format, device is a torch.device object representing the device (e.g. CPU or GPU) on which the model will be stored and operated, torch.channels_last specifies that the memory layout will be such that the channels (i.e. the color channels) will be stored in the last dimension of the tensor
    #MobileNetV1 is a type of deep convolutional neural network architecture designed for mobile and embedded vision applications
    return mobilenet_v1_model


def preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:##a method that takes 3 parameters and return a tensor
    image = cv2.imread(image_path)##read the image

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)##convert bgr image to RGB

    # OpenCV convert PIL
    image = Image.fromarray(image)

    # Resize to 224
    image = Resize([image_size, image_size])(image)##resize the image
    # Convert image data to pytorch format data
    tensor = imgproc.image_to_tensor(image, False, False).unsqueeze_(0)##imgproc has 3 paramters: image: the input image that needs to be converted to a tensor, bgr_mean: a boolean value that indicates if the mean values of the BGR channels should be subtracted from the image, normalize: a boolean value that indicates whether the pixel values of the image should be normalized to the range [0, 1] before converting it to a tensor, unsqueeze_() method is then called on the resulting tensor object to add a new dimension to the tensor at index 0, effectively creating a tensor batch of size 1.
    # Convert a tensor image to the given ``dtype`` and scale the values accordingly
    tensor = ConvertImageDtype(torch.float)(tensor)
    # Normalize a tensor image with mean and standard deviation.
    tensor = Normalize(args.model_mean_parameters, args.model_std_parameters)(tensor)##Normalize a tensor image with mean and standard deviation.

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)##device=device: specifies the device where the tensor should be moved to for computation, memory_format=torch.channels_last: specifies that the tensor should use a channels-last memory format, non_blocking=True: allows for non-blocking data transfers, which means the program can continue executing while the data is being transferred, potentially improving performance

    return tensor


def main():
    # Get the label name corresponding to the drawing
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes)##The load_class_label function reads the class label mapping from the file and returns a dictionary where the keys are the class IDs and the values are the corresponding class labels

    device = choice_device(args.device_type)##takes a string argument device_type and returns the corresponding PyTorch device object. The device object represents the device (CPU or GPU) on which the computation is executed

    # Initialize the model
    mobilenet_v1_model = build_model(args.model_arch_name, args.model_num_classes, device)##build the model
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    mobilenet_v1_model, _, _, _, _, _ = load_state_dict(mobilenet_v1_model, args.model_weights_path)##loads the pre-trained weights for the MobileNetV1 model from the specified file path and assigns the loaded model to the variable mobilenet_v1_model
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    mobilenet_v1_model.eval()##puts the model in evaluation mode, This is a necessary step before running inference because it tells PyTorch to disable certain layers or operations that are only used during training, such as dropout and batch normalization.During evaluation mode, PyTorch will not update the model's parameters, so the model will not learn anything new. Instead, it will simply use the trained parameters to make predictions based on the input data

    tensor = preprocess_image(args.image_path, args.image_size, device)##and uses it to preprocess an image before it is fed into a neural network for inference

    # Inference
    with torch.no_grad():##with torch.no_grad(): is a context manager that temporarily sets all the requires_grad flags to false. This means that no gradient computations will be performed in the forward pass of the model
        output = mobilenet_v1_model(tensor)##output = mobilenet_v1_model(tensor) passes the tensor tensor through the model mobilenet_v1_model to get the output tensor output. The no_grad() context is used here to save memory and computation time during inference as gradients do not need to be computed in this phase
    #Gradients refer to the derivative of a function with respect to its input parameters. In the context of deep learning, gradients are used to optimize the parameters of a neural network during the training process.

    # Calculate the five categories with the highest classification probability
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist()##squeeze(0) - This method removes the first dimension (which has size 1) of the tenso

    # Print classification results
    for class_index in prediction_class_index:##for each prediction class
        prediction_class_label = class_label_map[class_index]##get label from dictionary
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()##the output of the neural network represents the probability distribution over all the possible classes. Softmax is a function that takes this output and normalizes it into a probability distribution over all the classes, torch.softmax(output, dim=1) computes the softmax activation function over the output tensor, along the dim=1 axis (i.e., the classes axis). The resulting tensor contains the probabilities of each class. Then, prediction_class_prob is computed by selecting the probability of the class_index-th class from the resulting tensor using indexing and calling the .item() method to convert the tensor to a Python float.
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()##creates an argument parser object from the argparse module.
    parser.add_argument("--model_arch_name", type=str, default="mobilenet_v1")##add the architecture name
    parser.add_argument("--model_mean_parameters", type=list, default=[0.485, 0.456, 0.406])##the mean parameters
    parser.add_argument("--model_std_parameters", type=list, default=[0.229, 0.224, 0.225])##the standard deviation parameters
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")##the class label file
    parser.add_argument("--model_num_classes", type=int, default=1000)##number of classes
    parser.add_argument("--model_weights_path", type=str, default="./results/pretrained_models/MobileNetV1-ImageNet_1K.pth.tar")##the wights path
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")##image path
    parser.add_argument("--image_size", type=int, default=224)##image size
    parser.add_argument("--device_type", type=str, default="cpu", choices=["cpu", "cuda"])##device type
    args = parser.parse_args()

    main()
