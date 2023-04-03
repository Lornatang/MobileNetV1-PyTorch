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
import argparse##undefined
import json##undefined
import os

import cv2
import torch
from PIL import Image##undefined
from torch import nn
from torchvision.transforms import Resize, ConvertImageDtype, Normalize

import imgproc
import model
from utils import load_state_dict

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def load_class_label(class_label_file: str, num_classes: int) -> list:##undefined
    class_label = json.load(open(class_label_file))##undefined
    class_label_list = [class_label[str(i)] for i in range(num_classes)]##undefined

    return class_label_list


def choice_device(device_type: str) -> torch.device:##undefined
    # Select model processing equipment type
    if device_type == "cuda":##undefined
        device = torch.device("cuda", 0)##undefined
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, model_num_classes: int, device: torch.device) -> [nn.Module, nn.Module]:##undefined
    mobilenet_v1_model = model.__dict__[model_arch_name](num_classes=model_num_classes)##undefined
    mobilenet_v1_model = mobilenet_v1_model.to(device=device, memory_format=torch.channels_last)##undefined

    return mobilenet_v1_model


def preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:##undefined
    image = cv2.imread(image_path)##undefined

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)##undefined

    # OpenCV convert PIL
    image = Image.fromarray(image)

    # Resize to 224
    image = Resize([image_size, image_size])(image)##undefined
    # Convert image data to pytorch format data
    tensor = imgproc.image_to_tensor(image, False, False).unsqueeze_(0)##undefined
    # Convert a tensor image to the given ``dtype`` and scale the values accordingly
    tensor = ConvertImageDtype(torch.float)(tensor)
    # Normalize a tensor image with mean and standard deviation.
    tensor = Normalize(args.model_mean_parameters, args.model_std_parameters)(tensor)##undefined

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)##undefined

    return tensor


def main():
    # Get the label name corresponding to the drawing
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes)##undefined

    device = choice_device(args.device_type)##undefined

    # Initialize the model
    mobilenet_v1_model = build_model(args.model_arch_name, args.model_num_classes, device)##undefined
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    mobilenet_v1_model, _, _, _, _, _ = load_state_dict(mobilenet_v1_model, args.model_weights_path)##undefined
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    mobilenet_v1_model.eval()##undefined

    tensor = preprocess_image(args.image_path, args.image_size, device)##undefined

    # Inference
    with torch.no_grad():##undefined
        output = mobilenet_v1_model(tensor)##undefined

    # Calculate the five categories with the highest classification probability
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist()##undefined

    # Print classification results
    for class_index in prediction_class_index:##undefined
        prediction_class_label = class_label_map[class_index]##undefined
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()##undefined
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()##undefined
    parser.add_argument("--model_arch_name", type=str, default="mobilenet_v1")##undefined
    parser.add_argument("--model_mean_parameters", type=list, default=[0.485, 0.456, 0.406])##undefined
    parser.add_argument("--model_std_parameters", type=list, default=[0.229, 0.224, 0.225])##undefined
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")##undefined
    parser.add_argument("--model_num_classes", type=int, default=1000)##undefined
    parser.add_argument("--model_weights_path", type=str, default="./results/pretrained_models/MobileNetV1-ImageNet_1K.pth.tar")##undefined
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")##undefined
    parser.add_argument("--image_size", type=int, default=224)##undefined
    parser.add_argument("--device_type", type=str, default="cpu", choices=["cpu", "cuda"])##undefined
    args = parser.parse_args()

    main()
