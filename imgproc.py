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
from typing import Any
from torch import Tensor
from numpy import ndarray
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F_vision

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "center_crop", "random_crop", "random_rotate", "random_vertically_flip", "random_horizontally_flip",
]


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor: ##converts a NumPy array representing an image to a PyTorch tensor
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("example_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, False, False)

    """
    # Convert image data type to Tensor data type
    tensor = F_vision.to_tensor(image) ##converts the data type to tensor

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm: ##checks if in normal range
        tensor = tensor.mul(2.0).sub(1.0) #if it is then scale it

    # Convert torch.float32 image data type to torch.half image data type
    if half: ##if half is true
        tensor = tensor.half() #halve the tensor

    return tensor


def tensor_to_image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any: #convert the tensor into an image datatype
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_tensor = torch.randn([1,3, 256, 256], dtype=torch.float)
        >>> example_image = tensor_to_image(example_tensor, False, False)

    """
    # Scale the image data from [-1, 1] to [0, 1]
    if range_norm: #checks if in normal range
        tensor = tensor.add(1.0).div(2.0) #if it is then scale it

    # Convert torch.float32 image data type to torch.half image data type
    if half: #if half is true
        tensor = tensor.half() #halve the tensor

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8") #construct the image

    return image


def center_crop( ##undefined
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        patch_size: int,
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if input_type == "Tensor": #if it is a tensor
        image_height, image_width = images[0].size()[-2:] #then set the image with/height from the last two elements
    else:
        image_height, image_width = images[0].shape[0:2] #if it is not then set the shape

    # Calculate the start indices of the crop
    top = (image_height - patch_size) // 2 #calculates the top regarding patch size
    left = (image_width - patch_size) // 2 #calculates the left regarding patch size

    # Crop lr image patch
    if input_type == "Tensor": #if input is tensor
        images = [image[ #create a new tuple for the image
                  :,
                  :,
                  top:top + patch_size, #use the calculated top with patch size
                  left:left + patch_size] for image in images] #and left with patch size 
                  #do this for all images
    else:
        images = [image[ #if input is an image
                  top:top + patch_size, #initialize with the calculated top
                  left:left + patch_size, #and left
                  ...] for image in images] #for all images

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_crop( #create a random crop in image
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        patch_size: int,
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if input_type == "Tensor":
        image_height, image_width = images[0].size()[-2:]
    else:
        image_height, image_width = images[0].shape[0:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - patch_size)
    left = random.randint(0, image_width - patch_size)

    # Crop lr image patch
    if input_type == "Tensor":
        images = [image[ #creates the cropped images if tensor
                  :,
                  :,
                  top:top + patch_size,
                  left:left + patch_size] for image in images]
    else:
        images = [image[ #and if image
                  top:top + patch_size,
                  left:left + patch_size,
                  ...] for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_rotate( #rotates randomly
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        angles: list,
        center: tuple = None,
        rotate_scale_factor: float = 1.0
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Random select specific angle
    angle = random.choice(angles)

    if not isinstance(images, list): #if the image is not in list add it
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy" #detect the datatype of the image

    if input_type == "Tensor": #if it is tensor
        image_height, image_width = images[0].size()[-2:] #then set the image with/height from the last two elements
    else:
        image_height, image_width = images[0].shape[0:2] #set the shape if it is not tensor

    # Rotate LR image
    if center is None: #if center is not initialized
        center = (image_width // 2, image_height // 2) #calculate center

    matrix = cv2.getRotationMatrix2D(center, angle, rotate_scale_factor) #create rotation matrix

    if input_type == "Tensor": #if it is a tensor
        images = [F_vision.rotate(image, angle, center=center) for image in images] #rotate all images by a set angle
    else:
        images = [cv2.warpAffine(image, matrix, (image_width, image_height)) for image in images] #if not tensor warp

    # When image number is 1
    if len(images) == 1: #if only one image
        images = images[0]

    return images


def random_horizontally_flip( #random horizontal flip
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Get horizontal flip probability
    flip_prob = random.random()

    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            images = [F_vision.hflip(image) for image in images]
        else:
            images = [cv2.flip(image, 1) for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_vertically_flip( #random vertical flip
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Get vertical flip probability
    flip_prob = random.random()

    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            images = [F_vision.vflip(image) for image in images]
        else:
            images = [cv2.flip(image, 0) for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images
