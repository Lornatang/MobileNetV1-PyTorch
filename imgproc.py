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


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor: ##define image_to_tensor method with image: a NumPy array that represents an image, range_norm a boolean flag indicating whether to normalize the pixel values of the image to a certain range. If this flag is set to True, the function will normalize the pixel values of the image to the range [0, 1]. If it is set to False, the function will not normalize the pixel values, half: a boolean flag indicating whether to convert the pixel values to half-precision floating-point format
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
    tensor = F_vision.to_tensor(image) ##convert the image to a tensor. PyTorch tensor has the same shape as the image

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm: ##if you should scale the image
        tensor = tensor.mul(2.0).sub(1.0) ##multiplies all elements of the tensor by 2.0 and subtracts 1.0 from all elements of the tensor

    # Convert torch.float32 image data type to torch.half image data type
    if half: ##if you should do the conversion
        tensor = tensor.half() ##Convert torch.float32 image data type to torch.half image data type. In deep learning applications, it is often necessary to perform computations on large tensors with many elements. Using half-precision floating point format can significantly reduce the amount of memory needed to store the tensor, allowing for larger models and more efficient computations. However, half-precision floating point format has lower precision than single-precision floating point format, which can result in some loss of accuracy.

    return tensor


def tensor_to_image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any: ##define a method wich convert a tensor to an image
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
    if range_norm: ##if you should do the scale
        tensor = tensor.add(1.0).div(2.0) ##add 1 to all ements and divide all of them by 2

    # Convert torch.float32 image data type to torch.half image data type
    if half: ##if you should do the conversion
        tensor = tensor.half() ##Convert torch.float32 image data type to torch.half image data type

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8") ##transforms a PyTorch tensor tensor back into an image in the form of a NumPy array image
    #squeeze(0) removes the first dimension of the tensor (if it exists), which is often the batch dimension
    #permute(1, 2, 0) reorders the dimensions of the tensor so that the color channel is the last dimension
    #clamp(0, 255) clips the pixel values to the range [0, 255] to ensure that they are within the valid range for an 8-bit image
    #.cpu() moves the tensor from the GPU (if it was on the GPU) to the CPU. This is necessary if the tensor was created and manipulated on the GPU, but the subsequent processing requires the tensor to be on the CPU.
    #.numpy() converts the tensor to a NumPy array.
    #.astype("uint8") changes the data type of the array to uint8
    #In machine learning and deep learning, a batch size is the number of samples or examples that are processed together in a single forward/backward pass of a neural network during training.
    return image


def center_crop( ##crops an image to its center region
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        patch_size: int,
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    #The size of the tensor is usually represented in the following format: [batch_size, num_channels, height, width]
    if input_type == "Tensor": ##if input_type is a Tensor
        image_height, image_width = images[0].size()[-2:] ##extract the last two dimensions of the tensor.
    else:
        image_height, image_width = images[0].shape[0:2] ##images[0].shape returns a tuple of integers representing the dimensions of the input numpy array. In this case, [0:2] is used to extract the first two dimensions of the array.
    #The dimensions of the numpy array are usually represented in the following format: [height, width, num_channels]

    # Calculate the start indices of the crop
    top = (image_height - patch_size) // 2 ##The value of image_height is the height of the input image, while patch_size is the desired size of the cropped patch. Subtracting patch_size from image_height gives the maximum vertical position of the crop. To center the crop, we need to take half of the remaining space above and below the crop. We do this by dividing the difference by 2 and rounding down using the // operator
    left = (image_width - patch_size) // 2 ## calculates the starting index for the crop along the horizontal axis of the input image to center the crop

    # Crop lr image patch
    if input_type == "Tensor": ##if input type is Tensor
        images = [image[ ##slices the tensor to extract the desired crop region
                  :,
                  :,
                  top:top + patch_size, ##slicing operation selects all channels :, all rows :, and a range of columns from top to top + patch_size - 1
                  left:left + patch_size] for image in images] ## and a range of rows from left to left + patch_size - 1 and uses list comprehention
    else: #NumPy array
        images = [image[ ##The slicing operation
                  top:top + patch_size, ##selects a range of rows from top to top + patch_size - 1
                  left:left + patch_size, ##a range of columns from left to left + patch_size - 1
                  ...] for image in images] ##and all channels using ... and use a list comprehension

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_crop( ##do random crop
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
        images = [image[ ##same as above
                  :,
                  :,
                  top:top + patch_size,
                  left:left + patch_size] for image in images]
    else:
        images = [image[ ##same as above
                  top:top + patch_size,
                  left:left + patch_size,
                  ...] for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_rotate( ##a method to rotate an image
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        angles: list,
        center: tuple = None,
        rotate_scale_factor: float = 1.0
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Random select specific angle
    angle = random.choice(angles)

    if not isinstance(images, list): ##If images is not already a list, it's converted to a list to make it easier to handle
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy" ##If it's a PyTorch tensor, input_type is set to "Tensor", otherwise it's set to "Numpy"

    if input_type == "Tensor": ##if input_type is a tensor
        image_height, image_width = images[0].size()[-2:] ##extract height and width
    else:
        image_height, image_width = images[0].shape[0:2] ##extract height and width

    # Rotate LR image
    if center is None: ##If the center parameter is not provided,
        center = (image_width // 2, image_height // 2) ##the center of rotation is set to the center of the input image

    matrix = cv2.getRotationMatrix2D(center, angle, rotate_scale_factor) ##rotation matrix

    if input_type == "Tensor": ##if input_type is a tensor
        images = [F_vision.rotate(image, angle, center=center) for image in images] ##rotate all images
    else:
        images = [cv2.warpAffine(image, matrix, (image_width, image_height)) for image in images] ##rotate all images if input_type is Numpy arrays

    # When image number is 1
    if len(images) == 1: ##If the length of the output images list is 1, it's converted to a single image and returned. Otherwise, the list of images is returned as is.
        images = images[0]

    return images


def random_horizontally_flip( ##a method to random horizontally flip the image
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


def random_vertically_flip( ##a method to random vertically flip an image
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
