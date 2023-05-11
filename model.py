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
from typing import Callable, Any, Optional

import torch
from torch import Tensor ##imports Tensor class from torch module
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation##imports Conv2dNormActivation class from torchvision.ops.misc Conv2dNormActivation is a PyTorch module in the torchvision.ops.misc package that performs a convolutional operation with normalization and activation

__all__ = [ ##used to define a public interface of a module
    "MobileNetV1",
    "DepthWiseSeparableConv2d",
    "mobilenet_v1",
]


class MobileNetV1(nn.Module):##this shows that MobileNetV1 is a subclass of the class nn.Module

    def __init__(
            self,
            num_classes: int = 1000,##initialise the parameter num_classes with value 1000 inside the constructor
    ) -> None:
        super(MobileNetV1, self).__init__()##calls the constructor of the nn.Module class to initialise the instance of MobileNetV1 class
        self.features = nn.Sequential(##initialise the features using nn.Sequential container
            Conv2dNormActivation(3,##creates an instance of Conv2dNormActivation where 3 will be the number of input channels for the convolutional layer
                                 32,##initialise the number of output channels for the convolution layer
                                 kernel_size=3,##initialise the size of the convolution kernel
                                 stride=2,##initialise the stride of the convolution operation. In the context of convolutional neural networks (CNNs), the stride of a convolution operation refers to the number of pixels that the kernel or filter is shifted by in each step while traversing the input image
                                 padding=1,##initialise the amount of zero-padding to add to the input before the convolution operation is applied
                                 norm_layer=nn.BatchNorm2d,##initialise the normalization layer used after the convolutional operation.  the batch normalization operation will be used after the convolution operation on a 2D input tensor. BatchNorm2d is a module that normalizes the output of the previous layer by subtracting the batch mean and dividing by the batch standard deviation
                                 activation_layer=nn.ReLU,##initialise the activation function applied to the output of the normalization layer. ) activation function object from the torch.nn module. ReLU is a commonly used activation function in deep learning which applies an element-wise operation to the input tensor, setting any negative values to zero and leaving positive values unchanged
                                 inplace=True,##specifies to perform the activation function "in-place". that the activation function will modify the input tensor in-place, without allocating a new output tensor. This can help save memory by reusing the memory occupied by the input tensor.
                                 bias=False,##specifies to not include a bias term in the convolutional layer. A bias term is a learnable parameter in a neural network that is added to the weighted sum of inputs to a neuron, before passing it through an activation function
                                 ),
            #The DepthWiseSeparableConv2d(64, 128, 2) represents a depthwise separable convolution layer in a neural network, which applies two types of convolution operations to the input data. The first operation is a depthwise convolution, which applies a single filter to each input channel separately. The second operation is a pointwise convolution, which applies a 1x1 filter to combine the outputs of the depthwise convolution.
            DepthWiseSeparableConv2d(32, 64, 1),##reates a depthwise separable convolutional layer with 32 input channels, 64 output channels and a kernel size of 1x1
            DepthWiseSeparableConv2d(64, 128, 2),##reates a depthwise separable convolutional layer with 64 input channels, 128 output channels and a kernel size of 2x2
            DepthWiseSeparableConv2d(128, 128, 1),##reates a depthwise separable convolutional layer with 128 input channels, 128 output channels and a kernel size of 1x1
            DepthWiseSeparableConv2d(128, 256, 2),##reates a depthwise separable convolutional layer with 128 input channels, 256 output channels and a kernel size of 2x2
            DepthWiseSeparableConv2d(256, 256, 1),##reates a depthwise separable convolutional layer with 256 input channels, 256 output channels and a kernel size of 1x1
            DepthWiseSeparableConv2d(256, 512, 2),##reates a depthwise separable convolutional layer with 256 input channels, 512 output channels and a kernel size of 2x2
            DepthWiseSeparableConv2d(512, 512, 1),##reates a depthwise separable convolutional layer with 512 input channels, 512 output channels and a kernel size of 1x1
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 1024, 2),
            DepthWiseSeparableConv2d(1024, 1024, 1),
        )

        self.avgpool = nn.AvgPool2d((7, 7))##the average pooling operation is performed over a 7x7 window. The module applies a 2D kernel to the input tensor, computing the average value of each kernel-sized window of the input. The kernel size is specified as a tuple (kernel_size) and the stride (the step size of the kernel) is inferred from the kernel size.

        self.classifier = nn.Linear(1024, num_classes)##This line of code creates a linear layer (fully connected layer) in the neural network with an input size of 1024 and an output size of num_classes

        # Initialize neural network weights
        self._initialize_weights()##used to initialize the weights of the neural network. Initializing the weights is important because it can have a significant impact on the training of the model.

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)##called to compute the forward pass of the model

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:##define _forward_impl method which takes a tensor and return another tensor
        out = self.features(x)##computes the output feature map by passing the input tensor x through the convolutional and pooling layers defined in the features module
        out = self.avgpool(out)##performs average pooling operation on the input tensor. The size of the output tensor of self.features(x) is expected to be B x C x H x W (where B is batch size, C is number of channels, H is height and W is width) and self.avgpool applies average pooling operation on each channel of size (7,7). As a result, the output tensor is reduced to a size of B x C x 1 x 1, where 1 x 1 is the spatial dimension of the output tensor.
        out = torch.flatten(out, 1)##torch.flatten is a PyTorch function that takes an input tensor and returns a flattened version of the tensor. The second argument to the function indicates the starting dimension from where the flattening should begin. In this case, torch.flatten(out, 1) takes the out tensor and flattens all the dimensions except the first one
        out = self.classifier(out)##self.classifier is a linear layer that takes in the flattened feature tensor out and maps it to the number of classes in the dataset.

        return out

    def _initialize_weights(self) -> None:##initializes the weights of the neural network using different initialization methods depending on the type of layer
        for module in self.modules():##This loops through all the layers in the neural network
            if isinstance(module, nn.Conv2d):##checks if the layer is a 2D convolutional layer
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")##initializes the weights of the convolutional layer using the Kaiming initialization method with a normal distribution. The mode parameter specifies the distribution of the weights, and nonlinearity specifies the activation function.  In fan-out mode, the standard deviation is computed as sqrt(2.0 / fan_out), which is used to initialize the weights.
                if module.bias is not None:##checks if the layer has a bias term
                    nn.init.zeros_(module.bias)##initializes the bias term to zeros.
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):##checks if the layer is a batch normalization or group normalization layer
                nn.init.ones_(module.weight)## initializes the weights of the normalization layer to ones
                nn.init.zeros_(module.bias)## initializes the bias term of the normalization layer to zeros
            elif isinstance(module, nn.Linear):##checks if the layer is a linear layer.
                nn.init.normal_(module.weight, 0, 0.01)##initializes the weights of the linear layer using a normal distribution with a mean of zero and standard deviation of 0.01.
                nn.init.zeros_(module.bias)##initializes the bias term of the linear layer to zeros.


class DepthWiseSeparableConv2d(nn.Module):##defines a depth-wise separable convolution operation for 2D inputs
    def __init__( # It initializes the attributes of the module including the number of input and output channels, stride, and normalization layer
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None ##Optional is a type hinting class that indicates that the argument can be None or a value of the annotated type.Callable[..., nn.Module] is a type hint that specifies a function that takes any number of positional arguments and returns an nn.Module instance.... is a special syntax that indicates that the function can take any number of additional arguments, but their type is not specified
    ) -> None:
        super(DepthWiseSeparableConv2d, self).__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None: ##checks if the normalization layer argument is None
            norm_layer = nn.BatchNorm2d ##assigns nn.BatchNorm2d

        self.conv = nn.Sequential( ## is a container module in PyTorch which allows us to put multiple layers sequentially
            Conv2dNormActivation(in_channels, ##number of input channels to the convolutional layer.
                                 in_channels, ##same
                                 kernel_size=3, ##initialise kernel size with 3
                                 stride=stride,
                                 padding=1,
                                 groups=in_channels, ## number of groups in which the input channels are divided. This is used to perform depthwise separable convolutions.means that each input channel is handled separately as a group, which corresponds to a depthwise convolution operation.
                                 norm_layer=norm_layer, ##normalization layer used after the convolution.
                                 activation_layer=nn.ReLU, ##activation layer used after the normalization is ReLU
                                 inplace=True, ##This parameter specifies whether to modify the input tensor in place. When inplace is set to True, the input tensor is modified in place and the operation returns the modified tensor. When inplace is set to False (which is the default), the operation returns a new tensor and leaves the input tensor unchanged
                                 bias=False, ##there is no bias
                                 ),
            Conv2dNormActivation(in_channels,##number of input channels to the convolutional layer.
                                 out_channels,##number of output channels to the convolutional layer.
                                 kernel_size=1,##the kernel size is 1
                                 stride=1,##the stride is 1. A stride of 1 means that the filter slides over the input image one pixel at a time
                                 padding=0,##refers to the number of pixels added to the input tensor before applying the convolutional operation. In this case, the value of padding is 0, which means that no padding is added.
                                 norm_layer=norm_layer,##normalization layer used after the convolution.
                                 activation_layer=nn.ReLU,##activation layer used after the normalization is ReLU
                                 inplace=True,##This parameter specifies whether to modify the input tensor in place
                                 bias=False,##there is no bias
                                 ),

        )

    def forward(self, x: Tensor) -> Tensor: ##define forward method which takes a tensor and returns a tensor
        out = self.conv(x)##applies the two convolutional layers to the input tensor x and returns the resulting tensor out

        return out


def mobilenet_v1(**kwargs: Any) -> MobileNetV1:##It uses the "**kwargs" syntax to allow any number of keyword arguments to be passed to the function
    model = MobileNetV1(**kwargs) ##create an instance of MobileNetV1

    return model##return the model
