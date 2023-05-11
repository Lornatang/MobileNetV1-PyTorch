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
from torch import Tensor #import tensor class from the pytorch library
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation #imports convenience class hat combines a 2D convolutional layer with normalization and activation layers

__all__ = [ #public interface of a module (list variable)
    "MobileNetV1",
    "DepthWiseSeparableConv2d",
    "mobilenet_v1",
]


class MobileNetV1(nn.Module): #mobilenet class definition

    def __init__(
            self,
            num_classes: int = 1000, #number of classes in the classification
    ) -> None:
        super(MobileNetV1, self).__init__() #calls the super class (nn.Module) init
        self.features = nn.Sequential( #container for sequential layers
            Conv2dNormActivation(3, #input channels
                                 32,#output channels
                                 kernel_size=3,# kernel size
                                 stride=2,#stride of the convolution
                                 padding=1,#padding added to all four sides of the input
                                 norm_layer=nn.BatchNorm2d,#norm layer that will be stacked on top of the convolution layer
                                 activation_layer=nn.ReLU,#activation function that will be stacked on top of the normalization layer
                                 inplace=True,#parameter for the activation layer, which can optionally do the operation in-place
                                 bias=False,#bias in the convolution layer
                                 ),

            #depthwise convolution layer that applies a single filter per input channel, 
            #followed by a pointwise convolution layer that combines the outputs of the depthwise convolution using 1x1 filters.
            #batch normalization and ReLU activation.
            DepthWiseSeparableConv2d(32, 64, 1), #input channels=32, output channels=64, stride=1
            DepthWiseSeparableConv2d(64, 128, 2),#input channels=64, output channels=128, stride=2
            DepthWiseSeparableConv2d(128, 128, 1),#input channels=128, output channels=128, stride=1
            DepthWiseSeparableConv2d(128, 256, 2),#input channels=128, output channels=256, stride=2
            DepthWiseSeparableConv2d(256, 256, 1),#input channels=256, output channels=256, stride=1
            DepthWiseSeparableConv2d(256, 512, 2),#input channels=256, output channels=512, stride=2
            DepthWiseSeparableConv2d(512, 512, 1),#input channels=512, output channels=512, stride=1
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 1024, 2),
            DepthWiseSeparableConv2d(1024, 1024, 1),
        )

        self.avgpool = nn.AvgPool2d((7, 7))#average pooling layer, reduces spatial dimensions of the output feature maps to 1x1 (applies a 7x7 kernel)

        self.classifier = nn.Linear(1024, num_classes)#defines a fully connected linear layer that takes as input the output tensor of the global average pooling layer and produces as output a tensor of shape

        # Initialize neural network weights
        self._initialize_weights()#initializes the weights

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)#creates a tensor by applying the forward impl function

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:#defines forward pass
        out = self.features(x)#pass through features module
        out = self.avgpool(out)#pass through avgpool layer
        out = torch.flatten(out, 1)#flatten to one-dimension
        out = self.classifier(out)#pass through classifier layer

        return out

    def _initialize_weights(self) -> None:#weight initialization function
        for module in self.modules():#loop through all modules
            if isinstance(module, nn.Conv2d):#if conv2d layer
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")#use kaiming initialization
                if module.bias is not None:#if bias is not turned off
                    nn.init.zeros_(module.bias)#init the bias module
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):#if batchnorm2d layer or groupnorm layer
                nn.init.ones_(module.weight)#init weights with one
                nn.init.zeros_(module.bias)#init bias with zeros
            elif isinstance(module, nn.Linear):#if linear layers
                nn.init.normal_(module.weight, 0, 0.01)#normal distribution mean of 0, std dev of 0.01
                nn.init.zeros_(module.bias)#init bias with zeros


class DepthWiseSeparableConv2d(nn.Module):#class definition of depthwiseseperableconv2d
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None #normalization layer used in the convolutional layer (None by default)
    ) -> None:
        super(DepthWiseSeparableConv2d, self).__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None: #if normalization layer is None
            norm_layer = nn.BatchNorm2d # set it to BatchNorm2d

        self.conv = nn.Sequential( #initializes the sequential block
            Conv2dNormActivation(in_channels, #nr of input channels
                                 in_channels, #nr of output channels
                                 kernel_size=3, #size of convolution kernel
                                 stride=stride,
                                 padding=1,
                                 groups=in_channels, #number of groups to split into
                                 norm_layer=norm_layer, #normalization layer to use
                                 activation_layer=nn.ReLU, #activation layer  set to ReLU
                                 inplace=True, #modify a tensor inplace or not
                                 bias=False, #use bias or not
                                 ),
            Conv2dNormActivation(in_channels,#nr of input channels
                                 out_channels,#nr of output channels
                                 kernel_size=1,#size of convolution kernel
                                 stride=1,#stride of convolution
                                 padding=0,#amount of padding around the input
                                 norm_layer=norm_layer,#normalization layer
                                 activation_layer=nn.ReLU,#actiavtion layer
                                 inplace=True,#inplace or not
                                 bias=False,#bias or not
                                 ),

        )

    def forward(self, x: Tensor) -> Tensor: #input tensor x returns a tensor passed through convolution kernel
        out = self.conv(x)#pass through conv kernel

        return out


def mobilenet_v1(**kwargs: Any) -> MobileNetV1:#returns an instance of MobileNetV1 class, takes keywordarguments
    model = MobileNetV1(**kwargs) #creates the model by parsing the keywordarguments

    return model#returns the model
