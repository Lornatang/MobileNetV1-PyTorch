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
from torch import Tensor ##undefined
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation##undefined

__all__ = [ ##undefined
    "MobileNetV1",
    "DepthWiseSeparableConv2d",
    "mobilenet_v1",
]


class MobileNetV1(nn.Module):##undefined

    def __init__(
            self,
            num_classes: int = 1000,##undefined
    ) -> None:
        super(MobileNetV1, self).__init__()##undefined
        self.features = nn.Sequential(##undefined
            Conv2dNormActivation(3,##undefined
                                 32,##undefined
                                 kernel_size=3,##undefined
                                 stride=2,##undefined
                                 padding=1,##undefined
                                 norm_layer=nn.BatchNorm2d,##undefined
                                 activation_layer=nn.ReLU,##undefined
                                 inplace=True,##undefined
                                 bias=False,##undefined
                                 ),

            DepthWiseSeparableConv2d(32, 64, 1),##undefined
            DepthWiseSeparableConv2d(64, 128, 2),##undefined
            DepthWiseSeparableConv2d(128, 128, 1),##undefined
            DepthWiseSeparableConv2d(128, 256, 2),##undefined
            DepthWiseSeparableConv2d(256, 256, 1),##undefined
            DepthWiseSeparableConv2d(256, 512, 2),##undefined
            DepthWiseSeparableConv2d(512, 512, 1),##undefined
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 1024, 2),
            DepthWiseSeparableConv2d(1024, 1024, 1),
        )

        self.avgpool = nn.AvgPool2d((7, 7))##undefined

        self.classifier = nn.Linear(1024, num_classes)##undefined

        # Initialize neural network weights
        self._initialize_weights()##undefined

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)##undefined

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:##undefined
        out = self.features(x)##undefined
        out = self.avgpool(out)##undefined
        out = torch.flatten(out, 1)##undefined
        out = self.classifier(out)##undefined

        return out

    def _initialize_weights(self) -> None:##undefined
        for module in self.modules():##undefined
            if isinstance(module, nn.Conv2d):##undefined
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")##undefined
                if module.bias is not None:##undefined
                    nn.init.zeros_(module.bias)##undefined
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):##undefined
                nn.init.ones_(module.weight)##undefined
                nn.init.zeros_(module.bias)##undefined
            elif isinstance(module, nn.Linear):##undefined
                nn.init.normal_(module.weight, 0, 0.01)##undefined
                nn.init.zeros_(module.bias)##undefined


class DepthWiseSeparableConv2d(nn.Module):##undefined
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None ##undefined
    ) -> None:
        super(DepthWiseSeparableConv2d, self).__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None: ##undefined
            norm_layer = nn.BatchNorm2d ##undefined

        self.conv = nn.Sequential( ##undefined
            Conv2dNormActivation(in_channels, ##undefined
                                 in_channels, ##undefined
                                 kernel_size=3, ##undefined
                                 stride=stride,
                                 padding=1,
                                 groups=in_channels, ##undefined
                                 norm_layer=norm_layer, ##undefined
                                 activation_layer=nn.ReLU, ##undefined
                                 inplace=True, ##undefined
                                 bias=False, ##undefined
                                 ),
            Conv2dNormActivation(in_channels,##undefined
                                 out_channels,##undefined
                                 kernel_size=1,##undefined
                                 stride=1,##undefined
                                 padding=0,##undefined
                                 norm_layer=norm_layer,##undefined
                                 activation_layer=nn.ReLU,##undefined
                                 inplace=True,##undefined
                                 bias=False,##undefined
                                 ),

        )

    def forward(self, x: Tensor) -> Tensor: ##undefined
        out = self.conv(x)##undefined

        return out


def mobilenet_v1(**kwargs: Any) -> MobileNetV1:##undefined
    model = MobileNetV1(**kwargs) ##undefined

    return model##undefined
