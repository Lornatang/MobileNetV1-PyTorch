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
from torch import Tensor
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation

__all__ = [
    "MobileNetV1",
    "DepthWiseSeparableConv2d",
    "mobilenet_v1",
]


class MobileNetV1(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
    ) -> None:
        super(MobileNetV1, self).__init__()
        self.features = nn.Sequential(
            Conv2dNormActivation(3,
                                 32,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1,
                                 norm_layer=nn.BatchNorm2d,
                                 activation_layer=nn.ReLU,
                                 inplace=True,
                                 bias=False,
                                 ),

            DepthWiseSeparableConv2d(32, 64, 1),
            DepthWiseSeparableConv2d(64, 128, 2),
            DepthWiseSeparableConv2d(128, 128, 1),
            DepthWiseSeparableConv2d(128, 256, 2),
            DepthWiseSeparableConv2d(256, 256, 1),
            DepthWiseSeparableConv2d(256, 512, 2),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 1024, 2),
            DepthWiseSeparableConv2d(1024, 1024, 1),
        )

        self.avgpool = nn.AvgPool2d((7, 7))

        self.classifier = nn.Linear(1024, num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)


class DepthWiseSeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DepthWiseSeparableConv2d, self).__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            Conv2dNormActivation(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 padding=1,
                                 groups=in_channels,
                                 norm_layer=norm_layer,
                                 activation_layer=nn.ReLU,
                                 inplace=True,
                                 bias=False,
                                 ),
            Conv2dNormActivation(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 norm_layer=norm_layer,
                                 activation_layer=nn.ReLU,
                                 inplace=True,
                                 bias=False,
                                 ),

        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)

        return out


def mobilenet_v1(**kwargs: Any) -> MobileNetV1:
    model = MobileNetV1(**kwargs)

    return model
